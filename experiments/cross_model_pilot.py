"""Prepare and run a serial Qwen -> Llama prompt migration pilot.

The default stage only freezes a plan and prompt inputs. Execution requires
the already-running llama31 serving layout; this script never changes jobs,
starts servers, or queues GPU allocations. See CROSS_MODEL_PILOT.md.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from urllib.request import urlopen


PTX = Path(__file__).resolve().parents[1]
SMALL = "Llama-3.1-8B-Instruct"
BIG = "Llama-3.1-70B-Instruct-FP8"
SOURCE = "Qwen3-8B"
ARM_IDS = ("base", "own_big_reference", "direct", "warm_small", "cold_small", "warm_big", "cold_big")
LABELS = {
    "base": "Original benchmark instructions",
    "own_big_reference": "Llama70 own GEPA (historical full budget)",
    "direct": "Qwen8 instructions unchanged",
    "warm_small": "Qwen8 -> GEPA Llama8 -> Llama70",
    "cold_small": "Base -> GEPA Llama8 -> Llama70",
    "warm_big": "Qwen8 -> GEPA Llama70",
    "cold_big": "Base -> GEPA Llama70 (matched pilot budget)",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_prompt(path, dest, expected_keys=None):
    data = read_json(path)
    if not isinstance(data, dict) or not data or any(not isinstance(v, str) or not v.strip() for v in data.values()):
        raise ValueError(f"Invalid instruction mapping: {path}")
    if expected_keys is not None and set(data) != set(expected_keys):
        raise ValueError(f"Predictor keys do not match benchmark: {path}")
    if dest.exists() and dest.read_bytes() != Path(path).read_bytes():
        raise ValueError(f"Frozen prompt changed: {dest}")
    dest.write_bytes(Path(path).read_bytes())
    return {"path": str(dest), "sha256": digest(dest), "original_path": str(path)}


def prepare(args):
    root = args.out.resolve()
    gepa = args.gepa_root.resolve()
    metric, optroot, series = {
        "hotpotqa": ("f1", "gepa_f1", "llama31_f1"),
        "hover": ("recall", "gepa_hover_recall", "hover_llama31"),
    }[args.bench]
    source_dir = gepa / "local/runs" / optroot / f"{SOURCE}-thinkoff-s0"
    own_dir = gepa / "local/runs" / optroot / f"{BIG}-thinkoff-s0"
    small_dir = gepa / "local/runs" / optroot / f"{SMALL}-thinkoff-s0"
    base_config = gepa / "local/runs" / f"{series}_eval" / f"{BIG}-base-r0/config.json"
    for file in (gepa / "local/run_gepa.py", gepa / "local/eval_bench.py", gepa / ".venv/bin/python",
                 source_dir / "best_instructions.json", own_dir / "best_instructions.json", base_config):
        if not file.exists():
            raise ValueError(f"Required existing artifact missing: {file}")
    base_instructions = read_json(base_config)["instructions"]
    for optdir in (source_dir, own_dir):
        cfg = read_json(optdir / "config.json")
        if cfg.get("metric") != metric or cfg.get("bench", "hotpotqa") != args.bench:
            raise ValueError(f"Source benchmark or metric mismatch: {optdir}")
        if not (optdir / "done.json").exists():
            raise ValueError(f"Source optimization incomplete: {optdir}")
    settings = {
        "bench": args.bench, "metric": metric, "gepa_root": str(gepa),
        "seed": args.seed, "split_seed": 0, "train_size": args.train_size,
        "val_size": args.val_size, "budget": args.budget, "threads": args.threads,
        "test_n": args.test_n, "small_port": args.small_port, "big_port": args.big_port,
        "source_model": SOURCE, "target_small_model": SMALL, "target_model": BIG,
    }
    manifest_path = root / "manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    if previous and previous["settings"] != settings:
        raise ValueError("Pilot settings changed; use a new --out directory to preserve the comparison")
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    source = freeze_prompt(source_dir / "best_instructions.json", inputs / "qwen8.json", base_instructions)
    own = freeze_prompt(own_dir / "best_instructions.json", inputs / "llama70_full.json", base_instructions)
    # Use the historical evaluator's exact benchmark instructions, not a rewritten baseline.
    base_file = inputs / "base.json"
    if base_file.exists() and read_json(base_file) != base_instructions:
        raise ValueError("Historical base instructions changed")
    write_json(base_file, base_instructions)
    source["optimization_dir"] = str(source_dir)
    registry = (PTX / "serving/models.sh").read_text()
    revisions_text = re.search(r"declare -A MODEL_REV=\((.*?)\n\)", registry, re.S).group(1)
    revisions = dict(re.findall(r"\[([^]]+)\]=([a-f0-9]+)", revisions_text))
    arms = []
    for armid in ARM_IDS:
        optimized = armid.startswith(("cold_", "warm_"))
        odir = root / "optimize" / armid if optimized else own_dir if armid == "own_big_reference" else None
        prompt = str(odir / "best_instructions.json") if optimized else {
            "base": str(base_file), "direct": source["path"], "own_big_reference": own["path"],
        }[armid]
        arms.append({
            "id": armid, "label": LABELS[armid], "program": prompt,
            "eval_dir": str(root / "eval" / armid),
            "optimization_dir": str(odir) if odir else None,
            "historical": armid == "own_big_reference",
            "source_inherited": armid in ("direct", "warm_small", "warm_big"),
            "optimizer_model": SMALL if armid.endswith("small") else BIG if optimized else None,
            "initial_program": source["path"] if armid.startswith("warm_") else str(base_file),
        })
    # The paper's 300 test items are the prefix of our historical 1000-item set.
    # Freeze IDs now so evaluator/report mismatches cannot be hidden by intersections.
    historical_rows = [json.loads(line) for line in (base_config.parent / "results.jsonl").read_text().splitlines()]
    test_ids = [r["id"] for r in historical_rows[: args.test_n]]
    if len(test_ids) != args.test_n or len(set(test_ids)) != args.test_n:
        raise ValueError("Historical test IDs missing or duplicated")
    manifest = {
        **settings, "settings": settings, "source_prompt": source, "own_big_prompt": own,
        "base_prompt": {"path": str(base_file), "sha256": digest(base_file)},
        "model_revisions": {m: revisions[m] for m in (SOURCE, SMALL, BIG)},
        "frozen_test_ids": test_ids,
        "test_ids_sha256": hashlib.sha256(json.dumps(test_ids).encode()).hexdigest(),
        "arms": arms, "out": str(root),
        "historical_small_optimization_dir": str(small_dir),
        "created": previous["created"] if previous else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": [
            "All predictions use Llama70 FP8; bridge prompts are selected only on Llama8 validation.",
            "Pilot rollouts are matched, not tokens or FLOPs; GEPA checks budget at iteration boundaries.",
            "Historical own-big prompt used the full 300-val, ~6900-rollout run; it is not budget matched.",
            "Source optimization is sunk for migration, included separately in end-to-end cost.",
            "Each optimizer uses its executor as reflection LM; measure both roles.",
            "Forward FLOPs are an analytical arithmetic count, not measured device FLOPs or GPU time.",
            "One optimizer seed and one sampling repeat: screening evidence, not a final significance claim.",
        ],
    }
    if previous and manifest != previous:
        raise ValueError("Frozen pilot inputs/provenance changed; choose a new --out")
    write_json(manifest_path, manifest)
    return manifest


def commands(manifest, selected):
    gepa = Path(manifest["gepa_root"])
    python = str(gepa / ".venv/bin/python")
    for arm in manifest["arms"]:
        if arm["id"] not in selected:
            continue
        if arm["optimizer_model"]:
            port = manifest["small_port"] if arm["optimizer_model"] == SMALL else manifest["big_port"]
            command = [python, "local/run_gepa.py", "--bench", manifest["bench"],
                       "--model", arm["optimizer_model"], "--port", str(port), "--thinking", "off",
                       "--metric", manifest["metric"], "--seed", str(manifest["seed"]), "--split-seed", "0",
                       "--train-size", str(manifest["train_size"]), "--val-size", str(manifest["val_size"]),
                       "--threads", str(manifest["threads"]), "--max-metric-calls", str(manifest["budget"]),
                       "--initial-program", arm["initial_program"], "--out", arm["optimization_dir"]]
            yield "train", arm, command
        yield "eval", arm, [python, "local/eval_bench.py", "--bench", manifest["bench"],
                            "--model", BIG, "--port", str(manifest["big_port"]), "--thinking", "off",
                            "--metric", manifest["metric"], "--program", arm["program"],
                            "--n", str(manifest["test_n"]), "--threads", str(manifest["threads"]),
                            "--repeat", "0", "--out", arm["eval_dir"]]


def endpoint_info(model, port):
    base = f"http://localhost:{port}"
    try:
        with urlopen(base + "/v1/models", timeout=10) as response:
            served = json.load(response)
        if model not in {row["id"] for row in served.get("data", [])}:
            raise ValueError(f"Expected {model}; endpoint serves another model: {base}")
        with urlopen(base + "/get_server_info", timeout=10) as response:
            info = json.load(response)
        context = info.get("context_length", info.get("server_args", {}).get("context_length", 0))
        if not isinstance(context, int) or context < 32768:
            raise ValueError(f"{model} requires served context >=32768; found {context}")
        server_args = info.get("server_args", info)
        if server_args.get("disable_radix_cache") is not True:
            raise ValueError(f"{model}: disable radix cache to match token-based FLOP accounting")
        if server_args.get("enable_deterministic_inference") is not True:
            raise ValueError(f"{model}: deterministic inference is required by this serving protocol")
        # Store only serving configuration, never request contents.
        return {"model": model, "port": port, "context_length": context,
                "model_path": info.get("model_path", server_args.get("model_path")),
                "tp_size": server_args.get("tp_size"), "dp_size": server_args.get("dp_size"),
                "disable_radix_cache": True, "enable_deterministic_inference": True}
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(f"{model} unavailable at {base}; start the llama31 layout only after GPUs are free: {exc}") from exc


def validate_completed_train(manifest, arm):
    cfg = read_json(Path(arm["optimization_dir"]) / "config.json")
    expected = {"bench": manifest["bench"], "metric": manifest["metric"], "model": arm["optimizer_model"],
                "thinking": "off", "seed": manifest["seed"], "split_seed": 0,
                "n_train": manifest["train_size"], "n_val": manifest["val_size"],
                "effective_max_metric_calls": manifest["budget"], "effective_threads": manifest["threads"]}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed optimization settings changed: {arm['id']}")
    saved = read_json(Path(arm["optimization_dir"]) / "initial_instructions.json")
    if saved != read_json(arm["initial_program"]):
        raise ValueError(f"Completed optimization initialization changed: {arm['id']}")


def validate_completed_eval(manifest, arm):
    directory = Path(arm["eval_dir"])
    cfg = read_json(directory / "config.json")
    expected = {"bench": manifest["bench"], "metric": manifest["metric"], "model": BIG,
                "thinking": "off", "split": "test", "n_questions": manifest["test_n"], "repeat": 0,
                "decode": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384}}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed evaluation settings changed: {arm['id']}")
    if cfg.get("instructions") != read_json(arm["program"]):
        raise ValueError(f"Evaluated instructions changed: {arm['id']}")
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(manifest["frozen_test_ids"]):
        raise ValueError(f"Completed evaluation test IDs changed: {arm['id']}")


@contextmanager
def pilot_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another cross-model pilot is running; use the serial queue") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute(manifest, stage, selected):
    jobs = [(s, a, c) for s, a, c in commands(manifest, selected) if stage == "all" or stage == s]
    # Finish all optimizations before the first test query; test never selects an instruction.
    jobs.sort(key=lambda job: job[0] != "train")
    if not jobs:
        raise ValueError("No jobs selected for this stage")
    needed = {(arm["optimizer_model"], manifest["small_port"] if arm["optimizer_model"] == SMALL else manifest["big_port"])
              for s, arm, _ in jobs if s == "train"}
    if any(s == "eval" for s, _, _ in jobs):
        needed.add((BIG, manifest["big_port"]))
    with pilot_lock(PTX / "artifacts/cross_model_pilot/.cluster.lock"):
        checks = [endpoint_info(model, port) for model, port in sorted(needed)]
        write_json(Path(manifest["out"]) / "endpoint_preflight.json", {"checked": time.time(), "endpoints": checks})
        for step, arm, command in jobs:
            directory = Path(arm["optimization_dir"] if step == "train" else arm["eval_dir"])
            done = directory / ("done.json" if step == "train" else "summary.json")
            if done.exists():
                (validate_completed_train if step == "train" else validate_completed_eval)(manifest, arm)
                print(f"Completed, skipping: {step} {arm['id']}", flush=True)
                continue
            if step == "eval" and not Path(arm["program"]).exists():
                raise ValueError(f"Optimize {arm['id']} before evaluation: missing {arm['program']}")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Running serially: {step} {arm['id']}; log {directory / 'driver.log'}", flush=True)
            with (directory / "driver.log").open("a") as logfile:
                subprocess.run(command, cwd=manifest["gepa_root"], stdout=logfile, stderr=subprocess.STDOUT, check=True)
            (validate_completed_train if step == "train" else validate_completed_eval)(manifest, arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=("hotpotqa", "hover"), default="hotpotqa")
    parser.add_argument("--gepa-root", type=Path, default=PTX.parent / "gepa-artifact")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", choices=("plan", "train", "eval", "all", "report"), default="plan")
    parser.add_argument("--arms", default=",".join(ARM_IDS))
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--test-n", type=int, choices=(300, 1000), default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small-port", type=int, default=18014)
    parser.add_argument("--big-port", type=int, default=18015)
    args = parser.parse_args()
    if args.out is None:
        args.out = PTX / "artifacts/cross_model_pilot" / f"{args.bench}_s{args.seed}"
    if not (3 <= args.train_size <= 150 and 1 <= args.val_size <= 300 and args.threads >= 1 and args.budget > args.val_size):
        parser.error("Require train size 3..150, val size 1..300, threads>=1, and budget>val size")
    selected = set(args.arms.split(","))
    if not selected or selected - set(ARM_IDS):
        parser.error(f"Unknown arms: {selected - set(ARM_IDS)}")
    manifest = prepare(args)
    manifest_path = Path(manifest["out"]) / "manifest.json"
    print(f"Frozen manifest: {manifest_path}")
    if args.stage == "plan":
        for stage, arm, command in commands(manifest, selected):
            print(f"{stage:5s} {arm['id']}: (cd {shlex.quote(manifest['gepa_root'])} && {shlex.join(command)})")
        print("Plan only; no server queries, GPU jobs, or inference requests were launched.")
    elif args.stage == "report":
        from cross_model_report import build_report, write_report
        for arm in manifest["arms"]:
            if (Path(arm["eval_dir"]) / "summary.json").exists():
                validate_completed_eval(manifest, arm)
        write_report(build_report(manifest), Path(manifest["out"]) / "report")
    else:
        execute(manifest, args.stage, selected)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
