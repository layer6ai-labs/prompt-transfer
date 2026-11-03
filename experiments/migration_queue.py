"""Evaluate methods 1–3, optimize method 4, then evaluate method 4.

All six cross-family directions and both benchmarks use the same frozen 300
test IDs per benchmark. One owned serving job uses the existing two H100s.
The three campaign stages are global barriers, including after a resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

from bridge_queue import (BENCHES, PAIRS, Queue, atomic, digest, instructions_hash,
                          process_identity, read, validate_completed,
                          validate_prompt, verify_manifest as verify_bridges)
from cross_model_pilot import PTX, endpoint_info, pilot_lock


FAMILIES = {
    "llama": {"small": "Llama-3.1-8B-Instruct", "tag": "llama8",
              "big": "Llama-3.1-70B-Instruct-FP8", "port": 38015, "remote_port": 8015,
              "series": {"hotpotqa": "llama31_f1", "hover": "hover_llama31"}},
    "qwen": {"small": "Qwen3-8B", "tag": "qwen8", "big": "Qwen3-32B",
             "port": 38010, "remote_port": 8010,
             "series": {"hotpotqa": "qwen3_f1", "hover": "hover_qwen3"}},
    "gemma": {"small": "gemma-3-12b-it", "tag": "gemma12", "big": "gemma-3-27b-it",
              "port": 38006, "remote_port": 8006,
              "series": {"hotpotqa": "gemma3_f1", "hover": "hover_gemma3"}},
}
TARGET_ORDER = ["llama", "qwen", "gemma"]
STAGES = ["eval123", "train4", "eval4"]
SETTINGS = {"train_size": 150, "val_size": 100, "budget": 1200, "seed": 0,
            "split_seed": 0, "threads": 8, "test_n": 300}
DECODE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384}
ARMS123 = ["own_big_reference", "direct", "warm_small"]


def freeze(path, destination, bench, optimization_dir=None):
    prompt = validate_prompt(read(path), bench)
    if destination.exists() and read(destination) != prompt:
        raise ValueError(f"Frozen instruction conflict: {destination}")
    atomic(destination, prompt)
    return {"path": str(destination), "sha256": digest(destination),
            "instructions_sha256": instructions_hash(prompt),
            "original_path": str(path), "original_sha256": digest(path),
            "optimization_dir": str(optimization_dir) if optimization_dir else None}


def prepare(out, gepa, bridge_root):
    out, gepa, bridge_root = (Path(p).resolve() for p in (out, gepa, bridge_root))
    path = out / "manifest.json"
    if path.exists():
        manifest = read(path)
        if (manifest["gepa_root"] != str(gepa) or manifest["bridge_root"] != str(bridge_root)
                or manifest["settings"] != SETTINGS):
            raise ValueError("Frozen campaign settings changed; choose a new output directory")
        verify_manifest(manifest)
        initialize_status(manifest)
        return manifest
    bridge = read(bridge_root / "manifest.json")
    verify_bridges(bridge)
    for run in bridge["runs"]:
        validate_completed(bridge, run)
    registry = (PTX / "serving/models.sh").read_text()
    revisions = dict(re.findall(r"\[([^]]+)\]=([a-f0-9]+)",
                               re.search(r"declare -A MODEL_REV=\((.*?)\n\)", registry, re.S).group(1)))
    sources, references, bridges, inputs, test_ids = {}, {}, {}, {}, {}
    for bench, (metric, optroot, _) in BENCHES.items():
        for family, model in FAMILIES.items():
            source = bridge["sources"][f"{bench}/{model['tag']}"]
            meta = freeze(Path(source["path"]), out / "inputs" / f"{bench}_{model['tag']}.json",
                          bench, source["optimization_dir"])
            sources[f"{bench}/{model['tag']}"] = meta
            inputs[meta["path"]] = meta
            optdir = gepa / "local/runs" / optroot / f"{model['big']}-thinkoff-s0"
            cfg, done = read(optdir / "config.json"), read(optdir / "done.json")
            if (cfg.get("model") != model["big"] or cfg.get("metric") != metric
                    or cfg.get("bench", "hotpotqa") != bench or "usage" not in done):
                raise ValueError(f"Invalid historical own-big reference: {optdir}")
            meta = freeze(optdir / "best_instructions.json", out / "inputs" / f"{bench}_{family}_own_big.json",
                          bench, optdir)
            references[f"{bench}/{family}"] = meta
            inputs[meta["path"]] = meta
            base = gepa / "local/runs" / f"{model['series'][bench]}_eval" / f"{model['big']}-base-r0"
            ids = [json.loads(line)["id"] for line in (base / "results.jsonl").read_text().splitlines()
                   if line.strip()][:SETTINGS["test_n"]]
            if len(ids) != SETTINGS["test_n"] or len(set(ids)) != len(ids):
                raise ValueError(f"Invalid historical held-out IDs: {base}")
            if bench in test_ids and test_ids[bench] != ids:
                raise ValueError(f"Historical test IDs differ across target families: {bench}")
            test_ids[bench] = ids
    for run in bridge["runs"]:
        directory = Path(run["directory"])
        meta = freeze(directory / "best_instructions.json", out / "inputs" / "bridges" / (run["id"] + ".json"),
                      run["bench"], directory)
        bridges[run["id"]] = meta
        inputs[meta["path"]] = meta
    routes = []
    for run in bridge["runs"]:
        family = next(f for f, m in FAMILIES.items() if m["tag"] == run["target"])
        target = FAMILIES[family]
        bench = run["bench"]
        source_model = next(m["small"] for m in FAMILIES.values() if m["tag"] == run["source"])
        source = sources[f"{bench}/{run['source']}"]
        own = references[f"{bench}/{family}"]
        bridge_meta = bridges[run["id"]]
        bridge_config = read(Path(run["directory"]) / "config.json")
        route_name = run["id"].split("/", 1)[1]
        odir = out / "optimize" / bench / route_name
        arms = []
        specs = [
            ("own_big_reference", "1: Target large model's own GEPA", own["path"], own["optimization_dir"], False, True, None),
            ("direct", "2: Source instructions unchanged", source["path"], None, True, False, None),
            ("warm_small", "3: Source → target-small adaptation → target-large", bridge_meta["path"],
             bridge_meta["optimization_dir"], True, False, target["small"]),
            ("warm_big", "4: Source → target-large adaptation", str(odir / "best_instructions.json"),
             str(odir), True, False, target["big"]),
        ]
        for aid, label, program, optdir, inherited, historical, optimizer in specs:
            edir = out / "eval" / bench / (family if aid == "own_big_reference" else route_name) / aid
            arms.append({"id": aid, "label": label, "program": program, "eval_dir": str(edir),
                         "optimization_dir": optdir, "source_inherited": inherited, "historical": historical,
                         "optimizer_model": optimizer, "initial_program": source["path"]})
        routes.append({"id": run["id"], "pair": run["pair"], "bench": bench, "metric": run["metric"],
                       "source_model": source_model, "target_small_model": target["small"],
                       "target_model": target["big"], "target_family": family,
                       "source_prompt": source, "frozen_test_ids": test_ids[bench],
                       "test_ids_sha256": instructions_hash(test_ids[bench]),
                       "train_ids_sha256": bridge_config["train_ids_sha256"],
                       "val_ids_sha256": bridge_config["val_ids_sha256"], "settings": SETTINGS, "arms": arms})
    files = [PTX / "serving/models.sh", PTX / "serving/serve.sh", PTX / "serving/migration_serve.sh",
             PTX / "sky/migration-serve.yaml", PTX / "experiments/migration_queue.py",
             PTX / "experiments/bridge_queue.py", PTX / "experiments/cross_model_pilot.py",
             PTX / "experiments/migration_report.py", PTX / "experiments/cross_model_report.py"]
    files += [gepa / "local" / f for f in ("eval_bench.py", "eval_setup.py", "run_gepa.py", "gepa_setup.py",
              "benchmarks.py", "metrics.py", "cost.py", "compat.py", "series.py")]
    provenance = {str(p): digest(p) for p in files}
    for meta in inputs.values():
        optdir = meta["optimization_dir"]
        if optdir:
            for name in ("config.json", "done.json"):
                p = Path(optdir) / name
                provenance[str(p)] = digest(p)
    manifest = {"schema_version": 1, "queue_id": uuid.uuid4().hex, "out": str(out),
                "gepa_root": str(gepa), "bridge_root": str(bridge_root), "cluster": "prompt-transfer",
                "settings": SETTINGS, "decode": DECODE, "stages": STAGES, "target_order": TARGET_ORDER,
                "model_revisions": {m: revisions[m] for spec in FAMILIES.values() for m in (spec['small'], spec['big'])},
                "inputs": inputs, "provenance": provenance, "routes": routes, "created": time.time(),
                "notes": ["All methods 1–3 evaluations finish globally before method 4 optimization starts.",
                          "All method 4 optimizations finish globally before method 4 test evaluation starts.",
                          "Method 1 is the full-budget historical reference, not a budget-matched cold control.",
                          "No cold controls or extra baseline runs are included in this four-method campaign.",
                          "Source prompts are frozen independently, and test outcomes never select instructions.",
                          "HoVer inherits one train/validation overlap; neither overlaps the frozen test set.",
                          "One optimizer seed and one sampling repeat; bootstrap intervals cover question sampling only."]}
    verify_manifest(manifest)
    atomic(path, manifest)
    initialize_status(manifest)
    return manifest


def initialize_status(manifest):
    out = Path(manifest["out"])
    if (out / "status.json").exists():
        return
    state = {"queue_id": manifest["queue_id"], "state": "planned", "pairs": {}, "stages": {}}
    for stage in STAGES:
        state["stages"][stage] = "pending"
        for family in TARGET_ORDER:
            state["pairs"][f"{stage}-{family}"] = {"state": "pending", "queue_id": manifest["queue_id"]}
    atomic(out / "status.json", state)


def verify_manifest(manifest):
    if (manifest["settings"] != SETTINGS or manifest["stages"] != STAGES
            or manifest["target_order"] != TARGET_ORDER or manifest["decode"] != DECODE
            or len(manifest["routes"]) != 12):
        raise ValueError("Campaign protocol changed")
    if len({r["id"] for r in manifest["routes"]}) != 12:
        raise ValueError("Duplicate route IDs")
    for path, meta in manifest["inputs"].items():
        if digest(path) != meta["sha256"] or instructions_hash(read(path)) != meta["instructions_sha256"]:
            raise ValueError(f"Frozen instruction changed: {path}")
    for path, expected in manifest["provenance"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen code or optimization provenance changed: {path}")
    for route in manifest["routes"]:
        ids = route["frozen_test_ids"]
        if len(ids) != SETTINGS["test_n"] or len(set(ids)) != len(ids) or instructions_hash(ids) != route["test_ids_sha256"]:
            raise ValueError(f"Invalid frozen test IDs: {route['id']}")
        if [a["id"] for a in route["arms"]] != ARMS123 + ["warm_big"]:
            raise ValueError(f"Missing comparison method: {route['id']}")


def jobs_for(manifest, stage, family=None):
    seen = set()
    for route in manifest["routes"]:
        if family is not None and route["target_family"] != family:
            continue
        for arm in route["arms"]:
            if arm["id"] not in (ARMS123 if stage == "eval123" else ["warm_big"]):
                continue
            directory = arm["optimization_dir"] if stage == "train4" else arm["eval_dir"]
            if directory in seen:
                continue
            seen.add(directory)
            yield route, arm


def command_for(manifest, stage, route, arm):
    model = FAMILIES[route["target_family"]]
    command = [str(Path(manifest["gepa_root"]) / ".venv/bin/python"),
               "local/run_gepa.py" if stage == "train4" else "local/eval_bench.py",
               "--bench", route["bench"], "--model", model["big"], "--port", str(model["port"]),
               "--thinking", "off", "--metric", route["metric"], "--threads", str(SETTINGS["threads"])]
    if stage == "train4":
        command += ["--initial-program", arm["initial_program"], "--out", arm["optimization_dir"]]
        for flag, key in (("train-size", "train_size"), ("val-size", "val_size"),
                          ("max-metric-calls", "budget"), ("seed", "seed"), ("split-seed", "split_seed")):
            command += ["--" + flag, str(SETTINGS[key])]
    else:
        command += ["--program", arm["program"], "--n", str(SETTINGS["test_n"]),
                    "--repeat", "0", "--out", arm["eval_dir"]]
    return command


def validate_evaluation(route, arm):
    directory = Path(arm["eval_dir"])
    cfg, summary = read(directory / "config.json"), read(directory / "summary.json")
    expected = {"bench": route["bench"], "metric": route["metric"], "model": route["target_model"],
                "thinking": "off", "split": "test", "n_questions": SETTINGS["test_n"],
                "repeat": 0, "decode": DECODE}
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get("instructions") != read(arm["program"]):
        raise ValueError(f"Evaluation settings/instructions changed: {directory}")
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if (len(ids) != len(set(ids)) or set(ids) != set(route["frozen_test_ids"])
            or summary.get("n") != len(ids) or summary.get("metric") != route["metric"]):
        raise ValueError(f"Evaluation IDs or summary changed: {directory}")
    for row in rows:
        score = row.get(route["metric"])
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"Invalid per-item score: {directory}")
    if summary.get("usage_history_complete") is not True:
        raise ValueError(f"Evaluation usage accounting incomplete: {directory}")


def validate_training(route, arm):
    directory = Path(arm["optimization_dir"])
    cfg, done = read(directory / "config.json"), read(directory / "done.json")
    expected = {"bench": route["bench"], "metric": route["metric"], "model": route["target_model"],
                "thinking": "off", "seed": 0, "split_seed": 0, "n_train": 150, "n_val": 100,
                "effective_max_metric_calls": 1200, "effective_threads": 8, "decode": DECODE}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Method 4 optimization settings changed: {directory}")
    initial = read(arm["initial_program"])
    if (read(directory / "initial_instructions.json") != initial
            or cfg.get("initial_instructions_sha256") != instructions_hash(initial)):
        raise ValueError(f"Method 4 initialization changed: {directory}")
    if any(cfg.get(key) != route[key] for key in ("train_ids_sha256", "val_ids_sha256")):
        raise ValueError(f"Method 4 train/validation split changed: {directory}")
    validate_prompt(read(arm["program"]), route["bench"])
    if done.get("metric_calls_actual", 0) < 1200 or not isinstance(done.get("usage", {}).get("all"), dict):
        raise ValueError(f"Method 4 accounting incomplete: {directory}")


def is_complete(stage, route, arm):
    path = Path(arm["optimization_dir"] if stage == "train4" else arm["eval_dir"])
    sentinel = path / ("done.json" if stage == "train4" else "summary.json")
    if not sentinel.exists():
        return False
    (validate_training if stage == "train4" else validate_evaluation)(route, arm)
    return True


class MigrationQueue(Queue):
    def report(self):
        from migration_report import build_report, write_report
        write_report(build_report(self.manifest), self.out / "report")

    def deploy_target(self, key, family):
        self.guard_jobs()
        model = FAMILIES[family]
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", model["port"]))
        entry = self.state["pairs"][key]
        entry.update(state="deploying", queue_id=self.manifest["queue_id"], serve_cancelled=False,
                     serve_job_id=None, submission_started=time.time())
        self.save()
        output = self.call(["sky", "exec", self.cluster, "sky/migration-serve.yaml",
                            "--env", f"TARGET_FAMILY={family}", "--env", f"CAMPAIGN_ID={self.manifest['queue_id']}", "-d"],
                           self.out / f"launch_{key}.log", timeout=600, include_stderr=True)
        from bridge_queue import parse_job_id
        entry["serve_job_id"] = parse_job_id(output)
        self.save()
        cmd = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
               "-o", "ConnectTimeout=90", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
               "-L", f"127.0.0.1:{model['port']}:localhost:{model['remote_port']}", self.cluster]
        with (self.out / f"tunnel_{key}.log").open("a") as log:
            self.tunnel = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        entry["tunnel"] = {"pid": self.tunnel.pid, "start_ticks": process_identity(self.tunnel.pid)}
        self.save()
        deadline, last_check, error = time.monotonic() + 1800, 0, None
        while time.monotonic() < deadline:
            if self.tunnel.poll() is not None:
                raise ValueError(f"SSH forwarding failed: tunnel_{key}.log")
            if time.monotonic() - last_check > 60:
                self.guard_jobs(entry["serve_job_id"])
                last_check = time.monotonic()
            try:
                evidence = endpoint_info(model["big"], model["port"])
                if evidence.get("tp_size") != 2:
                    raise ValueError(f"Expected TP2 on the existing two GPUs: {evidence}")
                atomic(self.out / f"endpoints_{key}.json", {"checked": time.time(), "endpoint": evidence,
                       "revision": self.manifest["model_revisions"][model["big"]]})
                entry["state"] = "ready"
                self.save()
                return
            except ValueError as exc:
                error = str(exc)
                time.sleep(5)
        raise ValueError(f"Large-model serving startup exceeded 30 minutes: {error}")

    def execute_one(self, stage, key, route, arm):
        if is_complete(stage, route, arm):
            print(f"Verified complete: {stage} {route['id']} {arm['id']}", flush=True)
            return
        if stage != "train4":
            validate_prompt(read(arm["program"]), route["bench"])
        directory = Path(arm["optimization_dir"] if stage == "train4" else arm["eval_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        command = command_for(self.manifest, stage, route, arm)
        self.state["current_job"] = {"stage": stage, "route": route["id"], "arm": arm["id"],
                                     "directory": str(directory), "started": time.time()}
        self.save()
        import shlex
        print(f"Starting {stage} {route['id']} {arm['id']}", flush=True)
        with (directory / "driver.log").open("a") as log:
            log.write("\n$ " + shlex.join(command) + "\n")
            log.flush()
            env = {**os.environ, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
            proc = subprocess.Popen(command, cwd=self.manifest["gepa_root"], stdout=log, stderr=subprocess.STDOUT, env=env)
        self.workers.append(proc)
        record = {"id": stage + "/" + route["id"] + "/" + arm["id"],
                  "pid": proc.pid, "start_ticks": process_identity(proc.pid)}
        self.state["pairs"][key].setdefault("workers", []).append(record)
        atomic(directory / "status.json", {**record, "state": "running", "started": time.time()})
        self.save()
        last_check = time.monotonic()
        while proc.poll() is None:
            if self.tunnel.poll() is not None:
                raise ValueError("Campaign tunnel exited while client was running")
            if time.monotonic() - last_check > 60:
                self.guard_jobs(self.state["pairs"][key]["serve_job_id"])
                last_check = time.monotonic()
            time.sleep(2)
        if proc.returncode:
            atomic(directory / "status.json", {**record, "state": "failed", "returncode": proc.returncode, "finished": time.time()})
            raise ValueError(f"Client failed ({proc.returncode}); inspect {directory / 'driver.log'}")
        if not is_complete(stage, route, arm):
            raise ValueError(f"Client exited without a valid completion artifact: {directory}")
        self.workers.remove(proc)
        atomic(directory / "status.json", {**record, "state": "completed", "returncode": 0, "finished": time.time()})
        self.state.pop("current_job", None)
        self.save()
        self.report()

    def run(self):
        verify_manifest(self.manifest)
        with pilot_lock(PTX / "artifacts/cross_model_pilot/.cluster.lock"):
            try:
                self.recover()
                self.state["state"] = "running"
                self.save()
                for stage in STAGES:
                    # Revalidate every prior stage, not merely its saved status.
                    for previous in STAGES[:STAGES.index(stage)]:
                        if not all(is_complete(previous, r, a) for r, a in jobs_for(self.manifest, previous)):
                            raise ValueError(f"Global phase barrier incomplete: {previous}")
                    self.state["stage"] = stage
                    self.state["stages"][stage] = "running"
                    self.save()
                    for family in TARGET_ORDER:
                        key = f"{stage}-{family}"
                        jobs = list(jobs_for(self.manifest, stage, family))
                        entry = self.state["pairs"][key]
                        if all(is_complete(stage, r, a) for r, a in jobs):
                            entry["state"] = "completed"
                            self.save()
                            continue
                        try:
                            self.deploy_target(key, family)
                            entry["state"] = stage
                            self.save()
                            for route, arm in jobs:
                                self.execute_one(stage, key, route, arm)
                            entry["state"] = "completed"
                        except BaseException as exc:
                            entry.update(state="failed", error=str(exc))
                            raise
                        finally:
                            self.stop_children()
                            self.save()
                            self.cancel_owned(key)
                    if not all(is_complete(stage, r, a) for r, a in jobs_for(self.manifest, stage)):
                        raise ValueError(f"Stage did not finish: {stage}")
                    self.state["stages"][stage] = "completed"
                    self.save()
                    self.report()
                self.state["state"] = "completed"
                self.state.pop("current_job", None)
                self.save()
            except BaseException as exc:
                self.state.update(state="failed", error=str(exc))
                self.save()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "status", "report"))
    parser.add_argument("--out", type=Path, default=PTX / "artifacts/migration_20261101")
    parser.add_argument("--gepa-root", type=Path, default=PTX.parent / "gepa-artifact")
    parser.add_argument("--bridge-root", type=Path, default=PTX / "artifacts/bridge_queue_20261031")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(read(args.out / "status.json"), indent=2))
        return
    manifest = prepare(args.out, args.gepa_root, args.bridge_root)
    if args.command == "plan":
        print(f"Frozen manifest: {args.out / 'manifest.json'}")
        for stage in STAGES:
            print(stage, len(list(jobs_for(manifest, stage))), "unique jobs")
        print("Plan only; no serving changes or inference.")
    elif args.command == "report":
        MigrationQueue(manifest).report()
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Coordinator interrupted by signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        MigrationQueue(manifest).run()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
