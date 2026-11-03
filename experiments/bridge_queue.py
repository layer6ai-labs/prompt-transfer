"""Optimize six directed small-model prompt bridges; never evaluate large models.

plan freezes original cached prompts. run owns one two-GPU serving job at a
time, with two concurrent GEPA clients per benchmark. status is read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from cross_model_pilot import PTX, endpoint_info, pilot_lock

MODELS = {"qwen8": ("Qwen3-8B", 28005, 8005),
          "llama8": ("Llama-3.1-8B-Instruct", 28014, 8014),
          "gemma12": ("gemma-3-12b-it", 28007, 8007)}
PAIRS = (("qwen-llama", "qwen8", "llama8"), ("qwen-gemma", "qwen8", "gemma12"),
         ("llama-gemma", "llama8", "gemma12"))
BENCHES = {"hotpotqa": ("f1", "gepa_f1", "final_answer.predict"),
           "hover": ("recall", "gepa_hover_recall", "create_query_hop3.predict")}
SETTINGS = {"train_size": 150, "val_size": 100, "budget": 1200, "seed": 0,
            "split_seed": 0, "threads": 8, "worker_stagger_seconds": 60}
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "FAILED_SETUP"}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instructions_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_prompt(value, bench):
    keys = {"summarize1.predict", "create_query_hop2.predict", "summarize2.predict", BENCHES[bench][2]}
    if not isinstance(value, dict) or set(value) != keys or any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError(f"Invalid complete {bench} instruction mapping")
    return value


def prepare(out, gepa):
    out, gepa = Path(out).resolve(), Path(gepa).resolve()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["gepa_root"] != str(gepa) or manifest["settings"] != SETTINGS:
            raise ValueError("Frozen queue settings changed; choose a new --out")
        verify_manifest(manifest)
        initialize_statuses(manifest)
        return manifest
    for path in (gepa / ".venv/bin/python", gepa / "local/run_gepa.py"):
        if not path.exists():
            raise ValueError(f"Required optimizer missing: {path}")
    registry = (PTX / "serving/models.sh").read_text()
    revisions = dict(re.findall(r"\[([^]]+)\]=([a-f0-9]+)",
                               re.search(r"declare -A MODEL_REV=\((.*?)\n\)", registry, re.S).group(1)))
    sources = {}
    for bench, (metric, optroot, _) in BENCHES.items():
        for tag, (model, _, _) in MODELS.items():
            source_dir = gepa / "local/runs" / optroot / f"{model}-thinkoff-s0"
            source = source_dir / "best_instructions.json"
            cfg, done = read(source_dir / "config.json"), read(source_dir / "done.json")
            if cfg.get("model") != model or cfg.get("metric") != metric or cfg.get("bench", "hotpotqa") != bench or "usage" not in done:
                raise ValueError(f"Incomplete or incompatible cached source: {source_dir}")
            prompt = validate_prompt(read(source), bench)
            dest = out / "inputs" / f"{bench}_{tag}.json"
            atomic(dest, prompt)
            sources[f"{bench}/{tag}"] = {
                "path": str(dest), "sha256": digest(dest), "instructions_sha256": instructions_hash(prompt),
                "original_path": str(source), "original_sha256": digest(source), "optimization_dir": str(source_dir),
                "config_sha256": digest(source_dir / "config.json"), "done_sha256": digest(source_dir / "done.json"),
                "model": model, "model_revision": revisions[model]}
    runs = []
    for pair, a, b in PAIRS:
        for bench in BENCHES:
            for source, target in ((a, b), (b, a)):
                runs.append({"id": f"{bench}/{source}-to-{target}", "pair": pair, "bench": bench,
                             "metric": BENCHES[bench][0], "source": source, "target": target,
                             "initial_program": sources[f"{bench}/{source}"]["path"],
                             "directory": str(out / "optimize" / bench / f"{source}-to-{target}")})
    manifest = {"schema_version": 1, "queue_id": uuid.uuid4().hex, "out": str(out),
                "gepa_root": str(gepa), "cluster": "prompt-transfer", "settings": SETTINGS,
                "model_revisions": {m: revisions[m] for m, _, _ in MODELS.values()},
                "sources": sources, "runs": runs, "created": time.time(),
                "scope": "optimization only; original cached prompts initialize all twelve warm runs; large evaluation deferred"}
    atomic(manifest_path, manifest)
    initialize_statuses(manifest)
    return manifest


def initialize_statuses(manifest):
    out = Path(manifest["out"])
    pairs = {name: {"state": "pending", "queue_id": manifest["queue_id"]} for name, _, _ in PAIRS}
    if not (out / "status.json").exists():
        atomic(out / "status.json", {"queue_id": manifest["queue_id"], "state": "planned", "pairs": pairs})
    for name, entry in pairs.items():
        if not (out / "pairs" / name / "status.json").exists():
            atomic(out / "pairs" / name / "status.json", entry)
    for run in manifest["runs"]:
        if not (Path(run["directory"]) / "status.json").exists():
            atomic(Path(run["directory"]) / "status.json", {"id": run["id"], "state": "pending"})


def verify_manifest(manifest):
    if manifest["settings"] != SETTINGS or len(manifest["runs"]) != 12 or len(manifest["sources"]) != 6:
        raise ValueError("Unexpected frozen queue protocol")
    for source in manifest["sources"].values():
        if digest(source["path"]) != source["sha256"] or instructions_hash(read(source["path"])) != source["instructions_sha256"]:
            raise ValueError(f"Frozen source changed: {source['path']}")
    registry = (PTX / "serving/models.sh").read_text()
    for model, revision in manifest["model_revisions"].items():
        if f"[{model}]={revision}" not in registry:
            raise ValueError(f"Pinned model revision changed: {model}")


def train_command(manifest, run):
    settings, gepa = manifest["settings"], Path(manifest["gepa_root"])
    command = [str(gepa / ".venv/bin/python"), "local/run_gepa.py", "--bench", run["bench"],
               "--model", MODELS[run["target"]][0], "--port", str(MODELS[run["target"]][1]),
               "--thinking", "off", "--metric", run["metric"], "--initial-program", run["initial_program"],
               "--out", run["directory"]]
    for flag, key in (("train-size", "train_size"), ("val-size", "val_size"), ("max-metric-calls", "budget"),
                      ("seed", "seed"), ("split-seed", "split_seed"), ("threads", "threads")):
        command.extend(["--" + flag, str(settings[key])])
    return command


def validate_completed(manifest, run):
    directory, settings = Path(run["directory"]), manifest["settings"]
    cfg, done = read(directory / "config.json"), read(directory / "done.json")
    expected = {"bench": run["bench"], "metric": run["metric"], "model": MODELS[run["target"]][0],
                "thinking": "off", "seed": settings["seed"], "split_seed": settings["split_seed"],
                "n_train": settings["train_size"], "n_val": settings["val_size"],
                "effective_max_metric_calls": settings["budget"], "effective_threads": settings["threads"]}
    expected["decode"] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384}
    init = read(run["initial_program"])
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get("initial_instructions_sha256") != instructions_hash(init):
        raise ValueError(f"Completed run settings/initialization changed: {run['id']}")
    if read(directory / "initial_instructions.json") != init:
        raise ValueError(f"Saved initialization changed: {run['id']}")
    validate_prompt(read(directory / "best_instructions.json"), run["bench"])
    if done.get("metric_calls_actual", 0) < settings["budget"] or not isinstance(done.get("usage", {}).get("all"), dict):
        raise ValueError(f"Incomplete optimization accounting: {run['id']}")


def parse_queue(output, cluster):
    # Sky's logger may precede its JSON object on stdout.
    start = output.find("{")
    data = json.loads(output[start:]) if start >= 0 else None
    if not isinstance(data, dict) or cluster not in data or not isinstance(data[cluster], list):
        raise ValueError("Sky queue did not return the requested cluster; refusing deployment")
    rows = data[cluster]
    if any(not isinstance(r, dict) or "job_id" not in r or "status" not in r for r in rows):
        raise ValueError("Malformed Sky job records")
    return [r for r in rows if r["status"] not in TERMINAL]


def parse_job_id(output):
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    ids = set(re.findall(r"(?:Job submitted, ID:|Job ID:)\s*(\d+)", output))
    if len(ids) != 1:
        raise ValueError("Cannot identify the submitted serving job; inspect launch.log before proceeding")
    return int(ids.pop())


def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def recorded_process_alive(record):
    return bool(record and record.get("start_ticks") and process_identity(record["pid"]) == record["start_ticks"])


class Queue:
    def __init__(self, manifest):
        self.manifest, self.out = manifest, Path(manifest["out"])
        self.cluster = manifest["cluster"]
        self.state_path = self.out / "status.json"
        self.state = read(self.state_path) if self.state_path.exists() else {"queue_id": manifest["queue_id"], "pairs": {}}
        if self.state["queue_id"] != manifest["queue_id"]:
            raise ValueError("Queue status belongs to another manifest")
        self.workers, self.tunnel = [], None

    def save(self):
        self.state.update(updated=time.time(), owner_pid=os.getpid(), owner_start_ticks=process_identity(os.getpid()))
        atomic(self.state_path, self.state)
        for pair, entry in self.state["pairs"].items():
            atomic(self.out / "pairs" / pair / "status.json", {**entry, "updated": self.state["updated"], "owner_pid": os.getpid()})

    def call(self, command, log, timeout=180, include_stderr=False):
        print(shlex.join(command), flush=True)
        with Path(log).open("a") as handle:
            handle.write("\n$ " + shlex.join(command) + "\n")
            handle.flush()
            try:
                result = subprocess.run(command, cwd=PTX, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                for output in (exc.stdout, exc.stderr):
                    if output:
                        handle.write(output.decode(errors="replace") if isinstance(output, bytes) else output)
                raise
            handle.write(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError(f"Command failed ({result.returncode}); inspect {log}")
        return result.stdout + (result.stderr if include_stderr else "")

    def jobs(self):
        return parse_queue(self.call(["sky", "queue", self.cluster, "--all-users", "--skip-finished", "--output", "json"],
                                     self.out / "cluster_checks.log", timeout=120), self.cluster)

    def guard_jobs(self, own=None):
        jobs = self.jobs()
        unexpected = [r for r in jobs if r["job_id"] != own]
        if unexpected or (own is not None and not any(r["job_id"] == own for r in jobs)):
            raise ValueError(f"Unexpected active cluster jobs or own serving job ended: {jobs}")
        return jobs

    def cancel_owned(self, pair):
        entry = self.state["pairs"][pair]
        job = entry.get("serve_job_id")
        if job is None or entry.get("queue_id") != self.manifest["queue_id"]:
            return
        if any(r["job_id"] == job for r in self.jobs()):
            self.call(["sky", "cancel", self.cluster, str(job), "-y"], self.out / "cluster_actions.log")
            deadline = time.monotonic() + 180
            while any(r["job_id"] == job for r in self.jobs()):
                if time.monotonic() > deadline:
                    raise ValueError(f"Owned serving job {job} did not stop; refusing another deployment")
                time.sleep(5)
        entry["serve_cancelled"] = True
        self.save()

    def recover(self):
        for pair, entry in self.state["pairs"].items():
            if any(recorded_process_alive(p) for p in entry.get("workers", [])) or recorded_process_alive(entry.get("tunnel")):
                raise ValueError(f"Prior queue processes still live for {pair}; inspect recorded PIDs before resuming")
            if entry.get("serve_job_id") is not None and not entry.get("serve_cancelled"):
                self.cancel_owned(pair)
        self.guard_jobs()

    def stop_children(self):
        for proc in self.workers + ([self.tunnel] if self.tunnel else []):
            if proc.poll() is None:
                proc.terminate()
        for proc in self.workers + ([self.tunnel] if self.tunnel else []):
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        self.workers, self.tunnel = [], None

    def deploy(self, pair, tags):
        self.guard_jobs()
        entry = self.state["pairs"][pair]
        entry.update(state="deploying", queue_id=self.manifest["queue_id"], serve_cancelled=False,
                     serve_job_id=None, submission_started=time.time())
        self.save()
        output = self.call(["sky", "exec", self.cluster, "sky/bridge-serve.yaml", "--env", "LAYOUT=bridge-" + pair, "-d"],
                           self.out / f"launch_{pair}.log", timeout=600, include_stderr=True)
        entry["serve_job_id"] = parse_job_id(output)
        self.save()
        # Dedicated local ports must be unbound; never adopt an existing tunnel.
        reservations = []
        try:
            for tag in tags:
                sock = socket.socket()
                reservations.append(sock)
                sock.bind(("127.0.0.1", MODELS[tag][1]))
        finally:
            for sock in reservations:
                sock.close()
        command = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=90",
                   "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]
        for tag in tags:
            command += ["-L", f"127.0.0.1:{MODELS[tag][1]}:localhost:{MODELS[tag][2]}"]
        command.append(self.cluster)
        with (self.out / f"tunnel_{pair}.log").open("a") as log:
            self.tunnel = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        entry["tunnel"] = {"pid": self.tunnel.pid, "start_ticks": process_identity(self.tunnel.pid)}
        self.save()
        deadline, last_check, error = time.monotonic() + 1200, 0, None
        while time.monotonic() < deadline:
            if self.tunnel.poll() is not None:
                raise ValueError(f"SSH forwarding failed; inspect tunnel_{pair}.log")
            if time.monotonic() - last_check > 60:
                self.guard_jobs(entry["serve_job_id"])
                last_check = time.monotonic()
            try:
                checks = [endpoint_info(MODELS[t][0], MODELS[t][1]) for t in tags]
                atomic(self.out / f"endpoints_{pair}.json", {"checked": time.time(), "endpoints": checks})
                entry["state"] = "optimizing"
                self.save()
                return
            except ValueError as exc:
                error = str(exc)
                time.sleep(5)
        raise ValueError(f"Serving startup exceeded 20 minutes: {error}")

    def batch(self, pair, runs):
        entry, active, failures = self.state["pairs"][pair], [], []
        for run in runs:
            directory = Path(run["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "done.json").exists():
                validate_completed(self.manifest, run)
                atomic(directory / "status.json", {"state": "completed", "verified": time.time(), "reused": True})
                continue
            if active:
                deadline = time.monotonic() + self.manifest["settings"]["worker_stagger_seconds"]
                while time.monotonic() < deadline:
                    if active[0][1].poll() not in (None, 0):
                        raise ValueError(f"First optimizer failed during stagger: {active[0][0]['id']}")
                    time.sleep(1)
            command = train_command(self.manifest, run)
            with (directory / "driver.log").open("a") as log:
                log.write("\n$ " + shlex.join(command) + "\n")
                log.flush()
                env = {**os.environ, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
                proc = subprocess.Popen(command, cwd=self.manifest["gepa_root"], stdout=log,
                                        stderr=subprocess.STDOUT, env=env)
            self.workers.append(proc)
            record = {"id": run["id"], "pid": proc.pid, "start_ticks": process_identity(proc.pid)}
            entry.setdefault("workers", []).append(record)
            atomic(directory / "status.json", {**record, "state": "running", "owner_pid": os.getpid(), "started": time.time()})
            active.append((run, proc))
            self.save()
        last_check = time.monotonic()
        while active:
            if self.tunnel is not None and self.tunnel.poll() is not None:
                raise ValueError("Bridge tunnel exited while optimizers were running")
            for run, proc in list(active):
                result = proc.poll()
                if result is None:
                    continue
                error = None
                try:
                    if result:
                        raise ValueError(f"Optimizer exit {result}; inspect {run['directory']}/driver.log")
                    validate_completed(self.manifest, run)
                except (ValueError, OSError, KeyError) as exc:
                    error = str(exc)
                    failures.append(run["id"] + ": " + error)
                atomic(Path(run["directory"]) / "status.json", {"state": "failed" if error else "completed",
                       "returncode": result, "finished": time.time(), "error": error})
                active.remove((run, proc))
            if active and time.monotonic() - last_check > 60:
                self.guard_jobs(entry["serve_job_id"])
                last_check = time.monotonic()
            if active:
                time.sleep(2)
        if failures:
            raise ValueError("Batch failed; other active optimizer finished before stopping: " + "; ".join(failures))

    def run(self):
        verify_manifest(self.manifest)
        with pilot_lock(PTX / "artifacts/cross_model_pilot/.cluster.lock"):
            try:
                self.recover()
            except BaseException as exc:
                self.state.update(state="blocked", error=str(exc))
                self.save()
                raise
            self.state["state"] = "running"
            self.save()
            for pair, a, b in PAIRS:
                runs = [r for r in self.manifest["runs"] if r["pair"] == pair]
                if all((Path(r["directory"]) / "done.json").exists() for r in runs):
                    for run in runs:
                        validate_completed(self.manifest, run)
                    self.state["pairs"].setdefault(pair, {}).update(state="completed")
                    self.save()
                    continue
                self.state["pairs"].setdefault(pair, {"queue_id": self.manifest["queue_id"]})
                entry = self.state["pairs"][pair]
                try:
                    self.deploy(pair, (a, b))
                    for bench in BENCHES:
                        self.batch(pair, [r for r in runs if r["bench"] == bench])
                    entry["state"] = "completed"
                except BaseException as exc:
                    entry.update(state="failed", error=str(exc))
                    self.state.update(state="failed", error=str(exc))
                    raise
                finally:
                    self.stop_children()
                    self.save()
                    try:
                        self.cancel_owned(pair)
                    except BaseException as exc:
                        entry.update(state="failed", cleanup_error=str(exc))
                        self.state.update(state="failed", cleanup_error=str(exc))
                        self.save()
                        raise
            self.state["state"] = "completed"
            self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "status", "run"))
    parser.add_argument("--out", type=Path, default=PTX / "artifacts/bridge_queue_20261031")
    parser.add_argument("--gepa-root", type=Path, default=PTX.parent / "gepa-artifact")
    args = parser.parse_args()
    if args.command == "status":
        path = args.out / "status.json"
        print(json.dumps(read(path) if path.exists() else {"state": "not_started", "out": str(args.out)}, indent=2))
        return
    manifest = prepare(args.out, args.gepa_root)
    if args.command == "plan":
        print(f"Frozen optimization-only manifest: {args.out / 'manifest.json'}")
        for run in manifest["runs"]:
            print(shlex.join(train_command(manifest, run)))
    else:
        def stop(_signal, _frame):
            raise KeyboardInterrupt("Queue interrupted; cleaning up owned processes/job")
        signal.signal(signal.SIGTERM, stop)
        Queue(manifest).run()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        sys.exit(str(exc))
