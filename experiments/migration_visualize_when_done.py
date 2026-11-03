"""Generate final scientific plots after the existing campaign completes.

This is a read-only observer of inference/optimization. It never starts,
restarts, or cancels campaign jobs, and never edits the frozen experiment.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from bridge_queue import atomic, read, recorded_process_alive
from cross_model_pilot import PTX


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def observer_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A visualization observer is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def campaign_readiness(manifest, state):
    """Completion requires the full campaign, cleanup, and validated outputs."""
    if state.get("queue_id") != manifest["queue_id"]:
        raise ValueError("Campaign state belongs to a different frozen manifest")
    if state.get("state") != "completed":
        return {"state": "waiting_for_campaign", "campaign_state": state.get("state"),
                "campaign_stage": state.get("stage"), "campaign_error": state.get("error")}
    from migration_queue import STAGES, is_complete, jobs_for, verify_manifest
    if any(state.get("stages", {}).get(stage) != "completed" for stage in STAGES):
        raise ValueError("Campaign claims completion before all three stages finished")
    for entry in state.get("pairs", {}).values():
        if entry.get("serve_job_id") is not None and not entry.get("serve_cancelled"):
            return {"state": "waiting_for_cleanup", "campaign_state": "completed"}
    owner = {"pid": state.get("owner_pid"), "start_ticks": state.get("owner_start_ticks")}
    if owner["pid"] and recorded_process_alive(owner):
        # An orphaned zombie may retain its PID until the container reaps it;
        # it can no longer write reports and must not delay plots indefinitely.
        try:
            process_state = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            process_state = None
        if process_state not in (None, "Z", "X"):
            return {"state": "waiting_for_coordinator_exit", "campaign_state": "completed"}
    verify_manifest(manifest)
    for stage in STAGES:
        if not all(is_complete(stage, route, arm) for route, arm in jobs_for(manifest, stage)):
            raise ValueError(f"Missing validated campaign output in stage {stage}")
    return {"state": "ready", "campaign_state": "completed"}


def _existing_artifacts_valid(status, queue_id):
    files = status.get("artifact_sha256", {})
    return bool(status.get("state") == "completed" and status.get("queue_id") == queue_id and files
                and all(Path(path).is_file() and digest(path) == expected for path, expected in files.items()))


def process_once(campaign):
    campaign = Path(campaign).resolve()
    destination = campaign / "visualization"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign / "manifest.json"
    manifest, state = read(manifest_path), read(campaign / "status.json")
    request = read(destination / "request.json") if (destination / "request.json").exists() else None
    if request:
        if request["queue_id"] != manifest["queue_id"] or request["manifest_sha256"] != digest(manifest_path):
            raise ValueError("Visualization request no longer matches the frozen campaign")
        for path, expected in request.get("code_sha256", {}).items():
            if digest(path) != expected:
                raise ValueError(f"Visualization code changed since observer launch: {path}")
    status_path = destination / "status.json"
    previous = read(status_path) if status_path.exists() else {}
    if _existing_artifacts_valid(previous, manifest["queue_id"]):
        return previous
    readiness = campaign_readiness(manifest, state)
    status = {"queue_id": manifest["queue_id"], "updated": time.time(),
              "observer_pid": os.getpid(), **readiness}
    if readiness["state"] != "ready":
        atomic(status_path, status)
        return status
    atomic(status_path, {**status, "state": "generating"})
    from migration_report import build_report, write_report
    from migration_plots import generate_plots
    report = build_report(manifest)
    if report["method_cells"] != 48 or report["completed_method_cells"] != 48:
        raise ValueError("Final visualization requires all 48 method cells")
    if report["unique_evaluations"] != 42 or report["completed_unique_evaluations"] != 42:
        raise ValueError("Final visualization requires all 42 unique evaluations")
    write_report(report, campaign / "report")
    metadata = generate_plots(report, destination / "figures", allow_partial=False)
    atomic(destination / "artifacts.json", metadata)
    lines = ["# Four-method accuracy and FLOPs comparisons", "",
             "All four methods are evaluated on the same 300 held-out questions per benchmark. "
             "HotpotQA uses answer F1; HoVer uses article recall.", "",
             "Optimization costs are analytical PFLOP. Inference costs are analytical TFLOP per question. "
             "Method 1 includes historical full-budget optimization. Migration costs exclude source "
             "optimization; end-to-end costs include it. Confidence intervals cover question sampling.", ""]
    for artifact in metadata["artifacts"]:
        png, pdf = Path(artifact["png"]), Path(artifact["pdf"])
        lines += [f"## {artifact['bench']}: {artifact['kind']}", "",
                  f"![{artifact['kind']}]({os.path.relpath(png, destination)})", "",
                  f"[PNG]({os.path.relpath(png, destination)}) · [PDF]({os.path.relpath(pdf, destination)})", ""]
    lines += ["[Paired numerical report](../report/report.md)", ""]
    (destination / "README.md").write_text("\n".join(lines))
    files = [p for p in (destination / "figures").rglob("*") if p.is_file()]
    files += [destination / "README.md", destination / "artifacts.json"]
    status.update(state="completed", finished=time.time(), index=str(destination / "README.md"),
                  artifact_sha256={str(p): digest(p) for p in files})
    atomic(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=PTX / "artifacts/migration_20261101")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--once", action="store_true", help="Check once without waiting")
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("Polling interval must be at least 10 seconds")
    destination = args.campaign.resolve() / "visualization"
    with observer_lock(destination / ".observer.lock"):
        while True:
            try:
                status = process_once(args.campaign)
            except Exception as exc:
                atomic(destination / "status.json", {"state": "failed", "updated": time.time(), "error": str(exc),
                       "observer_pid": os.getpid()})
                raise
            print(json.dumps({key: status.get(key) for key in ("state", "campaign_state", "campaign_stage", "index")}), flush=True)
            if args.once or status["state"] == "completed":
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        sys.exit(str(exc))
