"""Report the four-method, six-route prompt migration campaign.

Only a complete evaluation on the frozen question IDs receives an accuracy
estimate. Every difference is paired against the target large model's own
prompt evaluation, never against whichever other arm happens to finish first.
The historical *prompts* are reused; their new evaluation is still required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

from cross_model_report import (
    COST_FIELDS, _add_cost, _break_even, _ci, _fmt, _inference_cost,
    _load_evaluation, bootstrap_ci, optimization_cost, optimization_rollouts,
)


ARM_IDS = ("own_big_reference", "direct", "warm_small", "warm_big")
METHOD_NAMES = {
    "own_big_reference": "1. Target large model's own GEPA prompts",
    "direct": "2. Source prompts transferred unchanged",
    "warm_small": "3. Source prompts adapted on target small model",
    "warm_big": "4. Source prompts adapted on target large model",
}
DECODE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384}


def _read(path):
    return json.loads(Path(path).read_text())


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _cost_scope(directory):
    if directory is None:
        return "No optimization required"
    done = Path(directory) / "done.json"
    if not done.exists():
        return "Pending optimization"
    data = _read(done)
    return data.get("seconds_scope", "Legacy reported session duration; cumulative duration unverified")


def _validate_evaluation(route, arm, summary, rows):
    """Reject wrong models/prompts/settings as well as wrong question sets."""
    path = Path(arm["eval_dir"]) / "config.json"
    config = _read(path)
    expected = {
        "bench": route["bench"], "model": route["target_model"],
        "metric": route["metric"], "thinking": "off", "split": "test",
        "repeat": 0, "n_questions": len(route["frozen_test_ids"]), "decode": DECODE,
    }
    wrong = [key for key, value in expected.items() if config.get(key) != value]
    if wrong:
        raise ValueError(f"Evaluation configuration mismatch for {route['id']}/{arm['id']}: {wrong}")
    if config.get("instructions") != _read(arm["program"]):
        raise ValueError(f"Evaluated instructions changed for {route['id']}/{arm['id']}")
    expected_ids = set(route["frozen_test_ids"])
    if set(rows) != expected_ids:
        raise ValueError(f"Question ID mismatch for {route['id']}/{arm['id']}: "
                         f"missing={len(expected_ids - rows.keys())}, extra={len(rows.keys() - expected_ids)}")


def build_route_report(route, settings=None):
    metric = route["metric"]
    if metric not in ("f1", "recall"):
        raise ValueError(f"Unsupported primary metric: {metric}")
    frozen = route["frozen_test_ids"]
    if not frozen or len(set(frozen)) != len(frozen):
        raise ValueError(f"Frozen test IDs must be nonempty and unique: {route['id']}")
    specs = route["arms"]
    if len(specs) != len(ARM_IDS) or {arm["id"] for arm in specs} != set(ARM_IDS):
        raise ValueError(f"Expected exactly four method IDs for {route['id']}")
    specs = sorted(specs, key=lambda arm: ARM_IDS.index(arm["id"]))
    if any("optimization_dir" not in arm for arm in specs):
        raise ValueError("Each method must explicitly identify its optimization directory or null")
    loaded = {arm["id"]: _load_evaluation(arm, metric) for arm in specs}
    for arm in specs:
        value = loaded[arm["id"]]
        if value is not None:
            _validate_evaluation(route, arm, *value)
    reference_id = "own_big_reference" if loaded["own_big_reference"] is not None else None
    source_directory = route.get("source_prompt", {}).get("optimization_dir")
    source_cost = optimization_cost(source_directory) if source_directory is not None else {
        key: None for key in COST_FIELDS}
    result = {
        "id": route["id"], "bench": route["bench"], "metric": metric,
        "source_model": route["source_model"], "target_small_model": route["target_small_model"],
        "target_model": route["target_model"], "target_family": route.get("target_family"),
        "source_prompt": route.get("source_prompt"), "source_optimization": source_cost,
        "source_optimization_seconds_scope": _cost_scope(source_directory),
        "settings": route.get("settings", settings), "reference_id": reference_id,
        "n": len(frozen), "test_ids_sha256": hashlib.sha256(json.dumps(
            sorted(frozen), separators=(",", ":")).encode()).hexdigest(),
        "ci_method": "95% percentile bootstrap over paired question scores, 2000 resamples, seed 0",
        "cost_method": "Analytical forward FLOPs from local/cost.py, task plus reflection; no hardware measurement",
        "arms": [], "contrasts": [],
    }
    for spec in specs:
        marginal = optimization_cost(spec["optimization_dir"])
        inherited = bool(spec.get("source_inherited", False))
        arm = {
            **spec, "method": ARM_IDS.index(spec["id"]) + 1,
            "status": "complete" if loaded[spec["id"]] is not None else "pending",
            "prompt_ready": Path(spec["program"]).is_file(),
            "marginal_optimization": marginal,
            "end_to_end_optimization": _add_cost(marginal, source_cost) if inherited else dict(marginal),
            "actual_rollouts": optimization_rollouts(spec["optimization_dir"]),
            "optimization_seconds_scope": _cost_scope(spec["optimization_dir"]),
            "score_pct": None, "score_ci_pct": None, "delta_pct": None, "delta_ci_pct": None,
            "errors": None, "inference": None, "lifecycle_vs_reference": None,
        }
        if loaded[spec["id"]] is not None:
            summary, rows = loaded[spec["id"]]
            scores = [100 * rows[key][metric] for key in sorted(rows)]
            arm["score_pct"], arm["score_ci_pct"] = mean(scores), bootstrap_ci(scores)
            if reference_id is not None:
                reference_rows = loaded[reference_id][1]
                delta = [100 * (rows[key][metric] - reference_rows[key][metric]) for key in sorted(rows)]
                arm["delta_pct"], arm["delta_ci_pct"] = mean(delta), bootstrap_ci(delta)
            errors_key = "errors_no_answer" if metric == "f1" else "errors_no_output"
            arm["errors"] = summary.get(errors_key)
            if arm["errors"] is None:
                arm["errors"] = sum(row.get("answer") is None if metric == "f1" else bool(row.get("no_output"))
                                    for row in rows.values())
            arm["inference"] = _inference_cost(summary, len(rows))
        result["arms"].append(arm)
    if reference_id:
        reference = next(arm for arm in result["arms"] if arm["id"] == reference_id)
        for arm in result["arms"]:
            if arm["status"] == "complete" and arm["id"] != reference_id:
                arm["lifecycle_vs_reference"] = _break_even(arm, reference)
    for arm_id, comparison in (("warm_small", "direct"), ("warm_big", "direct"), ("warm_big", "warm_small")):
        if loaded[arm_id] is None or loaded[comparison] is None:
            continue
        left, right = loaded[arm_id][1], loaded[comparison][1]
        delta = [100 * (left[key][metric] - right[key][metric]) for key in sorted(left)]
        result["contrasts"].append({"arm": arm_id, "versus": comparison,
                                     "delta_pct": mean(delta), "delta_ci_pct": bootstrap_ci(delta)})
    return result


def build_report(manifest):
    routes = manifest["routes"]
    if len({route["id"] for route in routes}) != len(routes):
        raise ValueError("Route IDs must be unique")
    reports = [build_route_report(route, manifest.get("settings")) for route in routes]
    evaluations = {}
    for route in reports:
        for arm in route["arms"]:
            directory = str(Path(arm["eval_dir"]).resolve())
            descriptor = (route["bench"], route["target_model"], route["test_ids_sha256"],
                          str(Path(arm["program"]).resolve()))
            if directory in evaluations and evaluations[directory][0] != descriptor:
                raise ValueError(f"Shared evaluation has incompatible consumers: {directory}")
            evaluations[directory] = (descriptor, arm["status"])
    return {
        "schema_version": 1, "queue_id": manifest.get("queue_id"),
        "settings": manifest.get("settings"), "stage_order": manifest.get("stages"),
        "route_count": len(reports), "method_cells": len(reports) * len(ARM_IDS),
        "completed_method_cells": sum(arm["status"] == "complete" for route in reports for arm in route["arms"]),
        "unique_evaluations": len(evaluations),
        "completed_unique_evaluations": sum(value[1] == "complete" for value in evaluations.values()),
        "routes": reports,
    }


def route_markdown(report):
    metric = "HotpotQA answer F1" if report["metric"] == "f1" else "HoVer article recall"
    lines = [f"# {report['id']}", "",
             f"Source: **{report['source_model']}**. Bridge: **{report['target_small_model']}**. "
             f"All answers use **{report['target_model']}**.", "",
             f"Primary metric: **{metric} (%)**; frozen test set: **{report['n']} items**.", "",
             "| Method | Prompt ready | Evaluation | Score [95% CI] | Difference from method 1 [95% CI], pp | Errors |",
             "|---|---|---|---:|---:|---:|"]
    for arm in report["arms"]:
        lines.append(f"| {METHOD_NAMES[arm['id']]} | {'yes' if arm['prompt_ready'] else 'no'} | {arm['status']} | "
                     f"{_ci(arm['score_pct'], arm['score_ci_pct'])} | "
                     f"{_ci(arm['delta_pct'], arm['delta_ci_pct'], True)} | "
                     f"{arm['errors'] if arm['errors'] is not None else 'pending'} |")
    lines += ["", f"{report['ci_method']}. Confidence intervals describe question sampling, "
              "not optimization-seed or inference-repeat variation. Pending means no completed frozen-set evaluation.", "",
              "Method 1 reuses historical full-budget target-model prompts, but its accuracy above comes from the new "
              "paired evaluation. It is a reference with a different optimization budget, not an accuracy ceiling.", ""]
    settings = report.get("settings")
    if settings:
        lines += [f"New adaptation runs: {settings['train_size']} train / {settings['val_size']} validation items, "
                  f"requested {settings['budget']} rollouts, optimizer seed {settings['seed']}, split seed "
                  f"{settings['split_seed']}. The actual budget can overshoot at GEPA iteration boundaries.", ""]
    if report["contrasts"]:
        lines += ["| Additional paired comparison | Difference [95% CI], pp |", "|---|---:|"]
        for contrast in report["contrasts"]:
            lines.append(f"| Method {ARM_IDS.index(contrast['arm']) + 1} minus method "
                         f"{ARM_IDS.index(contrast['versus']) + 1} | "
                         f"{_ci(contrast['delta_pct'], contrast['delta_ci_pct'], True)} |")
        lines.append("")
    lines += ["| Method | Actual rollouts | Migration tokens in / out (M) | Migration PFLOP | Reported optimizer h | "
              "End-to-end PFLOP | Inference tokens in / out per item | Inference TFLOP/item |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in report["arms"]:
        cost, total, inf = arm["marginal_optimization"], arm["end_to_end_optimization"], arm["inference"] or {}
        lines.append(f"| {arm['method']} | {_fmt(arm['actual_rollouts'], digits=0)} | "
                     f"{_fmt(cost['prompt_tokens'], 1e6, 2)} / {_fmt(cost['completion_tokens'], 1e6, 2)} | "
                     f"{_fmt(cost['flops'], 1e15, 2)} | {_fmt(cost['seconds'], 3600, 2)} | "
                     f"{_fmt(total['flops'], 1e15, 2)} | "
                     f"{_fmt(inf.get('prompt_tokens_per_item'), digits=0)} / "
                     f"{_fmt(inf.get('completion_tokens_per_item'), digits=0)} | "
                     f"{_fmt(inf.get('flops_per_item'), 1e12, 2)} |")
    lines += ["", "Migration cost excludes the original source optimization. End-to-end cost adds it to methods 2–4. "
              "Method 1's optimization cost is historical spending, not new campaign spending. Shared method 1 "
              "evaluations are run once per target and benchmark and reused by both incoming routes.", "",
              report["cost_method"] + ". Tokens use each model's tokenizer. Legacy optimizer hours may cover only "
              "one session; inspect the JSON duration scopes before drawing wall-time conclusions.", "",
              "HoVer retains one historical train/validation overlap; no train/test or validation/test overlap. "
              "HoVer measures retrieval recall, not claim-verdict accuracy.", ""]
    return "\n".join(lines) + "\n"


def markdown_report(report):
    lines = ["# Four-method cross-family prompt migration", "",
             f"Completed **{report['completed_method_cells']}/{report['method_cells']} method cells** across "
             f"**{report['route_count']} route/benchmark combinations**. Shared reference evaluations are counted "
             f"once for execution: **{report['completed_unique_evaluations']}/{report['unique_evaluations']}** complete.", "",
             "All scores below are held-out large-model inference results. Pending cells have no accuracy estimate.", "",
             "Execution order: evaluate methods 1–3 for every route; optimize method 4; evaluate method 4.", "",
             "| Route / benchmark | Target large model | Method 1 | Method 2 | Method 3 | Method 4 |",
             "|---|---|---:|---:|---:|---:|"]
    for route in report["routes"]:
        arms = {arm["id"]: arm for arm in route["arms"]}
        scores = [_ci(arms[key]["score_pct"], arms[key]["score_ci_pct"]) for key in ARM_IDS]
        lines.append(f"| [{route['id']}](routes/{route['id']}/report.md) | {route['target_model']} | "
                     + " | ".join(scores) + " |")
    lines += ["", "Methods: 1 = historical target-large own prompts; 2 = source prompts unchanged; "
              "3 = source prompts adapted on target-small; 4 = source prompts adapted on target-large. "
              "Each route report contains paired differences, error counts, and optimization/inference costs. "
              "Do not average different benchmark metrics or interpret validation-selection scores as test results.", ""]
    return "\n".join(lines) + "\n"


def write_report(report, out_dir):
    out = Path(out_dir)
    for route in report["routes"]:
        route_path = Path(route["id"])
        if route_path.is_absolute() or ".." in route_path.parts:
            raise ValueError(f"Invalid route ID for report output: {route['id']}")
        destination = out / "routes" / route_path
        _atomic(destination / "report.json", json.dumps(route, indent=2) + "\n")
        _atomic(destination / "report.md", route_markdown(route))
    _atomic(out / "report.json", json.dumps(report, indent=2) + "\n")
    _atomic(out / "report.md", markdown_report(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(_read(args.manifest))
        write_report(report, args.out)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.out / "report.md"),
                      "completed_method_cells": report["completed_method_cells"],
                      "method_cells": report["method_cells"]}))


if __name__ == "__main__":
    main()
