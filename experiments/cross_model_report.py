"""Paired accuracy and cost report for the cross-model prompt-transfer pilot.

Only completed evaluations are compared. All of them must contain exactly the
same question IDs. Optimization costs include task and reflection calls; FLOPs
are the analytical forward-pass estimates written by local/cost.py, not device
measurements. The inherited source optimization is a sunk cost for migration
and is added separately in the end-to-end column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean


COST_FIELDS = ("prompt_tokens", "completion_tokens", "flops", "seconds")


def bootstrap_ci(values, *, samples=2000, seed=0):
    """Percentile CI of the mean, in the same units as the input values."""
    if not values:
        raise ValueError("Cannot bootstrap an empty evaluation")
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return [draws[int(0.025 * samples)], draws[min(samples - 1, int(0.975 * samples))]]


def _number(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative number for {name}: {value!r}")
    return value


def optimization_cost(directory):
    """None means not measured yet; an absent optimization is a known zero."""
    if directory is None:
        return {key: 0 for key in COST_FIELDS}
    done_path = Path(directory) / "done.json"
    if not done_path.exists():
        return {key: None for key in COST_FIELDS}
    done = json.loads(done_path.read_text())
    usage = done.get("usage", {}).get("all", {})
    return {
        **{key: _number(usage.get(key), f"{done_path}:{key}") for key in COST_FIELDS[:3]},
        "seconds": _number(done.get("seconds"), f"{done_path}:seconds"),
    }


def optimization_rollouts(directory):
    if directory is None:
        return 0
    path = Path(directory) / "done.json"
    if not path.exists():
        return None
    return _number(json.loads(path.read_text()).get("metric_calls_actual"), f"{path}:metric_calls_actual")


def _add_cost(a, b):
    return {key: a[key] + b[key] if a[key] is not None and b[key] is not None else None for key in COST_FIELDS}


def _load_evaluation(arm, metric):
    directory = Path(arm["eval_dir"])
    result_path, summary_path = directory / "results.jsonl", directory / "summary.json"
    if not result_path.exists() or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    rows = {}
    for line in result_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = row["id"]
        if not isinstance(item_id, str) or item_id in rows:
            raise ValueError(f"Invalid or duplicate question ID in {result_path}: {item_id!r}")
        score = _number(row.get(metric), f"{result_path}:{item_id}:{metric}")
        if score is None or score > 1:
            raise ValueError(f"Expected {metric} in [0, 1] for {item_id} in {result_path}")
        rows[item_id] = row
    if not rows or summary.get("n") != len(rows):
        raise ValueError(f"Summary/question count mismatch in {directory}: n={summary.get('n')}, rows={len(rows)}")
    if summary.get("metric", metric) != metric:
        raise ValueError(f"Metric mismatch in {summary_path}: {summary.get('metric')} != {metric}")
    return summary, rows


def _inference_cost(summary, n):
    return {
        "prompt_tokens_per_item": _number(summary.get("prompt_tokens_total"), "prompt_tokens_total") / n
        if summary.get("prompt_tokens_total") is not None else None,
        "completion_tokens_per_item": _number(summary.get("completion_tokens_total"), "completion_tokens_total") / n
        if summary.get("completion_tokens_total") is not None else None,
        "flops_per_item": _number(summary.get("flops_total"), "flops_total") / n
        if summary.get("flops_total") is not None else None,
        "seconds": _number(summary.get("seconds"), "evaluation seconds"),
    }


def _break_even(arm, reference):
    """Describe C_arm + Q*I_arm <= C_ref + Q*I_ref for nonnegative Q."""
    c_arm, c_ref = arm["marginal_optimization"]["flops"], reference["marginal_optimization"]["flops"]
    i_arm, i_ref = arm["inference"]["flops_per_item"], reference["inference"]["flops_per_item"]
    if any(x is None for x in (c_arm, c_ref, i_arm, i_ref)):
        return {"status": "unknown", "reason": "Optimization or inference FLOPs are missing"}
    delta_c, delta_i = c_arm - c_ref, i_arm - i_ref
    base = {"optimization_difference_flops": delta_c, "inference_difference_flops_per_item": delta_i}
    if delta_i == 0:
        return {**base, "status": "always_cheaper_or_equal" if delta_c <= 0 else "never_cheaper"}
    crossing = -delta_c / delta_i
    if delta_i > 0:
        if delta_c > 0:
            return {**base, "status": "never_cheaper"}
        return {**base, "status": "cheaper_until", "queries": crossing,
                "condition": f"0 <= Q <= {crossing:.1f}"}
    if delta_c <= 0:
        return {**base, "status": "always_cheaper_or_equal"}
    return {**base, "status": "cheaper_after", "queries": crossing,
            "condition": f"Q >= {crossing:.1f}"}


def build_report(manifest):
    metric = manifest["metric"]
    if metric not in ("f1", "recall"):
        raise ValueError(f"Unsupported primary metric: {metric}")
    arms = manifest["arms"]
    if len({arm["id"] for arm in arms}) != len(arms):
        raise ValueError("Arm IDs must be unique")
    source_dir = manifest.get("source_prompt", {}).get("optimization_dir")
    source_cost = optimization_cost(source_dir) if source_dir is not None else {key: None for key in COST_FIELDS}
    if any("optimization_dir" not in arm for arm in arms):
        raise ValueError("Every arm must explicitly set optimization_dir (null means no optimization)")
    loaded = {arm["id"]: _load_evaluation(arm, metric) for arm in arms}
    complete = {arm_id: value for arm_id, value in loaded.items() if value is not None}
    ids = None
    frozen = manifest.get("frozen_test_ids", manifest.get("test_ids"))
    if frozen is not None:
        if not frozen or len(frozen) != len(set(frozen)):
            raise ValueError("Frozen test IDs must be a nonempty unique list")
        ids = set(frozen)
    for arm_id, (_, rows) in complete.items():
        if ids is None:
            ids = set(rows)
        elif set(rows) != ids:
            raise ValueError(f"Question ID mismatch for {arm_id}: expected {len(ids)} IDs, got {len(rows)}; "
                             f"missing={len(ids - rows.keys())}, extra={len(rows.keys() - ids)}")
    reference_id = next((key for key in ("cold_big", "own_big_reference", "direct") if key in complete), None)
    result = {
        "bench": manifest["bench"], "metric": metric, "target_model": manifest["target_model"],
        "source_prompt": manifest.get("source_prompt"), "source_optimization": source_cost,
        "settings": manifest.get("settings"),
        "reference_id": reference_id, "n": len(ids) if ids is not None else None,
        "test_ids_sha256": hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
        if ids is not None else None,
        "ci_method": "95% percentile bootstrap over paired question scores, 2000 resamples, seed 0",
        "cost_method": "Analytical forward FLOPs from local/cost.py, task plus reflection; no hardware measurement",
        "arms": [], "contrasts": [],
    }
    for spec in arms:
        marginal = optimization_cost(spec.get("optimization_dir"))
        inherited = bool(spec.get("source_inherited", False))
        end_to_end = _add_cost(marginal, source_cost) if inherited else dict(marginal)
        arm = {
            **spec, "status": "complete" if loaded[spec["id"]] is not None else "pending",
            "marginal_optimization": marginal, "end_to_end_optimization": end_to_end,
            "actual_rollouts": optimization_rollouts(spec.get("optimization_dir")),
            "score_pct": None, "score_ci_pct": None, "delta_pct": None, "delta_ci_pct": None,
            "errors": None, "inference": None, "lifecycle_vs_reference": None,
        }
        if loaded[spec["id"]] is not None:
            summary, rows = loaded[spec["id"]]
            scores = [100 * rows[key][metric] for key in sorted(rows)]
            arm["score_pct"], arm["score_ci_pct"] = mean(scores), bootstrap_ci(scores)
            if reference_id:
                reference_rows = complete[reference_id][1]
                differences = [100 * (rows[key][metric] - reference_rows[key][metric]) for key in sorted(rows)]
                arm["delta_pct"], arm["delta_ci_pct"] = mean(differences), bootstrap_ci(differences)
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
    for comparison in ("direct", "cold_small", "own_big_reference", "warm_big", "cold_big"):
        if "warm_small" not in complete or comparison not in complete:
            continue
        bridge_rows, other_rows = complete["warm_small"][1], complete[comparison][1]
        delta = [100 * (bridge_rows[key][metric] - other_rows[key][metric]) for key in sorted(bridge_rows)]
        result["contrasts"].append({"arm": "warm_small", "versus": comparison,
                                    "delta_pct": mean(delta), "delta_ci_pct": bootstrap_ci(delta)})
    return result


def _fmt(value, divisor=1, digits=1):
    return "pending" if value is None else f"{value / divisor:,.{digits}f}"


def _ci(value, interval, signed=False):
    if value is None:
        return "pending"
    sign = "+" if signed else ""
    return f"{value:{sign}.1f} [{interval[0]:{sign}.1f}, {interval[1]:{sign}.1f}]"


def markdown_report(report):
    metric_name = "answer F1" if report["metric"] == "f1" else "article recall"
    lines = [f"# Cross-model prompt transfer: {report['bench']}", "",
             f"Target: **{report['target_model']}**. Primary metric: **{metric_name}**, %. "
             f"Paired evaluation items: **{report['n'] if report['n'] is not None else 'pending'}**.", "",
             "Historical references use the original full optimization budget and are not budget matched to the pilot. "
             "Pending rows have no accuracy estimate. Every completed row must have the same question IDs.", "",
             "| Arm | Status | Score [95% CI] | Difference from reference [95% CI], pp | Errors |",
             "|---|---|---:|---:|---:|"]
    for arm in report["arms"]:
        label = arm.get("label", arm["id"]) + (" (historical)" if arm.get("historical") else "")
        lines.append(f"| {label} | {arm['status']} | {_ci(arm['score_pct'], arm['score_ci_pct'])} | "
                     f"{_ci(arm['delta_pct'], arm['delta_ci_pct'], True)} | "
                     f"{arm['errors'] if arm['errors'] is not None else 'pending'} |")
    lines += ["", f"Reference: `{report['reference_id'] or 'pending'}`. {report['ci_method']}. "
              "These intervals describe question sampling and do not measure optimizer seed variation.", ""]
    if report.get("settings"):
        settings = report["settings"]
        lines += [f"Pilot: {settings['train_size']} train / {settings['val_size']} validation items; "
                  f"requested budget {settings['budget']} rollouts, optimizer seed {settings['seed']}, "
                  f"split seed {settings['split_seed']}. Actual rollout counts can exceed the request at "
                  "GEPA iteration boundaries.", ""]
    if report.get("contrasts"):
        lines += ["| Paired comparison | Difference [95% CI], pp |", "|---|---:|"]
        for contrast in report["contrasts"]:
            lines.append(f"| `{contrast['arm']}` minus `{contrast['versus']}` | "
                         f"{_ci(contrast['delta_pct'], contrast['delta_ci_pct'], True)} |")
        lines.append("")
    lines += ["| Arm | Actual rollouts | Migration optimization tokens in / out (M) | Migration PFLOP | Optimization h | "
              "End-to-end tokens in / out (M) | End-to-end PFLOP | Inference tokens in / out per item | Inference TFLOP/item |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in report["arms"]:
        cost, total = arm["marginal_optimization"], arm["end_to_end_optimization"]
        inf = arm["inference"] or {}
        lines.append(f"| {arm.get('label', arm['id'])} | {_fmt(arm['actual_rollouts'], digits=0)} | "
                     f"{_fmt(cost['prompt_tokens'], 1e6, 2)} / "
                     f"{_fmt(cost['completion_tokens'], 1e6, 2)} | {_fmt(cost['flops'], 1e15, 2)} | "
                     f"{_fmt(cost['seconds'], 3600, 2)} | {_fmt(total['prompt_tokens'], 1e6, 2)} / "
                     f"{_fmt(total['completion_tokens'], 1e6, 2)} | {_fmt(total['flops'], 1e15, 2)} | "
                     f"{_fmt(inf.get('prompt_tokens_per_item'), digits=0)} / "
                     f"{_fmt(inf.get('completion_tokens_per_item'), digits=0)} | "
                     f"{_fmt(inf.get('flops_per_item'), 1e12, 2)} |")
    lines += ["", "Migration optimization excludes the existing Qwen source optimization. End-to-end adds that source "
              "cost to every arm that inherits the Qwen prompt. Historical optimization costs are the original research "
              "costs, not newly incurred pilot spending. Zero means no optimization is required; pending means cost has "
              "not been measured. All optimization costs include task and reflection calls. Token counts are measured "
              "with each model's tokenizer and are not directly interchangeable across families.", "",
              report["cost_method"] + ". Precision changes runtime and memory without changing the counted arithmetic. "
              "Wall times depend on serving layout, traffic and hardware contention; they are reported separately. "
              "Historical done.json seconds may describe only the final session; unknown cumulative time after "
              "an interruption is left pending, and no wall-time saving is inferred from that field.", "",
              "Lifecycle FLOPs for Q future target-model queries are `C_optimization + Q × I_per_query`. "
              "The conditions below compare marginal migration optimization with the reference and apply for Q ≥ 0. "
              "They compare compute cost only; accuracy remains a separate measurement.", ""]
    for arm in report["arms"]:
        crossing = arm["lifecycle_vs_reference"]
        if crossing is None:
            continue
        status = crossing["status"]
        text = {"unknown": "pending cost measurements", "always_cheaper_or_equal": "cheaper or equal for all Q ≥ 0",
                "never_cheaper": "never cheaper for Q ≥ 0"}.get(status)
        if text is None:
            text = f"cheaper or equal when {crossing['condition']}"
        lines.append(f"- {arm.get('label', arm['id'])}: {text}.")
    return "\n".join(lines) + "\n"


def plot_report(report, path):
    """Return False when matplotlib is unavailable or no complete costs exist."""
    complete = [arm for arm in report["arms"] if arm["status"] == "complete"
                and arm["marginal_optimization"]["flops"] is not None]
    if not complete:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for arm in complete:
        x, y = arm["marginal_optimization"]["flops"] / 1e15, arm["score_pct"]
        lo, hi = arm["score_ci_pct"]
        historical = bool(arm.get("historical"))
        label = arm.get("label", arm["id"]) + (" (historical)" if historical else "")
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="s" if historical else "o", capsize=3, label=label)
    # A linear axis preserves the direct-transfer point's genuine zero cost.
    ax.set(xlabel="Marginal migration optimization (PFLOP; task + reflection)",
           ylabel="Answer F1 (%)" if report["metric"] == "f1" else "Article recall (%)",
           title=f"{report['bench']}: prompts evaluated on {report['target_model']}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def write_report(report, out_dir, *, make_plot=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.md").write_text(markdown_report(report))
    return plot_report(report, out / "accuracy_vs_optimization_flops.png") if make_plot else False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        report = build_report(manifest)
        plotted = write_report(report, args.out, make_plot=not args.no_plot)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.out / "report.md"), "complete_arms": sum(
        arm["status"] == "complete" for arm in report["arms"]), "plot_written": plotted}))


if __name__ == "__main__":
    main()
