"""Export standalone figures for the four-method migration comparison.

The default requires all 48 held-out method cells. ``--allow-partial`` is an
explicit preview: pending accuracy remains absent and is labelled as pending.
No optimization and inference costs are combined into a campaign total.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


METHODS = ("own_big_reference", "direct", "warm_small", "warm_big")
COLORS = {"own_big_reference": "#0072B2", "direct": "#E69F00",
          "warm_small": "#009E73", "warm_big": "#CC79A7"}
MARKERS = dict(zip(METHODS, ("s", "o", "^", "D")))
LABELS = {"own_big_reference": "M1  Target-large own prompts (historical full budget)",
          "direct": "M2  Direct transfer (zero migration optimization)",
          "warm_small": "M3  Adapt on target-small",
          "warm_big": "M4  Adapt on target-large"}
DIRECTIONS = ("qwen8-to-llama8", "llama8-to-qwen8", "qwen8-to-gemma12",
              "gemma12-to-qwen8", "llama8-to-gemma12", "gemma12-to-llama8")
ROUTE_LABELS = {
    "qwen8-to-llama8": "Qwen8 → Llama70\nQwen–Llama",
    "llama8-to-qwen8": "Llama8 → Qwen32\nQwen–Llama",
    "qwen8-to-gemma12": "Qwen8 → Gemma27\nQwen–Gemma",
    "gemma12-to-qwen8": "Gemma12 → Qwen32\nQwen–Gemma",
    "llama8-to-gemma12": "Llama8 → Gemma27\nLlama–Gemma",
    "gemma12-to-llama8": "Gemma12 → Llama70\nLlama–Gemma",
}
METRICS = {"hotpotqa": ("f1", "HotpotQA answer F1 (%)"),
           "hover": ("recall", "HoVer article recall (%)")}
OFFSETS = (-0.27, -0.09, 0.09, 0.27)
CAPTION = ("FLOPs are analytical forward arithmetic, not measured GPU work. "
           "Shared M1 references appear in both incoming routes; these plots do not sum campaign costs.")


def _number(value, name, maximum=None):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative finite {name}: {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}: {value}")
    return value


def records_for_plotting(report, allow_partial=False):
    """Validate coverage independently of the report's cached progress counts."""
    routes = report["routes"]
    expected = {f"{bench}/{direction}" for bench in METRICS for direction in DIRECTIONS}
    ids = [route["id"] for route in routes]
    if len(ids) != len(set(ids)) or set(ids) - expected:
        raise ValueError("Route IDs must be unique recognized benchmark/direction combinations")
    if not allow_partial and set(ids) != expected:
        raise ValueError("Complete plots require all 12 route/benchmark combinations and 48 method cells")
    rows = []
    for route in routes:
        bench = route["bench"]
        if bench not in METRICS or route["metric"] != METRICS[bench][0] or not route["id"].startswith(bench + "/"):
            raise ValueError(f"Benchmark/metric mismatch: {route['id']}")
        if route.get("n") != 300:
            raise ValueError(f"Expected the frozen 300-question evaluation: {route['id']}")
        arms = route["arms"]
        if len(arms) != 4 or {arm["id"] for arm in arms} != set(METHODS):
            raise ValueError(f"Expected four unique methods in {route['id']}")
        for arm in sorted(arms, key=lambda value: METHODS.index(value["id"])):
            complete = arm["status"] == "complete"
            score = _number(arm.get("score_pct"), "score_pct", 100)
            interval = arm.get("score_ci_pct")
            if complete:
                if score is None or not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise ValueError(f"Complete evaluation needs a score and 95% CI: {route['id']}/{arm['id']}")
                low, high = [_number(value, "score CI", 100) for value in interval]
                if low is None or high is None or low > high:
                    raise ValueError("Invalid accuracy confidence interval")
            else:
                if score is not None or interval is not None:
                    raise ValueError(f"Pending evaluation must not carry an accuracy estimate: {route['id']}/{arm['id']}")
                if not allow_partial:
                    raise ValueError("Complete plots require all 48 method cells; use allow_partial for a pending preview")
                low = high = None
            migration = arm.get("marginal_optimization") or {}
            end_to_end = arm.get("end_to_end_optimization") or {}
            inference = arm.get("inference") or {}
            if not complete and any(inference.get(key) is not None for key in (
                    "flops_per_item", "prompt_tokens_per_item", "completion_tokens_per_item")):
                raise ValueError("Pending evaluation must not carry completed inference costs")
            rows.append({
                "bench": bench, "metric": route["metric"], "route": route["id"],
                "source_model": route["source_model"], "target_small_model": route["target_small_model"],
                "target_model": route["target_model"], "method": METHODS.index(arm["id"]) + 1,
                "arm": arm["id"], "evaluation_status": arm["status"],
                "historical_full_budget_prompt": arm["id"] == "own_big_reference",
                "prompt_ready": arm.get("prompt_ready"), "n": route["n"],
                "score_pct": score, "score_ci_low_pct": low, "score_ci_high_pct": high,
                "paired_difference_from_method1_pp": arm.get("delta_pct"),
                "paired_difference_ci_low_pp": (arm.get("delta_ci_pct") or [None, None])[0],
                "paired_difference_ci_high_pp": (arm.get("delta_ci_pct") or [None, None])[1],
                "migration_flops": _number(migration.get("flops"), "migration FLOPs"),
                "migration_pflop": None if migration.get("flops") is None else migration["flops"] / 1e15,
                "end_to_end_flops": _number(end_to_end.get("flops"), "end-to-end FLOPs"),
                "end_to_end_pflop": None if end_to_end.get("flops") is None else end_to_end["flops"] / 1e15,
                "inference_flops_per_item": _number(inference.get("flops_per_item"), "inference FLOPs/item"),
                "inference_tflop_per_item": None if inference.get("flops_per_item") is None else inference["flops_per_item"] / 1e12,
                "inference_prompt_tokens_per_item": _number(inference.get("prompt_tokens_per_item"), "inference input tokens/item"),
                "inference_completion_tokens_per_item": _number(inference.get("completion_tokens_per_item"), "inference output tokens/item"),
                "actual_rollouts": arm.get("actual_rollouts"), "errors": arm.get("errors"),
                "eval_dir": arm.get("eval_dir"), "optimization_dir": arm.get("optimization_dir"),
            })
    return sorted(rows, key=lambda row: (list(METRICS).index(row["bench"]),
                                         DIRECTIONS.index(row["route"].split("/", 1)[1]), row["method"]))


def _library():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    return plt, Line2D


def _legend(fig, line_class):
    handles = [line_class([], [], color=COLORS[method], marker=MARKERS[method],
                          linestyle="none", markersize=6, label=LABELS[method]) for method in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2,
               frameon=False, fontsize=9, columnspacing=2, handletextpad=0.6)


def _decorate(ax, rows, ylabel, ylim=None):
    directions = [direction for direction in DIRECTIONS if any(row["route"].endswith("/" + direction) for row in rows)]
    ax.set_xticks(range(len(directions)), [ROUTE_LABELS[direction] for direction in directions], fontsize=9)
    ax.set_xlim(-0.55, len(directions) - 0.45)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    for boundary in (1.5, 3.5):
        if boundary < len(directions) - 0.5:
            ax.axvline(boundary, color="#cccccc", linewidth=0.6)
    return directions


def _pending(ax, rows, directions, field):
    for x, direction in enumerate(directions):
        missing = [f"M{row['method']}" for row in rows if row["route"].endswith("/" + direction) and row[field] is None]
        if missing:
            ax.text(x, 0.025, "Pending: " + ", ".join(missing), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color="#666666", rotation=0)


def _save(fig, output, bench, kind):
    base = output / f"{bench}_{kind}"
    fig.savefig(base.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    return {"bench": bench, "kind": kind, "png": str(base.with_suffix(".png")), "pdf": str(base.with_suffix(".pdf"))}


def _title(fig, title, partial):
    fig.suptitle(title + (" — partial preview" if partial else ""), y=0.99, fontsize=14, fontweight="semibold")


def _footer(fig, text):
    import textwrap
    fig.text(0.06, 0.018, textwrap.fill(text, 155), ha="left", va="bottom", fontsize=8, color="#444444")


def _accuracy(plt, line_class, rows, bench, output, partial):
    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.07, right=0.985)
    _title(fig, METRICS[bench][1].replace(" (%)", "") + ": four transfer methods", partial)
    _legend(fig, line_class)
    directions = _decorate(ax, rows, METRICS[bench][1], (0, 100))
    for row in rows:
        if row["score_pct"] is None:
            continue
        x = directions.index(row["route"].split("/", 1)[1]) + OFFSETS[row["method"] - 1]
        method, y = row["arm"], row["score_pct"]
        lo, hi = row["score_ci_low_pct"], row["score_ci_high_pct"]
        ax.vlines(x, lo, hi, color=COLORS[method], linewidth=1.3)
        ax.hlines([lo, hi], x - 0.025, x + 0.025, color=COLORS[method], linewidth=1.3)
        ax.plot(x, y, marker=MARKERS[method], color=COLORS[method], markersize=6, linestyle="none")
    _pending(ax, rows, directions, "score_pct")
    _footer(fig, "Whiskers: 95% paired-question bootstrap confidence intervals; question uncertainty only, "
            "not optimization-seed or sampling-repeat variation. M1 prompts used a historical full optimization budget. "
            "Each target answers the same frozen 300 questions; missing evaluations remain pending.")
    artifact = _save(fig, output, bench, "accuracy")
    plt.close(fig)
    return artifact


def _cost_bars(ax, rows, field):
    directions = _decorate(ax, rows, "Optimization compute (PFLOP)")
    values = [row[field] for row in rows if row[field] is not None]
    max_value = max(values, default=1) or 1
    ax.set_ylim(0, max_value * 1.17)
    for row in rows:
        value = row[field]
        if value is None:
            continue
        x = directions.index(row["route"].split("/", 1)[1]) + OFFSETS[row["method"] - 1]
        ax.bar(x, value, width=0.15, color=COLORS[row["arm"]], edgecolor="white", linewidth=0.3,
               hatch="//" if row["method"] == 1 else None)
        if value == 0:
            ax.plot(x, 0, marker="_", color=COLORS[row["arm"]], markersize=9, clip_on=False)
            ax.annotate("0", (x, 0), xytext=(0, 4), textcoords="offset points", fontsize=7, ha="center")
    _pending(ax, rows, directions, field)


def _optimization(plt, line_class, rows, bench, output, partial):
    fig, axes = plt.subplots(2, 1, figsize=(13.6, 10.2))
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.985, hspace=0.37)
    _title(fig, METRICS[bench][1].split(" (")[0] + ": optimization compute", partial)
    _legend(fig, line_class)
    for ax, field, title in zip(axes, ("migration_pflop", "end_to_end_pflop"), (
            "Migration only: source optimization is sunk; direct transfer is exactly zero",
            "End-to-end: includes source optimization for methods 2–4")):
        _cost_bars(ax, rows, field)
        ax.set_title(title, fontsize=10, loc="left", pad=10)
    _footer(fig, CAPTION + " M1's hatched bars show historical full-budget optimization, not new campaign spending. "
            "M3 and M4 request 1,200 rollouts, so optimization budgets are unequal. "
            "Unknown costs stay pending; linear axes preserve true zero cost. Costs include task and reflection.")
    artifact = _save(fig, output, bench, "optimization_flops")
    plt.close(fig)
    return artifact


def _inference(plt, line_class, rows, bench, output, partial):
    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.08, right=0.985)
    _title(fig, METRICS[bench][1].split(" (")[0] + ": large-model inference compute", partial)
    _legend(fig, line_class)
    directions = _decorate(ax, rows, "Inference compute per question (TFLOP)")
    values = [row["inference_tflop_per_item"] for row in rows if row["inference_tflop_per_item"] is not None]
    ax.set_ylim(0, (max(values, default=1) or 1) * 1.17)
    for row in rows:
        value = row["inference_tflop_per_item"]
        if value is None:
            continue
        x = directions.index(row["route"].split("/", 1)[1]) + OFFSETS[row["method"] - 1]
        ax.bar(x, value, width=0.15, color=COLORS[row["arm"]], edgecolor="white", linewidth=0.3)
        if value == 0:
            ax.plot(x, 0, marker="_", color=COLORS[row["arm"]], markersize=9, clip_on=False)
    _pending(ax, rows, directions, "inference_tflop_per_item")
    _footer(fig, CAPTION + " Mean compute covers the complete multi-module program per held-out question. "
            "Inference compute is separate from optimization compute. Missing inference measurements stay pending.")
    artifact = _save(fig, output, bench, "inference_flops_per_question")
    plt.close(fig)
    return artifact


def _tradeoff(plt, line_class, rows, bench, output, partial, scope):
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.0))
    fig.subplots_adjust(top=0.79, bottom=0.17, left=0.065, right=0.985, hspace=0.42, wspace=0.27)
    scope_label = "migration-only" if scope == "migration" else "end-to-end"
    field = "migration_pflop" if scope == "migration" else "end_to_end_pflop"
    _title(fig, METRICS[bench][1].split(" (")[0] + f": accuracy vs {scope_label} optimization", partial)
    _legend(fig, line_class)
    for ax, direction in zip(axes.flat, DIRECTIONS):
        current = [row for row in rows if row["route"].endswith("/" + direction)]
        ax.set_title(ROUTE_LABELS[direction].replace("\n", " · "), fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel(METRICS[bench][1], fontsize=9)
        ax.set_xlabel("Optimization compute (PFLOP)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2, linewidth=0.6)
        costs = [row[field] for row in current if row[field] is not None]
        xmax = max(costs, default=1) or 1
        ax.set_xlim(-0.035 * xmax, 1.12 * xmax)
        missing = []
        for row in current:
            if row["score_pct"] is None or row[field] is None:
                missing.append(f"M{row['method']}")
                continue
            x, method = row[field], row["arm"]
            ax.vlines(x, row["score_ci_low_pct"], row["score_ci_high_pct"], color=COLORS[method], linewidth=1.2)
            ax.plot(x, row["score_pct"], color=COLORS[method], marker=MARKERS[method], linestyle="none", markersize=6)
            ax.annotate(f"M{row['method']}", (x, row["score_pct"]), xytext=(4, 5),
                        textcoords="offset points", fontsize=8, color=COLORS[method])
        if missing or not current:
            ax.text(0.03, 0.03, "Pending: " + (", ".join(missing) if current else "route"),
                    transform=ax.transAxes, fontsize=8, color="#666666")
        ax.tick_params(labelsize=8)
    _footer(fig, CAPTION + " Vertical whiskers show 95% question-bootstrap CIs. "
            + ("Source optimization is excluded; direct transfer is exactly at x = 0. " if scope == "migration"
               else "Source optimization is included for methods 2–4. ")
            + "M1 used its historical full budget; M3/M4 request 1,200 rollouts. "
            "Linear x axes use route-specific ranges; no artificial epsilon or missing-cost zero is introduced.")
    artifact = _save(fig, output, bench, f"accuracy_vs_{scope}_optimization_flops")
    plt.close(fig)
    return artifact


def generate_plots(report, out_dir, allow_partial=False):
    rows = records_for_plotting(report, allow_partial=allow_partial)
    complete = sum(row["evaluation_status"] == "complete" for row in rows)
    partial = complete != 48 or len(rows) != 48
    if not rows:
        raise ValueError("No routes available to plot")
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt, line_class = _library()
    artifacts = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for bench in METRICS:
            current = [row for row in rows if row["bench"] == bench]
            if not current:
                continue
            artifacts.append(_accuracy(plt, line_class, current, bench, output, partial))
            artifacts.append(_optimization(plt, line_class, current, bench, output, partial))
            artifacts.append(_inference(plt, line_class, current, bench, output, partial))
            for scope in ("migration", "end_to_end"):
                artifacts.append(_tradeoff(plt, line_class, current, bench, output, partial, scope))
    csv_path = output / "plot_data.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output / "plot_data.json"
    json_path.write_text(json.dumps({"records": rows, "source_report": report}, indent=2) + "\n")
    metadata = {
        "schema_version": 1, "queue_id": report.get("queue_id"),
        "status": "partial_preview" if partial else "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete_method_cells": complete, "expected_method_cells": 48,
        "source_report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
        "artifacts": artifacts, "data_csv": str(csv_path), "data_json": str(json_path),
        "units": {"optimization": "PFLOP = 10^15 forward FLOPs", "inference": "TFLOP/question = 10^12 forward FLOPs/question"},
        "notes": [CAPTION, "Accuracy CIs cover question sampling only.",
                  "M1 uses historical full-budget prompts with new held-out inference.",
                  "M2 has exactly zero migration optimization, but includes source cost end-to-end.",
                  "Unknown costs and pending evaluations are missing, never filled with zero."],
    }
    (output / "plots.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = ["# Four-method prompt-transfer figures", "", f"Status: **{metadata['status']}**; "
             f"completed evaluation cells: **{complete}/48**.", "", CAPTION, "",
             "Each figure is exported as PNG and PDF. Underlying values and missing entries are preserved "
             "in `plot_data.csv` and `plot_data.json`.", ""]
    for item in artifacts:
        lines.append(f"- {item['bench']} · {item['kind']}: [{Path(item['png']).name}]({Path(item['png']).name}) "
                     f"/ [PDF]({Path(item['pdf']).name})")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        metadata = generate_plots(json.loads(args.report.read_text()), args.out, args.allow_partial)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
