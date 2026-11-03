"""Export one scientific overview sheet from fresh four-method campaign evidence.

Only complete frozen-test evaluations receive accuracy or inference cost. A
completed optimizer can have a known cost while its held-out evaluation remains
pending. Existing campaign state/reports/plots are read and never modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from migration_report import build_report
from migration_plots import (COLORS, DIRECTIONS, MARKERS, METHODS,
                             METRICS, OFFSETS, records_for_plotting)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "artifacts/migration_20261101"
SHORT_ROUTES = {
    "qwen8-to-llama8": "Q8 → L70", "llama8-to-qwen8": "L8 → Q32",
    "qwen8-to-gemma12": "Q8 → G27", "gemma12-to-qwen8": "G12 → Q32",
    "llama8-to-gemma12": "L8 → G27", "gemma12-to-llama8": "G12 → L70",
}
LEGEND_LABELS = {
    "own_big_reference": "M1  Target-large own prompts · historical full budget",
    "direct": "M2  Source prompts · direct transfer",
    "warm_small": "M3  Adapt on target-small",
    "warm_big": "M4  Adapt on target-large",
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overview_records(report):
    """Keep test-only statistics and independently gate completed optimizer cost."""
    rows = records_for_plotting(report, allow_partial=True)
    specs = {(route["id"], arm["id"]): arm for route in report["routes"] for arm in route["arms"]}
    for row in rows:
        spec = specs[row["route"], row["arm"]]
        directory = spec.get("optimization_dir")
        done = Path(directory) / "done.json" if directory else None
        training_complete = directory is None or done.is_file()
        row["optimization_status"] = ("not_required" if directory is None else
                                      "complete" if training_complete else "pending")
        row["optimization_done_path"] = str(done) if done else None
        row["optimization_done_sha256"] = _digest(done) if done and training_complete else None
        row["training_complete_evaluation_pending"] = bool(
            training_complete and directory is not None and row["evaluation_status"] != "complete")
        # Incomplete optimization must not become a completed-cost estimate,
        # even if a future upstream helper reports live partial consumption.
        if not training_complete:
            for field in ("migration_flops", "migration_pflop", "end_to_end_flops", "end_to_end_pflop"):
                row[field] = None
        row["optimization_display_status"] = (
            "training complete / evaluation pending" if row["training_complete_evaluation_pending"] else
            "optimization pending / cost unknown" if not training_complete else "complete")
    return rows


def _axis(ax, ylabel, title, ceiling=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=12)
    ax.set_xticks(range(6), [SHORT_ROUTES[direction] for direction in DIRECTIONS], fontsize=8.5)
    ax.set_xlim(-0.55, 5.55)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b5b5")
    ax.tick_params(axis="y", labelsize=9, colors="#404040")
    ax.tick_params(axis="x", length=0, pad=7)
    ax.grid(axis="y", color="#d9dfe6", linewidth=0.6)
    ax.set_axisbelow(True)
    if ceiling is not None:
        ax.set_ylim(0, ceiling)
    for boundary in (1.5, 3.5):
        ax.axvline(boundary, color="#d9dfe6", linewidth=0.8)


def _location(row):
    return DIRECTIONS.index(row["route"].split("/", 1)[1]) + OFFSETS[row["method"] - 1]


def _missing(ax, rows, field, label):
    for index, direction in enumerate(DIRECTIONS):
        missing = [f"M{row['method']}" for row in rows
                   if row["route"].endswith("/" + direction) and row[field] is None]
        if missing:
            ax.text(index, 0.025, ", ".join(missing) + "\n" + label,
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7, color="#777f88", linespacing=1.1,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})


def _accuracy(ax, rows, bench):
    name = "HotpotQA · answer F1" if bench == "hotpotqa" else "HoVer · article recall"
    _axis(ax, "Test score (%)", name, 100)
    for row in rows:
        if row["score_pct"] is None:
            continue
        x, color = _location(row), COLORS[row["arm"]]
        lo, hi = row["score_ci_low_pct"], row["score_ci_high_pct"]
        ax.vlines(x, lo, hi, color=color, linewidth=1.4)
        ax.hlines([lo, hi], x - 0.025, x + 0.025, color=color, linewidth=1.2)
        ax.plot(x, row["score_pct"], color=color, marker=MARKERS[row["arm"]],
                markersize=5.8, linestyle="none")
    _missing(ax, rows, "score_pct", "test pending")


def _compute(ax, rows, field, title, ylabel, optimization=False):
    values = [row[field] for row in rows if row[field] is not None]
    ymax = max(values, default=1) or 1
    _axis(ax, ylabel, title, ymax * 1.20)
    for row in rows:
        value = row[field]
        if value is None:
            continue
        x, color = _location(row), COLORS[row["arm"]]
        pending = optimization and row["training_complete_evaluation_pending"]
        ax.bar(x, value, width=0.15, color="white" if pending else color,
               edgecolor=color, linewidth=1.0 if pending else 0.35,
               hatch="...." if pending else None, zorder=3)
        if value == 0:
            ax.plot(x, 0, marker="_", color=color, markersize=8, clip_on=False, zorder=4)
            ax.annotate("0", (x, 0), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color=color)
        if pending:
            ax.annotate("†", (x, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=10, color=color, fontweight="bold")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    _missing(ax, rows, field, "optimization\npending" if optimization else "test pending")


def _render_sheet(report, rows, output, cost_scope, created):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    complete = sum(row["evaluation_status"] == "complete" for row in rows)
    m4_complete = sum(row["arm"] == "warm_big" and row["evaluation_status"] == "complete" for row in rows)
    m4_training = sum(row["arm"] == "warm_big" and row["optimization_status"] == "complete" for row in rows)
    local_time = created.astimezone(ZoneInfo("America/Toronto")).strftime("%b %d, %Y · %I:%M %p %Z")
    scope_text = "End-to-end optimization · source included" if cost_scope == "end_to_end" else "Migration optimization · source excluded"
    field = "end_to_end_pflop" if cost_scope == "end_to_end" else "migration_pflop"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.subplots_adjust(left=0.058, right=0.985, top=0.81, bottom=0.17, hspace=0.48, wspace=0.27)
        fig.suptitle("Prompt transfer: test performance and compute", x=0.058, ha="left",
                     y=0.982, fontsize=21, fontweight="semibold", color="#172a3b")
        fig.text(0.058, 0.943, f"{complete}/48 method cells evaluated  ·  M4: {m4_training}/12 optimizations complete, "
                 f"{m4_complete}/12 test evaluations complete  ·  {local_time}",
                 ha="left", fontsize=11, color="#586777")
        handles = [Line2D([], [], color=COLORS[arm], marker=MARKERS[arm], linestyle="none",
                          markersize=7, label=LEGEND_LABELS[arm]) for arm in METHODS]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.915),
                   ncol=2, frameon=False, fontsize=10.2, columnspacing=4.5, handletextpad=0.6)
        for index, bench in enumerate(METRICS):
            current = [row for row in rows if row["bench"] == bench]
            _accuracy(axes[index, 0], current, bench)
            _compute(axes[index, 1], current, field, scope_text, "Optimization compute (PFLOP)", True)
            _compute(axes[index, 2], current, "inference_tflop_per_item",
                     "Large-model inference per question", "Inference compute (TFLOP / question)")
        footer = [
            "Every plotted score is a completed 300-question frozen test evaluation. Whiskers: 95% question-bootstrap CIs; no validation/dev scores are shown.",
            "† Dotted optimization bars: training complete / test evaluation pending. Missing test scores and inference costs stay pending; unfinished optimization costs remain unknown, never zero.",
            "FLOPs are analytical forward arithmetic (task + reflection), not measured GPU work. M1 uses historical full-budget prompts; M3/M4 request 1,200 rollouts. Budgets are unequal.",
            ("Optimization includes source training for M2–M4. " if cost_scope == "end_to_end" else "Migration excludes source training; M2 is exactly zero. ") +
            "Shared M1 references repeat across incoming routes; do not sum plotted bars as campaign spending. Accuracy and inference always use the target-large model.",
            "Routes show source-small → target-large. Q = Qwen3 (8B/32B), L = Llama3.1 (8B/70B), G = Gemma3 (12B/27B); M3 adapts on the target family's small model. HoVer measures article recall.",
        ]
        for index, line in enumerate(footer):
            fig.text(0.058, 0.119 - index * 0.022, line, ha="left", va="top", fontsize=8.3,
                     color="#44515f" if index != 1 else "#674b68")
        png, pdf = output / f"overview_{cost_scope}.png", output / f"overview_{cost_scope}.pdf"
        fig.savefig(png, dpi=220, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)
    return {"scope": cost_scope, "png": str(png), "pdf": str(pdf), "notes": footer}


def generate_overview(manifest_path, out_dir, cost_scope="both"):
    """Build one fresh report, then draw both cost views from that same snapshot."""
    if cost_scope not in ("both", "end_to_end", "migration"):
        raise ValueError("Cost scope must be both, end_to_end, or migration")
    manifest_path = Path(manifest_path).resolve()
    manifest_raw = manifest_path.read_bytes()
    report = build_report(json.loads(manifest_raw))
    rows = overview_records(report)
    if len(rows) != 48:
        raise ValueError("Overview requires all12routes and four method slots; pending slots are allowed")
    output = Path(out_dir).resolve()
    if output == manifest_path.parent or output in (manifest_path.parent / "report", manifest_path.parent / "eval"):
        raise ValueError("Use a dedicated visualization output directory")
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    scopes = ("end_to_end", "migration") if cost_scope == "both" else (cost_scope,)
    artifacts = [_render_sheet(report, rows, output, scope, created) for scope in scopes]
    csv_path = output / "plot_data.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data_path = output / "plot_data.json"
    snapshot = {"created_utc": created.isoformat(), "cost_scopes": list(scopes),
                "manifest_path": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "records": rows, "source_report": report}
    data_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    complete = sum(row["evaluation_status"] == "complete" for row in rows)
    m4_complete = sum(row["arm"] == "warm_big" and row["evaluation_status"] == "complete" for row in rows)
    m4_training = sum(row["arm"] == "warm_big" and row["optimization_status"] == "complete" for row in rows)
    metadata = {"schema_version": 1, "created_utc": created.isoformat(),
        "status": "complete" if complete == 48 else "partial_held_out_results",
        "complete_method_cells": complete, "expected_method_cells": 48,
        "m4_optimizations_complete": m4_training, "m4_evaluations_complete": m4_complete,
        "cost_scopes": list(scopes), "page_inches": [18, 11], "image_pixels": [3960, 2420],
        "artifacts": artifacts, "png": artifacts[0]["png"], "pdf": artifacts[0]["pdf"],
        "data_csv": str(csv_path), "data_json": str(data_path),
        "manifest_path": str(manifest_path), "manifest_sha256": snapshot["manifest_sha256"],
        "data_snapshot_sha256": _digest(data_path),
        "source_sha256": {str(path): _digest(path) for path in
            [Path(__file__).resolve(), Path(__file__).with_name("migration_report.py"),
             Path(__file__).with_name("migration_plots.py")]},
        "units": {"accuracy": "percent", "optimization": "PFLOP = 10^15 analytical forward FLOPs",
                  "inference": "TFLOP/question = 10^12 analytical forward FLOPs/question"},
        "notes": artifacts[0]["notes"]}
    (output / "overview_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cost-scope", choices=["both", "end_to_end", "migration"], default="both")
    args = parser.parse_args()
    manifest = args.manifest or args.campaign / "manifest.json"
    output = args.out or args.campaign / "visualization/overview_20261102"
    print(json.dumps(generate_overview(manifest, output, args.cost_scope)))


if __name__ == "__main__":
    main()
