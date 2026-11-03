"""Package the scientific overview as a self-contained local HTML page."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo


METHODS = {
    1: "Target-large own prompts",
    2: "Direct transfer",
    3: "Adapt on target-small",
    4: "Adapt on target-large",
}


def number(value, digits=1):
    return "Pending" if value is None else f"{value:,.{digits}f}"


def render_page(directory: Path):
    directory = directory.resolve()
    data_path = directory / "plot_data.json"
    data = json.loads(data_path.read_text())
    rows = data["records"]
    complete = sum(row["evaluation_status"] == "complete" for row in rows)
    m4 = [row for row in rows if row["method"] == 4]
    trained = sum(row["migration_pflop"] is not None for row in m4)
    created = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Toronto"))
    images = {}
    for scope in ("end_to_end", "migration"):
        path = directory / f"overview_{scope}.png"
        images[scope] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    table = []
    for row in rows:
        score = "Pending"
        if row["score_pct"] is not None:
            score = f"{row['score_pct']:.2f} [{row['score_ci_low_pct']:.2f}, {row['score_ci_high_pct']:.2f}]"
        values = ["HotpotQA · F1" if row["bench"] == "hotpotqa" else "HoVer · article recall",
                  row["source_model"], row["target_model"], f"M{row['method']} · {METHODS[row['method']]}",
                  score, number(row["end_to_end_pflop"]), number(row["migration_pflop"]),
                  number(row["inference_tflop_per_item"])]
        cells = "".join("<td>" + html.escape(value) + "</td>" for value in values)
        table.append(f'<tr data-bench="{row["bench"]}">{cells}</tr>')
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prompt transfer — performance and compute</title>
<style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172338;background:#eef2f6}
*{box-sizing:border-box}body{margin:0}main{max-width:1600px;margin:24px auto;padding:0 24px 32px}
header{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}h1{font-size:28px;letter-spacing:-.7px;margin:0 0 8px;font-weight:700}
p{line-height:1.55;margin:6px 0}.subtle{color:#526277;font-size:14px}.badge{display:inline-block;background:#fff0d4;color:#78510c;border:1px solid #e8cd96;padding:7px 11px;border-radius:7px;font-size:13px;white-space:nowrap}
.toolbar{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin:20px 0 12px}
.switch{display:flex;gap:4px;padding:4px;border:1px solid #cad3df;border-radius:9px;background:white}
button{font:inherit;font-size:13px;cursor:pointer;border:0;border-radius:6px;background:transparent;color:#465970;padding:9px 14px}
button[aria-pressed=true]{background:#172e4e;color:white}.downloads{display:flex;gap:16px;font-size:13px}a{color:#254f81;text-underline-offset:3px}
.sheet{background:white;border:1px solid #d7dfe9;box-shadow:0 6px 28px #152b4510;border-radius:10px;overflow:hidden}.sheet img{display:block;width:100%;height:auto}
.scope{font-size:13px;margin:0 0 12px;color:#526277}.note{border-left:3px solid #d0a35c;padding:9px 13px;background:#fffaf0;font-size:13px;margin:14px 0}
details{background:white;border:1px solid #d7dfe9;border-radius:8px;margin-top:16px;padding:14px 16px}summary{font-weight:600;font-size:14px;cursor:pointer}
.definitions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 28px;margin-top:14px;font-size:13px}.definitions p{margin:0}
.table-wrap{overflow:auto;margin-top:14px}table{border-collapse:collapse;min-width:1120px;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:10px 9px;border-bottom:1px solid #e3e8ee;text-align:left;vertical-align:top}th{background:#f5f7fa;color:#405269;font-weight:600}td:nth-child(n+5){text-align:right;white-space:nowrap}.table-tools{margin-top:14px;font-size:13px}select{margin-left:8px;padding:5px;border:1px solid #cad3df;border-radius:5px}
footer{margin-top:16px;font-size:12px;color:#617186}.hidden{display:none!important}
@media(max-width:700px){main{padding:0 12px}header{display:block}h1{font-size:23px}.badge{margin-top:10px}.definitions{grid-template-columns:1fr}.downloads{gap:10px}.switch button{padding:8px 10px}}
@media print{body{background:white}main{padding:0;margin:0;max-width:none}header,.toolbar,.scope,.note,details,footer{display:none}.sheet{border:0;box-shadow:none;border-radius:0} @page{size:A3 landscape;margin:4mm}}
</style></head><body><main>
<header><div><h1>Prompt transfer · performance &amp; compute</h1>
<p class="subtle">@@STAMP@@ Toronto · 6 transfer directions · 2 benchmarks · 300 frozen test questions per benchmark</p></div>
<span class="badge">@@COMPLETE@@/48 test scores available</span></header>
<div class="toolbar"><div class="switch" role="group" aria-label="Optimization cost basis">
<button type="button" data-scope="end_to_end" aria-pressed="true">End-to-end optimization</button>
<button type="button" data-scope="migration" aria-pressed="false">Migration only</button></div>
<nav class="downloads" aria-label="Download results"><a id="pdf-link" href="overview_end_to_end.pdf">One-page PDF</a><a id="png-link" href="overview_end_to_end.png">PNG</a><a href="plot_data.csv">Data CSV</a></nav></div>
<p class="scope" id="scope-description">Includes the original source optimization for methods 2–4. Inference compute is shown separately.</p>
<section class="sheet" aria-label="Performance and FLOPs comparison"><img id="overview-image" alt="Two benchmark rows comparing all four methods on test performance, optimization FLOPs, and inference FLOPs per question. Pending evaluations are labeled." src="@@DEFAULT_IMAGE@@"></section>
<p class="note"><strong>Method 4:</strong> @@TRAINED@@/12 optimizations finished; @@M4SCORES@@/12 test evaluations finished. “Optimization pending” means the optimization is still running or queued, so its final FLOPs are unavailable. “Test pending” means the test score and inference FLOPs await test evaluation. Completed optimization costs are shown even when testing is pending.</p>
<details><summary>Metric and cost definitions</summary><div class="definitions">
<p><strong>Performance.</strong> HotpotQA uses answer F1; HoVer uses article recall, not claim-verdict accuracy. Whiskers show 95% question-bootstrap confidence intervals. Scores are never averaged across benchmarks.</p>
<p><strong>Optimization.</strong> End-to-end adds source optimization for methods 2–4; migration excludes it. Direct transfer therefore has zero migration optimization, while its end-to-end cost includes its source prompt.</p>
<p><strong>Inference.</strong> TFLOP per test question for the full multi-module program. It is separate from optimization PFLOP. Lifetime cost needs an explicit query count: optimization + queries × inference/query.</p>
<p><strong>Accounting.</strong> FLOPs are analytical forward-pass estimates including task and reflection calls, not measured GPU work. PFLOP = 10¹⁵ FLOPs; TFLOP = 10¹² FLOPs.</p>
<p><strong>Budgets.</strong> M1 uses historical full-budget target-large prompts. M3/M4 adaptations request 1,200 rollouts; the GEPA budget may overshoot at iteration boundaries. The optimization budgets are unequal.</p>
<p><strong>Shared references.</strong> M1 is evaluated once per target family and benchmark and reused by its two incoming routes. Repeated M1 cells are not separate campaign spending.</p>
</div></details>
<details><summary>Exact values · 48 method cells</summary><div class="table-tools"><label>Benchmark<select id="benchmark-filter"><option value="all">Both benchmarks</option><option value="hotpotqa">HotpotQA</option><option value="hover">HoVer</option></select></label></div>
<div class="table-wrap"><table><thead><tr><th>Benchmark / metric</th><th>Source</th><th>Target executor</th><th>Method</th><th>Test score % [95% CI]</th><th>End-to-end PFLOP</th><th>Migration PFLOP</th><th>Inference TFLOP/question</th></tr></thead><tbody>@@ROWS@@</tbody></table></div></details>
<footer>Snapshot of validated completed outputs. The experiment queue and its evaluation order are unchanged. <a href="plot_data.json">Source data JSON</a></footer>
</main><script>
const images = @@IMAGES@@;
document.querySelectorAll('button[data-scope]').forEach(button => button.addEventListener('click', () => {
  const scope = button.dataset.scope;
  document.querySelectorAll('button[data-scope]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
  document.getElementById('overview-image').src = images[scope];
  document.getElementById('pdf-link').href = 'overview_' + scope + '.pdf';
  document.getElementById('png-link').href = 'overview_' + scope + '.png';
  document.getElementById('scope-description').textContent = scope === 'end_to_end'
    ? 'Includes the original source optimization for methods 2–4. Inference compute is shown separately.'
    : 'Excludes the original source optimization. Direct transfer has exactly zero migration optimization. Inference compute is shown separately.';
}));
document.getElementById('benchmark-filter').addEventListener('change', event => {
  document.querySelectorAll('tbody tr').forEach(row => row.classList.toggle('hidden', event.target.value !== 'all' && row.dataset.bench !== event.target.value));
});
</script></body></html>
"""
    replacements = {
        "STAMP": created.strftime("%B %d, %Y · %-I:%M %p"),
        "COMPLETE": str(complete), "TRAINED": str(trained),
        "M4SCORES": str(sum(row["evaluation_status"] == "complete" for row in m4)),
        "DEFAULT_IMAGE": images["end_to_end"], "ROWS": "".join(table),
        "IMAGES": json.dumps(images).replace("<", "\\u003c"),
    }
    for name, value in replacements.items():
        template = template.replace("@@" + name + "@@", value)
    if "@@" in template:
        raise ValueError("Unresolved HTML template field")
    target = directory / "index.html"
    target.write_text(template)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    print(render_page(args.input_dir))
