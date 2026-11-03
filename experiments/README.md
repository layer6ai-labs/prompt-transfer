# Migration experiment drivers

These scripts coordinate an existing GEPA/DSPy working tree and model-serving
pod. They do not include the optimizer implementation, retrieval corpora, source
optimization artifacts, or historical evaluations. The scientific comparison is
defined in [CROSS_MODEL_PILOT.md](../CROSS_MODEL_PILOT.md).

## Dependencies

Python 3.11+ runs orchestration, reporting, HTML packaging and the offline unit
tests. Figure rendering additionally needs matplotlib:

```bash
python -m unittest discover -s experiments -p 'test_*.py'
# Install only when rendering figures, preferably inside a virtual environment.
uv pip install --python .venv/bin/python -r requirements-analysis.txt
```

Run these commands from the repository root. Unit tests use synthetic artifacts
and mocks; they neither contact model endpoints nor submit GPU jobs. The standard
pytest configuration includes both the Tier 1 tests and these
campaign tests; the unittest command above runs the campaign suite alone.

Execution requires a compatible external `gepa-artifact` working tree, default
`../gepa-artifact`, including:

- `.venv/bin/python` with that program's dependencies and data/index setup;
- `local/run_gepa.py` supporting `--initial-program`, matched split and rollout
  settings, resumable output, and task/reflection usage accounting;
- `local/eval_bench.py` supporting frozen instruction files and the expected
  evaluation configuration/results schema;
- `local/{eval_setup,gepa_setup,benchmarks,metrics,cost,compat,series}.py`;
- historical `local/runs/gepa_f1/` and `local/runs/gepa_hover_recall/` own-model
  optimizations containing `config.json`, `done.json` with usage, and
  `best_instructions.json`;
- the historical `<series>_eval/<large-model>-base-r0/results.jsonl` records
  used to freeze identical held-out question IDs across target families.

Those `local/` drivers are workspace extensions, not files guaranteed by an
upstream GEPA checkout. An upstream Git revision alone is therefore insufficient
to recreate the environment. Preserve these drivers and their dependency lock
alongside an experiment archive. The full migration manifest hashes the driver
files, local serving/orchestration sources and inherited optimization metadata;
resume rejects changes to that recorded provenance. Existing cached inputs are
required even for `plan`; it is not a synthetic dry run.

Actual execution also requires the `sky` and `ssh` commands, configured access
to the existing `prompt-transfer` cluster, and the required checkpoints already
prepared on the pod. The queues do not download missing weights. Their serving
jobs reuse the existing two-H100 cluster.

## Script map

| Script | Purpose |
|---|---|
| `bridge_queue.py` | Freeze source prompts; optimize 12 small-model bridges across six directions and two benchmarks |
| `migration_queue.py` | Evaluate methods 1–3 globally, train method 4 globally, then evaluate method 4 |
| `cross_model_pilot.py` | Serial Qwen8 → Llama8 → Llama70 pilot with additional baseline and cold controls; uses already-running servers |
| `cross_model_report.py` | Shared paired statistics and optimization/inference cost accounting for the serial pilot |
| `migration_report.py` | Four-method route reports and campaign report |
| `migration_plots.py` | Validated four-method PNG/PDF exports with CSV/JSON data; requires matplotlib |
| `migration_visualize_when_done.py` | Observe completion, validate artifacts and cleanup, then generate final figures |
| `migration_overview.py` | Generate a separate overview snapshot, including explicitly pending cells; requires matplotlib |
| `migration_overview_page.py` | Package an overview snapshot as a local self-contained HTML file |
| `test_*.py` | Offline regression checks for protocol, resume, accounting and figure inputs |

## Plan, run, inspect, report

Use explicit output paths when starting a new study. The scripts retain dated
default paths for compatibility with earlier local runs; an existing directory
identifies that frozen campaign, not a request for a new one.

```bash
# Freeze the optimization-only bridge plan from existing own-model artifacts.
python experiments/bridge_queue.py plan \
  --gepa-root ../gepa-artifact --out artifacts/my_bridges
python experiments/bridge_queue.py status --out artifacts/my_bridges

# Starts serving jobs and model requests. Run one coordinator per campaign.
python experiments/bridge_queue.py run \
  --gepa-root ../gepa-artifact --out artifacts/my_bridges

# Requires all 12 completed, validated bridges plus historical large references.
python experiments/migration_queue.py plan \
  --gepa-root ../gepa-artifact --bridge-root artifacts/my_bridges \
  --out artifacts/my_migration
python experiments/migration_queue.py run \
  --gepa-root ../gepa-artifact --bridge-root artifacts/my_bridges \
  --out artifacts/my_migration
python experiments/migration_queue.py status --out artifacts/my_migration
python experiments/migration_queue.py report \
  --gepa-root ../gepa-artifact --bridge-root artifacts/my_bridges \
  --out artifacts/my_migration
```

`plan` freezes inputs and writes a manifest/status files, without model requests
or serving changes. `status` reads status only. `run` owns serving jobs, SSH
tunnels and optimization/evaluation workers, validates outputs before reuse,
and cleans up its owned jobs. `report` rebuilds reports from the frozen campaign;
it writes local output but makes no model requests. The serial pilot's
`--stage plan`/`train`/`eval`/`all`/`report` interface differs from the queues and
never manages serving jobs itself.

Read `status.json` and worker logs to distinguish waiting, loading, running and
completed stages. Do not launch a duplicate coordinator. Resume with the same
paths and code only after confirming the previous coordinator/workers exited;
changed settings or code provenance require a new campaign directory. Keep the
original source prompts and completed bridge outputs unchanged.

## Reports and figures

The migration queue writes `report/report.{md,json}` and per-route reports under
`report/routes/<benchmark>/<source>-to-<target>/`. Pending test cells have no
accuracy estimate. Method 1 is a historical full-budget optimization reference;
its new held-out evaluation is paired with the other methods. Costs distinguish
migration optimization, inherited source optimization and target inference.

```bash
# Observe once; no model requests or serving changes. Writes observer status.
.venv/bin/python experiments/migration_visualize_when_done.py \
  --campaign artifacts/my_migration --once

# Read completed report data and export standalone figures.
.venv/bin/python experiments/migration_plots.py \
  --report artifacts/my_migration/report/report.json \
  --out artifacts/my_migration/visualization/manual

# Export a separate, clearly labeled snapshot while cells may still be pending.
.venv/bin/python experiments/migration_overview.py \
  --campaign artifacts/my_migration --out artifacts/my_migration/overview
python experiments/migration_overview_page.py \
  --input-dir artifacts/my_migration/overview
```

The final observer waits for all stage barriers, 42 unique evaluations, 12
method-4 optimizations and serving cleanup. It writes beneath `visualization/`.
Plots require complete coverage by default; `migration_plots.py --allow-partial`
explicitly enables a preview. The overview always labels pending cells. Outputs
include PNG/PDF figures and CSV/JSON evidence; keep them under ignored
`artifacts/` or another external results directory.
