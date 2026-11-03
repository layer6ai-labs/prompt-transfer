# Cross-model prompt migration protocol

Compare direct prompt transfer with adapting source instructions on a smaller
or larger model in the target family. The transfer chain is an empirical path;
accuracy and compute savings are hypotheses to measure. GEPA edits four DSPy
module instructions per benchmark. The program, retrieval index, parsers and
decoding remain fixed.

This document defines the protocol, not live execution state. Earlier local
campaigns used `artifacts/bridge_queue_20261031/` and
`artifacts/migration_20261101/`; their manifests, `status.json` files and logs are
the execution record. A launch message or historical timestamp does not
establish that inference is currently running or that results are complete.

## Models and tasks

| Family | Source / target-small model | Target-large executor |
|---|---|---|
| Qwen3 | Qwen3-8B | Qwen3-32B |
| Llama 3.1 | Llama-3.1-8B-Instruct | Llama-3.1-70B-Instruct-FP8 |
| Gemma 3 | gemma-3-12b-it | gemma-3-27b-it |

Both directions for every unordered pair produce six routes, each evaluated on
HotpotQA and HoVer. All methods on a route use the same target-large executor.
HotpotQA measures answer F1 in the external multi-hop retrieval program; HoVer
measures article recall in its three-search program, not a final claim verdict.
These are separate from the Tier 1 single-prompt `ptx` tasks.

The external GEPA checkout, compatible local drivers, historical optimization
artifacts and retrieval data are prerequisites. They are not supplied by
installing this repository. See [experiments/README.md](experiments/README.md)
for the required files and commands.

## Four-method campaign

| Method | Target-large model evaluates | Optimization cost attributed |
|---|---|---|
| 1: `own_big_reference` | Its own historical GEPA instructions | Historical full-budget target-large optimization |
| 2: `direct` | Original source-small instructions, unchanged | Zero migration optimization; source cost inherited separately |
| 3: `warm_small` | Source instructions adapted on target-small | Small-model adaptation; source cost inherited separately |
| 4: `warm_big` | Source instructions adapted directly on target-large | Large-model adaptation; source cost inherited separately |

Method 4 starts from the original frozen source instructions, not method 3's
adapted output. Each incoming direction has its own source initialization;
reverse routes never initialize from a newly trained forward route. Method 1
is a full-budget historical reference, not a budget-matched cold control. The
four-method campaign does not include fresh cold-small or cold-large controls.

Every route uses the same 300 frozen held-out IDs for its benchmark. Each own
reference is evaluated once per target family and benchmark and reused by the
two incoming routes. This yields 48 route/method cells from 42 unique test
evaluations. Historical prompts still require a new paired test evaluation;
prompt availability does not mean the evaluation is complete.

The small-model bridge queue prepares 12 optimizations. The migration queue
then enforces three global stage barriers, also on resume:

| Stage | Work across all routes and both benchmarks |
|---|---|
| `eval123` | 30 unique evaluations: 6 own references, 12 direct transfers, 12 bridges |
| `train4` | 12 target-large adaptations, after every `eval123` evaluation completes |
| `eval4` | 12 target-large evaluations, after every method-4 optimization completes |

Within a stage, target families run in order Llama, Qwen, Gemma. The coordinator
uses one owned serving job at a time on the existing two-H100 cluster. The
migration launcher uses tensor parallelism across both GPUs for each target,
32,768-token context, eight server requests, deterministic kernels, disabled
prefix caching and 4,096-token prefill chunks. It keeps a private launcher
snapshot rather than modifying shared serving configuration.

The bridge queue serves one small-model pair at a time, one model per H100,
optimizing both directions concurrently with eight client threads per model.
Pair order is Qwen–Llama, Qwen–Gemma, Llama–Gemma; each pair runs HotpotQA then
HoVer. The queue cleans up only the serving jobs and workers it owns.

## Frozen settings and interpretation

Methods 3 and 4 use 150 training items, the first 100 validation items, 1,200
requested complete-program rollout metric calls, optimizer seed 0 and split
seed 0. Each target optimizer also performs reflection on the same model; the
small-model bridge has no hidden large-model critic. Task and reflection usage
are recorded separately. GEPA checks its budget at iteration boundaries, so
actual rollouts can exceed the requested cap. Equal rollout budgets do not
imply equal tokens, FLOPs or wall time.

Decoding is temperature 0.6, top-p 0.95, top-k 20 and a 16,384-token completion
cap, with thinking off for these model families. The optimizer seed controls
search; generation does not have an explicit decode seed. Deterministic server
kernels remove one source of batching variation without making sampled
responses deterministic.

Prompts are selected using the optimizer's validation set only. Test scores do
not select instructions. The pilot has one optimization seed and one sampling
repeat; 95% paired question-bootstrap intervals describe question sampling,
not optimizer-seed uncertainty.

The inherited HoVer split has one claim shared between train150 and val100
(train index 107, validation index 57), with no train/test or val/test overlap.
The exploratory campaign preserves this split for comparison. A confirmatory
study should use a clean split and regenerate source prompts and controls.

Report migration cost with source optimization treated as sunk, and end-to-end
optimization cost including the inherited source. Keep target inference
input/output tokens and FLOPs per question separate. A transferred prompt can
be cheaper to obtain yet more expensive over many deployment queries. Lifecycle
compute is `optimization_cost + query_count × target_inference_per_query`,
with optimization cost explicitly scoped to migration only or end-to-end. The
reports' FLOPs are analytical forward arithmetic, not measured GPU work;
precision affects runtime and memory separately. Historical optimizer wall time
may cover only a final resumed session, so unknown cumulative time stays unknown.
Shared own-reference costs must not be double-counted as campaign spending.

Inspect parsing failures as well as scores: instruction transfer can change
output formatting enough to break DSPy's parser. Do not use exploratory test
scores to choose a source, budget or checkpoint and then describe the same
questions as an untouched confirmatory test.

## Execution and output

Use the complete commands in [experiments/README.md](experiments/README.md).
Start with `bridge_queue.py plan`, inspect the frozen inputs, then use `run`
when the existing cluster is free. After all 12 bridges validate, prepare and
run `migration_queue.py`. Keep one coordinator per campaign, and resume only
when its prior coordinator and workers have exited.

`plan` writes local manifests and statuses but makes no model requests.
`status` reads state. `run` changes serving jobs and performs inference.
`report` validates artifacts and writes reports without model requests. Resume
checks input hashes, protocol settings and recorded code provenance; changing
those inputs requires a new output directory. Do not edit frozen campaign
sources in place while it runs.

Campaign outputs include:

- `manifest.json` and frozen `inputs/` with prompt and provenance hashes;
- `status.json`, per-stage/worker state and logs;
- `optimize/` outputs, configurations, usage and selected instructions;
- `eval/` paired item results and summaries;
- `report/report.{md,json}` plus per-route reports;
- `visualization/` final figures and source data, or separately labeled previews.

Pending cells have no accuracy estimate. The final visualization observer
requires all stages, 42 unique evaluations, 12 large adaptations and serving
cleanup to validate. It makes no model requests; it writes observer status and
figure output only. Check live status and logs for progress rather than relying
on this document or an ETA. All these run outputs remain local and ignored by
Git.

## Serial pilot and additional controls

`experiments/cross_model_pilot.py` retains a single-route Qwen8 → Llama8 →
Llama70 experiment. Its seven arms are original benchmark instructions,
historical own-large reference, direct source transfer, warm-small,
cold-small, warm-large and cold-large. Cold arms initialize from the original
benchmark instructions, using a matched pilot rollout budget. This runner uses
already-running Llama servers and does not start/cancel serving jobs or allocate
GPUs.

```bash
python experiments/cross_model_pilot.py --bench hotpotqa --stage plan
# With the llama31 layout already running and ports 18014/18015 forwarded:
python experiments/cross_model_pilot.py --bench hotpotqa --stage all
python experiments/cross_model_pilot.py --bench hotpotqa --stage report
```

The default output is `artifacts/cross_model_pilot/<benchmark>_s0/`; use `--out`
for a distinct study and repeat the same settings across stages. `--arms`
selects a subset; `--stage all` completes the selected optimization arms before
evaluating them. Reports compare paired accuracy and migration/end-to-end costs
with the same missing-versus-zero accounting rules.

A broader confirmatory comparison can share cold controls between incoming
routes for the same target family, benchmark and seed. Predeclare its clean
splits, multiple optimizer seeds, test sampling repeats and compute comparison
before reading the confirmatory test set.

## Offline verification

```bash
python -m unittest discover -s experiments -p 'test_*.py'
```

The suite checks frozen prompt/test IDs, endpoint identity, locking, resume and
stage barriers, cost accounting, paired statistics and figure input coverage
without model requests. It does not replace a serving smoke check or verify
availability of external datasets and historical artifacts.
