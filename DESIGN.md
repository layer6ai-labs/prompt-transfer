# Repo design: asymmetric prompt transfer

Companion to `PROPOSAL-prompt-matrix.en.md`. This document fixes the computational
object, the data model, the repo layout and the experiment sequence. It also records
the places where the proposal, as written, underspecifies something that the code has
to decide.

---

## 0. Positioning: open weights only

The proposal's stated edge is "the only route that can bring closed models into a fair
comparison." We have no closed-model API access, so that claim is not available and the
framing has to move. The honest reframe, which is also the stronger paper:

**From** "a fair-leaderboard method for closed models"
**to** "what actually transfers in a prompt, and why transfer is asymmetric" — with
leaderboard bias as the application rather than the headline.

What open-weights-only costs:
- Q1's leaderboard claim no longer covers the models leaderboards care most about.
- The "closed vs. open" contrast in Q2 disappears.

What it buys, and these are not small:
- **No hidden system prompts.** §5 admits provider system prompts are an uncontrollable
  confound on transfer loss. Self-serving removes it entirely. Every token the model sees
  is a token we wrote.
- **Within-family size ladders.** Wu et al.'s asymmetry result is base→large vs large→base
  within one family. We can run that design directly (Qwen 1.7B→4B→9B→27B) instead of
  approximating it with cross-family pairs.
- **Base vs. instruct of identical weights.** §5 hypothesises that "strong/weak" and
  "loose/strict instruction following" are two different axes. `gemma-3-4b-pt` vs
  `gemma-3-4b-it` holds capability fixed and varies only instruction tuning. That is the
  cleanest available test of the explicitness-mismatch mechanism, and it is only possible
  on open weights.
- **Q4 becomes a within-study comparison.** §6 proposes comparing against Hardt et al.'s
  *published* LoRA numbers. On open weights we can run the LoRA arm and the soft-prompt
  arm ourselves, on our models, our tasks, our test set. "Black-box adaptation recovers
  X% of the harmonization that LoRA achieves" is a controlled result instead of a
  cross-paper gesture.
- **Cost.** The N x N fan-out is GPU time, not invoice. That is what makes the test-set
  sizes and repeat counts Coin Flip demands actually affordable.

Keep the provider layer abstract so that adding closed models later is a config entry,
not a refactor. The architecture below does that; nothing else assumes open weights.

---

## 1. The computational object

One matrix `S[i][j]` = score of executor model `i` under the prompt optimized for model
`j`, plus a shared-prompt column `S[i][0]` and a joint-optimized column `S[i][*]`.
Replicated over: task x scorer x optimizer x critic-design x seed.

Three invariants follow, and they drive everything else.

**Invariant 1 — a matrix cell is a GROUP BY, not a job.**
The unit of work is a single `(executor_model, prompt, question)` generation. Cells,
rows, columns, rankings and losses are all aggregations over a long table. Nothing
aggregate is ever stored as primary data. This is what makes Coin Flip's "pair by
question" floor structural rather than something you remember to do.

**Invariant 2 — generation and scoring are separate passes.**
Generations are expensive and immutable. Scores are cheap, plural, and recomputable from
stored generations. Q3 *is* the comparison of multiple scorers over one fixed set of
generations; if scoring ever runs inline with generation, Q3 costs a full re-run and the
comparison is no longer paired at the output level.

**Invariant 3 — content-addressed everything.**
`prompt_id = sha256(canonical_json(PromptSpec))`. Two optimizers that converge on the
same text collide into one id, which is both a free cache hit and a finding worth
reporting. `generation_key = sha256(model_id, prompt_id, task_id, question_id,
decode_params, render_version)`. Resumption, deduplication and provenance are then the
same mechanism.

---

## 2. Data model

Four artifact stores and one table.

### `PromptSpec` (content-addressed)
```
system:        str | None
instruction:   str
demos:         [{input, output}]        # ordered, possibly empty
output_format: str | None
render_version: str                     # bump invalidates generations
provenance:    {source_model, optimizer, critic_model, task, seed, trajectory_id, step}
```
Provenance is metadata, outside the hash. A prompt is defined by what the model sees.

### `Trajectory`
Per optimizer run: ordered steps of `{step, candidate_prompt_id, train_score,
edit_from_prev, critic_raw_output}`. **Every optimizer adapter must emit this.** The
Gong-and-Wen edit analysis in Q2 needs step-to-step diffs, and they cannot be
reconstructed from final prompts. This is the one thing in the whole pipeline that is
impossible to backfill without re-running optimization — get it right in the first
adapter.

### `Generation`
Keyed as above. `{text, prompt_tokens, completion_tokens, finish_reason, served_by,
timestamp}`. The only expensive artifact in the system.

### `Score`
`(generation_key, scorer_id) -> {score, extracted_answer, scorer_metadata}`. Many per
generation.

### `results.parquet` (long format)
One row per `(task, question_id, executor_model, prompt_id, scorer_id)` -> score, with
denormalized provenance columns for grouping. Every analysis is a groupby on this file.

---

## 3. Repo layout

```
prompt-transfer/
├── DESIGN.md
├── PROPOSAL-prompt-matrix.en.md
├── pyproject.toml
├── src/ptx/
│   ├── promptspec.py          # canonical object, deterministic renderer, hashing
│   ├── tasks/
│   │   ├── base.py            # Task protocol: splits, render, gold, rule_scorer
│   │   └── gsm8k.py  mmlu.py  helpsteer2.py  ...
│   ├── models/
│   │   ├── registry.py        # model cards: id, family, params, path, serve recipe
│   │   ├── client.py          # ported from smart-memory llm_client.py
│   │   └── serving.py         # vLLM lifecycle + GPU lease
│   ├── optimize/
│   │   ├── base.py            # Optimizer protocol -> (PromptSpec, Trajectory)
│   │   ├── random_search.py   # headroom gate + noise floor
│   │   ├── gepa.py  mipro.py  textgrad.py
│   │   └── joint.py           # the multi-source column
│   ├── execute/
│   │   ├── plan.py            # cartesian expansion -> work units, cost estimate
│   │   ├── runner.py          # model-major consumer, resumable
│   │   └── store.py           # content-addressed generation cache
│   ├── score/
│   │   ├── rule.py            # the benchmark's own script, verbatim
│   │   ├── permissive.py      # loose extraction
│   │   └── judge.py           # fixed judge model, batch pass
│   ├── analysis/
│   │   ├── matrix.py          # long table -> S, with paired bootstrap CIs
│   │   ├── q1_leaderboard.py  q2_asymmetry.py  q3_scorer.py
│   │   ├── q4_harmonize.py    q5_critic.py
│   │   └── edits.py           # diff, classify, propensity-adjust
│   └── cli.py                 # ptx headroom | optimize | plan | execute | score | analyze
├── configs/
│   ├── models/*.yaml          # one card per model
│   ├── tasks/*.yaml
│   ├── optimizers/*.yaml
│   └── experiments/*.yaml     # an experiment = a full, reproducible spec
├── launch/                    # sky job wrappers
├── data/                      # frozen splits + their hashes
├── artifacts/                 # prompts/ trajectories/ generations/ scores/
└── results/                   # results.parquet + derived matrices + figures
```

`configs/experiments/*.yaml` is the reproducibility unit: models x tasks x optimizers x
critics x seeds x scorers, plus the split hashes. Everything else is derived.

### What to port from `smart-memory`
- `benchmarks/common/llm_client.py` — retry/backoff, rate limiting, TPM handling and
  context-length shrink-and-retry are all battle-tested; rewriting them wastes a week.
- The resumable-cache pattern in `benchmarks/common/utils.py` (`ModeResultCache`), but
  re-keyed for generations rather than mode results.
- `configs/loader.py` env-var expansion and the `configs/llm/*.yaml` backend split.
- `sky/sequential_remote.yaml` + `launch/*.sh` as the managed-job template.

### What not to port
The memory-benchmark concepts — cutoffs, ingestion checkpoints, memory clients, mode
evaluation. None of it maps, and carrying it in will confuse the schema.

---

## 4. Serving and scheduling: the 2 x H800 constraint

160 GB of VRAM and a roster of ~10 models means **at most one large model resident at a
time**. The scheduling consequence is the single most important practical decision in
the repo:

**The execution loop is model-major, not cell-major.**

Load model `i` once; drain *every* pending work unit for that model — all prompt
columns, all tasks, all questions, all seeds — then release the GPUs and load model
`i+1`. A cell-major loop ("for each matrix cell, launch a job") would reload weights
O(N x tasks) times and dominate the wall clock.

Concretely: `ptx plan` writes the full work queue up front. `ptx execute --model X`
acquires a GPU lease, serves X under vLLM, consumes every queue item for X, and exits.
The queue is content-addressed, so a crash mid-model costs only the un-flushed
generations, and re-running is a no-op on the completed ones.

Small models (<=4B) can co-reside one per GPU, so schedule the ladder's small end in
parallel and the 27B/31B/120B serially.

**The judge is a separate model-major pass.** After all generations exist, serve the
judge once and drain the entire judging queue. The judge must be a single fixed model
across every cell or Q3 is confounded by the judge as well as the scorer. Budget it as
one more model in the roster, and freeze its identity in the experiment config.

---

## 5. Model roster

All of these already have real weights on `/layer6share`, so there is nothing to
download. Sizes are the on-disk footprint.

**Qwen ladder — the asymmetry spine (4 models, 12 ordered pairs)**

| model | size | role |
|---|---|---|
| `Qwen3-1.7B`   | 3.8 GB | ladder floor |
| `Qwen3-4B`     | 7.6 GB | |
| `Qwen3.5-9B`   | 19 GB  | |
| `Qwen3.6-27B`  | 52 GB  | ladder ceiling, TP=2 |

One family, four capability levels, identical tokenizer and training style. This is the
direct natural-language analogue of Wu et al.'s base/large design and it is where the Q2
claim is cleanest.

**Cross-family (for Q1's rank statistics)**

| model | size | role |
|---|---|---|
| `gemma-3-4b-it`   | 8.1 GB | matched to Qwen3-4B |
| `gemma-4-31B-it`  | 59 GB  | matched to Qwen3.6-27B |
| `Meta-Llama-3.1-8B-Instruct` | 15 GB | matched to Qwen3.5-9B |
| `gpt-oss-120b`    | 61 GB  | MXFP4, fits TP=2; strong source, distinct training style |
| `SmolLM3-3B`      | 5.8 GB | deliberately weak, different recipe |

**Instruction-tuning axis (capability held fixed)**

| model | size | role |
|---|---|---|
| `gemma-3-4b-pt` | 8.1 GB | base counterpart of `gemma-3-4b-it` |
| `gemma-3-1b-pt` | 1.9 GB | optional second base point |

N = 9 for the main matrix (11 with the pt models). That is above the N=5 the proposal
criticises Sadjoli et al. for, which matters: with N=5 a single adjacent swap moves
Kendall's tau by ~0.2, so Q1's column-ranking claims need the larger roster to say
anything at all.

Note the size-matched triples (4B: Qwen/Gemma; ~9B: Qwen/Llama; ~30B: Qwen/Gemma). Those
let Q2 separate "loses because the source was stronger" from "loses because the source
was a different family" — which the proposal's design cannot currently do.

---

## 6. Statistical design the proposal needs and doesn't yet specify

### 6.1 The diagonal is biased upward. Budget for a null.
Each model's own prompt was *selected* to maximise that model's train score. On test, the
diagonal is inflated by winner's curse even if home advantage is exactly zero. Reporting
"the diagonal is higher than off-diagonal" without correcting for this proves nothing,
and it is Q1's headline claim.

**Twin-prompt design:** optimize **two** prompts per (model, task) under different seeds.
Prompt A defines the diagonal and fills the matrix. Prompt B is never used as a column —
it measures what model `i` scores on an independently-optimized prompt of its own. The
gap `S[i][A_i] - S[i][B_i]` is the selection bias; home advantage is what survives after
subtracting it. Add a permutation test over source labels as a second check.

This doubles optimization cost. It does not change matrix cost, since only prompt A
becomes a column. It is not optional — without it Q1 is unfalsifiable.

### 6.2 Three scorers, not two.
The proposal specifies rule-based and LLM judge. Add a **permissive rule** scorer (e.g.
"gold appears anywhere in the output" rather than "the last integer equals gold"). This
gives a three-way decomposition on identical generations:

- strict rule fails, permissive passes -> the answer was there, the script missed it
- permissive fails, judge passes -> the judge is being generous, or the answer is
  paraphrased
- all three fail -> the model was actually wrong

The Sadjoli "91 trees" example the proposal highlights is exactly the first bucket, and
the permissive scorer isolates it *without* having to trust the judge. It costs nothing:
pure post-processing over stored generations.

### 6.3 What is "a prompt" for transfer purposes?
This is the confound the proposal does not address, and it changes what Q2 means.

MIPRO optimizes instructions **and few-shot demonstrations**. TextGrad optimizes
instruction text. If "model A's prompt" includes demos whose outputs are written in A's
style, then transferring it to B mixes two different phenomena: instruction transfer and
style-of-demonstration transfer. They plausibly have opposite asymmetries.

**Decision: run the matrix in two variants.**
- `instruction_only` — demos stripped, instruction transfers alone. The clean claim.
- `full_prompt` — everything transfers. The realistic claim.

Cost is 2x on execution, which is the expensive axis, so the pilot should establish that
the two differ before paying for both at full N. If they do differ, that gap is a result
in its own right and closes a gap PromptBridge left open.

### 6.4 Ceiling effects are a gate, not a caveat.
§5 flags "strong models are near ceiling" as an alternative explanation to rule out. Make
it a task-admission criterion: reject any task where the strongest model's zero-shot score
exceeds ~85%, because there is no room for transfer loss to be visible.

### 6.5 Power.
Paired bootstrap over questions. For a paired accuracy difference with ~20% of questions
disagreeing, SE ~= sqrt(0.2/n): n=400 gives ~2.2 pp, n=900 gives ~1.5 pp. Transfer losses
in the literature are 10-30 pp, so **400-500 test questions per task** resolves the main
effects comfortably and is marginal for the subtle edit-level contrasts. Report a paired
bootstrap CI on every reported difference, never on a raw cell.

Training split must be **well above** Coin Flip's failure point of 20 examples — budget
200-500, held disjoint from test, split hash recorded in the experiment config.

---

## 7. Milestones

**M0 — pipeline smoke (no science).**
2 models x 2 prompts x 1 task x 50 questions, end to end through all three scorers and
into `results.parquet`. Proves the invariants hold before any GPU budget is spent.

**M1 — headroom gate.** Coin Flip's floor #3, as its own CLI stage. Random-search 10-20
candidate prompts per task on 3 models; drop tasks where no model's best candidate beats
zero-shot by more than the noise floor, and drop tasks that violate the ceiling rule in
6.4. Deliberately keep both format-sensitive and format-insensitive survivors — Q3 needs
both. Output: the task shortlist everything downstream uses.

**M2 — pilot matrix (the go/no-go).** Qwen ladder only (N=4), 2 tasks, 1 optimizer, twin
prompts, both prompt variants, all three scorers. 4 x 6 x 2 x 500 ~= 24K generations.
Answers: is asymmetry real, which direction, does instruction-only differ from
full-prompt, and how big is the selection bias. If asymmetry is absent here it will be
absent at N=9, and the project pivots before the expensive part.

**M3 — full matrix.** N=9, shortlisted tasks, + joint-optimized column, + the two critic
designs for Q5. This is the bulk of the GPU budget.

**M4 — mechanism and ceiling.** Edit-level analysis (Gong-and-Wen port, treatment =
edit type x transfer direction). Plus the white-box arms — LoRA per Hardt et al. and soft
prompts per Erkan et al. — run on our own models and tasks, turning Q4 into a controlled
comparison rather than a citation.

---

## 8. Cost model

Generations for one full matrix pass:

```
N_models x (N_prompts + 2) x N_tasks x N_test x N_variants
9 x 11 x 3 x 500 x 2  ~=  297K generations
```

Plus an equal number of judge calls. At ~400 output tokens that is ~120M output tokens
for the executor pass. Throughput on 2 x H800 varies by an order of magnitude across a
1.7B and a 120B, so plan per model rather than in aggregate; expect the 27B/31B/120B tier
to dominate wall clock and schedule it first.

`ptx plan` must print the generation count and a per-model time estimate **before**
anything launches, and refuse to run over a configured budget. The pilot's measured
throughput per model is what calibrates the estimator for M3.

---

## 9. Open questions to settle before M2

1. **Which tasks.** Needs real train/test splits, a rule-based scorer worth criticising,
   and a spread on format sensitivity. GSM8K and MMLU are the proposal's examples; at
   least one non-format-sensitive generative task is needed for Q3's contrast, and at
   least one task overlapping Hardt et al.'s benchmark set for M4.
2. **Which optimizer first.** GEPA has the cleanest trajectory semantics for the edit
   analysis and is the most recent; MIPRO forces the demos question from day one.
   Recommend GEPA for M2, add MIPRO at M3 as the demos contrast.
3. **Self-critic on small models.** Q5's symmetric design has each model critique itself.
   A 1.7B model is a poor critic, which confounds "critic bias" with "critic competence."
   The pilot should check whether the small end of the ladder can self-optimize at all
   before the design depends on it.
