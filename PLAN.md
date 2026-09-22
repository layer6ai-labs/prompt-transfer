# What to build, in what order

Companion to `DESIGN.md`. Nine build steps and four experiments, interleaved.
Each step says what to write, and what it lets you do.

---

## Why this order

Three facts about the cluster:

- The GPUs are **H100 80GB** (4 nodes x 8), not H800. Same 80 GB per card.
- Right now **0 of 32 are free**. Getting a GPU is harder than using one.
- The shared Qwen endpoint at `10.0.1.10` is **down**. There is no free endpoint to
  test against.

So: write everything that doesn't need a GPU first. Test it against a fake model. Spend
the first GPU lease on an experiment that produces real results, not on a smoke test.

---

## The order at a glance

| # | Build | So you can |
|---|---|---|
| 1 | Repo skeleton | commit and import |
| 2 | Prompt object, tasks, rule + permissive scorers | represent a prompt and score an answer |
| 3 | Generation store, results table | save answers once, score them many ways |
| 4 | Fake model backend | **run E0** — prove the pipeline, no GPU |
| 5 | Model registry, API client, vLLM serving | talk to a real model |
| 6 | Work planner, runner, CLI | run many models x prompts x questions |
| 7 | Judge pass | get the third score |
| — | | **run E1 + E2** — first GPU lease |
| 8 | Random-search optimizer | **run E3** — task shortlist + first matrix |
| 9 | Analysis code | read the matrix |
| — | | **run E4** — pilot matrix with GEPA (go/no-go) |

---

## Build steps

### 1. Repo skeleton
`pyproject.toml`, `src/ptx/`, first commit.

### 2. Core objects (no GPU)

**`promptspec.py`** — the prompt object. Fields: system, instruction, demos,
output_format. Serialize to canonical JSON, hash it to get `prompt_id`.

Two ways to render it, and this one bites if you miss it: instruct models take
`messages` at `/v1/chat/completions` and vLLM applies their own chat template. Base
models like `gemma-3-4b-pt` have no template and need raw text at `/v1/completions`. So
the same prompt becomes different tokens on different models. That is unavoidable, but
it is a transfer confound, so save the template hash with each generation.

**`tasks/base.py` + two tasks.** Each task provides: train/dev/test splits, how to render
a question, the gold answer, a strict scorer, a permissive scorer, and a
`format_sensitive` flag. Freeze the splits to `data/` and record their hash.

Start with one format-sensitive task and one format-insensitive one, so step 4 exercises
both scorers.

**`score/rule.py`** — copy the benchmark's own answer extraction exactly, bugs included.
The bugs are what Q3 studies.
**`score/permissive.py`** — the loose version (does gold appear anywhere in the output).

### 3. Storage (no GPU)

**`execute/store.py`** — save each generation under a key built from model, prompt,
question and decode settings. Copy the resumable-cache pattern from
`smart-memory/benchmarks/common/utils.py`. Save the full rendered prompt next to the
answer so you can audit a cell later.

Flush to disk every N answers. The cluster is busy and jobs get preempted; anything still
in a buffer is lost GPU time.

**`analysis/table.py`** — write one row per (task, question, model, prompt, scorer) to
`results.parquet`. Never store a matrix cell. A cell is a groupby over this table.

### 4. Fake backend (no GPU)

**`models/backends/fake.py`** — returns a canned answer based on the rendered prompt.
This is what makes the next experiment possible with zero GPUs, and it is what lets the
repo have tests at all.

> ### Run E0 — pipeline smoke
> 2 fake models x 2 prompts x 1 task x 50 questions, end to end.
> **You get:** nothing scientific. Proof the plumbing works.
> **Done when:** re-running produces zero new generations, and deleting
> `artifacts/scores/` then re-scoring rebuilds `results.parquet` identically.

### 5. Real model access

**`models/registry.py`** — one YAML per model: id, family, size, path on `/layer6share`,
tensor-parallel size, max length, whether it is a base model. Use the roster in
`DESIGN.md` section 5.

**`models/client.py`** — port `smart-memory/benchmarks/common/llm_client.py`. Keep the
retries, the backoff, and the shrink-and-retry on context overflow. Drop the Anthropic and
Azure branches and the memory-benchmark parts.

**`models/serving.py`** — start vLLM, wait for `/v1/models` to answer, tear down when
done. Run it as a SkyPilot **managed job** so it queues and recovers instead of failing
when no GPU is free.

### 6. Planner, runner, CLI

**`execute/plan.py`** — expand an experiment config into the full list of work, drop
anything already in the store, and print the total count and a time estimate **before**
anything launches. Refuse to run over a set budget.

**`execute/runner.py`** — **load one model, run everything for it, then switch.** All
prompts, all tasks, all questions for that model in one go. Do not loop over matrix cells
and reload weights each time; with ~10 models that would reload weights dozens of times
and dominate the wall clock.

**`cli.py`** — `ptx plan | execute | score | analyze | headroom`.

**Freeze the decode settings** in the experiment config: temperature, top_p, max_tokens,
seed, stop. Hash them into the generation key. Use temperature 0 for all answer
generation. Optimizer critics sample hotter, but that is a separate call path and must not
leak in.

### 7. Judge pass

**`score/judge.py`** — a second pass over stored generations, after all generations exist.
One judge model, one judge prompt, temperature 0.

Fix both at first use. Changing either throws away every judgment you have. If the judge
differs across cells, Q3 measures the judge instead of the scorer. Pick a judge from
**outside** the executor roster: using a roster model gives its family an advantage and
re-opens Q5 inside Q3.

> ### Run E1 — zero-shot baseline and ceiling screen
> 3 models (`Qwen3-4B`, `Qwen3.5-9B`, `Qwen3.6-27B`) x 4 candidate tasks x ~150
> questions, using each benchmark's own shared prompt, all three scorers.
> The 4B and 9B share one 2-GPU lease, one card each.
>
> **You get four things, which is why this is the first GPU spend:**
> 1. The shared-prompt column `S[i][0]`. This is a required column of the final matrix,
>    not a warm-up.
> 2. **Which models are actually stronger, on these tasks.** Q2 is a claim about
>    strong-to-weak versus weak-to-strong. That is undefined until you measure it, and
>    parameter count is not the answer.
> 3. The ceiling screen: drop any task where the best model scores above ~85%. Expect
>    this to kill GSM8K, the proposal's own running example. That is the screen working.
> 4. Real throughput numbers, which calibrate the estimator in step 6.
>
> ### Run E2 — noise floor (same lease)
> (a) Run one cell 3 times. (b) Write 5-8 paraphrases of each shared prompt and run those.
>
> **You get:** the number every later claim is compared against. (b) is the spread across
> prompts that mean the same thing. "Optimization beat zero-shot" is meaningless until it
> means "beat zero-shot by more than (b)."
> **Watch for:** (a) should be identical three times. vLLM batching often makes
> temperature 0 non-reproducible. If it does, fix it before continuing — a
> non-reproducible generator breaks the paired design everything else rests on.

### 8. Random-search optimizer

**`optimize/base.py`** — the optimizer interface. Returns a prompt **and a trajectory**:
every step's candidate, its train score, the diff from the previous step, and the raw
critic output.

Emit the trajectory from the **first** adapter, not the third. It is the only artifact you
cannot rebuild without re-running optimization, and the Q2 edit analysis is built entirely
on it.

**`optimize/random_search.py`** — generate K=15 candidate prompts, score them on train,
keep the best. About 80 lines.

> ### Run E3 — headroom gate, and a free first matrix
> Same 3 models, same tasks, K=15 candidates each, scored on train. Keep tasks where at
> least one model's best candidate beats zero-shot by more than E2's noise floor. Keep at
> least one format-sensitive and one format-insensitive task, because Q3 needs both.
>
> **You get:** the task shortlist — plus a complete N x N matrix as a by-product.
>
> "Best of 15 random candidates" is a real, if weak, per-model optimizer. So this gives
> you the whole matrix shape — diagonal, off-diagonal, column rankings, home advantage —
> **before GEPA exists**. You can write and debug all of step 9 against real numbers while
> the optimizer integration is still unwritten. Random search also has no critic model, so
> it has no Q5 bias, which makes it the control arm you want anyway.

### 9. Analysis

**`analysis/matrix.py`** — long table to matrix `S`. Put a paired bootstrap confidence
interval on every reported **difference**, never on a raw cell.

**`analysis/q1_leaderboard.py`** — column rankings and home advantage, with the
twin-prompt correction from `DESIGN.md` 6.1 and a permutation test on source labels.

**`analysis/q2_asymmetry.py`** — the loss `L(i->j)`, using the capability ordering
measured in E1.

**`analysis/q3_scorer.py`** — the three-way split: strict fails but permissive passes
(the script missed it) / permissive fails but judge passes / all three fail (the model was
wrong).

> ### Run E4 — pilot matrix with GEPA (go/no-go)
> Needs `optimize/gepa.py` first. GEPA before MIPRO: cleaner trajectories for the edit
> analysis. MIPRO comes later, as the few-shot-demos contrast.
>
> Qwen ladder only (N=4), 2 surviving tasks, two seeds per model, both prompt variants
> (instruction-only and full prompt), 400-500 test questions. About 48K generations over
> 4 model loads.
>
> **You get answers to:** Is transfer asymmetric, and which way? How much of the diagonal
> is just winner's curse? Does instruction-only differ from full-prompt — if not, drop one
> and halve the next stage. How much does the judge move things?
>
> **Go/no-go:** if asymmetry does not show up on a single-family ladder with the system
> prompt fully under your control, it will not show up at N=9. Stop and rethink before
> paying for the full matrix.

---

## Tasks to screen in E1

Screen four, expect to keep two or three:

| task | format-sensitive? | why it is in the list |
|---|---|---|
| MATH500 | yes | not saturated at 27B; its answer extraction is exactly the failure mode Sadjoli describes |
| MMLU-Pro | yes | 10 options, well clear of ceiling, unlike plain MMLU |
| BBH (subset) | mixed | overlaps Hardt et al.'s tasks, which Q4 needs later |
| HelpSteer2 (or similar open-ended) | no | Q3 needs a format-insensitive task; also Coin Flip's one success case |

Screen GSM8K too, expecting it to fail on ceiling. A documented rejection of a task the
field still reports on is worth writing up.

---

## Still to decide

- **Executor temperature.** Use 0. Confirm with E2(a) that it is actually reproducible.
- **Judge model.** Something outside the executor roster. Fix it at first use.
- **Can `Qwen3-1.7B` critique itself?** Q5's design assumes every model can act as its own
  critic. If the smallest cannot, you are confounding critic bias with critic competence.
  E3 is the cheap place to check.
