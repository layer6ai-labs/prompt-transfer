# prompt-transfer

Asymmetric prompt transfer and leaderboard bias: optimize one prompt per model,
then have every model run every prompt, and read the resulting N x N matrix two
ways — by rows for transfer, by columns for leaderboards.

- `PROPOSAL-prompt-matrix.en.md` — the research proposal
- `DESIGN.md` — architecture, invariants, model roster, statistical design
- `PLAN.md` — build order and experiment sequence (E0 → E4)
- `BENCHMARKS.md` — benchmark survey from the three optimizer papers

## Status

Tier 1 benchmarks are ready: **gsm8k**, **mmlu_pro**, **bbh_word_sorting**.
Splits are frozen and hashed, base prompts and rule scorers are transcribed
verbatim from lm-evaluation-harness and the MMLU-Pro reference repo.

Not built yet: model serving, the work planner/runner, the LLM judge, the
optimizers, the analysis layer. See `PLAN.md` steps 4–9.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
HF_HOME=/layer6share/zehao/hf-cache .venv/bin/python -m ptx.data freeze all
.venv/bin/python -m pytest -q
```

## Usage

```bash
.venv/bin/python -m ptx.data verify              # re-check split hashes
.venv/bin/python scripts/show_prompt.py gsm8k    # render a real prompt
```

## Two rules that are not style preferences

**Base prompts and rule scorers are copied verbatim from a named harness, and
their provenance is recorded.** The project's claim is that shared prompts carry
bias; a shared prompt we wrote ourselves would be a straw man. Every scorer
carries a `source` field and a test asserts it is non-empty.

**Extractor bugs are preserved.** A strict rule that misses a correct answer is
not a defect in this code — it is the thing Q3 measures. `tests/test_scorers.py`
pins those failures so they cannot be silently "fixed".
