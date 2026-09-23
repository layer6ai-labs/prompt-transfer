# prompt-transfer

Asymmetric prompt transfer and leaderboard bias: optimize one prompt per model,
then have every model run every prompt, and read the resulting N x N matrix two
ways — by rows for transfer, by columns for leaderboards.

- `PROPOSAL-prompt-matrix.en.md` — the research proposal
- `DESIGN.md` — architecture, invariants, model roster, statistical design
- `PLAN.md` — build order and experiment sequence (E0 → E4)
- `BENCHMARKS.md` — benchmark survey from the three optimizer papers
- `MODELS.md` — model specs, endpoints, KV capacity and measured throughput

## Status

Tier 1 benchmarks are ready: **gsm8k**, **mmlu_pro**, **bbh_word_sorting**.
Splits are frozen and hashed, base prompts and rule scorers are transcribed
verbatim from lm-evaluation-harness and the MMLU-Pro reference repo.

Serving: the four executors (Qwen3.6-27B, Qwen3.5-4B, gemma-4-31B-it,
gemma-4-E4B-it) run under SGLang on the 2-GPU `prompt-transfer` cluster; see
Serving below.

Not built yet: the work planner/runner, the LLM judge, the optimizers, the
analysis layer. See `PLAN.md` steps 3–9.

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

## Serving

The `prompt-transfer` cluster is a Lambda Kubernetes pod (2x H100). It cannot
see `/layer6share`, so weights are downloaded onto its own disk, which is
ephemeral: if the pod restarts, re-run setup. Models, pinned revisions, ports
and GPU layouts are in `serving/models.sh`. Run from the repo root:

```bash
sky exec prompt-transfer sky/pod-setup.yaml -d        # venv + weights (idempotent)
sky exec prompt-transfer sky/serve.yaml -d            # layout all4; job holds the GPUs
sky exec prompt-transfer sky/smoke.yaml               # answers, thinking off, repeatability
sky cancel prompt-transfer <serve job id>             # stop serving, free the GPUs
```

Layouts: `all4` keeps all four executors up at once (one big + one small model
per GPU); `strong` and `weak` give each model its own GPU for bulk runs
(`--env LAYOUT=strong`). In `all4`, gemma-4-31B-it gets only 2 concurrent
requests, so it is the bottleneck. Endpoints listen on the pod only
(`http://localhost:800{1..4}/v1`, listed in `~/ptx/endpoints.json`). Two ways
to call them:

```bash
# from the login VM: forward to local ports 18001-18004 (keep it running in tmux;
# the first connect takes ~25 s through SkyPilot's proxy)
ssh -N -o ServerAliveInterval=30 -L 18001:localhost:8001 -L 18002:localhost:8002 \
    -L 18003:localhost:8003 -L 18004:localhost:8004 prompt-transfer
curl localhost:18001/v1/models

# or run a command on the pod (~40 s job overhead per call; fine for batch runs)
sky exec prompt-transfer 'curl -s localhost:8001/v1/models'
```

Qwen requests must pass `chat_template_kwargs={"enable_thinking": false}`.

Serving runs with SGLang deterministic inference, no prefix cache, 8192-token
context and 1024-token prefill chunks for every model. Without deterministic
mode, the 4B models gave different temperature-0 text for a request batched
with others than for the same request alone. Measured on `all4`
(2026-09-23, 16 concurrent 256-token requests): Qwen3.6-27B 191 tok/s,
gemma-4-31B-it 30, gemma-4-E4B-it 405, Qwen3.5-4B 682. Per-model limits, KV
capacity and the full measurements are in `MODELS.md`.

## Two rules that are not style preferences

**Base prompts and rule scorers are copied verbatim from a named harness, and
their provenance is recorded.** The project's claim is that shared prompts carry
bias; a shared prompt we wrote ourselves would be a straw man. Every scorer
carries a `source` field and a test asserts it is non-empty.

**Extractor bugs are preserved.** A strict rule that misses a correct answer is
not a defect in this code — it is the thing Q3 measures. `tests/test_scorers.py`
pins those failures so they cannot be silently "fixed".
