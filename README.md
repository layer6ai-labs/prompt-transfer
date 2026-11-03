# prompt-transfer

Study asymmetric prompt transfer and leaderboard bias by optimizing instructions
on one model and evaluating them on others. This repository contains a small
single-prompt benchmark toolkit, SGLang serving configurations, and orchestration
and analysis for the cross-family GEPA migration experiments.

| Component | Location | Scope |
|---|---|---|
| Tier 1 toolkit | `src/ptx/`, `tests/` | GSM8K, MMLU-Pro, BBH Word Sorting; frozen splits, prompt rendering, rule scorers |
| Model serving | `serving/`, `sky/` | Pinned model revisions, two-H100 layouts, checkpoint setup and smoke checks |
| Migration campaigns | `experiments/` | HotpotQA and HoVer orchestration, paired reports, plots; uses an external GEPA checkout |

The GEPA optimizer and multi-hop benchmark programs live in a separate
`gepa-artifact` working tree. Installing `ptx` does **not** install those drivers
or their datasets, retrieval index, historical prompts, and results. See
[experiments/README.md](experiments/README.md) for the dependency contract and
[CROSS_MODEL_PILOT.md](CROSS_MODEL_PILOT.md) for the comparison protocol.

## Tier 1 setup and checks

Use Python 3.11 or newer; the example uses Python 3.12 and `uv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q

# Downloads datasets and writes local frozen splits; requires network access.
.venv/bin/python -m ptx.data freeze all
.venv/bin/python -m ptx.data verify
.venv/bin/python scripts/show_prompt.py gsm8k
```

Dataset manifests already tracked in `data/` describe the original frozen
splits. Dataset payloads are local inputs. Do not replace a study's frozen data
or prompts while comparing or resuming runs. The original architecture and
planned research sequence are recorded in [DESIGN.md](DESIGN.md) and
[PLAN.md](PLAN.md); they include work beyond what is implemented here.

Base prompts and strict rule scorers retain their named harness provenance.
`tests/test_scorers.py` deliberately pins extractor failures: changing these
rules changes the research measurement.

## Serving on the existing pod

The SkyPilot configuration targets the `prompt-transfer` cluster with two H100s.
The pod has ephemeral storage and no `/layer6share` mount. Model weights and the
SGLang environment live under `$PTX_HOME` (default `~/ptx`). Run SkyPilot commands
from this repository's root with a configured cluster and access to the required
model repositories.

[`serving/models.sh`](serving/models.sh) is the source of truth for full model
revisions, ports, memory fractions, concurrency and layout entries. Setup only
downloads the selected layout, defaulting to the original four executors:

```bash
sky exec prompt-transfer sky/pod-setup.yaml -d
# Example alternative: download the Qwen3 layout's models.
sky exec prompt-transfer sky/pod-setup.yaml --env SETUP_LAYOUT=qwen3 -d

# Submit serving only when the existing allocation is free.
sky exec prompt-transfer sky/serve.yaml -d
sky exec prompt-transfer sky/smoke.yaml
# Use the serving job ID returned by SkyPilot when releasing its allocation.
sky cancel prompt-transfer <serve-job-id>
```

`sky/serve.yaml` defaults to `LAYOUT=all4` and `CONTEXT_LENGTH=8192`. Other
layouts include `qwen3`, `gemma3`, `gptoss`, `llama31`, and small-model bridge
pairs. Long GEPA reflection prompts require a 32,768-token served context and a
layout with enough KV memory. The migration coordinator uses its dedicated
`sky/migration-serve.yaml` configuration. The Llama checkpoint preparation path
uses the separate fetch recipe, requiring the official config and tokenizer files described in
`serving/fetch_llama31.sh` and `sky/fetch-llama31.yaml`.

Servers expose OpenAI-compatible endpoints on the pod; the active layout writes
`$PTX_HOME/endpoints.json`. Forward only the ports needed by a manually managed
client, for example the original four endpoints:

```bash
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -L 18001:localhost:8001 -L 18002:localhost:8002 \
  -L 18003:localhost:8003 -L 18004:localhost:8004 prompt-transfer
curl localhost:18001/v1/models
```

Deterministic kernels and disabled prefix caching are the defaults. Prefill
chunk sizes vary by model and layout: several models use uncut prefill (`-1`)
to avoid traffic-dependent output changes, while GEPA layouts use explicit
chunks to accommodate longer reflection prompts. Do not infer reproducibility
from temperature zero alone. See [MODELS.md](MODELS.md) for protocol details and
historical throughput measurements.

## Campaign checks and repository hygiene

The campaign orchestration and its unit tests use the standard library:

```bash
python -m unittest discover -s experiments -p 'test_*.py'
```

Optional figure rendering needs `requirements-analysis.txt`. These offline
checks do not launch models or alter the cluster. `plan`, `run`, and report
commands have different effects; read [experiments/README.md](experiments/README.md)
before running a campaign.

Commit source, tests, serving configuration and documentation. Keep model
weights, datasets, virtual environments, endpoints, credentials, run manifests,
predictions, logs and generated figures local. `artifacts/`, `results/`, `logs/`
and caches are ignored; retain existing tracked dataset manifests.

The research proposal is [PROPOSAL-prompt-matrix.en.md](PROPOSAL-prompt-matrix.en.md).
[BENCHMARKS.md](BENCHMARKS.md) records the benchmark survey and the distinction
between the Tier 1 single-prompt tasks and the external retrieval programs.
