# Models

[`serving/models.sh`](serving/models.sh) is the authoritative registry of model
repositories, full pinned revisions, ports and layout entries. The registry
lists supported configurations; it is not a live endpoint inventory. Read
`$PTX_HOME/endpoints.json` and serving logs for a running job. See
[README.md](README.md) for setup and [CROSS_MODEL_PILOT.md](CROSS_MODEL_PILOT.md)
for the migration protocol.

The original four-model tables below record measurements from **2026-09-23**
on the `all4` layout. Later prefill and layout changes mean those KV and
throughput figures are historical, not capacity guarantees for a new run.
Re-measure a chosen configuration with `sky/smoke.yaml` before relying on it.

## Additional registered models and layouts

| Models | Pod ports | Layouts / use |
|---|---|---|
| Qwen3-32B, Qwen3-8B, Qwen3-1.7B | 8010, 8005, 8011 | `qwen3`; 8B also in `qwen-gepa-paper` and bridge pairs |
| gemma-3-27b-it, gemma-3-12b-it, gemma-3-4b-it, gemma-3-1b-it | 8006–8009 | `gemma3`; 12B also in bridge pairs |
| gpt-oss-120b, gpt-oss-20b | 8013, 8012 | `gptoss`; 120B tensor parallel across both GPUs |
| Llama-3.1-70B-Instruct-FP8, Llama-3.1-8B-Instruct | 8015, 8014 | `llama31`; 70B tensor parallel, 8B data-parallel replicas |

Other layouts include the original `all4`, `strong` and `weak`, the Qwen/Gemma
pairs `qwen`, `gemma`, `gemma-gepa`, and the small-model pairs
`bridge-qwen-llama`, `bridge-qwen-gemma`, `bridge-llama-gemma`. Layout entries
are `name:gpu:memory_fraction:max_running_requests[:prefill_chunk]`. GPU indices
are relative to the job's `CUDA_VISIBLE_DEVICES`; `0+1` means one tensor-parallel
server, while `0,1` means data-parallel replicas behind one endpoint. Models
start in layout order because a cohosted server's KV pool depends on memory
available when it loads.

The dedicated migration launcher creates a private target-only layout with the
chosen large model tensor parallel across both H100s, memory fraction 0.88,
eight concurrent requests and 4,096-token chunks. It fixes context at 32,768,
deterministic kernels and disabled radix caching. All four methods for a route
use that same target configuration.

The `gptoss` layout uses both GPUs for 120B: an observed single-H100 load failed
because MXFP4 weight conversion required temporary memory beyond the stored
checkpoint size. The layout reserves room for 20B on GPU 1. Both servers use
`--reasoning-parser gpt-oss`; compatible experiment clients pin reasoning effort
`low`. Reasoning behavior is separate from the Qwen thinking-off template flag.
Smoke requests allow at least 2,048 completion tokens for gpt-oss, including
reasoning; other models retain the 256-token throughput cap. Compare measured
throughput only with the request budgets and reasoning mode recorded.

The Llama fetch path combines pinned 8B weight shards with official configuration
and tokenizer files and uses RedHatAI's pinned FP8 70B checkpoint with official
tokenizer files. Read `serving/fetch_llama31.sh` for the required vendor files and
hash checks. It is a separate preparation path from a normal registry download.
Setup checks safetensors headers and payload extents before writing revision
markers. These checks detect incomplete downloads; the 8B mirror recipe also
checks shard SHA256 hashes against the official checkpoint.

---

## Original executors: model specifications

| Model | Role | Params | Architecture | Layers | Native context | Vocab | Weights (bf16) | HF revision | License |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-27B** <br><sub>`Qwen/Qwen3.6-27B`</sub> | Qwen strong | 27.8B | Qwen3.5 hybrid (Gated DeltaNet linear attention + full attention) | 64 = 48 linear + 16 full | 262,144 | 248,320 | 55.6 GB | `6a9e13bd` | Apache-2.0 |
| **Qwen3.5-4B** <br><sub>`Qwen/Qwen3.5-4B`</sub> | Qwen weak | 4.66B | same as above | 32 = 24 linear + 8 full | 262,144 | 248,320 | 9.3 GB | `851bf6e8` | Apache-2.0 |
| **gemma-4-31B-it** <br><sub>`google/gemma-4-31B-it`</sub> | Gemma strong | 31.3B | Gemma 4 (sliding-window + full attention) | 60 = 50 sliding (window 1024) + 10 full | 262,144 | 262,144 | 62.5 GB | `842da379` | Apache-2.0 |
| **gemma-4-E4B-it** <br><sub>`google/gemma-4-E4B-it`</sub> | Gemma weak | 8.0B stored, ~4B effective ("E4B"; per-layer embeddings) | same as above | 42 = 35 sliding (window 512) + 7 full | 131,072 | 262,144 | 16.0 GB | `ee0ef602` | Apache-2.0 |

Within each family the two models share a tokenizer and architecture, so
strong-vs-weak is not confounded with generation. Qwen and Gemma pairs are
size-matched across families (~27-31B, ~4B). All four are multimodal
checkpoints; we only send text.

---

## Historical all4 capacity (2026-09-23)

On the `prompt-transfer` cluster (Lambda Kubernetes, 2x H100 80GB HBM3). The
endpoints are OpenAI-compatible and listen on the pod only; the same list is in
`~/ptx/endpoints.json`. From the login VM, reach them through the SSH tunnel in
`README.md` (Serving), which maps port 800N on the pod to 1800N locally, e.g.
`http://localhost:18001/v1` for Qwen3.6-27B.

| Model (served name) | Endpoint | GPU (`all4`) | Served context | Max concurrent requests | KV cache (tokens) | Linear-attention state slots | `mem_fraction_static` | Thinking by default | Startup |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen3.6-27B` | `http://localhost:8001/v1` | 0 | 8,192 | 16 | 47,799 | 16 (2.4 GB) | 0.72 | **on**: send `enable_thinking: false` | ~60 s |
| `Qwen3.5-4B` | `http://localhost:8002/v1` | 1 | 8,192 | 16 | 21,564 | 16 (0.8 GB) | 0.82 | **on**: send `enable_thinking: false` | ~40 s |
| `gemma-4-31B-it` | `http://localhost:8003/v1` | 1 | 8,192 | **2** | 7,144 full + 4,355 sliding | — | 0.81 | off | ~50 s |
| `gemma-4-E4B-it` | `http://localhost:8004/v1` | 0 | 8,192 | 16 | 39,994 full + 12,305 sliding | — | 0.87 | off | ~40 s |

**Served context** caps prompt + completion. The measured run used 8,192;
long reflection prompts in later GEPA layouts use 32,768. Native model context
alone does not establish that a layout has enough KV capacity.

**gemma-4-31B-it's limit of 2 concurrent requests** comes from memory, not
choice. In `all4` it shares GPU 1 with Qwen3.5-4B. SGLang reserves ~1.15K
sliding-window tokens (~0.8 MiB/token) per running request, plus two prefill
chunks, and only ~3.9 GiB is left. The `strong` layout gives it a whole GPU.

**Thinking:** pass `"chat_template_kwargs": {"enable_thinking": false}` in every
request. Qwen's chat template thinks unless told not to; Gemma 4 ignores the
flag when thinking is already off.

---

## Historical throughput (2026-09-23)

Measured with `serving/smoke.py`: 16 concurrent chat requests, short prompts
(< 100 tokens), `max_tokens` 256, temperature 0. The smoke test hits one model
at a time, so these are single-model numbers. When both models sharing a GPU are
busy, each gets less.

| Model | tok/s, deterministic | 16 requests wall time | tok/s, non-deterministic | Same text alone vs. in a batch, non-deterministic |
|---|---|---|---|---|
| Qwen3.6-27B | **191** | 13.8 s | 474 | yes |
| Qwen3.5-4B | **682** | 3.6 s | 1,583 | **no** (3 distinct outputs of 4) |
| gemma-4-31B-it | **30** | 67.0 s | 81 | yes |
| gemma-4-E4B-it | **405** | 5.2 s | 1,215 | **no** (2 distinct outputs of 4) |

Deterministic mode (`--enable-deterministic-inference`) is on because, without
it, the two 4B models gave different temperature-0 text for the same request
depending on what it was batched with. That breaks the paired design (PLAN
E2(a)). With it on, all four gave identical text in the original short-prompt smoke
cases, at roughly 2.5x lower throughput. Later long-prompt stress checks found
traffic-dependent differences with chunked prefill; see below.

The non-deterministic numbers come from an earlier `all4` run with two
differences: max prefill was 16,384, and Qwen3.5-4B's KV pool was only 2,462
tokens. Treat that column as indicative.

---

## Serving settings and long-prompt repeatability

The pod setup pins SGLang 0.5.18 and its environment with
`serving/sglang-0.5.18.constraints.txt`. `$PTX_HOME` defaults to `~/ptx` and
contains `models/<name>/`, `.venv-sglang/`, `logs/serve-latest/` and
`endpoints.json`. Pod storage is ephemeral; setup must be repeated after a pod
restart that loses those files.

`serve.sh` defaults to deterministic inference, disabled radix caching,
8,192-token context, a 1,024-token fallback prefill chunk and a 4,096-token
prefill batch budget. A layout entry's chunk override takes precedence over
`MODEL_CHUNKED_PREFILL`, which takes precedence over `CHUNKED_PREFILL`.
Qwen3.6-27B, Qwen3.5-4B, both Gemma 4 models and Qwen3-8B default to uncut
prefill (`-1`); uncut prefill raises the batch budget to the context length.
Explicit layout chunks, such as 4,096 for long GEPA reflection prompts, override
that default and can affect batching numerics. Memory limits still apply.

The launcher sets deterministic Triton prefill alignment per
server to its chunk size (or the fallback alignment for uncut prefill). This
avoids the observed Gemma 4 stall when 1,024-token chunks were smaller than the
backend's default 4,096-token alignment. The original single long-prompt smoke
passed with alignment fixed, but later stress tests found chunked outputs could
still vary with traffic. The model registry records those observations and
chosen uncut defaults. Use the long and reversed-wave stress checks in
`serving/smoke.py` when changing a layout; short temperature-zero tests alone
are insufficient.

Prefix caching remains disabled because cache hits change prefill computation.
On Qwen3.5/3.6 it also consumed additional linear-attention state slots. General
serving can override these settings through documented environment variables;
frozen campaign serving fixes its protocol and refuses arbitrary extra flags.
Weights are normally bf16, while gpt-oss uses MXFP4 and the Llama70 checkpoint
uses FP8. Do not infer precision from a default dtype flag alone.
