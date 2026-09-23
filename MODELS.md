# Models

The executor roster as deployed, plus the models planned but not yet served.
Serving config lives in `serving/models.sh` and `serving/serve.sh`; how to run it
is in `README.md` (Serving). Measured numbers are from 2026-09-23 on the `all4`
layout; after changing either file, re-measure with `sky/smoke.yaml`.

---

## Executors: what they are

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

## Executors: how they are served

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

**Served context** is the cap on prompt + completion. It is 8,192 for all four
models, well under what they support, to save memory; our tasks fit.

**gemma-4-31B-it's limit of 2 concurrent requests** comes from memory, not
choice. In `all4` it shares GPU 1 with Qwen3.5-4B. SGLang reserves ~1.15K
sliding-window tokens (~0.8 MiB/token) per running request, plus two prefill
chunks, and only ~3.9 GiB is left. The `strong` layout gives it a whole GPU.

**Thinking:** pass `"chat_template_kwargs": {"enable_thinking": false}` in every
request. Qwen's chat template thinks unless told not to; Gemma 4 ignores the
flag when thinking is already off.

---

## Throughput

Measured with `serving/smoke.py`: 16 concurrent chat requests, short prompts
(< 100 tokens), `max_tokens` 256, temperature 0. The smoke test hits one model
at a time, so these are single-model numbers. When both models sharing a GPU are
busy, each gets less.

| Model | tok/s, deterministic (current) | 16 requests wall time | tok/s, non-deterministic | Same text alone vs. in a batch, non-deterministic |
|---|---|---|---|---|
| Qwen3.6-27B | **191** | 13.8 s | 474 | yes |
| Qwen3.5-4B | **682** | 3.6 s | 1,583 | **no** (3 distinct outputs of 4) |
| gemma-4-31B-it | **30** | 67.0 s | 81 | yes |
| gemma-4-E4B-it | **405** | 5.2 s | 1,215 | **no** (2 distinct outputs of 4) |

Deterministic mode (`--enable-deterministic-inference`) is on because, without
it, the two 4B models gave different temperature-0 text for the same request
depending on what it was batched with. That breaks the paired design (PLAN
E2(a)). With it on, all four gave identical text in every case, at roughly 2.5x
lower throughput.

The non-deterministic numbers come from an earlier `all4` run with two
differences: max prefill was 16,384, and Qwen3.5-4B's KV pool was only 2,462
tokens. Treat that column as indicative.

---

## Serving stack

| | |
|---|---|
| Cluster | `prompt-transfer`: Lambda Kubernetes pod, 2x H100 80GB HBM3, driver 580.126.20, no `/layer6share` mount, ephemeral disk |
| On the pod | `/home/sky/ptx`: `models/<name>/` (with a `REVISION` file), `.venv-sglang/`, `logs/serve-latest/`, `endpoints.json` |
| Engine | SGLang 0.5.18, torch 2.13.0, transformers 5.12.1, flashinfer 0.6.17. All pinned via `serving/sglang-0.5.18.constraints.txt`, copied from the SGLang venv smart-memory already runs |
| Flags on every server | `--enable-deterministic-inference --disable-radix-cache --context-length 8192 --chunked-prefill-size 1024 --max-prefill-tokens 4096 --sampling-backend pytorch`; bf16 weights and KV cache |
| Layouts | `all4`: all four resident, tested (numbers above). `strong` (one 27-31B model per GPU) and `weak` (one 4B per GPU): defined in `serving/models.sh`, **not yet tested** |

Why prefix caching is off: a cache hit changes the prefill computation, which
can change temperature-0 output. On Qwen3.5/3.6 it also costs ~6
linear-attention state slots per request instead of 1, which had capped
Qwen3.6-27B at 3 concurrent requests.

Copies of the same revisions on `/layer6share`, for anything run off the pod:

| Model | Path |
|---|---|
| Qwen3.6-27B | `/layer6share/aida/hf-cache/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Qwen3.5-4B | `/layer6share/zehao/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| gemma-4-31B-it | `/layer6share/mz/models/gemma-4-31B-it` (no revision recorded; config, index and tokenizer files are identical to `842da379` and shard sizes match; shard hashes not checked) |
| gemma-4-E4B-it | `/layer6share/zehao/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2` |

---

## Planned, not yet served

| Model | Role | Size | Where | Notes |
|---|---|---|---|---|
| gpt-oss-120b | proposer / reviewer (fixed external critic, Q5) | 65.2 GB, MXFP4; fits one H100 | `/layer6share/mz/models/gpt-oss-120b` | Third family (OpenAI) and the largest model on disk; whether it out-critiques the strong executors is untested. Reasoning model: pin `reasoning_effort` (template default: medium). SGLang needs `--reasoning-parser gpt-oss`. |
| Meta-Llama-3.1-8B-Instruct | judge (optional; kept separate from the critic) | 16.1 GB | `/layer6share/aida/hf-cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct` | Fourth family. Gated on HF, so a pod download needs a token with Meta's license accepted. |

Neither fits alongside the four executors on 2 GPUs. Each needs its own layout
entry in `serving/models.sh` and a `setup.sh` download.

