# Model registry and GPU layouts for the prompt-transfer pod.
# Sourced by setup.sh and serve.sh; runs on the pod, not the login VM.

# Everything on the pod lives here. The pod disk is ephemeral (no PVC), so a
# restarted pod needs setup.sh again.
PTX_HOME="${PTX_HOME:-${HOME}/ptx}"
SGLANG_VENV="${PTX_HOME}/.venv-sglang"
SGLANG_VERSION="0.5.18"

# Served name -> HF repo, pinned revision, port. Revisions match the copies
# on /layer6share, so pod and share serve identical weights.
declare -A MODEL_REPO=(
  [Qwen3.6-27B]=Qwen/Qwen3.6-27B
  [Qwen3.5-4B]=Qwen/Qwen3.5-4B
  [gemma-4-31B-it]=google/gemma-4-31B-it
  [gemma-4-E4B-it]=google/gemma-4-E4B-it
)
declare -A MODEL_REV=(
  [Qwen3.6-27B]=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
  [Qwen3.5-4B]=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  [gemma-4-31B-it]=842da3794eaa0b77d5f08bae87a17459d91ff475
  [gemma-4-E4B-it]=ee0ef6023621cff504d758262d4e04895a5af4a2
)
declare -A MODEL_PORT=(
  [Qwen3.6-27B]=8001
  [Qwen3.5-4B]=8002
  [gemma-4-31B-it]=8003
  [gemma-4-E4B-it]=8004
)

# Layout entries are name:gpu:mem_fraction_static:max_running_requests,
# started in order. GPU is an index into the job's CUDA_VISIBLE_DEVICES.
#
# SGLang 0.5.18 sizes a server's KV pool from the memory free when it starts
# (pool = free_after_weights - free_before_load * (1 - fraction)), so on a
# shared GPU the big model goes first with a fraction low enough to leave
# room for its partner, and the partner takes most of what remains.
# max_running_requests is capped at 16 in all4 because the Qwen3.5/3.6
# linear-attention state is per request (~147 MiB for 27B, ~49 MiB for 4B).
# all4 fractions are fitted to the first run's logs (2026-09-23): Qwen3.5-4B
# needs ~2.5 GiB of slack for warmup, which leaves gemma-4-31B only ~3.9 GiB.
# Its sliding-window pool reserves ~1.15K tokens per running request plus two
# prefill chunks (~0.8 MiB/token), so it gets 2 concurrent requests here. Use
# strong/weak for bulk runs.
declare -A LAYOUTS=(
  # all four executors resident at once: one big + one small per GPU
  [all4]="Qwen3.6-27B:0:0.72:16 gemma-4-31B-it:1:0.81:2 gemma-4-E4B-it:0:0.87:16 Qwen3.5-4B:1:0.82:16"
  # one strong model per GPU, room for large batches
  [strong]="Qwen3.6-27B:0:0.88:64 gemma-4-31B-it:1:0.88:64"
  # one weak model per GPU
  [weak]="Qwen3.5-4B:0:0.85:64 gemma-4-E4B-it:1:0.85:64"
)
