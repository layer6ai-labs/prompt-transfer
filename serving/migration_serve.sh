#!/usr/bin/env bash
# Serve one migration target on the existing two-GPU prompt-transfer pod.
#
# Usage: bash migration_serve.sh llama|qwen|gemma
# No weights are downloaded and no existing servers are stopped. The campaign
# coordinator must establish that its two GPUs are free before submitting this
# task. A private launch snapshot adds a layout without changing models.sh.
set -euo pipefail
umask 0002

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/models.sh"

TARGET_FAMILY="${1:-${TARGET_FAMILY:-llama}}"
case "${TARGET_FAMILY}" in
  llama) model="Llama-3.1-70B-Instruct-FP8" ;;
  qwen) model="Qwen3-32B" ;;
  gemma) model="gemma-3-27b-it" ;;
  *) echo "ERROR: unknown migration target ${TARGET_FAMILY}; choose llama, qwen, gemma" >&2; exit 2 ;;
esac

# Tensor parallelism leaves substantially more KV room than loading Qwen32 or
# Gemma27 on one GPU; all four methods use this same target configuration. The
# FP8 Llama checkpoint already requires both GPUs. Eight client threads are the
# campaign limit; cap server concurrency to the same value.
IFS=, read -r -a migration_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
[ "${#migration_gpus[@]}" -eq 2 ] || {
  echo "ERROR: migration serving requires exactly two assigned GPUs" >&2
  exit 2
}
export CONTEXT_LENGTH=32768 CHUNKED_PREFILL=4096 MAX_PREFILL=4096
export DETERMINISTIC=1 RADIX_CACHE=0
# Scientific serving settings are fixed here; arbitrary extra flags could
# silently override them later in serve.sh's argument list.
[ -z "${SGLANG_EXTRA_ARGS:-}" ] || {
  echo "ERROR: SGLANG_EXTRA_ARGS must be empty for the migration campaign" >&2
  exit 2
}

# Keep an immutable copy of the exact launcher/registry for provenance. serve.sh
# sources its neighboring models.sh, so append the dedicated layout in this
# launch-specific directory rather than editing the shared serving files.
mkdir -p "${PTX_HOME}/launchers"
snapshot="$(mktemp -d "${PTX_HOME}/launchers/migration-${TARGET_FAMILY}-XXXXXXXX")"
cp "${HERE}/models.sh" "${snapshot}/models.sh"
cp "${HERE}/serve.sh" "${snapshot}/serve.sh"
printf '\nLAYOUTS[migration-%s]="%s:0+1:0.88:8:4096"\n' \
  "${TARGET_FAMILY}" "${model}" >> "${snapshot}/models.sh"
printf '%s\n' "${CAMPAIGN_ID:-unassigned}" > "${snapshot}/campaign_id.txt"
printf 'migration target: %s (%s); launch snapshot: %s; campaign: %s\n' \
  "${TARGET_FAMILY}" "${model}" "${snapshot}" "${CAMPAIGN_ID:-unassigned}"
sha256sum "${snapshot}/models.sh" "${snapshot}/serve.sh"
exec bash "${snapshot}/serve.sh" "migration-${TARGET_FAMILY}"
