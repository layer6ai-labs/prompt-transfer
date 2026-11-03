#!/usr/bin/env bash
# Fetch Llama 3.1 8B / 70B Instruct onto the pod without access to Meta's gated
# repos. CPU only; the operator must already have access to the official
# config/tokenizer files and comply with the model's license.
#
#   Llama-3.1-8B-Instruct: Meta's bf16 weights. The four safetensors shards come
#     from the unsloth/Llama-3.1-8B-Instruct mirror and must match the sha256 of
#     meta-llama/Llama-3.1-8B-Instruct@0e9e39f (HF LFS metadata, listed below);
#     every other file (config, generation config, tokenizer, chat template,
#     index) is Meta's own, copied from the official snapshot on /layer6share
#     (aida/hf-cache, same revision) to $PTX_HOME/vendor/llama-3.1-official by
#     scp first, because unsloth changed the tokenizer and config files.
#   Llama-3.1-70B-Instruct-FP8: RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 (FP8
#     weights and activations, static per-tensor scales; bf16 does not fit two
#     H100s with a usable KV pool). Its chat template equals Meta's; its
#     tokenizer_config and special_tokens_map only add a pad token, and are
#     replaced by Meta's (identical blobs in the official 8B and 70B repos).
#
#   usage: bash serving/fetch_llama31.sh   (sky exec prompt-transfer sky/fetch-llama31.yaml -d)
set -euo pipefail
umask 0002

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/models.sh"
export HF_HUB_DISABLE_TELEMETRY=1
HF="${SGLANG_VENV}/bin/hf"
VENDOR="${PTX_HOME}/vendor/llama-3.1-official"
for f in config.json generation_config.json model.safetensors.index.json special_tokens_map.json \
    tokenizer.json tokenizer_config.json; do
  [ -s "${VENDOR}/${f}" ] || { echo "ERROR: ${VENDOR}/${f} missing (scp Meta's files first)" >&2; exit 1; }
done

tokenizers_match() {
  local dir="$1" f
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
    cmp -s "${dir}/${f}" "${VENDOR}/${f}" || return 1
  done
}

checkpoint_ready() {
  local name="$1" dir="${PTX_HOME}/models/$1"
  [ "$(cat "${dir}/REVISION" 2>/dev/null)" = "${MODEL_REV[${name}]}" ] \
    && "${SGLANG_VENV}/bin/python" "${HERE}/validate_model.py" "${dir}" \
    && tokenizers_match "${dir}"
}

# --- 8B --------------------------------------------------------------------
name=Llama-3.1-8B-Instruct
dir="${PTX_HOME}/models/${name}"
if ! checkpoint_ready "${name}"; then
  rm -f "${dir}/REVISION"
  "${HF}" download unsloth/Llama-3.1-8B-Instruct --revision 4699cc75b550f9c6f3173fb80f4703b62d946aa5 \
    --include "model-0000?-of-00004.safetensors" --local-dir "${dir}"
  (cd "${dir}" && sha256sum -c - <<'EOF'
2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668  model-00001-of-00004.safetensors
09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15  model-00002-of-00004.safetensors
fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa  model-00003-of-00004.safetensors
92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b  model-00004-of-00004.safetensors
EOF
  )
  for f in config.json generation_config.json model.safetensors.index.json special_tokens_map.json \
      tokenizer.json tokenizer_config.json; do cp "${VENDOR}/${f}" "${dir}/${f}"; done
  "${SGLANG_VENV}/bin/python" "${HERE}/validate_model.py" "${dir}"
  tokenizers_match "${dir}"
  echo "${MODEL_REV[${name}]}" > "${dir}/REVISION"
fi
echo "${name}: $(cat "${dir}/REVISION")"

# --- 70B FP8 ---------------------------------------------------------------
name=Llama-3.1-70B-Instruct-FP8
dir="${PTX_HOME}/models/${name}"
if ! checkpoint_ready "${name}"; then
  rm -f "${dir}/REVISION"
  "${HF}" download "${MODEL_REPO[${name}]}" --revision "${MODEL_REV[${name}]}" --local-dir "${dir}"
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do cp "${VENDOR}/${f}" "${dir}/${f}"; done
  "${SGLANG_VENV}/bin/python" "${HERE}/validate_model.py" "${dir}"
  tokenizers_match "${dir}"
  echo "${MODEL_REV[${name}]}" > "${dir}/REVISION"
fi
echo "${name}: $(cat "${dir}/REVISION")"

# Both revision markers are written only after checkpoint completeness and
# official tokenizer equality pass. Cached markers receive the same checks.
