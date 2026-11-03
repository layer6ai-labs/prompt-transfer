#!/usr/bin/env bash
# One-time pod setup: pinned SGLang venv + executor weights at pinned
# revisions, all under $PTX_HOME. Needs no GPU. Safe to re-run: the venv is
# rebuilt only when the constraints change. A model is fetched when its REVISION
# differs from models.sh or the cached checkpoint fails completeness checks.
# Llama checkpoints use the dedicated fetch_llama31.sh recipe instead.
#   usage: bash serving/setup.sh [model ...]    (default: all4 layout)
#          bash serving/setup.sh --layout qwen3
set -euo pipefail
umask 0002

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/models.sh"

MODELS=()
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo "Usage: bash serving/setup.sh [--layout LAYOUT | MODEL ...] (default: --layout all4)"
  exit 0
elif [ "${1:-}" = "--layout" ] || [ "$#" -eq 0 ]; then
  [ "$#" -eq 0 ] || [ "$#" -eq 2 ] || { echo "ERROR: --layout needs one layout name" >&2; exit 2; }
  layout="${2:-all4}"
  entries="${LAYOUTS[${layout}]:?unknown layout ${layout}}"
  declare -A selected=()
  for entry in ${entries}; do
    name="${entry%%:*}"
    if [ -z "${selected[${name}]:-}" ]; then MODELS+=("${name}"); selected[${name}]=1; fi
  done
else
  MODELS=("$@")
fi
for name in "${MODELS[@]}"; do
  [ -n "${MODEL_REV[${name}]:-}" ] || { echo "ERROR: unknown model ${name}" >&2; exit 2; }
  case "${name}" in
    Llama-3.1-8B-Instruct|Llama-3.1-70B-Instruct-FP8)
      echo "ERROR: ${name} requires serving/fetch_llama31.sh (verified weights and official tokenizer files)." >&2
      exit 2
      ;;
  esac
done

UV="$(command -v uv || echo "${HOME}/.local/bin/uv")"
export UV_CACHE_DIR="${PTX_HOME}/.cache/uv"
mkdir -p "${PTX_HOME}/models" "${PTX_HOME}/logs"

# --- SGLang venv -----------------------------------------------------------
CONSTRAINTS="${HERE}/sglang-${SGLANG_VERSION}.constraints.txt"
RECIPE="sglang-${SGLANG_VERSION}-py312-$(sha256sum "${CONSTRAINTS}" | cut -c1-12)"
if [ ! -x "${SGLANG_VENV}/bin/python" ] || [ "$(cat "${SGLANG_VENV}/RECIPE" 2>/dev/null)" != "${RECIPE}" ]; then
  echo "Building ${SGLANG_VENV} (${RECIPE})"
  rm -rf "${SGLANG_VENV}"
  "${UV}" venv --python 3.12 "${SGLANG_VENV}"
  "${UV}" pip install --python "${SGLANG_VENV}/bin/python" --prerelease=allow \
    -c "${CONSTRAINTS}" "sglang==${SGLANG_VERSION}" ninja
  echo "${RECIPE}" > "${SGLANG_VENV}/RECIPE"
else
  echo "SGLang venv up to date (${RECIPE})"
fi
"${UV}" pip freeze --python "${SGLANG_VENV}/bin/python" > "${PTX_HOME}/sglang-freeze.txt"

# --- weights ---------------------------------------------------------------
export HF_HUB_DISABLE_TELEMETRY=1

pids=() names=()
for name in "${MODELS[@]}"; do
  dir="${PTX_HOME}/models/${name}"
  rev="${MODEL_REV[${name}]:?unknown model ${name}}"
  if [ "$(cat "${dir}/REVISION" 2>/dev/null)" = "${rev}" ] \
      && "${SGLANG_VENV}/bin/python" "${HERE}/validate_model.py" "${dir}"; then
    echo "${name}: already at ${rev}"
    continue
  fi
  echo "${name}: downloading ${MODEL_REPO[${name}]}@${rev}"
  rm -f "${dir}/REVISION"
  read -ra pats <<< "${MODEL_EXCLUDE[${name}]:-}"  # read: no glob expansion
  exclude=()
  for pat in "${pats[@]}"; do exclude+=(--exclude "${pat}"); done
  "${SGLANG_VENV}/bin/hf" download "${MODEL_REPO[${name}]}" --revision "${rev}" "${exclude[@]}" \
    --local-dir "${dir}" > "${PTX_HOME}/logs/download-${name}.log" 2>&1 &
  pids+=($!) names+=("${name}")
done

failed=0
for i in "${!pids[@]}"; do
  name="${names[$i]}"
  dir="${PTX_HOME}/models/${name}"
  if wait "${pids[$i]}" \
      && "${SGLANG_VENV}/bin/python" "${HERE}/validate_model.py" "${dir}"; then
    echo "${MODEL_REV[${name}]}" > "${PTX_HOME}/models/${name}/REVISION"
  else
    rm -f "${dir}/REVISION"
    echo "ERROR: ${name} download or checkpoint validation failed; see ${PTX_HOME}/logs/download-${name}.log" >&2
    tail -5 "${PTX_HOME}/logs/download-${name}.log" >&2
    failed=1
  fi
done

df -h "${PTX_HOME}" | tail -1
du -sh "${PTX_HOME}/models" "${SGLANG_VENV}" "${UV_CACHE_DIR}" 2>/dev/null || true
exit "${failed}"
