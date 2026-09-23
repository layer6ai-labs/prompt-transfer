#!/usr/bin/env bash
# One-time pod setup: pinned SGLang venv + executor weights at pinned
# revisions, all under $PTX_HOME. Needs no GPU. Safe to re-run: the venv is
# rebuilt only when the constraints change, and a model is fetched only when
# its REVISION file does not match models.sh.
#   usage: bash serving/setup.sh [model ...]    (default: every model)
set -euo pipefail
umask 0002

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/models.sh"

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
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${!MODEL_REPO[@]}")

pids=() names=()
for name in "${MODELS[@]}"; do
  dir="${PTX_HOME}/models/${name}"
  rev="${MODEL_REV[${name}]:?unknown model ${name}}"
  if [ "$(cat "${dir}/REVISION" 2>/dev/null)" = "${rev}" ]; then
    echo "${name}: already at ${rev}"
    continue
  fi
  echo "${name}: downloading ${MODEL_REPO[${name}]}@${rev}"
  rm -f "${dir}/REVISION"
  "${SGLANG_VENV}/bin/hf" download "${MODEL_REPO[${name}]}" --revision "${rev}" \
    --local-dir "${dir}" > "${PTX_HOME}/logs/download-${name}.log" 2>&1 &
  pids+=($!) names+=("${name}")
done

failed=0
for i in "${!pids[@]}"; do
  name="${names[$i]}"
  if wait "${pids[$i]}"; then
    echo "${MODEL_REV[${name}]}" > "${PTX_HOME}/models/${name}/REVISION"
  else
    echo "ERROR: ${name} download failed; see ${PTX_HOME}/logs/download-${name}.log" >&2
    tail -5 "${PTX_HOME}/logs/download-${name}.log" >&2
    failed=1
  fi
done

# Every shard named in the index must be present at full size.
"${SGLANG_VENV}/bin/python" - "${PTX_HOME}/models" "${MODELS[@]}" <<'EOF'
import json, os, sys
root, names = sys.argv[1], sys.argv[2:]
for name in names:
    d = os.path.join(root, name)
    files = os.listdir(d) if os.path.isdir(d) else []
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    index = os.path.join(d, "model.safetensors.index.json")
    need = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) else shards
    missing = sorted(set(need) - set(shards))
    size = sum(os.path.getsize(os.path.join(d, f)) for f in shards)
    rev = open(os.path.join(d, "REVISION")).read().strip() if os.path.exists(os.path.join(d, "REVISION")) else "NONE"
    status = "ok" if not missing and rev != "NONE" else f"INCOMPLETE missing={missing}"
    print(f"{name:16s} {len(shards)}/{len(need)} shards  {size/1e9:6.1f} GB  rev={rev[:12]}  {status}")
EOF

df -h "${PTX_HOME}" | tail -1
du -sh "${PTX_HOME}/models" "${SGLANG_VENV}" "${UV_CACHE_DIR}" 2>/dev/null || true
exit "${failed}"
