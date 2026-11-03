#!/usr/bin/env bash
# Serve one layout from models.sh. Meant to run as a `sky exec` job that holds
# both GPUs (sky/serve.yaml). Servers start one at a time, because a shared
# GPU's KV sizing depends on start order; then the script blocks, so the job
# holds the GPUs until it is cancelled. If any server exits, the others are
# stopped and the job fails.
#
#   usage: bash serving/serve.sh [layout]    (default all4; see LAYOUTS in models.sh)
#
# Optional env:
#   CONTEXT_LENGTH     max prompt + completion tokens (default 8192)
#   CHUNKED_PREFILL    prefill chunk size (default 1024) for models without a
#                      MODEL_CHUNKED_PREFILL entry in models.sh; small because
#                      Gemma 4's sliding-window pool reserves 2 chunks. Also
#                      sets the deterministic triton prefill alignment. Models
#                      set to -1 are never cut and get MAX_PREFILL =
#                      CONTEXT_LENGTH so any prompt fits in one prefill.
#   MAX_PREFILL        tokens per prefill batch (default 4096); bounds
#                      activation memory on shared GPUs
#   DETERMINISTIC      1 (default) adds --enable-deterministic-inference, so
#                      temperature-0 output does not depend on batching, at
#                      ~2.5x lower throughput. Set 0 only for throwaway runs.
#   RADIX_CACHE        1 re-enables prefix caching (default off, see below)
#   SGLANG_EXTRA_ARGS  appended to every server's command line (per-model flags:
#                      MODEL_EXTRA_ARGS in models.sh)
#
# Prefix caching is off by default: a cache hit changes the prefill path, which
# can change temperature-0 output, and on Qwen3.5/3.6 it costs ~6
# linear-attention state slots per request instead of 1.
set -euo pipefail
umask 0002

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/models.sh"

LAYOUT="${1:-all4}"
ENTRIES="${LAYOUTS[${LAYOUT}]:?unknown layout ${LAYOUT}; choose from: ${!LAYOUTS[*]}}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1024}"
MAX_PREFILL="${MAX_PREFILL:-4096}"
PY="${SGLANG_VENV}/bin/python"
[ -x "${PY}" ] || { echo "ERROR: ${PY} missing; run serving/setup.sh first." >&2; exit 1; }

LOG_DIR="${PTX_HOME}/logs/serve-$(date +%Y%m%d-%H%M%S)-${LAYOUT}"
mkdir -p "${LOG_DIR}"
ln -sfn "${LOG_DIR}" "${PTX_HOME}/logs/serve-latest"

# The job's GPUs, so layout index 0/1 maps onto whatever sky assigned.
IFS=, read -r -a JOB_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"

# SGLang JIT-compiles kernels with the venv's ninja, so its bin goes on PATH.
export PATH="${SGLANG_VENV}/bin:${PATH}"

# FlashInfer JIT and DeepGEMM need a CUDA toolkit. The image's /usr/local/cuda
# is 12.1 but torch is built for CUDA 13, so use the toolkit bundled in the
# venv, adding the lib64 and unversioned .so links the build expects.
bundled_cuda=""
for candidate in "${SGLANG_VENV}"/lib/python3.*/site-packages/nvidia/cu13; do
  if [ -x "${candidate}/bin/nvcc" ]; then bundled_cuda="${candidate}"; break; fi
done
if [ -x "${bundled_cuda}/bin/nvcc" ]; then
  export CUDA_HOME="${bundled_cuda}" PATH="${bundled_cuda}/bin:${PATH}"
  [ -e "${bundled_cuda}/lib64" ] || ln -s lib "${bundled_cuda}/lib64"
  for so in "${bundled_cuda}/lib"/lib*.so.*; do
    [ -e "${so}" ] || continue
    base="$(basename "${so}")"
    [ -e "${bundled_cuda}/lib/${base%%.so.*}.so" ] || ln -s "${base}" "${bundled_cuda}/lib/${base%%.so.*}.so"
  done
fi
export SGL_ENABLE_JIT_DEEPGEMM=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# The bundled nvcc is 13.4 but its cuda_runtime headers are 13.0, and CCCL
# refuses to compile when the minor versions differ ("CUDA compiler and CUDA
# toolkit headers are incompatible"). That broke FlashInfer's JIT build of the
# gpt-oss router GEMM (tinygemm2) on 2026-10-02 (job 42). Minor versions are
# compatible, so skip the check for FlashInfer JIT builds.
export FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

pids=()
cleanup() {
  trap - EXIT INT TERM
  [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_ready() {  # <name> <port> <pid> <logfile>
  local name="$1" port="$2" pid="$3" log="$4" waited=0
  until curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: ${name} exited during startup. Last log lines:" >&2
      tail -40 "${log}" >&2
      exit 1
    fi
    if [ "${waited}" -ge 1800 ]; then
      echo "ERROR: ${name} not healthy after ${waited}s." >&2
      tail -40 "${log}" >&2
      exit 1
    fi
    sleep 10; waited=$((waited + 10))
  done
  echo "${name}: healthy on :${port} after ~${waited}s"
}

extra=()
[ "${DETERMINISTIC:-1}" = "1" ] && extra+=(--enable-deterministic-inference)
# Deterministic mode cuts each prefill chunk down to a multiple of an alignment
# (4096 by default for the triton backend) and skips a request whose chunk is
# smaller than that. With 1024-token chunks, any prompt longer than one chunk
# then waits in the queue forever, with no error returned. Gemma 4 runs on
# triton (Qwen on fa3, which has no alignment), so match it to the chunk size.
# SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE is the same trap on flashinfer.
# Set per server below, to that server's chunk size.
[ "${RADIX_CACHE:-0}" = "1" ] || extra+=(--disable-radix-cache)
read -r -a user_extra <<< "${SGLANG_EXTRA_ARGS:-}"

echo "layout ${LAYOUT}: ${ENTRIES}"
echo "logs: ${LOG_DIR}"
served=()  # name=port, for endpoints.json
for entry in ${ENTRIES}; do
  IFS=: read -r name gpu frac max_running layout_chunk <<< "${entry}"
  port="${MODEL_PORT[${name}]}"
  served+=("${name}=${port}")
  model_dir="${PTX_HOME}/models/${name}"
  [ "$(cat "${model_dir}/REVISION" 2>/dev/null)" = "${MODEL_REV[${name}]}" ] \
    || { echo "ERROR: ${model_dir} is not at ${MODEL_REV[${name}]}; run serving/setup.sh ${name}" >&2; exit 1; }
  log="${LOG_DIR}/${name}.log"
  chunk="${layout_chunk:-${MODEL_CHUNKED_PREFILL[${name}]:-${CHUNKED_PREFILL}}}"
  max_prefill="${MAX_PREFILL}"
  [ "${chunk}" -gt 0 ] || max_prefill="${CONTEXT_LENGTH}"
  # gpu is one index; several joined by "+" for one tensor-parallel server, or by ","
  # for data-parallel replicas (one per GPU, one port, SGLang balances the requests)
  IFS='+,' read -r -a gidx <<< "${gpu}"
  devs=""
  for i in "${gidx[@]}"; do
    [[ "${i}" =~ ^[0-9]+$ ]] && [ "${i}" -lt "${#JOB_GPUS[@]}" ] || {
      echo "ERROR: layout ${LAYOUT} references unassigned GPU index ${i}" >&2; exit 2;
    }
    devs+="${JOB_GPUS[${i}]},"
  done
  devs="${devs%,}"
  tp=()
  [[ "${gpu}" == *+* ]] && tp=(--tp-size "${#gidx[@]}")
  [[ "${gpu}" == *,* ]] && tp=(--dp-size "${#gidx[@]}")
  echo "starting ${name} on GPU ${devs} :${port} (mem_fraction_static=${frac}, max_running_requests=${max_running}, chunked_prefill=${chunk}, max_prefill=${max_prefill})"
  align="${chunk}"
  [ "${chunk}" -gt 0 ] || align="${CHUNKED_PREFILL}"
  read -r -a model_extra <<< "${MODEL_EXTRA_ARGS[${name}]:-}"
  CUDA_VISIBLE_DEVICES="${devs}" SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE="${align}" \
    "${PY}" -m sglang.launch_server \
    --model-path "${model_dir}" \
    --served-model-name "${name}" \
    --host 0.0.0.0 --port "${port}" \
    --trust-remote-code \
    --mem-fraction-static "${frac}" \
    --max-running-requests "${max_running}" \
    --context-length "${CONTEXT_LENGTH}" \
    --chunked-prefill-size "${chunk}" \
    --max-prefill-tokens "${max_prefill}" \
    --sampling-backend pytorch \
    "${tp[@]}" "${extra[@]}" "${model_extra[@]}" "${user_extra[@]}" \
    > "${log}" 2>&1 &
  pids+=($!)
  wait_ready "${name}" "${port}" "$!" "${log}"
  grep -h -i -E 'KV Cache is allocated|max_total_num_tokens|mamba.*(cache|size)|Memory pool end' "${log}" | tail -4 | sed "s/^/  [${name}] /" || true
done

"${PY}" - "${LOG_DIR}/endpoints.json" "${served[@]}" <<'EOF'
import json, sys
out, pairs = sys.argv[1], [s.split("=") for s in sys.argv[2:]]
with open(out, "w") as f:
    json.dump({name: f"http://localhost:{port}/v1" for name, port in pairs}, f, indent=2)
EOF
cp "${LOG_DIR}/endpoints.json" "${PTX_HOME}/endpoints.json"
echo "ALL READY ($(date -u +%FT%TZ)):"
cat "${PTX_HOME}/endpoints.json"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Block until any server exits; the EXIT trap then stops the rest.
wait -n "${pids[@]}"
echo "ERROR: a server exited; stopping the layout." >&2
exit 1
