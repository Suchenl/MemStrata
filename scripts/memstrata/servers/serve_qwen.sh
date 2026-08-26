#!/usr/bin/env bash
# Launch the Qwen OpenAI-compatible MLLM server for the *memstrata* production loop.
# (Pipeline-scoped on purpose: each pipeline keeps its own service launchers under
#  scripts/<pipeline>/servers/, e.g. scripts/baselines/iamflow/servers/serve_iamflow_vllm.sh.)
#
# Text roles  -> Qwen3.5-9B         (served name matches roles.DEFAULT_MODEL)
# Vision roles -> Qwen3-VL-8B-Instruct (served name matches runner.vision_model)
#
# Usage (under tmux on a GPU node, or auto-launched by memstrata.production.services):
#   bash serve_qwen.sh text  <gpu> <port>   # e.g. text 0 8000
#   bash serve_qwen.sh vision <gpu> <port>  # e.g. vision 1 8001
set -euo pipefail

KIND="${1:?usage: serve_qwen.sh <text|vision> <gpu> <port>}"
GPU="${2:?gpu id}"
PORT="${3:?port}"

VLLM_ENV="${VLLM_ENV:-${CONDA_ENVS_ROOT}/vllm}"
PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-${PUBLIC_MODELS_ROOT}}"

case "${KIND}" in
  text)
    MODEL_PATH="${PUBLIC_MODELS_ROOT}/Qwen/Qwen3.5-9B"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B-Instruct}"
    # Qwen3.5-9B is multimodal, so this single server can play ALL roles (text R1/R2/R3/R5
    # and vision R4/R7/R8...). We keep video=0 (only the max-size *video* profiling is slow;
    # image profiling is cheap) and eager (low-QPS planning calls don't need CUDA graphs).
    IMG_LIMIT="${IMG_LIMIT:-8}"
    MM_ARGS=(--allowed-local-media-path ${ALLOWED_LOCAL_MEDIA_PATH:-.} --limit-mm-per-prompt "{\"image\":${IMG_LIMIT},\"video\":0}" --enforce-eager)
    ;;
  vision)
    MODEL_PATH="${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-8B-Instruct"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B-Instruct}"
    MM_ARGS=(--allowed-local-media-path ${ALLOWED_LOCAL_MEDIA_PATH:-.} --limit-mm-per-prompt '{"image":8,"video":1}')
    ;;
  *) echo "unknown kind: ${KIND}" >&2; exit 2 ;;
esac

[[ -d "${MODEL_PATH}" ]] || { echo "model path missing: ${MODEL_PATH}" >&2; exit 1; }

export PATH="${VLLM_ENV}/bin:${PATH}"
for d in "${VLLM_ENV}"/lib/python*/site-packages/nvidia/nvjitlink/lib; do
  [[ -d "${d}" ]] && { export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; break; }
done
export CUDA_VISIBLE_DEVICES="${GPU}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  "${MM_ARGS[@]}" \
  --trust-remote-code
