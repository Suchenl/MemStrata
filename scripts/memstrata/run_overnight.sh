#!/usr/bin/env bash
# Overnight MemStrata production launcher (one screenplay -> one run dir + stable log + EXIT sentinel).
#
# Designed to run under tmux on a multi-GPU node (gpu-a800 node1: 8x80GB). Pins the four services
# to distinct GPUs so nothing OOM-contends:
#   MLLM (Qwen3.5-9B) -> GPU0 ; crop-acquisition (S5) -> GPU1 ; FLUX + video auto-pick GPUs 2-7.
# Writes everything under a FIXED run dir so the local notification watcher can tail one path.
#
# Usage:  bash scripts/memstrata/run_overnight.sh <screenplay.json> <system> <chunks> <run_tag>
#   chunks=0 -> whole screenplay. run_tag names the output subdir (e.g. smoke3, full).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # benchmarks/MemStrata
cd "$HERE"

SCREENPLAY="${1:?screenplay json}"
SYSTEM="${2:-memstrata}"
CHUNKS="${3:-0}"
RUN_TAG="${4:-run}"
# Default = the clean 4-step distill (NO LoRA), matching production/run.py. To A/B the morphic
# interpolation LoRA, run `BACKEND=wan22_i2v_a14b_lightx2v_4step_morphic bash run_overnight.sh ...`;
# that variant applies morphic as a runtime LoRA (base model dir + separate .safetensors via
# lightx2v lora_configs), NOT a disk-baked merged checkpoint.
BACKEND="${BACKEND:-wan22_i2v_a14b_lightx2v_4step}"
PY="${PY:-python3}"

export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-${PUBLIC_MODELS_ROOT}}"
export PYTHONPATH="src:${PYTHONPATH:-}"

STORY="$(basename "$SCREENPLAY" .json)"
RUN_DIR="$HERE/production/outputs/${STORY}/${SYSTEM}/${RUN_TAG}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run.log"

echo "[overnight] story=$STORY backend=$BACKEND chunks=$CHUNKS run_dir=$RUN_DIR" | tee "$LOG"

# force-recompose: a fresh FLUX keyframe every shot -> film quality, no AR drift, and the memory
# read-path composes context every chunk (needed for the objective read-path metric).
"$PY" -m memstrata.production.run \
  --screenplay "$SCREENPLAY" --backend "$BACKEND" --system "$SYSTEM" \
  --chunks "$CHUNKS" --flux --force-recompose \
  --decompose crop_server --crop-acq-device 1 \
  --mllm-gpu 0 --mllm-port 8000 \
  --run-dir "$RUN_DIR" \
  >> "$LOG" 2>&1
echo "EXIT:$? run_dir=$RUN_DIR" | tee -a "$LOG"
