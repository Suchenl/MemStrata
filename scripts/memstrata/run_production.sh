#!/usr/bin/env bash
# MemStrata screenplay-driven production run.
#
# Thin wrapper: sets env + PYTHONPATH and calls the runner module `memstrata.production.run`
# (logic lives entirely in src/memstrata/, not here). Outputs land under
#   production/outputs/<story_id>/<system>/<timestamp>/
#
# Usage:
#   bash scripts/memstrata/run_production.sh \
#     [SCREENPLAY] [BACKEND] [SYSTEM]
# Env overrides:
#   PY           python3 interpreter (default: python3)
#   FLUX=1       add FLUX I2I keyframe fusion
#   FORCE=1      recompose a fresh keyframe every shot (film-quality)
#   CHUNKS=N     limit shots (0/unset = whole screenplay)
#   CROP_ACQ_GPU crop-acquisition server GPU index
#   MLLM_GPU     GPU index for the auto-served Qwen MLLM endpoint (default 0)
#   MLLM_PORT    Qwen MLLM port (default 8000)
#   NO_AUTOSERVE=1  do not auto-launch services (assume Qwen already up)
#   STOP_SERVICES=1 stop services this run launched when it finishes
#   EXTRA        extra args forwarded to `python -m memstrata.production.run`
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the MemStrata repository
cd "$HERE"

SCREENPLAY="${1:-production/screenplay/products/en/0000_detective_mystery.json}"
BACKEND="${2:-wan22_i2v_a14b_lightx2v_4step}"
SYSTEM="${3:-memstrata}"
PY="${PY:-python3}"

args=(-m memstrata.production.run
  --screenplay "$SCREENPLAY" --backend "$BACKEND" --system "$SYSTEM")
# FLUX keyframe fusion is ON by default (run.py); NO_FLUX=1 disables it.
[ "${NO_FLUX:-0}" = "1" ] && args+=(--no-flux)
[ "${FORCE:-0}" = "1" ] && args+=(--force-recompose)
n="${SEGMENTS:-${CHUNKS:-}}"
[ -n "$n" ] && args+=(--segments "$n")
[ -n "${CROP_ACQ_GPU:-}" ] && args+=(--crop-acq-device "$CROP_ACQ_GPU")
[ -n "${MLLM_GPU:-}" ]  && args+=(--mllm-gpu "$MLLM_GPU")
[ -n "${MLLM_PORT:-}" ] && args+=(--mllm-port "$MLLM_PORT")
[ "${NO_AUTOSERVE:-0}" = "1" ]  && args+=(--no-autoserve)
[ "${STOP_SERVICES:-0}" = "1" ] && args+=(--stop-services)
[ -n "${EXTRA:-}" ]     && args+=(${EXTRA})

echo "[run_production] screenplay=$SCREENPLAY backend=$BACKEND system=$SYSTEM py=$PY"
PYTHONPATH="src:${PYTHONPATH:-}" exec "$PY" "${args[@]}"
