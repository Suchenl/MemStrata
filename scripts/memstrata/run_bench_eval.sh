#!/usr/bin/env bash
# MemStrata bench-mode regeneration of a screenplay set.
#
# Every eval run MUST be bench-mode (no GT leakage): production/run.py now defaults to bench_mode,
# so we pass NO --oracle-assisted here. Each run writes run_manifest.json with gt_leakage=none; if
# anything leaks, run.py aborts. Output run-tag defaults to "bench" so the earlier oracle-assisted
# "optCA" runs are preserved for the oracle-vs-real A/B table.
#
# Designed for a multi-GPU quota node (gpu-h800 node1: 8 free GPUs). Pins MLLM->GPU0, crop(S5)->GPU1,
# FLUX+video auto-pick GPUs 2-7. The MLLM at :8000 is shared/reused across all stories (reuse-first).
#
# Usage:  bash scripts/memstrata/run_bench_eval.sh [run_tag] [screenplay1 screenplay2 ...]
#   Defaults to run_tag=bench over the three bundled CN stress screenplays.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the MemStrata repository
cd "$HERE"

RUN_TAG="${1:-bench}"; shift || true
SCREENPLAYS=("$@")
if [ "${#SCREENPLAYS[@]}" -eq 0 ]; then
  SCREENPLAYS=(
    production/screenplay/products/cn/0001_lighthouse_keeper.json
    production/screenplay/products/cn/0002_night_market_courier.json
    production/screenplay/products/cn/0003_desert_archaeologist.json
  )
fi

BACKEND="${BACKEND:-wan22_i2v_a14b_lightx2v_4step}"   # production default (no morphic)
PY="${PY:-python3}"
if [ -n "${PUBLIC_MODELS_ROOT:-}" ]; then
  export PUBLIC_MODELS_ROOT
fi
export PYTHONPATH="src:${PYTHONPATH:-}"

MASTER="$HERE/production/outputs/_bench_eval_${RUN_TAG}.log"
: > "$MASTER"
overall_rc=0
echo "[bench-eval] tag=$RUN_TAG backend=$BACKEND stories=${#SCREENPLAYS[@]} (BENCH-MODE / no oracle)" | tee -a "$MASTER"

for SP in "${SCREENPLAYS[@]}"; do
  STORY="$(basename "$SP" .json)"
  RUN_DIR="$HERE/production/outputs/${STORY}/memstrata/${RUN_TAG}"
  mkdir -p "$RUN_DIR"
  LOG="$RUN_DIR/run.log"
  echo "[bench-eval] >>> $STORY -> $RUN_DIR" | tee -a "$MASTER"
  # bench-mode is the DEFAULT (no --oracle-assisted). --force-recompose: fresh FLUX keyframe every
  # shot, exercising the memory read path each chunk.
  "$PY" -u -m memstrata.production.run \
    --screenplay "$SP" --backend "$BACKEND" --system memstrata \
    --flux --force-recompose \
    --decompose crop_server --crop-acq-device 1 \
    --mllm-gpu 0 --mllm-port 8000 \
    --run-dir "$RUN_DIR" \
    >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && [ "$overall_rc" -eq 0 ]; then
    overall_rc="$rc"
  fi
  # audit the manifest actually says no leakage
  LEAK="$($PY -c "import json,sys;print(json.load(open('$RUN_DIR/run_manifest.json')).get('gt_leakage'))" 2>/dev/null || echo "no_manifest")"
  echo "[bench-eval] <<< $STORY EXIT:$rc gt_leakage=$LEAK" | tee -a "$MASTER"
done

echo "[bench-eval] ALL_DONE tag=$RUN_TAG" | tee -a "$MASTER"
exit "$overall_rc"
