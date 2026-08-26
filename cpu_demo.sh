#!/usr/bin/env bash
# CPU demo copied from scripts/memstrata/run_production.sh, with recording defaults.
# No GPU, no PUBLIC_MODELS_ROOT, no Qwen. Writes mp4 + bank.json under production/outputs/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install ffmpeg, then re-run." >&2
  exit 2
fi
export PY="${PY:-python3}"
export NO_FLUX=1
export NO_AUTOSERVE=1
export SEGMENTS="${SEGMENTS:-2}"
exec bash "$HERE/scripts/memstrata/run_production.sh" \
  production/screenplay/products/en/0000_detective_mystery.json \
  recording \
  memstrata
