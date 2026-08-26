#!/usr/bin/env bash
# Manual launcher for the persistent crop-acquisition server
# (GroundingDINO + SAM3 + DINOv3, loaded once).
#
# The production loop normally auto-starts this via
# memstrata.skills.crop_acquisition.crop_client. This script is the
# manual/debug entrypoint.
#
# SAM3 needs transformers>=5.9 (vendored dir prepended on PYTHONPATH).
# Use a CPython 3.11 interpreter with torch: export MEMSTRATA_PYTHON=...
#
# Usage:
#   bash scripts/memstrata/servers/serve_crop_acq.sh <server_dir> [device] [idle_timeout]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SAM3_DEPS="${MEMSTRATA_SAM3_DEPS:-${HERE}/models/vendor/sam3_transformers59}"
PY="${MEMSTRATA_PYTHON:-python3}"

SERVER_DIR="${1:?usage: serve_crop_acq.sh <server_dir> [device] [idle_timeout]}"
DEVICE="${2:-}"
IDLE_TIMEOUT="${3:-1800}"

export PYTHONPATH="${SAM3_DEPS}:${HERE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec "${PY}" -m memstrata.skills.crop_acquisition.crop_server \
  --server_dir "${SERVER_DIR}" \
  --device "${DEVICE}" \
  --idle_timeout "${IDLE_TIMEOUT}"
