#!/usr/bin/env bash
# Manual launcher for the *memstrata* persistent crop-acquisition server
# (GroundingDINO + SAM3 + DINOv3, loaded once). Pipeline-scoped: memstrata's service
# launchers live under scripts/memstrata/servers/ (mirrors scripts/baselines/iamflow/servers/).
#
# NOTE: in the production loop this server is normally auto-started programmatically by
# memstrata.skills.crop_acquisition.crop_client (python -m ...crop_server). This script is the
# manual/debug entrypoint that reproduces that launch by hand.
#
# SAM3 needs transformers>=5.9, which lives in the vendored dir prepended on PYTHONPATH.
# Everything is overridable via env; defaults hard-resolve to the Montage monorepo.
#
# Usage:
#   bash scripts/memstrata/servers/serve_crop_acq.sh <server_dir> [device] [idle_timeout]
# Example:
#   bash scripts/memstrata/servers/serve_crop_acq.sh /path/to/run/crop_acq_server 0 1800
set -euo pipefail

MONTAGE_ROOT="${MONTAGE_ROOT:-${MONTAGE_ROOT}}"
SAM3_DEPS="${MEMSTRATA_SAM3_DEPS:-${MONTAGE_ROOT}/models/vendor/sam3_transformers59}"
# The vendored sam3_transformers59 bundle is cp311; run under a py3.11 + torch env (helios).
PY="${MEMSTRATA_PYTHON:-python3}"

SERVER_DIR="${1:?usage: serve_crop_acq.sh <server_dir> [device] [idle_timeout]}"
DEVICE="${2:-}"
IDLE_TIMEOUT="${3:-1800}"

# Vendored transformers>=5.9 MUST come first so SAM3 classes import; then the two src roots.
export PYTHONPATH="${SAM3_DEPS}:${MONTAGE_ROOT}/benchmarks/MemStrata/src:${MONTAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PUBLIC_MODELS_ROOT="${PUBLIC_MODELS_ROOT:-${PUBLIC_MODELS_ROOT}}"
export MONTAGE_ROOT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec "${PY}" -m memstrata.skills.crop_acquisition.crop_server \
  --server_dir "${SERVER_DIR}" \
  --device "${DEVICE}" \
  --idle_timeout "${IDLE_TIMEOUT}"
