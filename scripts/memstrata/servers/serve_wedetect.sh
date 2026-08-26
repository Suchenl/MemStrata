#!/usr/bin/env bash
# Manual launcher for the isolated WeDetect-Ref grounding service (describe -> bbox).
# Upstream: models/vendor/WeDetect (GPL-v3) — imported ONLY inside this process; memstrata
# talks to it over stdlib HTTP (memstrata.skills.crop_acquisition.wedetect_client).
#
# Usage:
#   bash scripts/memstrata/servers/serve_wedetect.sh            # port 8710, sdpa
#   PORT=8710 REF=2B bash scripts/memstrata/servers/serve_wedetect.sh
#
# Then point memstrata at it:
#   export MEMSTRATA_WEDETECT_URL=http://127.0.0.1:8710
set -euo pipefail

MONTAGE_ROOT="${MONTAGE_ROOT:-.}"
REPO="${WEDETECT_REPO:-${MONTAGE_ROOT}/models/vendor/WeDetect}"
WEIGHTS="${WEDETECT_WEIGHTS:-${PUBLIC_MODELS_ROOT}/_classified_by_task/Object_Detection}"
PY="${WEDETECT_PYTHON:-wedetect/bin/python}"

REF="${REF:-2B}"                                   # 2B | 4B
REF_CKPT="${WEDETECT_REF_CKPT:-${WEIGHTS}/WeDetect-Ref-${REF}}"
UNI_CKPT="${WEDETECT_UNI_CKPT:-${WEIGHTS}/WeDetect/wedetect_base_uni.pth}"
ATTN="${ATTN:-sdpa}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8710}"

# The service imports the GPL repo (generate_proposal, wedetect_ref.*) from REPO.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec "${PY}" "${MONTAGE_ROOT}/methods/MemStrata/scripts/memstrata/servers/serve_wedetect.py" \
  --repo "${REPO}" \
  --ref_checkpoint "${REF_CKPT}" \
  --uni_checkpoint "${UNI_CKPT}" \
  --attn "${ATTN}" \
  --host "${HOST}" \
  --port "${PORT}"
