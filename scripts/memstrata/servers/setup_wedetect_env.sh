#!/usr/bin/env bash
# Build an isolated conda env for WeDetect-Ref (referring-expression grounding).
# Upstream: models/vendor/WeDetect (GPL-v3, run as an isolated service).
# Ref pipeline only -> does NOT need mmcv/mmdet.
#
# Usage:
#   bash setup_wedetect_env.sh                # default: env name "wedetect", pip via tencent mirror
#   ENV_NAME=wedetect USE_MIRROR=0 bash setup_wedetect_env.sh
#   SKIP_FLASH=1 bash setup_wedetect_env.sh   # skip flash-attn (we fall back to sdpa)
set -euo pipefail

ENV_NAME="${ENV_NAME:-wedetect}"
PY_VER="${PY_VER:-3.10}"
USE_MIRROR="${USE_MIRROR:-1}"
SKIP_FLASH="${SKIP_FLASH:-0}"

if [[ "${USE_MIRROR}" == "1" ]]; then
  PIP_IDX="-i https://mirrors.cloud.tencent.com/pypi/simple"
else
  PIP_IDX=""
fi

echo "[1/5] create conda env: ${ENV_NAME} (python ${PY_VER})"
eval "$(conda shell.bash hook)"
conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
conda activate "${ENV_NAME}"

echo "[2/5] install torch 2.5.1 + cu124 (matches upstream README)"
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] install WeDetect-Ref core deps (no mmcv/mmdet for Ref)"
pip install ${PIP_IDX} \
  transformers==4.57.1 \
  trl==0.17.0 \
  accelerate==1.10.0

echo "[4/5] install misc runtime deps"
pip install ${PIP_IDX} \
  "pillow>=10" "numpy<2" einops safetensors sentencepiece

if [[ "${SKIP_FLASH}" == "1" ]]; then
  echo "[5/5] SKIP flash-attn (will run with attn_implementation=sdpa)"
else
  echo "[5/5] install flash-attn (needs nvcc + a few min to build; set SKIP_FLASH=1 to skip)"
  pip install ${PIP_IDX} ninja
  pip install flash-attn==2.7.4.post1 --no-build-isolation ${PIP_IDX} \
    || echo "WARN: flash-attn build failed -> run inference with attn_implementation=sdpa instead"
fi

echo
echo "DONE. Activate with:  conda activate ${ENV_NAME}"
echo "Smoke test (needs 1 GPU):"
cat <<'EOF'
  REPO=./models/vendor/WeDetect
  WEIGHTS=${PUBLIC_MODELS_ROOT}/_classified_by_task/Object_Detection
  cd "$REPO"
  python infer_wedetect_ref.py \
    --wedetect_ref_checkpoint "$WEIGHTS/WeDetect-Ref-2B" \
    --wedetect_uni_checkpoint "$WEIGHTS/WeDetect/wedetect_base_uni.pth" \
    --image assets/demo.jpeg \
    --query "红褐色的松鼠" \
    --visualize
  # -> writes pred.png in $REPO
EOF
