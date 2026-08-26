#!/usr/bin/env bash
# Build a native-Wan (official generate.py) ckpt_dir for the LightX2V 4-step distilled
# Wan2.2-I2V-A14B experts, reusing the SAME native_wan pipeline as wan22_i2v_a14b.toml.
#
# Only the DiT experts come from the distilled path; T5 / VAE / tokenizer / top-level config are
# symlinked from the original Wan2.2-I2V-A14B (per user: "wan weights from the new path, other
# modules from the previous path"). The distilled experts share the base DiT key names exactly
# (verified: 1095/1095 keys match), so they are a drop-in replacement.
#
# The official loader is diffusers `WanModel.from_pretrained(ckpt_dir, subfolder="high_noise_model")`,
# which resolves a SINGLE `diffusion_pytorch_model.safetensors` (no index.json) — so we name the
# symlink accordingly (this differs from the LightX2V-framework layout in setup_lightx2v_weights.sh).
#
# Usage:
#   bash setup_wan_distill_native_weights.sh [DISTILL_DIR] [BASE_DIR] [LAYOUT_DIR]
# Defaults target the paths the user provided. Idempotent (symlinks only, no copies).
set -euo pipefail

DISTILL="${1:-${PUBLIC_MODELS_ROOT}/lightx2v/Wan2.2-Distill-Models}"
BASE="${2:-${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B}"
LAYOUT="${3:-${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B-distill4step-native}"

HIGH="$DISTILL/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors"
LOW="$DISTILL/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors"

for f in "$HIGH" "$LOW"; do
  [ -f "$f" ] || { echo "ERROR: distilled weight not found: $f" >&2; exit 1; }
done
[ -d "$BASE" ] || { echo "ERROR: base Wan2.2-I2V-A14B not found: $BASE" >&2; exit 1; }

echo "[setup] layout dir: $LAYOUT"
mkdir -p "$LAYOUT/high_noise_model" "$LAYOUT/low_noise_model"

# Shared base components (tokenizer / T5 encoder / VAE / top-level config): from the ORIGINAL model.
for c in google Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth configuration.json README.md assets examples; do
  [ -e "$BASE/$c" ] && ln -sfn "$BASE/$c" "$LAYOUT/$c"
done

# Per-expert diffusers config.json from base (distilled weights keep the same architecture).
ln -sfn "$BASE/high_noise_model/config.json" "$LAYOUT/high_noise_model/config.json"
ln -sfn "$BASE/low_noise_model/config.json"  "$LAYOUT/low_noise_model/config.json"

# DiT experts from the distilled path, named for the diffusers single-file loader. Ensure no stale
# base shards / index leak into these subfolders (single-file load requires no index.json present).
rm -f "$LAYOUT/high_noise_model/diffusion_pytorch_model.safetensors" \
      "$LAYOUT/high_noise_model/diffusion_pytorch_model.safetensors.index.json" \
      "$LAYOUT/low_noise_model/diffusion_pytorch_model.safetensors" \
      "$LAYOUT/low_noise_model/diffusion_pytorch_model.safetensors.index.json"
ln -sfn "$(readlink -f "$HIGH")" "$LAYOUT/high_noise_model/diffusion_pytorch_model.safetensors"
ln -sfn "$(readlink -f "$LOW")"  "$LAYOUT/low_noise_model/diffusion_pytorch_model.safetensors"

echo "[setup] done. Contents:"
ls -l "$LAYOUT" "$LAYOUT/high_noise_model" "$LAYOUT/low_noise_model"
echo
echo "[setup] set   model = \"$LAYOUT\"   in configs/video_gen/wan22_i2v_a14b_distill4step.toml"
