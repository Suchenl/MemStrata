#!/usr/bin/env bash
# Build the LightX2V-layout model dir for Wan2.2-I2V-A14B 4-step distilled inference.
#
# The two distilled experts are single-file safetensors (~28GB BF16 each). LightX2V's
# wan2.2_moe loader wants a model dir with high_noise_model/ + low_noise_model/ (each holding
# ONE distilled safetensors + its config.json) plus the shared base components (t5 tokenizer,
# T5 encoder, VAE, top-level config) which we symlink from the original Wan2.2-I2V-A14B.
#
# Usage:
#   bash setup_lightx2v_weights.sh <high_noise.safetensors> <low_noise.safetensors> [LAYOUT_DIR]
#
# Idempotent: re-running relinks. Symlinks only (no copies) so it costs no extra disk.
set -euo pipefail

HIGH="${1:?path to wan2.2_i2v_A14b_high_noise_lightx2v_4step_*.safetensors}"
LOW="${2:?path to wan2.2_i2v_A14b_low_noise_lightx2v_4step_*.safetensors}"
BASE="${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B"
LAYOUT="${3:-${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step}"

for f in "$HIGH" "$LOW"; do
  [ -f "$f" ] || { echo "ERROR: weight file not found: $f" >&2; exit 1; }
done
[ -d "$BASE" ] || { echo "ERROR: base Wan2.2-I2V-A14B not found: $BASE" >&2; exit 1; }

echo "[setup] layout dir: $LAYOUT"
mkdir -p "$LAYOUT/high_noise_model" "$LAYOUT/low_noise_model"

# Shared base components (tokenizer/T5/VAE/top-level config).
for c in google Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth configuration.json README.md assets examples; do
  [ -e "$BASE/$c" ] && ln -sfn "$BASE/$c" "$LAYOUT/$c"
done

# Per-expert config.json from base (the distilled weights keep the same architecture config).
ln -sfn "$BASE/high_noise_model/config.json" "$LAYOUT/high_noise_model/config.json"
ln -sfn "$BASE/low_noise_model/config.json"  "$LAYOUT/low_noise_model/config.json"

# The distilled single-file transformer weights.
ln -sfn "$(readlink -f "$HIGH")" "$LAYOUT/high_noise_model/$(basename "$HIGH")"
ln -sfn "$(readlink -f "$LOW")"  "$LAYOUT/low_noise_model/$(basename "$LOW")"

echo "[setup] done. Contents:"
ls -l "$LAYOUT" "$LAYOUT/high_noise_model" "$LAYOUT/low_noise_model"
echo
echo "[setup] set   model = \"$LAYOUT\"   in configs/video_gen/wan22_i2v_a14b_lightx2v_4step.toml"
