# Model layout (do not put weights in git)

Set one root, Hugging Face style `<org>/<repo>`:

```bash
export PUBLIC_MODELS_ROOT="$HOME/public_models"
```

Track A MemStrata adapter (SAM3 + DINOv3), same ids as the internal write path:

```bash
huggingface-cli download facebook/sam3 --local-dir "$PUBLIC_MODELS_ROOT/facebook/sam3"
huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir "$PUBLIC_MODELS_ROOT/facebook/dinov3-vits16-pretrain-lvd1689m"
```

Default production generator (only if you drop `--backend recording`):

| Role | Local dir |
|---|---|
| Keyframes | `$PUBLIC_MODELS_ROOT/black-forest-labs/FLUX.2-klein-9B-kv` (see `configs/image_gen/`) |
| Video | Wan2.2-I2V-A14B LightX2V 4-step, as in `configs/video_gen/wan22_i2v_a14b_lightx2v_4step.toml` |

The optional install check `bash scripts/memstrata/cpu_demo.sh` does not read this file (no Wan / FLUX / Qwen). It is not a substitute for a GPU production run.
