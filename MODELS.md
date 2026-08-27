# Model weights — where to get them & where to put them

Weights are **not** in git. Set one Hugging Face–style root and lay every weight
out as `<org>/<repo>` underneath it:

```bash
export PUBLIC_MODELS_ROOT="$HOME/public_models"   # any writable dir
```

Everything below resolves under `$PUBLIC_MODELS_ROOT`. Codename checkpoints
(FLUX.2-klein, Qwen3.5, the distilled Wan2.2, WeDetect-Ref) come from each
vendor's official release and some are gated — download them from the vendor and
place them at the exact path shown; we do not mirror weights.

---

## 1. Perception — the faithful default path (Track A + production)

MemStrata's default crop backend is **WeDetect-Ref** (a `describe -> bbox`
grounding service), with **DINOv3** for identity/novelty. SAM3 is only a
fallback used when WeDetect-Ref is unreachable.

| Role | Where to get it | Local path |
|---|---|---|
| **WeDetect-Ref** (default grounder) | Vendor service — clone/serve separately, then `export MEMSTRATA_WEDETECT_URL=http://127.0.0.1:8710` (see `scripts/memstrata/servers/serve_wedetect.sh`). Not shipped (vendor-licensed). | served on `:8710` |
| **DINOv3** (identity / novelty / keyframes) | `huggingface-cli download` (below) | `$PUBLIC_MODELS_ROOT/facebook/dinov3-vitb16-pretrain-lvd1689m` |
| **SAM3** (fallback grounder only) | `huggingface-cli download` (below); needs the vendored `transformers>=5.9` bundle at `models/vendor/sam3_transformers59` on `PYTHONPATH` | `$PUBLIC_MODELS_ROOT/facebook/sam3` |

WeDetect-Ref is intentionally not copied into this repository. Obtain the
upstream GPL-licensed service separately, install its dependencies in its own
environment, and point the launcher at it:

```bash
git clone https://github.com/WeChatCV/WeDetect.git "$HOME/vendor/WeDetect"
export WEDETECT_REPO="$HOME/vendor/WeDetect"
export WEDETECT_PYTHON=/path/to/your/wedetect-env/bin/python
bash scripts/memstrata/servers/serve_wedetect.sh
export MEMSTRATA_WEDETECT_URL=http://127.0.0.1:8710
```

```bash
huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir "$PUBLIC_MODELS_ROOT/facebook/dinov3-vitb16-pretrain-lvd1689m"
# SAM3 fallback (optional; only used when WeDetect-Ref is down):
huggingface-cli download facebook/sam3 --local-dir "$PUBLIC_MODELS_ROOT/facebook/sam3"
```

> Crop-backend policy: `MEMSTRATA_ENABLE_SAM3=auto` (default) loads SAM3 only if
> WeDetect-Ref is down; `off` = WeDetect-only; `on` = always load SAM3 (the
> grounder still wins when up). The value actually used is reported as
> `crop_backend` in the adapter's `finalize()` output.

## 2. Named-entity path (`name_source=mllm`) — required for faithful reproduction

When binding visible prompt names, MemStrata routes them through its **own VLM**
(never string-matching gold). This is the setting used for the paper Track A
numbers (`MEMSTRATA_TRACKA_NAME_SOURCE=mllm`).

| Role | Local path |
|---|---|
| MemStrata VLM (entity decompose / intent) | `$PUBLIC_MODELS_ROOT/Qwen/Qwen3.5-9B` |

```bash
huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir "$PUBLIC_MODELS_ROOT/Qwen/Qwen3.5-9B"
```

## 3. Production generators (only if you drop `--backend recording`)

The closed-loop production run needs the real keyframe + video generators. The
config files under `configs/image_gen/` and `configs/video_gen/` reference these
exact paths:

| Role | Local path | Config |
|---|---|---|
| Keyframes (FLUX.2 [klein] 9B-KV FP8) | `$PUBLIC_MODELS_ROOT/black-forest-labs/FLUX.2-klein-9b-kv` (+ `.../FLUX.2-klein-9b-kv-fp8/flux-2-klein-9b-kv-fp8.safetensors`) | `configs/image_gen/flux.2-klein-9b-kv-fp8.toml` |
| Video (Wan2.2-I2V-A14B, LightX2V 4-step distill) | `$PUBLIC_MODELS_ROOT/Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step` (`high_noise_model/` + `low_noise_model/`) | `configs/video_gen/wan22_i2v_a14b_lightx2v_4step.toml` |
| Intent / decompose LLM | `$PUBLIC_MODELS_ROOT/Qwen/Qwen3.5-9B` (same as §2) | — |

Build the LightX2V-layout video dir from the two distilled safetensors + base
Wan2.2-I2V-A14B components with the helper (no source edits):

```bash
bash src/memstrata/steps/generate/backends/setup_lightx2v_weights.sh \
  <high_noise.safetensors> <low_noise.safetensors>
```

The Wan video backend runs LightX2V in its own interpreter. Point at it via the
config `python = ...` **or** `export MEMSTRATA_LIGHTX2V_PYTHON=/path/to/lightx2v-env/bin/python`
(see `src/memstrata/steps/generate/backends/README_wan22_lightx2v.md`).

---

The optional install check `bash scripts/memstrata/cpu_demo.sh` reads **none** of
the above (no Wan / FLUX / Qwen) and is not a substitute for a GPU production run.
