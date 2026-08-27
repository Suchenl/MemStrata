# Wan2.2-I2V-A14B (MoE) 4-step distilled backend — LightX2V

Higher-fidelity replacement for the Helios i2v backend. Wan2.2-I2V-A14B is a **dual-expert**
(high-noise + low-noise) 14B MoE diffusion model; the [LightX2V](https://github.com/ModelTC/LightX2V)
step-distilled release runs it in **4 steps** (2 high + 2 low), **CFG-free** (`guidance_scale=1.0`),
Euler scheduler, `sample_shift=5.0`. Same MemStrata contract as Helios: it animates the composed
FLUX keyframe into a chunk video; the external stratified memory supplies long-range identity.

## Files
- `wan22_lightx2v_backend.py` — adapter (provider `wan_lightx2v`), mirrors `HeliosBackend`.
- `wan22_lightx2v_server.py` — persistent file-queue server; keeps both 14B experts resident.
- `../../../../configs/video_gen/wan22_i2v_a14b_lightx2v_4step.toml` — config.
- `scripts/memstrata/servers/setup_lightx2v_weights.sh` — builds the LightX2V-layout model dir from the two weights (repo-relative from the MemStrata root).

## One-time setup (blocked on the distilled weights the user is downloading)

1. **Install LightX2V** into a Python env that already has FA2/FA3 (a dedicated
   LightX2V env is cleaner):
   ```bash
   git clone https://github.com/ModelTC/LightX2V && cd LightX2V
   pip install -e .          # + follow its README for the attention kernel (sage_attn2 / flash-attn)
   ```
   Then tell MemStrata which interpreter runs LightX2V, one of:
   - set `python = ".../envs/<env>/bin/python"` in the TOML, or
   - `export MEMSTRATA_LIGHTX2V_PYTHON=.../envs/<env>/bin/python` (overrides the TOML), or
   - leave the default `python = "python3"` and launch the loop *from* the LightX2V env
     (the bare command is resolved on `PATH`).

2. **Lay out the weights.** The two files
   (`wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors`,
   `wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors`) are single-file 14B
   experts. Symlink them + the base Wan2.2-I2V-A14B components into a LightX2V-layout dir:
   ```bash
   bash scripts/memstrata/servers/setup_lightx2v_weights.sh <HIGH.safetensors> <LOW.safetensors>
   # -> /data/.../Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step/
   #      high_noise_model/<HIGH>.safetensors + config.json
   #      low_noise_model/<LOW>.safetensors + config.json
   #      google/  models_t5_*.pth  Wan2.1_VAE.pth  configuration.json   (symlinked from base)
   ```
   Set `model = "<that dir>"` in the TOML (default already points there).

3. **Smoke test** one i2v call before wiring into the loop:
   ```bash
   <env-python> wan22_lightx2v_server.py --model_path <layout_dir> --server_dir /tmp/wx2v_smoke &
   # then submit a job with first_frame_path=<a keyframe png>, prompt=..., save_file=/tmp/out.mp4
   ```

## Use in the closed loop
```bash
python experiments/e2e/memstrata_helios_loop/run.py \
  --backend wan22_i2v_a14b_lightx2v_4step --flux --force-recompose ...
```
i2v (`recompose_*`) is the only supported mode (the loop runs `--force-recompose`, a fresh
keyframe per beat, which also sidesteps AR drift). `continue_ar`/v2v continuation is a TODO
(LightX2V supports `--video_path` v2v; wire it only if AR chaining is revisited).

## Engine interface (for reference)
```python
from lightx2v import LightX2VPipeline
pipe = LightX2VPipeline(model_path=<layout_dir>, model_cls="wan2.2_moe", task="i2v")
pipe.create_generator(infer_steps=4, height=720, width=1280, num_frames=81,
                      guidance_scale=[1.0, 1.0], sample_shift=5.0, attn_mode="sage_attn2")
pipe.generate(seed=2026, image_path=<keyframe>, prompt=..., negative_prompt=..., save_result_path=<out.mp4>)
```
