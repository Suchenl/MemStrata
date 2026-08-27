# Generator wiring notes

Date: 2026-07-17

## 已纳入的具体后端

代码在 `steps/generate/backends/`（自包含，无外部包 import）：

| 模块 | 覆盖 |
|---|---|
| `diffusers_backend.py` | wan_vace / wan_t2v / ltx / cogvideox / hunyuan / mochi |
| `native_vace_backend.py` | provider=`native_vace`（含 persistent server / job queue） |
| `ltx23_backend.py` | provider=`ltx23` |
| `longcat_backend.py` | provider=`longcat` |
| `magref_backend.py` | provider=`magref` |
| `multishotmaster_backend.py` | provider=`multishotmaster` |
| `oracle.py` / `recording.py` | 轻量本地后端 |
| `factory.py` | `build_video_backend(name, output_dir=...)` |

TOML 配置镜像：`methods/MemStrata/configs/video_gen/*.toml`

## 图像关键帧后端（keyframe-first 第一阶段，2026-07-22 vendored）

主生产管线是 **keyframe-first 两阶段解耦**：图像模型先合成 Layout Anchor 关键帧，视频后端
（Wan2.2-I2V-A14B + SVI / Morphic LoRA）再展开成视频。图像这一环 vendored 自 Montage
`src/montage/models/image_generation/`（import 重写、零 `montage` 依赖）：

| 模块（`steps/generate/image_backends/`） | 覆盖 |
|---|---|
| `flux_klein_backend.py` | FLUX.2 Klein 9B-KV：T2I / I2I(reference-conditioned) / 关键帧；含持久服务 |
| `flux_persistent_server.py` | FLUX 常驻服务（文件队列 IPC，复用 `backends/vace_job_queue`） |
| `base.py` | `ImageGenerationModel` 协议 + `apply_photographic_grain` / `preprocess_de_ai_prompt` |
| `factory.py` | `build_image_backend(name, output_dir=...)` / `list_image_backend_names()` |

TOML 配置：`methods/MemStrata/configs/image_gen/*.toml`（当前 `flux.2-klein-9b-kv-fp8`，
`family=flux`；`python` 指向 `envs/MultiShotMaster`，与视频侧 `envs/vace` 分属两个环境 / 两张卡）。
`build_image_backend` 消费 `MediaTaskType.{REFERENCE_IMAGE,KEYFRAME,IMAGE_EDIT}` 任务，产出
`media_type="image"` 的 `GenerationArtifact`。跨 env/GPU 编排（FLUX 出关键帧 → Wan 展开）属
Phase 3，尚未落地。

## 调用方式

```python
from memstrata import MemStrata, MediaTaskGenerator, build_video_backend

backend = build_video_backend("wan_t2v", output_dir=run_dir / "media")
gen = MediaTaskGenerator(backend, bank=bank, model_name="wan_t2v")
mem = MemStrata(bank=bank, generator=gen, run_dir=run_dir)
mem.run_chunk(prompt, chunk_id=0, generation_controls={...})
```

CLI：

运行逻辑在 `src/memstrata/production/run.py`（模块 `memstrata.production.run`），`scripts/memstrata/run_production.sh` 只是薄 bash 入口。管线依赖的服务由 `src/memstrata/production/services.py` 声明式管理：真实闭环所需的 Qwen MLLM 端点会被 **reuse-first** 自动拉起（已在跑则复用、绝不 kill 别人；`--mllm-gpu/--mllm-port` 可配，`--no-autoserve` 关闭）；FLUX/Helios/Wan/crop-acq server 各自在 backend 内自启，无需外部拉。

```bash
cd MemStrata
PYTHONPATH=src python3 -m memstrata.production.run --list-backends
# 无 GPU 后端冒烟（--decompose none 跳过 S5 crop server）：
PYTHONPATH=src python3 -m memstrata.production.run --backend recording --decompose none --chunks 2
PYTHONPATH=src python3 -m memstrata.production.run --backend oracle --decompose none --chunks 2
# 真模型闭环（需 GPU + 权重；默认 --decompose crop_server，记忆从生成视频增长）：
PYTHONPATH=src python3 -m memstrata.production.run --backend wan_t2v --crop-acq-device 7
PYTHONPATH=src python3 -m memstrata.production.run --backend helios_distilled_i2v --flux --force-recompose
# 或直接用 bash 入口（输出落 production/outputs/<story>/<system>/<时间戳>/）：
bash scripts/memstrata/run_production.sh data/Screenplay/products/cn/0000_detective_mystery.json helios_distilled_i2v memstrata
```

## 协议

`MediaTaskGenerator` 把 Composed Context 物化为：

- `controls['composed_references']`
- `controls['continuation']`（transition=continue 时）

再调用 `backend.generate(MediaGenerationTask) -> GenerationArtifact`。
