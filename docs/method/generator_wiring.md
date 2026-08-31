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

TOML 配置文件：`configs/video_gen/*.toml`

## 图像关键帧后端（keyframe-first 第一阶段，2026-07-22 vendored）

主生产管线是 **keyframe-first 两阶段解耦**：图像模型先合成 Layout Anchor 关键帧，视频后端
（Wan2.2-I2V-A14B + SVI / Morphic LoRA）再展开成视频。图像后端已直接随本仓库提供，
位于 `src/memstrata/steps/generate/image_backends/`，不依赖 Montage：

| 模块（`steps/generate/image_backends/`） | 覆盖 |
|---|---|
| `flux_klein_backend.py` | FLUX.2 Klein 9B-KV：T2I / I2I(reference-conditioned) / 关键帧；含持久服务 |
| `flux_persistent_server.py` | FLUX 常驻服务（文件队列 IPC，复用 `backends/vace_job_queue`） |
| `base.py` | `ImageGenerationModel` 协议 + `apply_photographic_grain` / `preprocess_de_ai_prompt` |
| `factory.py` | `build_image_backend(name, output_dir=...)` / `list_image_backend_names()` |

TOML 配置：`configs/image_gen/*.toml`（当前 `flux.2-klein-9b-kv-fp8`）。
`python` 应指向包含 FLUX 依赖的解释器；视频侧可以使用另一个包含 Wan
依赖的解释器，具体通过 `MEMSTRATA_FLUX_PYTHON` 与
`MEMSTRATA_LIGHTX2V_PYTHON` 配置。
`build_image_backend` 消费 `MediaTaskType.{REFERENCE_IMAGE,KEYFRAME,IMAGE_EDIT}` 任务，产出
`media_type="image"` 的 `GenerationArtifact`。跨 env/GPU 编排（FLUX 出关键帧 → Wan 展开）属
Phase 3，尚未落地。

### 关键帧合成：默认原生 FLUX 多图组合

`KeyframeComposer`（`steps/keyframe.py`）把该段选中的记忆 crop 合成一张场景关键帧。
**默认走原生 FLUX 多图组合**（`MEMSTRATA_KEYFRAME_MODE=native`）：直接把多张 crop 作为
FLUX.2 klein 的多张参考图喂入，由 FLUX 自行组合，不再用 Qwen 规划色块布局 / 拼贴 collage
——因此关键帧阶段**不需要 Qwen MLLM 端点**（见 `production/services.py`）。冷启动 / 全首次
出现（该段无 crop）时退化为 FLUX 文生图 bootstrap。

旧的 Qwen-canvas 路径（R3 色块布局 → R4 crop→region 分配 → collage → FLUX I2I 融合）保留为
`MEMSTRATA_KEYFRAME_MODE=collage` 可选项，仅该模式需要 Qwen MLLM 端点。

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
PYTHONPATH=src python3 -m memstrata.production.run --backend recording --decompose none --segments 2
PYTHONPATH=src python3 -m memstrata.production.run --backend oracle --decompose none --segments 2
# 真模型闭环（需 GPU + 权重；默认 --decompose crop_server，记忆从生成视频增长）：
PYTHONPATH=src python3 -m memstrata.production.run --backend wan_t2v --crop-acq-device 7
PYTHONPATH=src python3 -m memstrata.production.run --backend helios_distilled_i2v --flux --force-recompose
# 或直接用 bash 入口（输出落 production/outputs/<story>/<system>/<时间戳>/）：
bash scripts/memstrata/run_production.sh production/screenplay/products/en/0000_detective_mystery.json helios_distilled_i2v memstrata
```

## 协议

`MediaTaskGenerator` 把 Composed Context 物化为：

- `controls['composed_references']`
- `controls['continuation']`（transition=continue 时）

再调用 `backend.generate(MediaGenerationTask) -> GenerationArtifact`。
