# MemStrata · 模型权重 / 环境 / 运行要求（唯一权威清单）

> 目的：把 MemStrata（SUT `memstrata` + 基准 `vmem_bench` + baselines）用到的**所有**
> 权重位置、conda 环境、环境变量、运行要求集中到一处，便于管理与**独立发布**。
> 新增/换权重/换环境时**只改本文件**，其它文档引用此处，不要另立清单。
>
> 发布红线（见 [`../../AGENTS.md`](../../AGENTS.md) 规则 1）：`src/memstrata/` 与
> `src/vmem_bench/` **不得 import `benchmarks/VMem-Bench/` 之外的代码**。第三方库与
> **按路径加载的权重**不算耦合。本文件末尾维护「发布自包含检查单」，列出当前尚存的越界依赖。

## 0. 权重根目录（PUBLIC_MODELS_ROOT）

代码统一通过环境变量 `PUBLIC_MODELS_ROOT` 解析本地权重（见
`src/memstrata/encoders/base.py`、`scripts/**/serve_*.sh`、crop server env）。

| 根 | 路径 | 用途 |
|---|---|---|
| 默认根（代码 default） | `${PUBLIC_MODELS_ROOT}` | 编码器 / MLLM / 视频权重的默认解析根 |
| 用户扩展根 | `${PUBLIC_MODELS_ROOT}` | dinov3 全档 + **音频模型**都在这里 |

> 两个根都存在同名 `facebook/dinov3-vitb16-pretrain-lvd1689m`。跑校准/闭环时用哪个根，就
> `export PUBLIC_MODELS_ROOT=<root>`；音频相关必须用扩展根（默认根下没有 `Audio/`）。

## 1. 权重清单（按角色）

路径 = `<PUBLIC_MODELS_ROOT>/<相对路径>`，除非另注绝对路径。

### 1.1 视觉编码器（相似度门 / 检索 / 打分）
| 角色 | provider | 相对路径 | 说明 |
|---|---|---|---|
| 通用图像 embed（写路径 gate②/去冗/自审、检索关键帧多样性） | `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m`（另有 `-vitl16-`/`-vits16-`，仅在扩展根） | 写路径阈值**编码器相对**，换它必须重跑校准（见 §4） |
| 文本↔帧跨模态 | `siglip2` | `google/siglip2-base-patch16-512`（打分侧默认 `-224`，`MEMSTRATA_SIGLIP2_MODEL`） | text→frame 检索 / 视觉打分 |
| 文本↔文本 | `qwen3_embedding` | `Qwen/Qwen3-Embedding-4B` | 可走本地权重，或设 `MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT` 走 server |
| 人脸（可选路由） | `insightface` | 由 `MEMSTRATA_FACE_EMBEDDER_WEIGHTS` / `weights=` 指定 | 仅在开启 face 路由时 |

### 1.2 感知（crop 获取 / grounding）
| 组件 | 权重/依赖 | 位置 | 说明 |
|---|---|---|---|
| SAM3 概念分割 + DINOv3 + GroundingDINO | vendored `sam3_transformers59`（transformers 5.9，cp311 .so） | `<montage_root>/models/vendor/sam3_transformers59` | **⚠️ 当前在 MemStrata 之外**（发布需内置，见 §6）。crop server 子进程用 py3.11 env |
| GroundingDINO 权重 | 随 crop server 解析（缺权重则优雅降级为纯 kind 检索） | `PUBLIC_MODELS_ROOT` 下 | bbox-only 兜底标 `bbox_high_recall_no_mask` |

### 1.3 MLLM（planner / 角度属性 / 打分判分）
由 `scripts/memstrata/servers/serve_qwen.sh` 用 vLLM 起 OpenAI 兼容服务：
| 角色 | 模型 | 相对路径 | served name |
|---|---|---|---|
| 文本 R1/R2/R3/R5（也可全角色） | Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `Qwen3.5-9B-Instruct` |
| 视觉 R4/R7/R8（角度属性 / 视觉判分） | Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen3-VL-8B-Instruct` |
| 判分（Stage2 视觉覆盖） | qwen3-vl-32b（外部判分 API） | 判分服务 | **调用必须带 `/chat/completions` 全路径**（曾 404） |

### 1.4 视频生成 / 关键帧
| 组件 | 权重 | 位置 | 说明 |
|---|---|---|---|
| 视频 i2v（默认后端） | Wan2.2-I2V-A14B lightx2v 4-step distill | `${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step` | 由 `setup_lightx2v_weights.sh` 从两份 distilled safetensors + 基座组装；配置见 `configs/video_gen/wan22_i2v_a14b_lightx2v_4step.toml` |
| 关键帧融合 | FLUX.2 klein（`flux.2-klein-9b-kv-fp8`） | 见对应 image backend config | R3/R4 collage → FLUX I2I，可 `--no-flux` 关闭 |

### 1.5 音频（showcase / audio MVP，权重在扩展根 `Audio/` 下）
根：`${PUBLIC_MODELS_ROOT}/Audio`
| 用途 | 模型 | 子路径 |
|---|---|---|
| 对白 TTS | CosyVoice2-0.5B | `FunAudioLLM/CosyVoice2-0.5B` |
| 对白 TTS（备选/新版） | Fun-CosyVoice3-0.5B-2512 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| CosyVoice 文本前端 | CosyVoice-ttsfrd | `FunAudioLLM/CosyVoice-ttsfrd` |
| 情感/双语 TTS（备选） | IndexTTS-2 | `IndexTeam/IndexTTS-2` |
| 背景音乐 BGM | ACE-Step v1.5 | `ACE-Step/acestep-v15-sft`、`ACE-Step/acestep-v15-xl-turbo-diffusers` |
| 音效 foley | Stable Audio Open | `stabilityai/stable-audio-open-1.0`、`stabilityai/stable-audio-open-small` |

> 方案设计见 [`../showcase/audio_pipeline_PLAN.md`](../showcase/audio_pipeline_PLAN.md)（CosyVoice2 对白 /
> ACE-Step BGM / Stable Audio Open foley + ducking + 对齐）。权重已下载，MVP 待接线。

## 2. Conda 环境与归属

根：``

| 环境 | 用途 | 关键约束 |
|---|---|---|
| `vace` | **主环境**：pytest、`montage`/`memstrata` 主流程、Wan2.2 lightx2v（editable 装入，torch 2.5.1+cu124 + flash_attn2） | 系统 python **无 pytest**，测试必须用它：`python3` |
| `helios` | **crop-acquisition server 子进程**（py3.11 + torch 2.10，匹配 vendored transformers 5.9 的 cp311 .so） | crop_client 默认 `python=.../envs/helios/bin/python`；client 端可用任意 env |
| `vllm` | MLLM 服务（`serve_qwen.sh` 默认 `VLLM_ENV`） | vLLM serve Qwen3.5-9B / Qwen3-VL-8B |
| `qwen` / `slotmem` / `diffsynth*` / `flux2` 等 | 特定 baseline / 后端 | 按需，见各自 serve 脚本 |

## 3. 环境变量（MEMSTRATA_*）

| 变量 | 作用 | 默认 |
|---|---|---|
| `PUBLIC_MODELS_ROOT` | 本地权重解析根 | `${PUBLIC_MODELS_ROOT}` |
| `MEMSTRATA_GENERAL_EMBEDDER_PROVIDER` | 通用图像 embed provider（`hash`\|`dinov3`\|...） | `hash`（非语义，gate②/自审关闭） |
| `MEMSTRATA_FACE_EMBEDDER_PROVIDER` / `MEMSTRATA_PLACE_EMBEDDER_PROVIDER` | 人脸 / 场所路由 provider | 空（不路由） |
| `MEMSTRATA_SIGLIP2_WEIGHTS` / `MEMSTRATA_SIGLIP2_MODEL` | siglip2 权重/模型名覆盖 | 相对默认 |
| `MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT` / `_MODEL` / `_API_KEY` / `_WEIGHTS` | qwen3 文本 embed 走 server 或本地 | 空→本地权重 |
| `MEMSTRATA_ANGLE_CLASSIFIER` | 角度/属性分类器 `null`\|`heuristic`\|`vlm` | `null`（每 rep 角度未知，**不可报分层结果**） |
| `MEMSTRATA_CONTEXT_JUDGER_BASE_URL` / `MEMSTRATA_CROP_ATTR_BASE_URL` / `..._MODEL` | planner / 角度属性 MLLM 端点与模型 | 见 `mllm/planner.py`、`mllm/crop_attributes.py` |
| `MEMSTRATA_SCORING_EMBEDDER_WEIGHTS` / `MEMSTRATA_WEIGHTS_ROOT` | 打分侧编码器权重 | 见 `vmem_bench/scoring/embedder.py` |
| `MEMSTRATA_ALLOW_HF_DOWNLOAD` | =1 允许 transformers 联网解析（默认离线） | 未设（离线；crop server 设 `HF_HUB_OFFLINE=1`） |
| `MEMSTRATA_RETRIEVAL_VARIANT` / `MEMSTRATA_RETRIEVAL_TOPK` | 检索 baseline 家族变体 / top-k | 见 `retrieval_family_DESIGN.md` |

## 4. 写路径阈值校准（换编码器必跑）

`MemoryPolicy.production()` 里的 per-type 阈值（B_τ / γ_τ / β_τ / cohesion floor）是**编码器相对**的
起点值，不是测量值。换编码器（如 hash→dinov3）后必须重跑：

```bash
PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} \
python3 \
  experiments/methods/MemStrata/20260725_memstrata_write_path_calibration/calibrate_write_path.py \
  --labels experiments/methods/MemStrata/20260722_memstrata_cohesion_calibration/labels_lsmdc.json \
  --encoder dinov3
```

- 标注对照集：`experiments/methods/MemStrata/20260722_memstrata_cohesion_calibration/labels_lsmdc.json`
- dinov3 缓存嵌入已存在于 20260722 目录（`embeddings_dinov3.json`，可复用免 GPU）
- 无 GPU 管线自检：`--self-test`
- 产出 `calibration_result.json` 的 `suggested_policy` 回填 `MemoryPolicy.production()`

## 5. 运行要求（GPU / 远程）

- GPU 优先级：大显存卡优先；**只在你自己分配到的节点上调度**。
- 远程作业必须 `setsid`（或等价）防止 SSH/tmux 断连。
- 判分 API 必须带 `/chat/completions` 全路径。

## 6. 发布自包含检查单（release blockers）

发布前需消除以下「MemStrata 之外」的依赖（`src/memstrata/` 不得 import 仓库其它路径）：

- [ ] `models/vendor/sam3_transformers59`（crop server 依赖）在 `benchmarks/VMem-Bench/` 之外——需内置或作为可选外部依赖显式声明。
- [ ] `src/memstrata/skills/embedding_deduplication/`（tests + README）`import montage.skills.embedding_deduplication`——需 vendored 化或删除。
- [ ] `src/memstrata/skills/focus_segmentation/`（tests + README）`import montage.skills.focus_segmentation`——同上。
- [ ] `src/memstrata/skills/layout_anchor_processing/` 标注为 "vendored from montage"——确认已真正 vendored、无运行期跨引用。
- [ ] 视频 lightx2v 权重目录为待组装 TODO：`.../Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step`（用户下载中）。

> 校验命令（应为空）：在 `src/memstrata/`、`src/vmem_bench/` 下搜 `import montage` /
> `from montage` / 指向仓库其它路径的产出写入。
