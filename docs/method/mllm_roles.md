# MLLM 角色目录（生产管线中的 MLLM 分工）

Date: 2026-07-23 · 单一事实源代码：`src/memstrata/mllm/roles.py`（纯 spec，无 HTTP/无副作用）

本文件枚举 MemStrata 生产管线里 MLLM 要扮演的所有角色/任务，以及每个角色使用的模型、
采样参数（temperature / top-p / thinking）、输入输出与 JSON schema 约束。这是**设计规格**，
用于后续讨论"哪些实现、哪些不实现"；`roles.py` 与本文件必须保持一致。

## 词汇对齐

论文四步的**官方命名**是：
`Intent Interpretation → Visual Generation → Evidence Acquisition → Stratified Update`。
代码里的模块名（`intent / compose / generate / decompose / curate`）是同一套环的内部工程词汇：

```
Intent  ->  Compose  ->  Generate  ->  Decompose  ->  Curate      (+ Offline)
```

- **Compose** = 记忆**读**路径（组装 Composed Context）。
- **Decompose** = 记忆**写**分析（把生成 chunk 拆回候选实体 crop + 属性），喂给 **Curate**。

> 说明：`Compose/Decompose` 只作**代码内部**词汇；论文正文用的是 read/write 记忆隐喻 + 上述四步
> 官方名，两者不在论文里混用。

## 铁律：默认读路径零模型调用

`memstrata.steps.intent` 的 **FAST 模式**用 name/alias + description 匹配 + identifier
dereference，**不调用任何模型**（`model_calls=0`）。MLLM 只出现在 **Compose 的推理读路径**
与 **写路径（Decompose / Curate）**。因此每个角色都标注了 `hot_path`（是否每 chunk 触发）。

## 采样参数原则

| 类别 | 预设 | temp / top-p | thinking | 说明 |
|---|---|---|---|---|
| 决策 / 规划（模糊） | `_DECIDE` | 0.0 / 1.0 | 开 | 需推理的选择/映射/去重/状态 |
| 分类（简单结构化） | `_CLASSIFY` | 0.0 / 1.0 | 关 | 视觉属性分类、枚举判定，省延迟 |
| 文案撰写 | `_AUTHOR` | 0.2 / 0.9 | 关 | caption / 生成 prompt，纯文本 |
| 离线元优化 | `_META` | 0.3 / 0.95 | 开 | prompt 优化，容许发散 |

- 决策/分类/映射类一律 `temperature=0.0` + `response_format=json_schema`，保证可复现、可对齐评测。
- 默认模型 **Qwen3.5-9B-Instruct**（每 chunk 热调用）；离线/高难推理（R12，或 R4 疑难）可切
  **Qwen3.5-27B-Instruct**。模型绑定 per-role 可覆盖，不全局写死。
- 视觉角色（R4/R7/R8/R9/R10/R13）需要一个具视觉能力的 Qwen（VL）后端。

## 角色清单（R1–R13）

状态：✅ implemented · 🟡 partial · ⬜ planned

### Intent
- **R1 Intent Parser / Director** ✅（hot）· 文本 · `_DECIDE`
  下一段提示词 → 结构化意图（点名实体 / continue·cut / scene-return / 所需能力）。
  代码：`steps/intent.py::IntentInterpreter`（SLOW 经 `MllmIntentResolver`）；FAST 为 model-free。

### Compose（记忆读 + 组场，MLLM 密集）
- **R2 Asset Retriever/Selector** ✅（hot）· 文本 · `_DECIDE`
  意图 + bank 清单 → 最小充分 asset 集（Sufficiency vs Parsimony）。
  代码：`llm/planner.py::MllmPlanner.select_assets`。
- **R3 Layout / Spatial Planner** ⬜（hot）· 文本 · `_DECIDE`
  场景 → 归一化 bbox 布局 `[{label, box_2d[ymin,xmin,ymax,xmax], shape}]`，渲染色块给 FLUX。
  待 vendor：`src/montage/skills/layout_anchor_processing`（`LayoutPlanner`）。
- **R4 Crop → Region Assigner** ⬜（hot）· **视觉** · `_DECIDE`
  把检索到的**真实 crop** 映射到 R3 的哪个 box + 选哪个 angle/state 变体。
  **该放置决策必须由 MLLM 在管线内做，不能由人工写死。** 喂给 Crop2Image 拼图 → FLUX I2I 融合。
- **R5 Generation-Prompt Composer** 🟡（hot）· 文本 · `_AUTHOR`
  意图 + 所选 crop → FLUX/视频正向 prompt，**与 crop 属性一致**（不许乱指定颜色/服装覆盖身份）。
  现状：`lib/prompt_standardizer.py` 确定性；MLLM 重写可选。

### Generate
- **R6 View / Angle Requester** ⬜（非 hot）· 文本 · `_CLASSIFY`
  计划需要新视角时，决定渲染哪个 viewpoint 的 reference_image。

### Decompose（分析生成输出，写路径）
- **R7 Entity Detector / Namer** 🟡（hot）· **视觉** · `_DECIDE`
  新 chunk 检测实体、切 crop、给 name/kind。Track A gold ObservationPacket 会绕过它。
- **R8 Crop Attribute / Angle-State Classifier** ✅（hot）· **视觉** · `_CLASSIFY`
  crop → spatial_angle/state_angle(+shot size/光照/遮挡)。
  代码：`llm/angle_classifier.py` + `llm/crop_attributes.py`。

### Curate（记忆写路径）
- **R9 Ingest / Dedup Judge** ✅（hot）· **视觉** · `_DECIDE`
  新观测 vs 候选 → merge 或新建 + caption + reasoning。
  代码：`llm/planner.py::MllmPlanner.make_ingest_decision`。
- **R10 Admission Gate** 🟡（hot）· 视觉 · `_CLASSIFY`
  crop 质量/新颖度是否够格入库。代码：`lib/crop_quality.py` + who-admission 门（tests）。
- **R11 State-Update / Deprecation Manager** ⬜（hot）· 文本 · `_DECIDE`
  更新实体状态（changed/damaged）、标记废弃证据、维护 avoidance（防复用废弃表征）。

### Offline / 跨切面
- **R12 Prompt Optimizer（元优化）** ✅（非 hot）· 文本 · `_META` · **Qwen3.5-27B**
  读评测历史 → 重写 planner 提示词。代码：`llm/planner.py::PromptOptimizer`。
- **R13 In-loop Quality / Consistency Judge** ⬜（非 hot）· 视觉 · `_DECIDE`
  生成 chunk vs 参考的定性一致性判定（仅 hard case；headline 指标由 Track A 确定性 scorer 负责）。

## 现状小结

- ✅ 已有：R1, R2, R8, R9, R12
- 🟡 部分：R5, R7, R10
- ⬜ 待建：R3, R4, R6, R11, R13
- hot path（每 chunk）：R1–R5, R7–R11

## 后续（待讨论后再落地）

1. 决定实现优先级（建议先 R3+R4：把 layout_anchor vendored 进来 + crop→region 由 MLLM 决定，
   打通"色块布局 + 记忆 crop 贴入 → FLUX I2I 融合"这条 Crop2Image 主链路）。
2. 起一个 Qwen3.5-9B 常驻服务（OpenAI 兼容，`MEMSTRATA_CONTEXT_JUDGER_BASE_URL`），
   per-role 绑定模型/采样参数/schema；视觉角色接 VL 后端。
3. 为每个要实现的角色补：prompt 模板 + JSON schema + 单测（延续 `tests/` 现有风格）。
