# MemStrata SUT 设计（对齐论文 §3 四步环路）

> **最高纲领见 [`philosophy.md`](philosophy.md)**（六条库质量公理 + WHO-before-WHERE
> 准入原则）。本文件是实现细节；与纲领冲突时以纲领为准。
>
> 状态：2026-07-16 按论文重写并整理包结构；同日落地 **视觉地层 `[image + angle]`**
> （spatial / temporal / state）。2026-07-17 补齐 **生产闭环**：crop → VLM/启发式
> 角度分类 → `AssetRepresentation` → 角度多样性门控。`vmem_bench` **评测协议冻结**——
> 本包可消费可选 angle 字段，但 **不要求** Bench 新增 angle 标注轴或新指标。对接仍是纯
> dict 契约（`PromptPacket` / `ObservationPacket` / `ComposedContextRecord`）。

## 0. 硬约束

1. `memstrata` **零 import `vmem_bench`**。
2. Asset 类型仅 **character / prop / location**。
3. 身份由 **命名信号**锚定；embedding 只做 representation 非冗余门控。
4. Step 1 产出 \(q_n\) 后，组 \(\mathcal{C}_n\) 是 **字典解引用**（读路径默认 name match，无 VLM /
   无全库相似检索）；可选 `preferred_spatial` / `preferred_state` 在已有 reps 上做 angle 优先。
5. Intent resolver（MLLM）仅名字缺失时的可选降级，**禁止返回全部资产**；主评测关。
6. 测试在 `src/memstrata/tests/`，assert-based，无 GPU / 真实权重。

## 1. 包布局

| 路径 | 职责 |
|---|---|
| `bank/` | Asset bank \(\mathcal{M}_n\)；`AssetRepresentation` 含 spatial/state/temporal |
| `steps/intent.py` | Step 1：\(g_n\to q_n+\tilde{p}_n\)（refs 可带 preferred angles） |
| `steps/compose.py` | Step 1 尾：model-free compose；angle 优先再 recency |
| `steps/generate/` | Step 2：`MediaGenerationTask` → backend |
| `steps/decompose.py` | Step 3：点名实体 + 类型路由编码；UNKNOWN 时调 angle classifier |
| `steps/curate.py` | Step 4：命名锚定 / 角度多样性 / deprecate；ingest 写入 angles |
| `llm/crop_attributes.py` | 生产闭环：crop → 全量属性包（spatial/state/shot_size/lighting/occlusion） |
| `llm/angle_classifier.py` | 兼容层：从属性包投影 spatial/state（`vlm` / `heuristic` / `null`） |
| `pipeline.py` | `MemStrata` 四步编排（含 run_ledger） |
| `adapters/bench.py` | Track A：`handle_prompt` / `handle_observation` |
| `encoders/` | face / place / general SSL |
| `llm/` | Intent 用的 MLLM 后端 |
| `lib/` | paths / weights / dedup / media |
| `extras/` | 非热路径（如 shot boundary） |

生成器接线细节见 [`generator_wiring.md`](generator_wiring.md)。

## 1.1 视觉地层（方法能力，非 Bench 新轴）

每个视觉证据 \(r\in\mathcal{R}_j\) 是分层记录：

| 字段 | 取值 | 用途 |
|---|---|---|
| `spatial_angle` | `front` / `side` / `back` / `top` / `unknown` | 跨视角召回 |
| `state_angle` | `default` / `changed` / `damaged` / `unknown` | 跨状态召回 |
| `temporal_tag` + `origin_chunk_id` | 时间标签 | 超长回忆：中间 filler chunk 不抹 id |

Compose 选择顺序：显式 `representation_id` → aspect/function 过滤 → **preferred spatial/state 命中** → 最新 chunk。缺 angle 时行为与冻结前一致（latest-first）。

### 生产闭环（2026-07-17）

```
crop → AngleClassifier → Observation(spatial, state, temporal)
     → AssetCurator → AssetRepresentation [image + angle]
```

| 开关 | 行为 |
|---|---|
| `MEMSTRATA_ANGLE_CLASSIFIER=null`（默认） | 不调模型；角度保持 unknown（离线/CI） |
| `=heuristic` | 从文件名 stem 解析（测试用：`hero_front_default.jpg`） |
| `=vlm` | 多模态 OpenAI 兼容 API；读图打标 |

- **生产路径**：`RoleAwareDecomposer.decompose` 与 `AssetCurator.ingest_observation` /
  `curate_observations` 在角度为 `unknown` 时分类；显式角度不被覆盖。
- **Track A**：`ingest_packet` **不调** classifier；packet 字段权威。
- **WHO 准入门控（先于 WHERE，`philosophy.md` §2；`AssetCurator._apply_rep_selection`）**：
  一条证据要先证明"是本实体且看得清"，才有资格参与角度多样性竞争。三道互补闸：
  - **① 确定性暗/低信息门禁**（`lib/crop_quality.is_dark_low_information`，默认开）：蒙版实体像素
    亮度 `mean<26 且 std<16` → 直接拒（不占库容）；不可读图当作可评估失败 → 放行。
  - **③ identity_visible**（`llm/crop_attributes` 属性包新增布尔字段，随原有 VLM 调用一次返回）：
    后脑勺/背影/严重模糊/遮挡看不出身份 → **不拒、降级为非锚**（剥离 `identity_anchor` reference
    aspect、记 `identity_anchor_eligible=false`），仍作 `spatial_angle=back` 等跨视角多样性保留。
  - **② embedding 内聚度**（`lib/dedup.similarity_to_set` / `medoid_cohesion`，`cohesion_floor` 默认
    0=关）：仅在身份可见证据之间、且已有 ≥`cohesion_min_refs` 条锚时启用；新证据与本 asset 可见簇
    余弦 < floor → 判为混入拒收。离线 HashEmbedding 无语义故默认关，生产用真实 encoder 显式开并校准。
- **库级 cohesion 自审（回溯式，`philosophy.md` §2.1；`AssetCurator.audit_cohesion`）**：
  入库闸守不住"首帧即内鬼 / ②启用前的历史污染"。周期性对每个 asset 复查，到"多数身份"参考基准
  相似度 < `selfaudit_cohesion_floor`（默认回退 `cohesion_floor`）的可见 rep 判为疑似异身份，
  **打 `deprecated_by="cohesion_selfaudit:*"` 隔离留痕**（不删、可回退、退出 compose）；`isolate=False`
  只出报告。floor≤0 空转。参考基准由 `selfaudit_reference` 选：`"medoid"`（默认，单点质心）或
  `"subcluster"`（最大内聚子簇集合参考，内鬼过半时更稳，见 `experiments/.../RESULTS.md` Findings #2）。
- **WHERE 门控**：同 `(spatial, state)` 已知桶只留一条；容量淘汰优先覆盖不同角度桶
  （`lib/dedup.select_attribute_diverse`，桶 = spatial×state×shot_size×lighting×**pose**；
  pose 为可选轴，缺失记 `unknown` 不减粒度；`select_angle_diverse` 为两维包装）；
  近重复 embedding 若带来**新的已知属性桶**则保留。
- **命名锚定别名感知（axiom 3；`bank.find_by_name` / `register_alias`）**：同一身份不同称呼经
  `metadata.aliases` 归一到同一 asset，读写路径对称，只认显式别名、不做模糊自动合并。
- **组合预算（axiom 6；`CompositionRequest.context_rep_budget` / `max_reps_per_asset`）**：可选硬上限，
  先砍关系扩展的连续性 rep，再砍点名实体多余 rep，**永不砍点名身份的最后一条 rep**。
  与 bench 契约见 [`benchmark/crop_contract.md`](../../../../benchmarks/VMem-Bench/docs/benchmark/crop_contract.md)；
  **Bench+SUT crop 总原则**见[`benchmark/crop_contract.md`](../../../../benchmarks/VMem-Bench/docs/benchmark/crop_contract.md)。

环境变量：`MEMSTRATA_ANGLE_CLASSIFIER_BASE_URL`（否则回退 `MEMSTRATA_CONTEXT_JUDGER_BASE_URL`）、
`MEMSTRATA_ANGLE_CLASSIFIER_MODEL`。

## 2. Track A

```
PromptPacket → IntentInterpreter → compose → ComposedContextRecord
ObservationPacket → AssetCurator.ingest_packet
```

Observation 可选透传 `spatial_angle` / `state_angle` / `temporal_tag`；缺省为 `unknown`，
与冻结 Bench 兼容。

## 3. 测试

```bash
cd benchmarks/MemStrata
PYTHONPATH=src python3 src/memstrata/tests/test_replay_roundtrip.py
PYTHONPATH=src python3 src/memstrata/tests/test_production_dedup.py
PYTHONPATH=src python3 src/memstrata/tests/test_pipeline_smoke.py
PYTHONPATH=src python3 src/memstrata/tests/test_angle_preference.py
PYTHONPATH=src python3 src/memstrata/tests/test_angle_closed_loop.py
```
