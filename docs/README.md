# MemStrata 文档索引（唯一入口）

> 项目已拆分为两个自包含子项目：**方法文档在本处 `methods/MemStrata/docs/`**（本索引），
> **基准 / baseline / bench 侧实验文档已迁至 `benchmarks/VMem-Bench/docs/`**，各自按角色分目录。
> 目的：消除语义漂移，让"方法 / 基准 / baseline / 实验 / 论文"各有唯一权威位置。
> 下方指向基准侧的条目均以跨包相对链接指向其新家。
> 新增、移动、重命名文档前先读本索引；**已有近义文档就编辑，不要新建**。
>
> 术语以 [`glossary.md`](glossary.md) 为**唯一真源**，不要在其它文档另立定义。

## 一图看清三个角色（别再混）

本项目有**三类东西**，文档目录与之一一对应：

| 角色 | 是什么 | 代码包 | 文档目录 |
|---|---|---|---|
| **方法 / SUT** | `memstrata`——**我们提出的**记忆管理 + 上下文组合方案，是**被评测的对象** | `src/memstrata/` | [`method/`](method/) |
| **基准 / Bench** | `vmem_bench`——**评测这套方法**的标注管线 → 冻结 gold → 确定性评分 | `src/vmem_bench/` | [`benchmark/`](../../../benchmarks/VMem-Bench/docs/benchmark/) |
| **baseline / 对照** | 与 SUT 在**同一基准**上对比的**其他系统**（因果 / 检索 / 诊断） | `baselines/`、`baseline_adapters/` | [`baselines/`](../../../benchmarks/VMem-Bench/docs/baselines/) |

> ⚠️ **`memstrata` 是 SUT（被测方法），不是 baseline 之一。** `method/` 与 `baselines/` 永远分开写。

## 目录结构

| 目录 | 内容 | 什么时候读 |
|---|---|---|
| [`overview/`](overview/) | 三个角色**各自怎么运作**的心智模型（本索引只管"在哪找"，overview 管"怎么运作"） | 想在十分钟内建立正确认知 |
| [`method/`](method/) | **SUT（`memstrata`）** 设计哲学与实现规格 | 改 `src/memstrata/` 前 |
| [`benchmark/`](../../../benchmarks/VMem-Bench/docs/benchmark/) | **基准（`vmem_bench`）** 协议 / 契约 / 标注 / 评分权威文档 | 改 `src/vmem_bench/` 或评分前 |
| [`baselines/`](../../../benchmarks/VMem-Bench/docs/baselines/) | baseline 策略、Track A 实现、**公平性决定** | 跑 / 改 / 评 baseline 前 |
| [`experiments/`](experiments/) | 实验计划、ablation、**公平性实验计划** | 规划 / 复现实验前 |
| [`paper/`](paper/) | 论文中文设计笔记（组织、insight、各节草稿） | 写 / 改论文前 |
| [`operations/`](operations/) | **权重 / 环境 / 运行要求唯一清单**（模型路径、conda 环境、env 变量、校准、发布自包含检查单） | 换权重 / 配环境 / 跑 GPU / 发布前 |
| [`glossary.md`](glossary.md) | 统一术语表（唯一真源） | 任何时候拿不准用词 |

## 各目录清单

### method/（SUT = `src/memstrata/` 权威设计；**不是 baseline**）
- [`philosophy.md`](method/philosophy.md)：**最高纲领**——六条库质量公理、WHO 先于 WHERE、命名锚定 vs embedding 的边界、保守拒绝/隔离。
- [`design.md`](method/design.md)：对齐论文四步环路的实现规格（含 D6 零 import 边界、planner fallback 约束、视觉地层、Track A 适配器）。
- [`generator_wiring.md`](method/generator_wiring.md)：视频生成器后端清单与接线。

### benchmark/（Bench = `vmem_bench` 权威协议；**已迁至 [`benchmarks/VMem-Bench/docs/benchmark/`](../../../benchmarks/VMem-Bench/docs/benchmark/)**）
- [`running_eval.md`](../../../benchmarks/VMem-Bench/docs/benchmark/running_eval.md)：**端到端运行手册**——怎么把一部影片从 frozen gold 跑到分数（Stage 1 产 context → Stage 2 VLM 打分）。要"跑一部影片"先读这里。
- [`design_principles.md`](../../../benchmarks/VMem-Bench/docs/benchmark/design_principles.md)：本基准设计原则（含通用基准方法论纲领）。
- [`schemas_and_contracts.md`](../../../benchmarks/VMem-Bench/docs/benchmark/schemas_and_contracts.md)：数据契约与指标定义（**schema 权威源**；headline 指标见 `scoring.md`）。
- [`annotation_pipeline.md`](../../../benchmarks/VMem-Bench/docs/benchmark/annotation_pipeline.md)：标注管线**阶段级权威**（S1–S7 产出 frozen gold）。
- [`annotation_tracking_internals.md`](../../../benchmarks/VMem-Bench/docs/benchmark/annotation_tracking_internals.md)：track-first 追踪 / re-ID / 身份消解**内部机制**（被 `pipeline_track_first` 代码引用）。
- [`scoring.md`](../../../benchmarks/VMem-Bench/docs/benchmark/scoring.md)：**评分权威**——VisualFidelity headline、多 embedder 路由、确定性回放、LSMDC 专项指标。
- [`crop_contract.md`](../../../benchmarks/VMem-Bench/docs/benchmark/crop_contract.md)：Bench+SUT 共用的 crop 总原则与属性字段契约。
- [`services_and_time.md`](../../../benchmarks/VMem-Bench/docs/benchmark/services_and_time.md)：常驻模型服务、GPU 放置、时间元数据。
- [`dashboard_and_review.md`](../../../benchmarks/VMem-Bench/docs/benchmark/dashboard_and_review.md)：SSE 监控 + 人审 UI 任务书与人机评审策略。
- [`staged_pipeline_plan.md`](../../../benchmarks/VMem-Bench/docs/benchmark/staged_pipeline_plan.md)：S2/S3/S4 分阶段管线优化计划。
- [`pitfalls.md`](../../../benchmarks/VMem-Bench/docs/benchmark/pitfalls.md)：运行事故与修复记录（机构记忆）。
- [`references.md`](../../../benchmarks/VMem-Bench/docs/benchmark/references.md)：参考文献与 2026 landscape。

### baselines/（在**同一基准**上对照 SUT 的其他系统；**已迁至 [`benchmarks/VMem-Bench/docs/baselines/`](../../../benchmarks/VMem-Bench/docs/baselines/)**）
- [`fairness_decisions.md`](../../../benchmarks/VMem-Bench/docs/baselines/fairness_decisions.md)：**baseline 公平对比的最终决定**（2026-07-22 固化，当前权威）。
- [`track_a.md`](../../../benchmarks/VMem-Bench/docs/baselines/track_a.md)：Track A = 真实检索 / 记忆在 GT 视觉上的实现（含各 baseline 权重来源与放置）。
- [`strategy.md`](../../../benchmarks/VMem-Bench/docs/baselines/strategy.md)：baseline 策略与历史选型记录（选型以 `fairness_decisions.md` 为准）。
- [`external_baseline_audit.md`](../../../benchmarks/VMem-Bench/docs/baselines/external_baseline_audit.md)：外部 baseline 历史核验表（已被 `fairness_decisions.md` 更新）。
- [`hook_recipes.md`](../../../benchmarks/VMem-Bench/docs/baselines/hook_recipes.md)：给外部系统插桩导出 evidence 的配方。

### experiments/
- [`buildplan.md`](experiments/buildplan.md)：paper-facing 交付顺序与可验收 gate。
- [`fairness_experiment_plan.md`](../../../benchmarks/VMem-Bench/docs/experiments/fairness_experiment_plan.md)：**公平性实验计划**（name-anchored / description-only 两套输入、多 embedder、k 扫描、因果覆盖）。**（bench 侧，已迁至 `benchmarks/VMem-Bench/docs/experiments/`）**
- [`generator_in_the_loop_eval_plan.md`](../../../benchmarks/VMem-Bench/docs/experiments/generator_in_the_loop_eval_plan.md)：**generator-in-the-loop 评测设计（讨论稿）**——长程一致性退化曲线、样本规模、与 Track A 主表分工。**（bench 侧，已迁至 `benchmarks/VMem-Bench/docs/experiments/`）**
- [`open_source_movie_track_decomposition.md`](experiments/open_source_movie_track_decomposition.md)、[`ablation_study/`](experiments/ablation_study/)。

### paper/
- [`paper_organization.md`](paper/paper_organization.md)：contribution / baseline / table 契约。
- [`_insight.md`](paper/_insight.md)、[`0_abstract.md`](paper/0_abstract.md)、[`1_introduction.md`](paper/1_introduction.md)、[`2_related_work.md`](paper/2_related_work.md)。

## 不在 docs/ 内（刻意留原地，此处仅登记指针）

这些是**运行时代码资产、包入口契约或构建产物**，移动会破坏功能或语义，故留在原地：

- `../AGENTS.md`：硬性架构边界声明（`memstrata` ↔ `vmem_bench` 零互相 import、对外零依赖）。
- `src/*/README.md`、`scripts/*/README.md`：包/目录入口契约（Design 文档一律指回本 `docs/`）。
- `src/vmem_bench/annotation/pipeline/stages/**/*.md`：被标注管线**运行时读取**的提示词模板与审核清单。
- `assets/paper/MemStrata/sections/`、`_figures/`、`notation.md`：论文 LaTeX 构建产物。
- `data/_runs/*/results.md`、`data/_vlm_rerun_kit_*/`：实验产物 / 临时标注数据，不是知识文档。
- `baselines/{Scripted,Causal}/**`：外部 vendored 源码 checkout（含其自带 README），不是本项目知识文档。
