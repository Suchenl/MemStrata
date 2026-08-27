# 项目速读：memstrata / vmem_bench / baselines 三个角色

> 本文读者：想在十分钟内建立正确心智模型的人。全项目有**三个不能混淆的角色**——被评测的方法
> （SUT）、评测它的基准、以及基准上的对照系统。术语以 [`../glossary.md`](../glossary.md) 为唯一真源。

## 一句话先分清角色

> **`memstrata` 是被评测的方法（SUT），不是 baseline 之一。** baseline 是**其它**被放到同一基准上
> 对比的系统；基准本身既不是方法也不是 baseline。

本文只讲**这三个角色各自怎么运作**；"哪个文档在哪"由 [`../README.md`](../README.md) 索引负责（含
角色 ↔ 目录映射），不在此重复。两包**永不互相 import**，只通过纯 JSON 契约（`PromptPacket` /
`ObservationPacket` / `ComposedContextRecord`）交互；硬边界见 [`../../AGENTS.md`](../../AGENTS.md)。

## memstrata（SUT / 我们的方法）

`memstrata` 把"被动检索 dump 出来的扁平历史"换成"**意图对齐的实体组合**"：写入时按实体分层沉淀
（视角 / 状态 / 时间 / 生命周期，embedding 只做去冗余）；读取时 `prompt 内稳定名字 → 关键词匹配 id
→ 字典解引用 → 组条件包`，**读路径默认无 VLM、无全库相似检索**。它是本基准要评测的对象，产出
`ComposedContextRecord`（具名资产 ⊕ role ⊕ lifecycle ⊕ instruction ⊕ forbidden）。详见
[`../method/design.md`](../method/design.md) 与 [`../method/philosophy.md`](../method/philosophy.md)。

## vmem_bench（基准）

基准是一个**离线构造冻结金标、再确定性回放评分**的系统，不是在线让 VLM 临场裁判。三段：

1. **标注**：从单一源、时间连续的长视频离线构造候选金标（检测/跟踪/re-ID 决定"谁在何处何时出现"，
   VLM/人审只在受控步骤给候选与裁决）。生产管线为 **S1–S7**。
2. **冻结**：只有通过人审 + 严格 lint 的金标才 `freeze`；`gold/*.json` + `embeddings.safetensors`
   是唯一评分真值，`present` / `forbidden` / scenario tag 绝不提前泄露给 SUT。
3. **评分**：`load_gold` 拒绝未冻结/hash 不一致的金标；`run_replay` 按时间顺序对每个 chunk
   先 `PromptPacket → SUT ComposedContextRecord → 打分`，再发 `ObservationPacket`（SUT 可从过去
   建立记忆，但组当前上下文前看不到当前金标）。headline 指标是 **VisualFidelity**（按实体类型路由的
   多 embedder）。

因果顺序、指标定义、schema 权威见 [`benchmarks/VMem-Bench/docs/benchmark/`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/)（`schemas_and_contracts.md` /
`scoring.md` / `design_principles.md`）。

## baselines（对照系统）

放到**同一 frozen gold、同一 harness、同一 pinned embedder** 上与 `memstrata` 比较的**其它**系统。
主定量表为**因果**系统（`helios / longlive_rag / memflow / iamflow / decmem`，与论文 setting 一致）；
脚本化 / agentic 系统（ViMax / MovieAgent / VideoMemory / StoryMem / Memento / MM-StoryAgent）为
非因果，移出定量主表、仅在附录做定性说明。选择与公平性的**当前权威**见
[`baselines/fairness_decisions.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/baselines/fairness_decisions.md)；实现见
[`baselines/track_a.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/baselines/track_a.md)、策略/历史见
[`baselines/strategy.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/baselines/strategy.md)。

## 当前状态与下一步去哪看

- **金标现状**：BBB 金标已 **FROZEN**，52 个 chunk；生产标注管线为 **S1–S7**。标注怎么造出来的权威
  事实底稿见 [`benchmark/annotation_pipeline.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/benchmark/annotation_pipeline.md)。
- **交付顺序与验收 gate**（freeze → 可复现实验 → 回填论文）见
  [`../experiments/buildplan.md`](../experiments/buildplan.md)。

> 心智模型三条：① SUT 是被评的方法，不是 baseline；② 金标离线冻结、评分确定性回放，不是在线 VLM
> 裁判；③ 一个 chunk 先打分再发 observation，保证 SUT 组当前上下文前看不到当前真值。
