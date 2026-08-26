# 语义召回 + 帧级自审 · 设计方案（DRAFT，待审）

> 状态：**提案，未实现**。记录一个读侧检索增强 + 记忆库自审的方向，先审后做。
> 关联：`methods/MemStrata/src/memstrata/skills/intent_understanding/`（读侧 fast/slow）、
> `skills/memory_retrieval/`（`name_match` + `retrievers.py` 基线臂）、
> `skills/memory_update/curator.py`（写侧 reconcile + cohesion 自审）、
> `skills/memory_update/snapshot.py`（`memory.json` 导出）、
> `benchmarks/VMem-Bench/docs/benchmark/sut_pixel_channel_DESIGN.md`（契约侧的姊妹提案）。

## 0. 现状（先对齐事实）

- **默认读侧是文本匹配 + MLLM**：`intent_understanding` 快路径 `name_match`（确定性同语言表层名/别名匹配）→ 描述重叠 → 慢路径 MLLM 意图解析 → `composition` 按 preferred angle / 最新挑 rep。**不做向量检索**。
- **`retrievers.py` 是检索类基线的臂**（`frame_text` / `seg_dinokey` / `seg_framererank` RRF / `bm25` / `recency` / `random`），它们在**原视频采样帧**上召回、返回时间戳。**不是 MemStrata 主路径**。
- **记忆库当前每个状态只落 crop**（`visual/<kind>/<slug>/states/<state>/*.png`）+ `appearances[{sec, segment}]`；**不落对应的源视频整帧**。
- **写侧已有 cohesion 自审**：`curator` 的 `selfaudit_cohesion_floor` / `selfaudit_each_segment` 用 crop 的 embedding 复查"某实体库里是否混进了别的实体"（现已切到 DINOv3 语义编码器）。

## 1. 用户提案（原话要点）

> 名字向量匹配 和 语义向量检索+帧召回重排 是能快速互补的：把每个用户意图提示词 embedding 化，
> 新意图来时做 embedding 对比，然后在召回的片段里做 text-frame 召回重排；也能和实体图做对比。
> 建议记忆库在落每个状态图时，不仅落 crop/分割图，还落对应的那**一个视频帧**；
> 这样能和前一个方案交互，比较帧之间相似度，还能用这个相似度**提醒系统记忆库是否存在问题**。

## 2. 判断（哪些成立 / 硬边界）

**总体成立**，但要卡一条硬边界，且部分能力与已有机制接上：

1. **语义召回 + text-frame 重排——合理，但只能做 opt-in / 鲁棒性轴，不能进默认，且不得改变最终产出形态。**
   - 便宜、对"精确名字匹配落空"（尤其 description-only 设定）是有效互补。
   - **硬边界**：它只能用来**缩小候选实体/片段**；MemStrata 最终 compose 出去的**必须仍是自己的实体记忆图（crop）**。一旦最终产出退化成"原视频采样帧的召回集合"，MemStrata 就等于 `retrievers.py` 的基线臂，novelty 与 `fairness_decisions.md`（name-anchored 主设定 / description 鲁棒性）的叙事同时崩掉。
   - 因此落点是：`name(快) → 语义召回/向量重排(opt-in 中间层) → MLLM(慢)` 的级联扩展，产出始终是实体 rep。

2. **每个状态图额外落对应源视频整帧——好，成本小，一举多得。**
   - 已有 `source_seconds`，落帧只是省去 read 时重新 cut。
   - 直接支撑 §3-D 的帧级自审；并与契约侧提案（`sut_pixel_channel_DESIGN.md`）呼应：库里同时有 crop 和原帧，bench 要 crop 口径还是整帧口径都能就地满足。

3. **相似度自审记忆库——成立，且是已有 cohesion 自审的自然扩展。**
   - 现有自审在 **crop embedding** 层复查实体内聚度；用户提议的**帧级**自审是扩展：
     (a) **crop↔源帧一致性**（这张 crop 是否真来自它声称的源帧的一个子区域）→ 抓串号/错抠；
     (b) **同实体多帧离群**（一个实体的若干源帧里有一张明显不合群）→ 抓污染。
   - 阈值是 encoder 相对的（与 cohesion floor 同一课：非语义编码器下会误杀），现已用 DINOv3，可用。

## 3. 设计（提案，未实现）

### A. Prompt 语义召回索引
- 每段观测时把该段**意图提示词**用文本编码器嵌入，持久化到**旁挂文件**（不塞进 `memory.json`；见 §4）。
- 新意图来时：先走确定性 `name_match`；**未命中**才对历史 prompt 嵌入做 cosine，召回 top-k 候选段。
- 输出仍映射到这些段里出现过的**实体**，交给 compose 组合实体 rep。

### B. 召回段内 text-frame 重排（opt-in）
- 仅在 A 召回的候选段范围内，用 query 文本对候选段的（本方法自己库里的）实体图/关键帧做重排，
  取更相关的实体 rep。**不新采原视频帧**、不返回裸帧。
- 复用 `retrievers.py` 已有的重排/RRF 纯函数，但作用对象换成**本方法的实体库**，不是源视频帧池。

### C. 记忆库落原帧
- rep 除 `object_uri`（crop）外，新增 `source_frame_uri`（该 crop 所在源帧，832×480）。
- `snapshot.py` 导出 `states.<state>.images` 时并列导出 `source_frames`（或每张图带 `crop`/`frame` 两路径）。
- 存储小、可选（无帧时留空，向后兼容）。

### D. 帧级自审（扩展现有 cohesion 自审）
- **crop↔源帧**：crop 应能在其 `source_frame` 里找到高相似子区域；否则标 `provenance_suspect`（只审计，不静默删）。
- **同实体源帧离群**：对实体的源帧集合做 medoid/离群检测，离群帧对应的 rep 标 `identity_suspect`，进人工/降权。
- 与写侧 `selfaudit_cohesion_floor` 合流：crop 层与帧层两条证据共同决定"这个实体是否被污染"，而非单一阈值。

## 4. 约束与风险

- **别让默认变基线**：A/B 是 opt-in / 消融 / 鲁棒性轴；默认仍 `name → MLLM`。
- **持久化用旁挂**：prompt 嵌入、（若需要）rep 视觉嵌入存 `embeddings.npz`/`prompt_index.npz`（按 `rep_id` / `segment` 键），**不塞进人读的 `memory.json`**。
- **阈值 encoder 相对**：自审/重排阈值只在语义编码器（DINOv3 / 文本模型）下有意义。
- **成本**：A 便宜（短文本嵌入）；B 限定在召回段内、且对象是已存的库图，不重采原视频；C 存储小。
- **因果**：召回/自审只能用 `<t` 的证据，沿用现有因果护栏。

## 5. 开放决策（待 owner 定）
1. A/B 是否只做 description-only 鲁棒性轴，还是也作为 name-anchored 下的兜底中间层？
2. C 的原帧是否默认落（增存储）还是按开关落？
3. D 的自审是"仅审计标记"还是允许自动降权/拒识？
4. **不实现，直到 owner 审定**。

## 6. 评审驱动的修订（gpt-5.5 + opus5 交叉印证，2026-07-27）

两份独立只读评审（含 BBB B16 实跑产物交叉验证）改变了本方案的**前提**与**顺序**。关键更正：

### 6.1 碎片化的真正根因 —— 之前的"自发现→χ 合并"修复在真实数据上不生效
- 实跑证据：`memory.json` 48 个资产**没有一个** `_disc_` 前缀，即 χ `_reconcile_identity` 一次都没触发；主角兔子仍碎成 `兔子/大兔子/棕色兔子/橙色兔子/灰色兔子/白兔` 六条。
- 机理：requested/discovered 的分流谓词是 `term_in_prompt`（CJK ≥2 字子串），在中文描述型提示词下**几乎恒真**，`_reconcile_names` 还会把漂移标签吸附回提示词原词 → 几乎所有实体判为 requested、按名字锚定 → χ 路径不可达。
- **更深一层**：碎片主要来自**跨段提示词用了不同名字**（兔子/大兔子/白兔 分别出现在各自段的提示词里，各自都是"requested"），不是 VLM 自发现漂移。所以"自发现→χ"这条**天然管不到**它。
- **正确修法（取代 §原 fix①的二元路由）**：让 **requested 也过一道 χ**，但 χ 的输出**降级为"别名/`REPLACES` 建议"**——名字锚定继续决定落哪条记录，χ 只在"名字不同但视觉同一"的记录间建立 alias 关系并回写 `register_alias`，读侧即可用任一说法召回同一身份。

### 6.2 必做前提（否则上面这些 + 本方案的 A/B/C/D 都建在流沙上）
- **P0｜Track A 从未启用 `MemoryPolicy.production()`**：curator 用默认 policy，β_τ/γ_τ/cohesion 准入/自审计/VLM 判官/全局预算全部关闭或误标定（character 合并跑在 0.55 而非标定的 0.75，prop 0.55 而非 0.21）。→ 适配器显式传 `policy=MemoryPolicy.production(...)`；**本方案 §3-D 的"帧级自审"正是 production 下才打开的 `audit_cohesion` 的扩展**，前提不开则自审无从谈起。
- **P1｜`_entity_images` 按名字查 asset_id**：discovered 资产查不到 → crop 层身份门与新颖性排序静默失效；且每段重复编码已存 embedding（数千次无谓 DINOv3 前向）。→ 按 asset_id 查 + 复用 `rep.annotations["embedding"]`。
- **P1｜读侧预算未接入**：MemStrata 适配器无 `set_budget`，`max_reps_per_asset` 默认 1，每 chunk 中位仅交 3 条 / 预算 16（20%），与填满预算的检索基线**非等预算**——这是"MemStrata 看起来分低"的大头之一。→ 加 `set_budget` + `context_rep_budget`。

### 6.3 与像素通道的协同（已落地）
- 像素通道让 MemStrata 交自己的 crop、不再按时间戳回切，**顺带把评审 P0"时间戳记成段中点导致切错打分帧"对_打分_的影响消掉了**（评分用真 crop，不再用错时刻的整帧）。该时间戳问题降级为"`memory.json` 时间轴元数据/因果护栏精度"问题，仍应修但不再是打分 P0。

### 6.4 读侧在中文上结构性失效（"分低"的另一大头）
- description 兜底：读侧 `_content_tokens` 不切 CJK（整句一个 token）→ 中文下恒不命中（实跑 `description=0/52`）；而写侧 dedup 却按字切（过松）。两侧分词需统一。
- alias 通道：`register_alias` 在 Track A 不可达 → 读侧别名恒空（实跑 aliases 全空）。修 §6.1 的 χ 建议合并后，alias 回写是把结果传给读侧的唯一通路。
- `recency` 兜底名不副实（miss 记成 recency 但空返回）、`retrieval_sources` 被写侧污染（mllm:53>52）——指标可信度问题。

### 6.5 顺序（时间有限时的高 ROI 优先）
1. **T0 让当前设计真的生效**：`production()` policy（6.2）+ requested 也过 χ 做别名建议 + `register_alias` 回写（6.1）+ `_entity_images` 按 id 查并复用 embedding（6.2）。
2. **T1 可比性/有效性**：读侧 `set_budget`（6.2）；snapshot 导出移出计时窗口。
3. **T2 中文读侧召回**：统一 CJK 分词；修 recency/counter 标签。
4. **T3 本方案的 A/B/C/D**：语义召回 + 落原帧 + 帧级自审，建在 T0–T2 之上；开 VLM 属性分类器（填状态）前先把"同桶硬拒"改成按质量替换。

> 已确认**不是**问题：无未来泄露（compose 先于 observe、三道因果护栏、`n_future_dropped=0`）；无 gold/roster 使用。
