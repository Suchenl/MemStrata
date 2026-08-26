# Screenplay Design Principles (剧本设计原则)

Montage 的 screenplay 不是单一扁平 JSON，而是一套分层创作与生产协议。它先让人类确认一个像电影剧本一样可读、可审、可改的创作真值，再从这个真值派生面向长视频生成管线的生产视图。这样既保留影视创作的中立性，也保留 MemCoRA-Bench 所需的资产一致性、生命周期与音画控制。

## 1. Human Layer First

用户可以从短目标、故事摘要、非格式化剧本、已有剧本片段，或可选资产开始。LLM/MLLM 的第一职责是把这些输入整理成固定格式、可读、可审的 `human_readable_screenplay`，而不是直接生成生产管线字段。

这一层应接近 Hollywood-inspired 剧本组织：scene heading / slugline、action、character cue、dialogue、parenthetical、scene/beat 节奏。它负责故事、人物、戏剧张力和创作判断。

## 2. Production Layer As Derived View

用户确认 human-readable screenplay 后，系统再把它编译为 `production_screenplay`。生产层是可再生成、可校验的执行视图，不是第二份独立剧本，也不应开放式改写故事。

生产层负责 `scenes`、`shots`、`visual_track`、`audio_track`、`planned_assets`、`preserve/transform/avoid` 生命周期、时长、转场、prompt scaffold 和下游 backend 所需的约束。

## 3. Bilingual Separated Dual-track

中英文剧本必须分成两份文件，例如 `*_en.json` 与 `*_zh.json`。两份文件使用完全相同的 schema、story id、scene id、beat id、shot id 和 entity id，但所有自然语言字段分别使用英文或中文。

不要在同一份 JSON 中混用 `*_en` / `*_zh` 字段。英文版本更适合作为模型生成侧输入，中文版本更适合人类导演、中文 MLLM 规划器和审阅界面。

## 4. Optional Assets As Inputs, Not Truth

可选资产可以包括角色参考图、地点/空间参考、关键道具图、风格图、历史视频片段、配乐或氛围参考。它们进入项目时只是创作约束和候选素材，不能自动等同于正式生产资产。

只有当用户确认剧本，且生产层编译、规划和资产更新通过后，这些素材才会以明确用途进入 asset store，例如身份参考、地点参考、道具状态、风格参考或 forbidden reference。

## 5. Scene Is Narrative, Location Is Asset

`scene` 是叙事层级：它表示一段相对连续的时间、地点和戏剧目标。`location` 是资产层级：它表示一个可以被引用、复用、变体化或排除的物理空间/地点资产。

剧本可以包含 narrative scenes；生产层中的资产类型必须使用 `location`，避免把叙事分段和资产库条目混为一谈。

## 6. Shot Is The Atomic Creative Unit

在影视语言中，`shot` 是最小的连续视听单元。生产层应以 shots 承载镜头动作、镜头语言、音画同步和资产调度。

底层生成器仍可能按固定时长 chunk 执行，也可能由 multi-shot 后端一次生成多个 shots。Chunk 是执行和资源约束概念，不是 screenplay 的创作基本单位。是否合并多个 shots 生成，应由 router 根据后端能力、参考输入能力和质量风险决定。

## 7. Entity Registration And Lifecycle

所有核心角色、关键道具和地点资产都必须有稳定的全局 ID，例如 `E1`、`E2`、`L1`。生产层通过这些 ID 把自然语言剧本、可选资产、生成结果和后续 asset update 绑定起来。

生产层必须显式记录生命周期操作：

- `preserve`：保持并复用已有身份、外观、状态或地点。
- `transform`：状态发生变化，例如换装、受伤、破损、发光、熄灭。
- `deprecate` / `avoid`：资产已经失效或当前必须规避，后续作为 forbidden / negative constraint 使用。

## 8. AV Script: Separate Visual And Audio Tracks

生产层采用 AV Script 风格，把视觉和声音分轨写清楚。

`visual_track` 承载动作、构图、景别、运镜、光影、画面状态和视觉连续性。`audio_track` 承载声音设计。Scene 级音轨用于持续元素，例如 ambience 与 bgm；shot 级音轨用于瞬时同步元素，例如 dialogue、foley 和 voiceover。

这种设计比把所有内容塞进一个 prompt 更稳定，也更接近真实制作中的画面、对白、拟音、环境声和配乐分工。

## 9. Determinism Before Model Calls

从 human layer 到 production layer 的转换应尽量由脚本、schema、ID 映射和校验完成。只有在确实需要语义判断时，才使用 LLM/MLLM，并且该步骤必须受约束、可检查、可重跑。

例如：解析 slugline、复制 ID、校验双语结构、检查 shot id 对齐、统计复用次数、验证 `avoid` 是否持续生效，这些都应确定性完成。把较长 action beat 拆成多个 shots 这类语义操作可以使用模型辅助，但结果必须被用户或验证器确认。

## 10. Screenplay Must Stay System-neutral

Human-readable screenplay 是创作真值，不能泄漏 Montage 的 Python 包结构、类名、内部 pipeline 或特定生成器实现。它应该像真实剧本一样可被人类导演、外部 baseline 或其它系统理解。

Production screenplay 可以包含执行字段，但也应保持 backend-agnostic。具体转成 Helios、StoryMem、ViMax、VACE、LongCat 或其它后端输入，应发生在 adapter / router 层，而不是污染原始剧本。
