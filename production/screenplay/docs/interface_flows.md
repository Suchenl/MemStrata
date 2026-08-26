# Screenplay Interface Flows (剧本接口对接与转换流程)

本文件记录 screenplay 如何从人类可读创作层进入生产层，并继续适配 Montage 与外部 baseline。核心原则是：**human-readable screenplay 是创作真值，production screenplay 是派生执行视图，baseline adapter 只读取自己需要的字段。**

## 1. 总体流程

```text
User rough input / optional assets
        │
        ▼
LLM normalizes to human_readable_screenplay
        │
        ▼
User approves creative truth
        │
        ▼
Constrained compiler derives production_screenplay
        │
        ├── Montage Planner / Composer
        ├── Helios-style single prompt adapter
        ├── StoryMem-style story JSON adapter
        └── ViMax / MovieAgent-style reference adapter
```

可选资产只在第一步作为创作约束和候选素材进入项目。它们不会因为出现在输入中就自动成为 production asset；是否被使用、如何使用、何时成为 required/optional/forbidden reference，必须由已确认剧本和生产层编译结果决定。

## 2. 双语文件约定

中英文使用两份独立文件，例如 `0000_detective_mystery_en.json` 和 `0000_detective_mystery_zh.json`。两份文件结构、ID 与 traceability 必须严格对齐。

- Model-facing adapter 默认读取英文文件。
- Human-facing UI 默认读取中文文件。
- 不使用 `*_en` / `*_zh` 字段后缀。
- 跨语言对齐依赖 `story_id`、`scene_id`、`beat_id`、`shot_id`、`entity_id`。

## 3. Human Layer To Production Layer

从 `human_readable_screenplay` 到 `production_screenplay` 的转换应以确定性规则为主：

1. 复制 story metadata、scene id、beat id、entity id。
2. 将 Hollywood-inspired scenes 映射到 production scenes。
3. 将 beats 拆为 shots，并为每个 shot 写入 `source_beat_ids`。
4. 从 action / dialogue / parenthetical 派生 `visual_track` 与 `audio_track`。
5. 从 cast/assets 与 referenced entities 派生 `planned_assets`。
6. 用生命周期规则补全 `preserve | transform | deprecate | avoid`。
7. 校验双语结构、ID 对齐、实体复用、地点回访、状态变化和 forbidden carryover。

只有 beat-to-shot 拆分、镜头语言补全、声音细节补全这类语义判断可以使用 LLM/MLLM 辅助；这些结果必须受 schema 和 validation 约束。

## 4. Baseline Adapter Flows

### A. 单提示词长视频生成器（Helios / Skywork Matrix 类）

下游期望一段完整连续的英文 T2V prompt。

转换规则：

1. 读取英文 `story_overview` 和 `global_plan.constraints`。
2. 遍历 `production_screenplay.shots`。
3. 拼接每个 shot 的 `visual_track.actions`、关键 `cinematography` 和必要 `continuity_requirements`。
4. 将 `planned_assets.operation == "avoid"` 的实体转为 negative constraints。
5. 输出一段连续英文提示词，不暴露 Montage 内部字段。

```python
def to_single_prompt(screenplay_json):
    production = screenplay_json["production_screenplay"]
    entities = {e["entity_id"]: e["name"] for e in screenplay_json["main_entities"]}
    parts = [screenplay_json["story_overview"]]
    for shot in production["shots"]:
        actions = " ".join(shot["visual_track"]["actions"])
        for entity_id, name in entities.items():
            actions = actions.replace(f"({entity_id})", f"({name})")
        parts.append(actions)
    return " ".join(parts)
```

### B. StoryMem-style 故事 JSON

下游期望 `story_name`、`story_overview`、`scenes`，以及每个场景的 `video_prompts`、`first_frame_prompt` 和 `cut`。

转换规则：

1. 将 production shots 按 `scene_id` 聚合。
2. 每个 shot 生成一个 StoryMem scene item 或 prompt item，具体取决于 baseline 的最小粒度。
3. `transition == "continue"` 映射为 `cut: false`，其它显式切换映射为 `cut: true`。
4. `video_prompts` 可以由规则拼接动作、实体外观、地点背景与镜头语言；只在需要风格化重写时使用受约束 MLLM。
5. `first_frame_prompt` 从首个 action、scene background 和 required assets 派生。

### C. Reference-based 生成器（ViMax / MovieAgent 类）

下游期望参考资产列表和逐镜头指令。

转换规则：

1. 从 `main_entities` 收集 character/object/location 候选。
2. 遍历 `production_screenplay.shots[*].planned_assets`。
3. `preserve` / `transform` 映射为 required 或 optional references。
4. `avoid` / `deprecate` 映射为 forbidden references 或 negative prompt。
5. `source_beat_ids` 保留为可追溯信息，方便人工审核和错误定位。

### D. Montage 原生生产管线

Montage 读取完整 production screenplay：

```text
production_screenplay
        │
        ▼
Seed candidate assets from main_entities + optional asset intent
        │
        ▼
For each shot/chunk boundary:
  - read visual_track and audio_track
  - resolve planned_assets into required / optional / forbidden directives
  - select backend by transition, duration, reference needs, and multi-shot capability
        │
        ▼
Generate media artifact
        │
        ▼
Observe, materialize, and update asset store
```

Multi-shot 后端只能在三个条件同时满足时合并 shots：叙事上可连续、后端支持所需参考输入、合并不会破坏 asset materialization 和评测边界。合并生成后仍必须拆回可评估、可落盘的 shot/chunk 单元。

## 5. 字段职责表

| 字段 | 所属层 | 职责 |
|---|---|---|
| `human_readable_screenplay.scenes[*].slugline` | human | 人类可读的场景标题 |
| `human_readable_screenplay.scenes[*].beats` | human | 可审阅的戏剧节拍，是生产层来源 |
| `cast_and_assets` | human | 创作层实体登记，不直接等于正式资产 |
| `optional_asset_intent` | human | 可选资产的创作意图和候选绑定 |
| `main_entities` | production | 生产层实体登记和初始视觉状态 |
| `production_screenplay.scenes` | production | 生产时序、场景背景、持续音轨 |
| `production_screenplay.shots` | production | 镜头级执行单元 |
| `source_beat_ids` | production | 回溯到 human layer 的依据 |
| `visual_track` | production | 动作、镜头语言、视觉连续性 |
| `audio_track` | production | scene 级 ambience/bgm 与 shot 级 dialogue/foley/voiceover |
| `planned_assets` | production | required/optional/forbidden 资产调度 |
| `video_prompts` / `first_frame_prompts` | adapter | 下游 prompt scaffold，由 adapter 最终生成 |
