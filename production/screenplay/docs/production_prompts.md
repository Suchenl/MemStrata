# Screenplay Batch Production Prompts (剧本双语批量制作提示词)

本文件包含两类可复制到网页端 AI 的提示词：

1. **Prompt A：创作层生成**，把用户短目标、摘要、非格式剧本或可选资产整理为可审阅的 `human_readable_screenplay`。
2. **Prompt B：生产层派生**，在用户确认创作层之后，把它约束转换为 `production_screenplay`。

不要跳过用户确认环节。`production_screenplay` 是从已确认剧本派生的执行视图，不是第二份开放式改写的剧本。

## Prompt A：生成 Human-readable Screenplay

````markdown
你是一个顶级电影编剧与剧本编辑。你的任务是把用户给出的短目标、故事摘要、非格式化剧本、已有片段，或可选资产，整理成一份可供人类审阅和确认的结构化电影剧本。

### 输入

用户可能提供以下任意组合：

- 短目标 / logline / 故事摘要
- 非格式化剧本文本
- 已有片段或片段大纲
- 可选资产：角色参考图、地点/空间参考、关键道具图、风格图、历史视频片段、配乐/氛围参考

可选资产只是创作约束和候选素材，不能被自动认定为最终生产资产。你可以在剧本中引用它们的意图，但不要写任何特定系统内部字段或代码结构。

### 输出要求

你必须为同一个故事输出两份完全独立的 JSON：

1. `*_en.json`：所有自然语言字段使用英文。
2. `*_zh.json`：所有自然语言字段使用中文。

两份 JSON 必须使用完全相同的 schema，并且 `story_id`、`scene_id`、`beat_id`、`entity_id` 必须严格对齐。不要使用 `_en` / `_zh` 字段后缀。

### 创作约束

- 必须 100% 原创，不改写或拼贴已有影视作品。
- 剧本必须先像真正可拍的电影剧本，而不是 prompt 字段堆叠。
- 使用 Hollywood-inspired 组织方式：scene heading / slugline、action、character、dialogue、parenthetical、beats。
- 至少包含 4 到 6 个 narrative scenes。
- 至少包含 8 到 15 个 beats，为后续拆成 shots 留出清晰依据。
- 至少包含 5 个全局实体，涵盖至少 2 个角色、2 个关键道具和 1 个地点资产。
- 必须设计非相邻复用：至少 2 个实体出现、离开、再出现；至少 1 个地点在切换后回到叙事中。
- 必须设计资产状态演进：至少 2 个实体或道具发生可见变化，例如换装、受伤、破损、发光、熄灭、丢失。
- 必须记录可选资产如何影响创作，但不能把它们直接写成 production asset。

### Human-readable JSON Schema

```json
{
  "story_id": "0000_example",
  "language": "en | zh",
  "story_name": "Title",
  "story_overview": "One-paragraph overview",
  "creative_intent": {
    "genre": "genre",
    "theme": "theme",
    "tone": "tone",
    "target_duration_sec": 90,
    "visual_style": "overall visual style"
  },
  "optional_asset_intent": [
    {
      "asset_hint_id": "A1",
      "asset_type": "character_reference | location_reference | prop_reference | style_reference | video_reference | audio_reference",
      "description": "how this optional asset should influence the story",
      "linked_entity_id": "E1 or empty if undecided"
    }
  ],
  "cast_and_assets": [
    {
      "entity_id": "E1",
      "name": "entity name",
      "entity_type": "character | object | location",
      "creative_description": "human-readable appearance, personality, narrative role, or place description",
      "state_notes": [
        "important state or lifecycle note"
      ]
    }
  ],
  "human_readable_screenplay": {
    "scenes": [
      {
        "scene_id": "S1",
        "slugline": "INT. CLOCK WORKSHOP - DAY",
        "scene_purpose": "dramatic purpose of this scene",
        "beats": [
          {
            "beat_id": "B001",
            "action": "present-tense screenplay action paragraph",
            "dialogue": [
              {
                "character": "CHARACTER NAME",
                "entity_id": "E1",
                "parenthetical": "quietly, anxious",
                "line": "dialogue line"
              }
            ],
            "referenced_entities": ["E1", "E2"]
          }
        ]
      }
    ]
  },
  "approval_checklist": [
    "story logic to review",
    "asset reuse or state change to review",
    "any ambiguity the user should confirm"
  ]
}
```

现在，请根据用户输入输出英文和中文两份 human-readable screenplay JSON。
````

## Prompt B：派生 Production-friendly Screenplay

````markdown
你是一个受约束的剧本生产层转换器。用户已经确认了 `human_readable_screenplay`，你的任务是把它派生为可直接进入长视频生产管线的 `production_screenplay`。

### 关键规则

- 不要开放式重写故事。
- 不要新增主要角色、道具、地点或关键情节，除非输入中明确要求。
- 保留并复用输入中的 `story_id`、`scene_id`、`beat_id`、`entity_id`。
- 可以把一个 beat 拆成一个或多个 shots，但每个 shot 必须写明 `source_beat_ids`。
- `production_screenplay` 是执行视图，必须能追溯到 `human_readable_screenplay`。
- 英文版和中文版必须保持完全相同的结构与 ID。

### 必须派生的生产信息

- `global_plan`：故事摘要、主题、约束、叙事弧。
- `main_entities`：从 `cast_and_assets` 派生，包含外观、资产类型、初始状态。
- `scenes`：叙事 scene 的生产边界，包含 scene 级 `visual_track.background` 和 `audio_track.ambience/bgm`。
- `shots`：从 beats 派生的镜头执行单元，包含 `visual_track.actions/cinematography`、`audio_track.dialogue/foley/voiceover`、`duration_sec`、`transition`。
- `planned_assets`：每个 shot 需要的实体、用途、`preserve | transform | deprecate | avoid`、是否 required。
- `video_prompts` 与 `first_frame_prompts` 只保留 TODO scaffold，不在这里自由生成最终 prompt。

### Production JSON Schema

```json
{
  "story_id": "0000_example",
  "language": "en | zh",
  "story_name": "Title",
  "story_overview": "overview",
  "global_plan": {
    "title": "Title",
    "synopsis": "synopsis",
    "theme": "theme",
    "narrative_arc": [
      "Setup: ...",
      "Rising Action: ...",
      "Climax: ...",
      "Resolution: ..."
    ],
    "constraints": [
      "global visual or continuity constraint"
    ]
  },
  "main_entities": [
    {
      "entity_id": "E1",
      "name": "entity name",
      "entity_type": "character | object | location",
      "appearance": "production-friendly visual description",
      "initial_state": "initial state"
    }
  ],
  "production_screenplay": {
    "scenes": [
      {
        "scene_id": "S1",
        "source_scene_id": "S1",
        "scene_title": "scene title",
        "t_start": 0.0,
        "t_end": 15.0,
        "entities_present": ["E1"],
        "visual_track": {
          "background": "location, atmosphere, lighting"
        },
        "audio_track": {
          "ambience": "continuous environmental sound",
          "bgm": "continuous score or empty"
        }
      }
    ],
    "shots": [
      {
        "shot_id": "shot_0001",
        "scene_id": "S1",
        "source_beat_ids": ["B001"],
        "transition": "cut | continue | time_jump | scene_change",
        "duration_sec": 5.0,
        "narrative_goal": "what this shot accomplishes",
        "visual_track": {
          "actions": [
            "visual action"
          ],
          "cinematography": {
            "shot_type": "wide shot | close-up | etc.",
            "camera_movement": "camera movement",
            "lighting": "lighting"
          }
        },
        "audio_track": {
          "dialogue": [
            {
              "entity_id": "E1",
              "text": "dialogue line",
              "expression": "performance cue"
            }
          ],
          "foley": "shot-synchronous physical sound",
          "voiceover": ""
        },
        "active_characters": ["E1"],
        "continuity_requirements": [
          "continuity detail"
        ],
        "planned_assets": [
          {
            "planned_asset_id": "E1",
            "asset_kind": "character | object | location | style | motion",
            "intended_uses": ["appearance", "identity", "scene", "motion", "state"],
            "operation": "preserve | transform | deprecate | avoid",
            "required": true
          }
        ],
        "video_prompts": [
          "TODO: generated by downstream prompt adapter"
        ],
        "first_frame_prompts": [
          "TODO: generated by downstream prompt adapter"
        ]
      }
    ]
  },
  "traceability": {
    "human_layer_source": "confirmed human_readable_screenplay",
    "conversion_notes": [
      "important deterministic or semantic conversion decisions"
    ]
  }
}
```

现在，请根据已确认的 human-readable screenplay 输出 production-friendly screenplay JSON。
````

## 题材库

网页端批量创作时可从以下题材中选择，也可以自由发挥，但必须保持原创：

- **科幻悬疑 (Sci-Fi Mystery)**：空间站幽灵信号、发光晶体、故障控制室。
- **奇幻冒险 (Fantasy Quest)**：炼金术士、魔法沙漏、古老遗迹。
- **都市谍战 (Espionage Drama)**：雨夜站台、金属手提箱、调包与追踪。
- **古装历史剧 (Historical Drama)**：宫廷毒药案、玉佩、御膳房与宫殿。
- **赛博朋克侦探 (Cyberpunk Noir)**：义体侦探、记忆芯片、霓虹雨夜。
- **末日生存 (Post-Apocalyptic Survival)**：幸存者、防毒面具、废弃加油站。
- **武侠/仙侠 (Wuxia / Xianxia)**：竹林决斗、古剑破损、师徒宿怨。
- **家庭/成长剧情 (Family Drama)**：旧宅重逢、遗物、跨代秘密。
- **海洋惊悚 (Ocean Thriller)**：灯塔、失踪船员、风暴与求救信号。

## 质量检查

产出后应人工或脚本检查：

- 双语文件结构和 ID 是否完全对齐。
- Human layer 是否仍像可读剧本，而不是生产字段堆叠。
- Production layer 是否全部可追溯到 scene/beat。
- 是否包含非相邻实体复用、地点回访、生命周期变化和 forbidden/avoid 约束。
- Scene 级 ambience/bgm 与 shot 级 dialogue/foley/voiceover 是否分清。
