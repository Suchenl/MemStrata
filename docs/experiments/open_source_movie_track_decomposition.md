# 开源影片结构化拆解方案调研

## 1. 目标定义

本文关注这样一类任务：

> 将一部影片自动拆解为剧本、场景、镜头、角色、物体、动作、对白、音频事件等多个相互对齐的结构化 Track。

理想输出不仅是一段视频描述，而应包含以下信息：

- 全局故事和风格
- 场景边界与地点
- 镜头边界与镜头语言
- 角色、物体和地点的永久实体 ID
- 角色与物体的视觉轨迹
- 动作、事件与状态变化
- 对白、说话人和时间戳
- 背景音乐、环境声和音效
- 不同 Track 之间的时间和身份关联

目前，完整解决上述问题的成熟开源系统仍然不存在。较现实的方案是组合多个开源模型，分别负责剧本式描述、视觉实体跟踪和角色身份统一。

---

## 2. 当前最值得关注的开源方案

### 2.1 TimeChat-Captioner

**定位：影片到结构化音视频描述的主干模型。**

TimeChat-Captioner 是目前最接近“影片自动拆解为剧本式时间轴”的开源方案之一。它公开了：

- 推理代码
- 训练代码
- 7B 模型权重
- TimeChatCap-40K 训练数据
- OmniDCBench 评测集
- BSD-3-Clause 许可证

模型名称：

```text
yaolily/TimeChat-Captioner-GRPO-7B
```

#### 输出内容

它可以对视频进行场景级拆解，并生成以下信息：

1. Audio-Visual Events：人物或物体发生了什么
2. Background / Environment：场景、地点、环境和空间背景
3. Camera State：运镜、景别和机位
4. Multi-shot Editing：镜头切换和剪辑方式
5. Dialogue：对白内容
6. Acoustic Cues：背景音乐、音效和环境声

示意输出：

```json
{
  "start": "00:12.0",
  "end": "00:18.5",
  "event": "A woman enters the room and looks toward the window.",
  "background": "A dimly lit bedroom at night.",
  "camera": "Medium shot, slowly panning right.",
  "editing": "Cuts from an exterior establishing shot.",
  "dialogue": "Is anyone here?",
  "audio": "Door creak, footsteps, low background music."
}
```

#### 优势

- 同时理解视频画面与音频
- 输出带时间区间的结构化描述
- 能覆盖事件、对白、声音和镜头语言
- 模型、代码和训练数据均已公开
- 适合作为 Scene、Event、Dialogue、Audio 和 Camera Track 的基础模型

#### 局限

- 官方更推荐处理约一分钟的视频片段
- 长影片需要提前切分
- 不维护可靠的全片永久角色 ID
- 不输出逐帧人物或物体的 bbox、mask
- 角色换装、跨镜头重识别能力有限
- 输出仍然偏“结构化描述”，而不是严格分离的多轨数据库

#### 建议用途

将其作为整个系统的语义主干，负责：

```text
Scene Track
Shot Description Track
Event Track
Dialogue Track
Audio Track
Camera Track
```

---

### 2.2 CaptionFormer

**定位：视觉实体检测、分割、跟踪和描述。**

CaptionFormer 是一个端到端的 Dense Video Object Captioning 模型，可以同时完成：

```text
Detect → Segment → Track → Caption
```

它公开了：

- 推理代码
- 训练代码
- 模型 checkpoint
- 评测脚本
- Demo
- Apache-2.0 许可证

#### 输出内容

对于每一个人物或物体轨迹，它可以输出：

```json
{
  "track_id": 17,
  "category": "person",
  "caption": "A child in a red jacket runs along the river bank.",
  "bboxes": [],
  "masks": []
}
```

#### 优势

- 输出真实的视觉空间 Track
- 支持人物和物体检测
- 支持逐帧 bbox
- 支持逐帧 mask
- 支持视频轨迹跟踪
- 支持轨迹级自然语言描述
- 可以弥补纯视频语言模型缺少视觉 grounding 的问题

#### 局限

- 不重点处理对白、说话人和音乐
- 不负责全局故事理解
- Track ID 通常只在局部 clip 内有效
- 跨场景、跨换装、跨长时间的角色身份一致性仍需额外模块
- 不生成完整剧本

#### 建议用途

将其作为：

```text
Entity Track
Visual Grounding Track
Object Track
Character Spatial Track
```

---

### 2.3 AutoAD-Zero

**定位：角色命名、Character Bank 和电影级描述。**

AutoAD-Zero 面向电影 Audio Description，提供了角色识别和角色命名能力。它公开了：

- 官方代码
- Character Recognition 模块
- 电影描述与摘要流程
- Apache-2.0 许可证

其典型输入包括：

- 角色名称
- 演员或角色参考图
- 当前片段中的人脸检测结果
- 视频片段

它可以将：

```text
A woman walks into the room.
```

转换为：

```text
Rachel walks into the room.
```

#### 优势

- 支持 Character Bank
- 可以根据演员或角色参考图进行身份命名
- 能把匿名人物描述替换为真实角色名
- 适合用作角色身份统一模块
- 支持电影级描述与摘要

#### 局限

- 往往依赖预先提供的演员表或角色参考图
- 更偏 Audio Description，而不是完整 screenplay
- 不输出多轨结构
- 不提供现代端到端长视频角色 tracking
- 对换装、遮挡和侧脸等情况仍可能不稳定

#### 建议用途

不建议将它作为整个系统的主干，而是拆出：

```text
Character Bank
Face Recognition
Character Naming
Identity Assignment
```

---

## 3. 其他相关开源方案

### 3.1 原始 TimeChat

原始 TimeChat 也公开了模型和代码，主要能力是：

- 时间感知视频理解
- Dense Video Captioning
- Temporal Grounding
- 视频问答

但相较于 TimeChat-Captioner：

- 音频建模较弱
- 输出结构不够丰富
- 模型架构和依赖相对较旧
- 不适合作为当前首选主干

更适合用作 temporal grounding baseline。

---

### 3.2 MovieChat

MovieChat 面向超长视频理解，主要支持：

- 长视频问答
- 视频摘要
- 长期记忆建模

它适合回答“整部影片讲了什么”，但不擅长输出：

- 完整时间轴
- 角色 Track
- 镜头 Track
- 音频 Track
- 逐帧视觉实体轨迹

因此不适合作为影片多轨拆解系统的核心模块。

---

### 3.3 Goldfish

Goldfish 通过长视频切片和检索实现任意长度视频问答。

其优势在于：

- 长视频检索
- 局部片段定位
- 面向长视频的问答和摘要

但它本质上更接近 Long Video RAG，而不是结构化拆解器。

---

### 3.4 StoryTeller

StoryTeller 的任务定义非常接近目标系统，包括：

- 场景切分
- 说话人聚类
- 角色识别
- 跨片段角色 ID 统一
- 长视频密集描述

但目前其关键特化 checkpoint 尚未完整公开，因此现阶段不能被视为完整可复现的开源方案。

---

### 3.5 LongVALE

LongVALE 提供了长音视频理解相关数据和部分资源。

但其完整自动标注 pipeline、推理模型和若干关键模块仍未全部公开，因此更适合作为数据资源，而不是直接运行的完整系统。

---

## 4. 推荐方案对比

| 方案 | 开源完整度 | 主要能力 | 主要缺陷 | 推荐级别 |
|---|---:|---|---|---:|
| TimeChat-Captioner | 高 | 场景、事件、对白、音频、镜头描述 | 无永久角色 ID，无 bbox/mask | 首选 |
| CaptionFormer | 高 | 检测、分割、跟踪、轨迹描述 | 不理解完整剧情和对白 | 首选补充 |
| AutoAD-Zero | 高 | Character Bank、角色命名 | 依赖参考图，模型相对较旧 | 建议拆模块 |
| TimeChat | 高 | 时间定位、Dense Caption | 音视频结构较弱 | Baseline |
| MovieChat | 高 | 长视频问答与摘要 | 不输出多 Track | 不作为主干 |
| Goldfish | 高 | 长视频检索和问答 | 偏 RAG，不是拆解 | 不作为主干 |
| StoryTeller | 中 | 角色一致性、长片描述 | 缺关键 checkpoint | 暂缓 |
| LongVALE | 中 | 数据和长音视频理解 | Pipeline 不完整 | 数据参考 |

---

## 5. 最推荐的组合式系统

目前最现实、能力最完整的开源组合是：

```text
完整影片
    │
    ├── Shot / Scene Detection
    │
    ├── TimeChat-Captioner
    │      ├── Scene Track
    │      ├── Event Track
    │      ├── Dialogue Track
    │      ├── Audio Track
    │      └── Camera Track
    │
    ├── CaptionFormer
    │      ├── Entity Track
    │      ├── Bounding Box Track
    │      ├── Mask Track
    │      └── Object / Person Caption Track
    │
    ├── Face Recognition / Re-ID
    │      ├── Cross-shot Character Linking
    │      ├── Character Naming
    │      └── Persistent Character ID
    │
    ├── Speaker Diarization
    │      └── Dialogue Speaker Assignment
    │
    └── Track Fusion
           ├── Temporal Alignment
           ├── Identity Alignment
           ├── Conflict Resolution
           └── Production Memory Construction
```

核心思路不是寻找一个万能模型，而是：

> 使用 TimeChat-Captioner 负责语义和音视频事件，使用 CaptionFormer 负责视觉实体空间轨迹，再通过人脸识别、说话人聚类和实体融合模块建立永久角色 ID。

---

## 6. 推荐的数据结构

最终可以统一为如下 JSON：

```json
{
  "global": {
    "title": "",
    "synopsis": "",
    "style": "",
    "duration": 0
  },
  "scenes": [
    {
      "scene_id": "",
      "start": 0,
      "end": 0,
      "location": "",
      "time_of_day": "",
      "environment": "",
      "mood": ""
    }
  ],
  "shots": [
    {
      "shot_id": "",
      "scene_id": "",
      "start": 0,
      "end": 0,
      "shot_size": "",
      "camera_motion": "",
      "camera_angle": "",
      "editing_transition": ""
    }
  ],
  "entities": [
    {
      "entity_id": "",
      "entity_type": "character | object | animal | location",
      "name": "",
      "semantic_description": "",
      "reference_images": [],
      "aliases": []
    }
  ],
  "visual_tracks": [
    {
      "track_id": "",
      "entity_id": "",
      "shot_id": "",
      "bboxes": [],
      "masks": [],
      "visual_caption": ""
    }
  ],
  "events": [
    {
      "event_id": "",
      "start": 0,
      "end": 0,
      "actor_id": "",
      "action": "",
      "target_id": "",
      "state_change": "",
      "confidence": 0
    }
  ],
  "dialogues": [
    {
      "start": 0,
      "end": 0,
      "speaker_id": "",
      "text": "",
      "emotion": "",
      "language": ""
    }
  ],
  "audio_events": [
    {
      "start": 0,
      "end": 0,
      "audio_type": "music | ambience | sound_effect",
      "description": "",
      "source_entity_id": ""
    }
  ]
}
```

---

## 7. 推荐的 Track 定义

### 7.1 Global Track

负责：

- 故事概要
- 题材
- 视觉风格
- 时代背景
- 全局世界设定

### 7.2 Scene Track

负责：

- 场景边界
- 地点
- 室内或室外
- 时间
- 环境
- 氛围

### 7.3 Shot Track

负责：

- 镜头起止时间
- 景别
- 运镜
- 机位
- 构图
- 转场

### 7.4 Entity Track

负责：

- 角色
- 道具
- 动物
- 地点
- 永久 ID
- Reference Image
- 语义描述

### 7.5 Visual Grounding Track

负责：

- bbox
- mask
- 轨迹
- 每帧可见性
- 遮挡状态

### 7.6 State Track

负责：

- 服装变化
- 姿态
- 情绪
- 手持物
- 所在地点
- 角色状态变化

### 7.7 Event Track

负责：

- 谁
- 在什么时间
- 做了什么
- 作用于谁
- 产生什么状态变化

### 7.8 Dialogue Track

负责：

- 说话人
- 对白文本
- 开始和结束时间
- 语气
- 情绪
- 语言

### 7.9 Audio Track

负责：

- 背景音乐
- 环境声
- 音效
- 声音来源
- 音画对应关系

---

## 8. 目前仍未解决的关键问题

### 8.1 长影片角色永久身份

当前模型很难稳定解决：

- 换装
- 年龄变化
- 光照变化
- 侧脸和背影
- 遮挡
- 多个相似角色
- 跨场景长时间消失后重新出现

局部 track ID 不等于全片永久角色 ID。

### 8.2 语义 Track 与视觉 Track 对齐

例如 TimeChat-Captioner 输出：

```text
A woman enters the room.
```

CaptionFormer 输出：

```text
track_id = 17
```

系统还需要判断：

```text
woman == track_id 17 == Character_A
```

该步骤需要跨模态实体对齐。

### 8.3 对白与角色匹配

ASR 和 Speaker Diarization 可以得到：

```text
speaker_03: "Where are you going?"
```

但仍需要确定：

```text
speaker_03 == Character_A
```

这通常需要联合使用：

- 面部可见性
- 口型
- 声纹
- 角色共现
- 剧情上下文

### 8.4 细粒度状态变化

现有模型很少稳定输出：

- 角色换衣服
- 道具从谁手里转移到谁手里
- 某物体被破坏或丢失
- 角色进入或离开某个地点
- 情绪、姿态和关系的持续变化

这部分对于 Production Memory 尤其重要。

### 8.5 长上下文误差累积

影片切片后处理会导致：

- 角色 ID 漂移
- 场景名称不一致
- 同一物体被重复创建
- 时间边界重叠或缺失
- 前后描述冲突

因此需要显式的全局 Memory Manager 和后处理纠错。

---

## 9. 实际落地建议

### 第一阶段：先实现语义拆解

优先跑通：

```text
Video
→ Scene / Shot Split
→ TimeChat-Captioner
→ Structured JSON
```

先获得：

- Scene
- Event
- Dialogue
- Audio
- Camera

### 第二阶段：加入视觉实体轨迹

接入 CaptionFormer：

```text
Clip
→ Person / Object Tracks
→ bbox / mask / caption
```

然后将视觉 Track 与语义 Event 对齐。

### 第三阶段：建立 Character Bank

增加：

- 人脸检测
- 人脸 embedding
- Person Re-ID
- Speaker Embedding
- Character Reference Image
- 全局聚类

为每个角色建立永久 ID。

### 第四阶段：构造 Production Memory

将所有结果融合为：

```text
Persistent Assets
+ Temporal Tracks
+ State Transitions
+ Identity Links
+ Event Relations
```

并支持后续查询：

- 当前镜头出现了哪些角色？
- 某个角色此前穿什么衣服？
- 某个道具最后出现在哪里？
- 下一段生成应该引用哪些角色和场景资产？
- 某个事件是否与已有剧情冲突？

---

## 10. 最终结论

目前没有一个开源模型能够独立、高质量地完成完整影片的多 Track 拆解。

现阶段最合理的开源方案是：

```text
TimeChat-Captioner
+ CaptionFormer
+ Character Bank
+ Face / Speaker Clustering
+ Global Track Fusion
```

其中：

- **TimeChat-Captioner** 负责剧本式语义、事件、对白、音频和镜头描述
- **CaptionFormer** 负责人物和物体的 bbox、mask、轨迹和局部描述
- **AutoAD-Zero 或自定义 Character Bank** 负责角色命名和身份统一
- **额外的全局融合模块** 负责建立跨镜头、跨场景的一致 Production Memory

如果目标是为长视频生成构造 Production Memory 数据，这一组合比单纯的长视频 caption 更合适，因为它同时包含：

- 语义信息
- 时间信息
- 身份信息
- 视觉空间信息
- 可持续更新的实体状态
