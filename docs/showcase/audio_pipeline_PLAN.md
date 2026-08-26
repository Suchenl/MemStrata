# MemStrata Showcase 生产级音频方案

> 目标：把 `production_screenplay.audio_track` 从“gTTS 对白 mux demo”升级为可替换后端的三轨音频流水线：对白 TTS、配乐 BGM、环境音/foley，最终与 `production/outputs/<movie>/memstrata/<variant>/review/long_video.mp4` 精确 mux。
>
> 本方案只面向 showcase / project page 生产，不改变 `src/memstrata/`、评分或 benchmark 协议。

## 0. 当前现状与约束

现有脚本：`scripts/showcase/mux_dialogue_audio.py`。

当前行为：

- 从 `data/Screenplay/products/cn/*.json` 的 `production_screenplay.shots[*].audio_track.dialogue[]` 抽取对白。
- 使用 `gTTS` 或 `espeak-ng` 生成单句音频。
- 按 shot `duration_sec` 累积计划时间，再把总时长等比缩放到实际视频时长。
- 用 `ffmpeg adelay + amix` 叠到静音底轨，最后 mux 到 long video。

主要问题：

- 声音自然度不足：gTTS 机械、不可控、角色音色不可分。
- 情绪表达不足：虽然 screenplay 有 `expression`，现有后端未使用。
- 缺少配乐和环境音：scene 级 `ambience` / `bgm`、shot 级 `foley` / `voiceover` 未真正生成。
- 时间轴粗对齐：只按总时长缩放，未用 scene `t_start` / `t_end`，也没有字幕/字词级时间戳。
- 无口型同步：对白音频不会修正角色特写镜头的嘴形。

本地资源检查：

- `models/model_weights/local_paths.md` 当前登记的 `PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}` 没有 TTS / text-to-music / text-to-audio 专用权重。
- 对 `${PUBLIC_MODELS_ROOT}` 做过 Ceph 文件数预检：`ceph.dir.rfiles="3887"`，低于 1,000,000 阈值；有针对性匹配未发现 CosyVoice、Fish-Speech、IndexTTS、MusicGen、AudioLDM、TangoFlux、ACE-Step 等音频权重。
- 结论：生产级音频需要新下载权重，统一放到 `${PUBLIC_MODELS_ROOT}/<org>/<repo>`。

## 1. 三条音轨的生产级选型

### 1.1 对白 TTS

需求：中文自然度高；能用 `expression` 控制情绪；支持多角色音色区分；最好可 zero-shot voice cloning；可离线批量生成。

| 候选 | 推荐级别 | 适配度 | 授权/中文支持 | 说明 |
|---|---:|---|---|---|
| `FunAudioLLM/CosyVoice2-0.5B` + `FunAudioLLM/CosyVoice-ttsfrd` | 首选 MVP | 高 | Apache-2.0；中文强，支持 9 种语言和多种中文方言/口音；支持 instruct 情绪、语速、音量、方言控制；支持 zero-shot / cross-lingual voice cloning | 参数量小、部署门槛低、许可友好、足够替换 gTTS。`expression` 可转成自然语言指令或控制 token，例如“低沉、克制”“轻声、坚定”。 |
| `IndexTeam/IndexTTS-2` | 完整版强候选 | 很高 | bilibili Model Use License；中文/英文强；支持 zero-shot 音色克隆、独立情绪控制和 duration-control | 对视频配音特别有价值：可指定生成 token 数/时长，更适合卡 shot 时间窗。许可不是标准 Apache/MIT，大规模商业主体需单独确认。 |
| `fishaudio/s2-pro` | 高质量候选 | 高 | Fish Audio Research License；中文属于强支持；支持自然语言情绪标签、multi-speaker/multi-turn、10-30s 参考音色克隆 | 质量和流式能力强，但权重更大，许可对商业使用更敏感。适合作为高级版本或对比后端。 |
| `RVC-Boss/GPT-SoVITS` / `SWivid/F5-TTS` | 备选 | 中高 | GPT-SoVITS 代码 MIT，中文/粤语/日英韩强；F5-TTS 代码 MIT 但官方权重 CC-BY-NC 4.0 | GPT-SoVITS 很适合少样本角色音色，但批处理和情绪控制工程成本更高。F5-TTS 零样本自然，但官方权重非商用，基础模型显式情绪控制弱于 CosyVoice2 / IndexTTS2。 |

推荐：

- MVP：`CosyVoice2-0.5B`。理由是中文强、许可友好、GPU 成本低、情绪 instruction 与 screenplay `expression` 对接直接。
- 完整版：评测 `IndexTTS-2`。理由是 duration-control 对时间轴和口型前处理最有价值。
- 高级/角色音色：`CosyVoice2` 或 `IndexTTS2` 使用每个 `entity_id` 的参考音频；如果需要更强流式或多轮角色对话，再加入 `Fish-Speech S2-Pro`。

### 1.2 配乐 BGM

需求：根据 scene `audio_track.bgm` 生成 10-30s 以上音乐床；可循环/续写；风格一致；不含人声优先；可被对白 sidechain ducking。

| 候选 | 推荐级别 | 适配度 | 授权/中文支持 | 说明 |
|---|---:|---|---|---|
| `ACE-Step/acestep-v15-sft` 或 `ACE-Step/acestep-v15-xl-turbo-diffusers` | 首选 | 高 | MIT；文本到音乐，官方标注 50+ 语言；面向商用品质音乐生成 | 许可最友好，生成时长可覆盖 scene，适合“低沉弦乐长音”“温暖内敛钢琴”等 scene prompt。建议把中文 `bgm` 规范化/翻译为英文音乐制作 prompt 后生成。 |
| `facebook/musicgen-medium` / `facebook/musicgen-large` | 备选 | 中高 | 权重 CC-BY-NC 4.0；英文 prompt 更稳 | 老牌稳定，生态成熟，适合研究展示；但非商用许可不适合公开商业传播，长结构一致性一般。 |
| `stabilityai/stable-audio-open-1.0` | 备选/短音乐素材 | 中 | Stability AI Community License；英文 prompt；更偏短样本、音效和 field recording | 可生成短音乐 riff / production elements，不建议作为完整 score 主后端。 |

推荐：

- BGM 主后端用 `ACE-Step`。它的许可、长时长能力、文本到音乐能力更适合 showcase 成片。
- 如果只做论文 demo 或内部研究展示，可保留 `MusicGen` 作为低风险成熟备选。

### 1.3 环境音 / Foley

需求：根据 scene `ambience` 生成连续环境床，根据 shot `foley` 生成短音效；应避免压对白；支持雷雨、海浪、脚步、纸页、玻璃碎裂等常见影视声音。

| 候选 | 推荐级别 | 适配度 | 授权/中文支持 | 说明 |
|---|---:|---|---|---|
| `stabilityai/stable-audio-open-1.0` 或 `stabilityai/stable-audio-open-small` | 首选 | 高 | Stability AI Community License；英文 prompt 更稳；模型卡说明更擅长 sound effects / field recordings than music | 对 ambience / foley 比对 BGM 更适合。可生成海风、浪、雷、玻璃、脚步、机械声等，输出 44.1kHz stereo。 |
| `cvssp/audioldm2-large` / `cvssp/audioldm2` | 备选 | 中高 | CC-BY-NC-SA-4.0；英文 prompt 更稳；支持 text-to-audio / sound effects / music | Diffusers 集成成熟，适合快速落地；许可非商用且 SA 传播约束更强。 |
| `declare-lab/TangoFlux` | 备选/快速实验 | 中高 | 非商业研究用途；英文 prompt；44.1kHz，最长 30s | 速度快、音效质量好，但许可限制较强。 |
| 免费音效库检索（Freesound / Pixabay / 自建 CC0 库） | 生产补充 | 高 | 取决于素材；优先 CC0 / CC-BY | 对“钟声、脚步、纸页、海鸥”等具体 foley，检索库常比生成模型稳定。建议完整版本加入 `sfx_retrieval` 后端作为 deterministic fallback。 |

推荐：

- 环境音主后端：`Stable Audio Open`。
- Foley：先用 `Stable Audio Open` 生成，失败或重复 artefact 明显时走免费音效库检索。
- 所有 SFX prompt 最好自动翻译/规范化为英文，例如“碎浪拍岩、海鸥鸣叫”转成 “clear coastal waves crashing against rocks, distant seagulls, no music, no speech”。

## 2. 权重下载清单

统一下载到：

```bash
export PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}
```

推荐优先级表：

| 优先级 | 模型名 | Repo id | 大致体积 | 用途 | 中文支持 | 许可备注 |
|---:|---|---|---:|---|---|---|
| P0 | CosyVoice2 | `FunAudioLLM/CosyVoice2-0.5B` | 约 1-2 GB | MVP 对白 TTS、情绪 instruction、角色音色参考 | 强 | Apache-2.0 |
| P0 | CosyVoice 文本前端资源 | `FunAudioLLM/CosyVoice-ttsfrd` | 约数百 MB 到 1 GB | 中文文本规范化/前端增强；未安装时可用 WeTextProcessing 兜底 | 强 | 随 CosyVoice 资源使用；按模型卡确认 |
| P1 | IndexTTS2 | `IndexTeam/IndexTTS-2` | 约 4-8 GB | 高质量 TTS、音色/情绪解耦、duration-control | 强 | bilibili Model Use License，大规模商业使用需确认 |
| P1 | ACE-Step SFT | `ACE-Step/acestep-v15-sft` | 约 4-8 GB | 主 BGM：scene 级 text-to-music | 可用，建议英文 prompt | MIT |
| P1 | ACE-Step Turbo Diffusers | `ACE-Step/acestep-v15-xl-turbo-diffusers` | 约 4-8 GB | 快速 BGM 草稿或多候选采样 | 可用，建议英文 prompt | MIT |
| P1 | Stable Audio Open | `stabilityai/stable-audio-open-1.0` | 约 5-10 GB | 环境音、foley、production elements | 英文 prompt 更稳 | Stability AI Community License |
| P1 | Stable Audio Open Small | `stabilityai/stable-audio-open-small` | 约 1-3 GB | 快速 SFX / foley 试跑 | 英文 prompt 更稳 | Stability AI Community License |
| P2 | Fish-Speech S2-Pro | `fishaudio/s2-pro` | 约 8-16 GB | 高级 TTS、流式、多说话人、多情绪标签 | 强 | Fish Audio Research License；商业需另行授权 |
| P2 | AudioLDM2 Large | `cvssp/audioldm2-large` | 约 5-8 GB | SFX / ambience 备选 | 英文 prompt 更稳 | CC-BY-NC-SA-4.0 |
| P2 | AudioLDM2 Base | `cvssp/audioldm2` | 约 3-6 GB | SFX / ambience 快速备选 | 英文 prompt 更稳 | CC-BY-NC-SA-4.0 |
| P2 | MusicGen Medium | `facebook/musicgen-medium` | 约 3-5 GB | BGM 备选 | 英文 prompt 更稳 | 权重 CC-BY-NC 4.0 |
| P2 | MusicGen Large | `facebook/musicgen-large` | 约 6-10 GB | BGM 高质量备选 | 英文 prompt 更稳 | 权重 CC-BY-NC 4.0 |
| P2 | TangoFlux | `declare-lab/TangoFlux` | 约 2-4 GB | 快速 text-to-audio SFX 备选 | 英文 prompt 更稳 | 非商业研究用途 |
| P3 | MFA 中文声学模型 | `mandarin_mfa` via MFA model manager | < 500 MB | 中文强制对齐，产出字/词/音素时间戳 | 强 | CC BY 4.0 |
| P3 | MFA 中文词典 | `mandarin_mfa` / `mandarin_china_mfa` dictionary | < 500 MB | 中文强制对齐词典 | 强 | CC BY 4.0 |
| P3 | LatentSync | `ByteDance/LatentSync-1.6` | 约 2-5 GB | 可选口型同步，仅用于角色特写 | 对中文视频有改进版本 | 代码 Apache-2.0；权重 OpenRAIL++，需单独确认 |

示例下载命令：

```bash
export PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}/Audio

hf download FunAudioLLM/CosyVoice2-0.5B \
  --local-dir "${PUBLIC_MODELS_ROOT}/FunAudioLLM/CosyVoice2-0.5B"
hf download FunAudioLLM/CosyVoice-ttsfrd \
  --local-dir "${PUBLIC_MODELS_ROOT}/FunAudioLLM/CosyVoice-ttsfrd"

hf download IndexTeam/IndexTTS-2 \
  --local-dir "${PUBLIC_MODELS_ROOT}/IndexTeam/IndexTTS-2"

hf download ACE-Step/acestep-v15-sft \
  --local-dir "${PUBLIC_MODELS_ROOT}/ACE-Step/acestep-v15-sft"
hf download ACE-Step/acestep-v15-xl-turbo-diffusers \
  --local-dir "${PUBLIC_MODELS_ROOT}/ACE-Step/acestep-v15-xl-turbo-diffusers"

hf download stabilityai/stable-audio-open-1.0 \
  --local-dir "${PUBLIC_MODELS_ROOT}/stabilityai/stable-audio-open-1.0"
hf download stabilityai/stable-audio-open-small \
  --local-dir "${PUBLIC_MODELS_ROOT}/stabilityai/stable-audio-open-small"
```

注意：

- 上表体积是规划级估计，最终以 HF snapshot 实际文件大小为准。
- 长下载应按 `models/model_weights/local_paths.md` 规则放后台执行，不写入项目目录或 `~/.cache` 的随机位置。
- 下载后建议补充 `models/model_weights/local_paths.md` 的 Audio / Music / SFX 小节，方便后续 agent 复用。

## 3. 时间轴、强制对齐与口型

### 3.1 时间轴不再只做全片等比缩放

建议生成一个独立 `timeline.json`，把 screenplay 计划时间和实际视频时间分开：

- `scene_timeline_planned`：来自 `production_screenplay.scenes[*].t_start/t_end`。
- `shot_timeline_planned`：来自 `production_screenplay.shots[*].duration_sec` 累积。
- `video_timeline_actual`：来自实际 long video 总时长；如果未来有每 shot clip manifest，则使用真实 clip start/end；没有时按 scene 或 shot 局部缩放。
- `cue_timeline`：对白、voiceover、foley、ambience、bgm 的最终 `t_start/t_end`。

原则：

- scene 级 ambience/bgm 使用 scene `t_start/t_end` 归一化到实际视频时间，保证场景氛围与画面段落一致。
- shot 级 dialogue/foley 使用 shot 时间窗，避免全片缩放导致后段 drift。
- TTS 先生成自然语速，再比较目标窗长：
  - 若差距在 5-8% 内，用 `ffmpeg atempo` 或高质量 time-stretch 微调。
  - 若差距更大，优先重生成：给 TTS 后端传 `speed` / `duration` / 更短停顿指令。
  - 避免把一句话强拉到不自然长度。

### 3.2 强制对齐生成字幕时间戳

比“按 shot 起点放整句”更好的做法：

1. 对每句 TTS 输出 wav 和原文运行强制对齐。
2. 产出 `dialogue_alignment.json`：
   - `entity_id`
   - `shot_id`
   - `text`
   - `utterance_start/end`
   - `char_segments` 或 `word_segments`
   - `confidence`
3. 用这些时间戳做三件事：
   - 生成字幕 `.srt` / `.ass`，供 showcase 检查。
   - 让 foley 避开对白重音处。
   - 口型同步只截取有可见嘴部的对白片段。

中文对齐推荐：

- 首选 Montreal Forced Aligner：`mandarin_mfa` acoustic model + dictionary，许可 CC BY 4.0。
- 轻量备选：aeneas 可做句级/短语级，但中文分词和音素级精度通常不如 MFA。
- 如果 TTS 后端能直接返回 token/phoneme duration，优先保存原生 duration；MFA 作为校准和质检。

### 3.3 可选口型同步

不要全片强行口型同步，只处理“角色嘴部可见且对白明显”的镜头：

- 先从 shot metadata / video analysis 标出 `lip_sync_required=true`：
  - 中近景、近景、特写；
  - active character 包含 dialogue 的 `entity_id`；
  - 人脸/嘴部可见。
- 对这些片段裁剪出视频窗口和对应 dialogue wav。
- 使用 LatentSync 或 Wav2Lip 类后端生成修正片段。
- 修正后按原时间窗 splice 回 long video，再 mux 全局混音。

推荐口型后端：

- `ByteDance/LatentSync-1.6`：画质和中文视频适配更好，适合高级阶段；权重许可需单独确认。
- Wav2Lip：工程成熟、速度快，但画质和生成视频风格融合较弱，适合作为兜底。

风险：

- 当前 long video 是生成视频，人物身份和面部稳定性不一定足够；口型同步可能引入脸部漂移。
- 只对少数 showcase hero shots 做 lip sync，整体收益/风险比更好。

## 4. 混音与 mux 设计

目标混音结构：

```text
dialogue_bus: per-shot TTS + voiceover
bgm_bus:      per-scene music bed, loop/extend/crossfade
amb_bus:      per-scene ambience bed
foley_bus:    per-shot foley/event sounds
master_bus:   loudnorm + limiter
```

推荐 ffmpeg 策略：

- 统一采样率：中间 wav 统一到 48kHz，最终 AAC 192k 或 256k。
- 对白优先：dialogue 保持清晰，峰值约 -3 dBFS，loudness 可目标 -18 到 -16 LUFS。
- BGM ducking：用 dialogue bus 做 sidechain，让 BGM 在对白时自动降低 6-10 dB。
- Ambience 更低：常态 -24 到 -30 LUFS，避免遮盖对白。
- Foley 局部突出：短音效按 cue 放置，可加 `afade` 避免切口。
- Master：`loudnorm` + `alimiter`，避免 amix 叠加爆音。

示意 filter：

```bash
# 概念示意，实际由 pipeline 生成 filter_complex
[bgm]sidechaincompress=threshold=0.04:ratio=8:attack=20:release=300[bgm_ducked];
[dialogue][bgm_ducked][amb][foley]amix=inputs=4:duration=first:dropout_transition=0,
loudnorm=I=-16:TP=-1.5:LRA=11,
alimiter=limit=0.95[master]
```

## 5. `scripts/showcase/` 落地架构

建议不要继续把所有逻辑塞进 `mux_dialogue_audio.py`，而是在 `scripts/showcase/audio_pipeline/` 下拆成可替换后端。先新增 showcase 专用代码，不 import `src/memstrata/` 或 `src/vmem_bench/`。

建议目录：

```text
scripts/showcase/audio_pipeline/
  __init__.py
  cli.py
  screenplay_io.py
  timeline.py
  prompts.py
  mixer.py
  mux.py
  schemas.py
  backends/
    tts_base.py
    tts_cosyvoice.py
    tts_indextts.py
    music_base.py
    music_acestep.py
    sfx_base.py
    sfx_stable_audio.py
    align_mfa.py
    lipsync_latentsync.py
```

建议产物目录：

```text
production/outputs/_showcase_audio/<story_id>/<variant>/
  audio_plan.json
  timeline.json
  dialogue/
    shot_0003_E1_000.wav
    shot_0007_E1_000.wav
  bgm/
    S1.wav
    S2.wav
  ambience/
    S1.wav
    S2.wav
  foley/
    shot_0001.wav
  align/
    dialogue_alignment.json
    subtitles.srt
  mix/
    dialogue_bus.wav
    bgm_bus.wav
    ambience_bus.wav
    foley_bus.wav
    master.wav
  mux/
    long_video_with_audio.mp4
```

核心接口伪代码：

```python
@dataclass
class AudioCue:
    cue_id: str
    kind: Literal["dialogue", "voiceover", "bgm", "ambience", "foley"]
    scene_id: str
    shot_id: str | None
    entity_id: str | None
    text: str
    expression: str | None
    t_start: float
    t_end: float
    target_lufs: float

class TTSBackend(Protocol):
    requires_gpu: bool

    def synthesize(
        self,
        cue: AudioCue,
        voice_profile: VoiceProfile,
        out_wav: Path,
        target_duration_sec: float | None = None,
    ) -> AudioAsset:
        ...

class MusicBackend(Protocol):
    requires_gpu: bool

    def generate_scene_bgm(
        self,
        scene: SceneAudioSpec,
        out_wav: Path,
        duration_sec: float,
    ) -> AudioAsset:
        ...

class SFXBackend(Protocol):
    requires_gpu: bool

    def generate_sfx(
        self,
        cue: AudioCue,
        out_wav: Path,
        duration_sec: float,
    ) -> AudioAsset:
        ...
```

流水线入口：

```python
def run_audio_pipeline(args: Args) -> Path:
    screenplay = load_screenplay(args.screenplay)
    video_info = probe_video(args.video, args.ffprobe)

    plan = build_audio_plan(screenplay, video_info, args.variant)
    cues = build_timeline(plan, strategy=args.timeline_strategy)

    dialogue_assets = tts_backend.batch_synthesize(cues.dialogue)
    bgm_assets = music_backend.batch_generate(cues.bgm)
    ambience_assets = sfx_backend.batch_generate(cues.ambience)
    foley_assets = sfx_backend.batch_generate(cues.foley)

    alignments = aligner.align_dialogue(dialogue_assets)
    master_wav = mixer.mix(
        dialogue=dialogue_assets,
        bgm=bgm_assets,
        ambience=ambience_assets,
        foley=foley_assets,
        alignments=alignments,
    )
    return mux_video_audio(args.video, master_wav, args.out_mp4)
```

GPU 标注：

- 需要 GPU：CosyVoice2 / IndexTTS2 / Fish-Speech / ACE-Step / Stable Audio Open / AudioLDM2 / TangoFlux / LatentSync。
- 可 CPU：screenplay 解析、timeline 生成、prompt 规范化、ffprobe、ffmpeg 混音/mux、SRT 生成。
- MFA：可 CPU，但批量时建议单独环境，避免污染 TTS 环境。

后端替换原则：

- 每个后端只接收纯 `AudioCue` / `SceneAudioSpec`，不直接读 screenplay。
- 每次生成写入 `audio_plan.json` 和 `manifest.json`，记录 repo id、local path、seed、prompt、duration、采样率、许可证备注。
- 失败时允许 fallback：TTS 可从 IndexTTS2 fallback 到 CosyVoice2；SFX 可从 Stable Audio fallback 到检索库；BGM 可从 ACE-Step fallback 到 MusicGen 或静音占位。

## 6. 分阶段路线

### 阶段 A：MVP（1-2 天）

目标：肉眼/耳朵明显优于当前 gTTS demo。

内容：

- 下载 `FunAudioLLM/CosyVoice2-0.5B` 和 `FunAudioLLM/CosyVoice-ttsfrd`。
- 新增 CosyVoice2 TTS backend，把 `entity_id + expression + text` 转为自然中文配音。
- 为 E1 / E2 配置默认 voice profile；没有参考音频时先用内置/zero-shot prompt，后续再替换为角色参考音。
- 重写 mix 层：保留现有 `adelay` 思路，但加入 `loudnorm`、limiter、dialogue bus、预留 sidechain ducking。
- 产出 `manifest.json`、`subtitles.srt`、`long_video_with_audio.mp4`。

预期效果：

- 机械音显著降低。
- 角色之间至少有音色/语气区分。
- 对白响度稳定，不再因为 amix 叠加爆音。

### 阶段 B：完整版（3-5 天）

目标：形成完整电影感音频床。

内容：

- 下载 `ACE-Step` 和 `Stable Audio Open`。
- scene `bgm` 生成 per-scene music bed；scene 边界做 crossfade。
- scene `ambience` 生成连续环境床；shot `foley` 生成短音效。
- 引入 prompt adapter，把中文音频字段翻译/规范化为英文音频 prompt，并保存原文/英文双语 manifest。
- 引入 MFA 对齐，生成字/词级字幕时间戳。
- 用 dialogue sidechain ducking 压 BGM，保证对白清晰。
- 增加人工快速审核清单：对白是否错字、BGM 是否有人声、SFX 是否突兀、响度是否爆音。

预期效果：

- 从“有对白的视频”升级为“有配乐、环境、动作声的 showcase 成片”。
- 时间轴和字幕可检查，后续可用于网页展示。

### 阶段 C：高级版（5-10 天）

目标：提升角色一致性和视觉-听觉同步。

内容：

- 为每个 `entity_id` 建立 `voice_profile.json`：参考音频、年龄/性别/性格、默认语速、默认情绪强度。
- 评测 `IndexTTS2` duration-control，对需要卡口型的镜头优先使用。
- 对角色特写镜头跑 LatentSync，生成 lip-synced video segments。
- 引入免费音效库检索 fallback，替换生成模型不稳定的 foley。
- 对 BGM 做主题复用：同一 story 的主旋律 prompt / reference audio 在多个 scene 中延续。

预期效果：

- 角色声音随剧情年龄/状态变化而保持身份一致。
- 重点对白镜头有口型同步。
- BGM 有主题延续，不只是独立短段拼接。

## 7. 推荐的 MVP 第一步

第一步只做一件事：把 gTTS 换成 CosyVoice2，并同步把混音 bus 抽出来。

具体执行：

1. 下载：

```bash
export PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}
hf download FunAudioLLM/CosyVoice2-0.5B \
  --local-dir "${PUBLIC_MODELS_ROOT}/FunAudioLLM/CosyVoice2-0.5B"
hf download FunAudioLLM/CosyVoice-ttsfrd \
  --local-dir "${PUBLIC_MODELS_ROOT}/FunAudioLLM/CosyVoice-ttsfrd"
```

2. 新增 showcase 专用 CLI：

```bash
python -m scripts.showcase.audio_pipeline.cli \
  --screenplay data/Screenplay/products/cn/0001_lighthouse_keeper.json \
  --video production/outputs/0001_lighthouse_keeper/memstrata/optCA/review/long_video.mp4 \
  --tts-backend cosyvoice2 \
  --out-dir production/outputs/_showcase_audio/0001_lighthouse_keeper/optCA
```

3. 保留旧 `mux_dialogue_audio.py` 作为 demo fallback，不在第一步删除。

4. 第一版验收标准：

- 三句样例对白都由 CosyVoice2 生成。
- `E1` 与 `E2` 声音不同，且 `expression` 至少影响语气。
- 输出 mp4 有音轨，响度稳定，无明显削波。
- `manifest.json` 记录每句的 `entity_id`、`expression`、TTS backend、local model path、planned/actual time。

