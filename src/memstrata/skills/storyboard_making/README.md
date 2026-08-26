# 故事板创作 Skill (Storyboard Making Skill)

> **核心定义**：本 Skill 规范了在 Montage 长视频生产管线中，如何利用 `FLUX.2-klein-9b-kv` 极速大模型，通过单次 Text-to-Image (T2I) 推理，直接生成高一致性、排版整齐的多镜头联合故事板（Storyboard/Grid），并进行高精度的自动切片。
> **适用场景**：Montage 是一个通用的长视频生产系统，支持电影、动漫、广告、游戏 CG、科幻、文艺、国风等多种视频风格。本 Skill 的设计原则和尺寸推导公式适用于所有视觉风格，不局限于写实电影风。

---

## 🔑 核心机制与设计原则 (Core Principles)

### 1. 窄白边物理分割 (Narrow White Gutters)
- **结论**：无白边（Seamless）的拼贴画虽然艺术感强，但画面元素高度混杂，边界模糊，在工程上极难进行自动化切片（Slicing）。
- **规范**：在实际生产中，**必须要求模型生成窄白边（Narrow White Gutters）**。高对比度的纯白分界线不仅使画面条理清晰，还便于下游 OpenCV/PIL 脚本通过轮廓检测或等分投影实现 100% 精准的自动切片。

### 2. 黄金排版网格 (3x2 或 2x2 Grid)
- **结论**：长条图（如 1x4 横向胶片）或超高密度网格（如 4x4 / 16格）在生成时容易触发模型的“漫画书先验”，导致排版崩溃、幽灵白边或语义严重重复。
- **规范**：
  - 首选 **3x2 网格（6个关键帧）**：叙事长度与画面细节的完美平衡点，最适合承载一个完整 Chunk 的镜头序列。
  - 次选 **2x2 网格（4个关键帧）**：适合短镜头序列，单格画面分辨率更高。
  - **严禁使用 4x4 或更高密度网格**。

### 3. 格子分辨率与宽高比绝对一致性 (Consistent Panel Aspect Ratio)
- **痛点**：若不加约束，模型在生成多格故事板时，部分格子的分辨率和宽高比会不一致（部分格子被拉伸或压缩，导致切片后变形）。
- **规范**：
  - 在 Prompt 中必须显式加入强力的排版和几何约束：`"Each panel must have an identical aspect ratio and resolution, forming a perfectly symmetrical grid."`（每个分格必须具有完全一致的宽高比和分辨率，形成完美对称的网格）。
  - **物理分辨率与网格比例必须数学对齐**。
  - **只嘱咐模型是不够的**：FLUX 的白边位置实测会漂移几个像素。因此 `slice_storyboard()`
    在切片后会 ① 裁掉每格残留的白边，② 强制把所有格子归一到同一分辨率
    （给了 `target_size` 就归到视频尺寸，否则归到各格的公共最小尺寸）。多关键帧视频调用要求
    所有参考图同分辨率，这一步是硬约束，不是美化。

### 4. 镜头范围必须显式声明 (Shot Scope Must Be Declared)
同一个 2x2 网格，可以是**一个镜头内的连续关键帧**，也可以是**多个镜头的分镜**，两者对模型的要求正好相反，
不声明就只能靠模型猜：

| `shot_scope` | 含义 | 帧间**允许**变化 | 帧间**禁止**变化 |
|---|---|---|---|
| `SHOT_SCOPE_WITHIN` | 一个镜头内的关键帧 | 人物姿态、动作推进 | 机位、镜头焦段、构图、景别、背景、光线 |
| `SHOT_SCOPE_ACROSS` | 多个镜头的分镜 | 机位、构图、景别（像电影切镜） | 人物身份与脸、服装、道具、场景 |

段内映射用 `WITHIN`；段间（跨镜头）用 `ACROSS`。MLLM 规划出图时**必须**给出这个字段。

### 5. 帧间时序差必须给到模型 (Explicit Inter-Panel Timing)
`panel_times_sec` 传入每格在该镜头时间轴上的秒数，Prompt 里会展开成逐格增量
（`Panel 2 (+1.7s after Panel 1)`）外加一条总时序约束，让"帧间变化幅度"与真实时间间隔挂钩：
间隔小就只推进一点姿态，间隔大就是明显更晚的时刻。不传则完全不提时序（不编造）。

### 6. 去 AI 味 (De-AI Film Anchors)
`raw_film=True`（默认）会剔除 `photorealistic / hyperrealistic / highly detailed / 8k /
flawless skin / studio lighting` 这类把画面推向塑料 CG 感的词，并追加独立的胶片段落
（`raw photo, un-retouched, shot on 35mm film, Fujifilm Superia, natural skin texture,
visible pores, slight film grain`）。措辞与视频提示词路径共用
`memstrata.lib.prompt_standardizer` 的 `AI_BUZZWORDS` / `FILM_ANCHORS`，两边不会漂移。

---

## 📐 尺寸反推与等比例缩放公式 (Dimension Reverse-Engineering)

在 Montage 生产管线中，**故事板的物理生成尺寸必须由目标视频的尺寸反向推导得出**，以确保切片后的每个 Panel 能够等比例（不拉伸、不裁剪）地直接作为视频生成模型（如 Wan2.1-I2V, LTX-2.3）的首帧条件。

### 1. 核心公式
设目标视频的单帧尺寸为 $W_{video} \times H_{video}$（例如 $832 \times 480$ 或 $720 \times 480$）。
设我们要求的窄白边宽度为 $G$（Gutters，通常设为 $16$ 到 $32$ 像素）。
设网格排版为 $C$ 列（Columns）$\times$ $R$ 行（Rows）（例如 3x2 网格中 $C=3, R=2$）。

为了让切片后的每个 Panel 完美等比例缩放至视频尺寸，故事板的整体生成物理尺寸 $W_{board} \times H_{board}$ 必须满足：

$$W_{board} = S \times W_{video} \times C + (C - 1) \times G$$
$$H_{board} = S \times H_{video} \times R + (R - 1) \times G$$

其中 $S$ 为缩放因子（Scale Factor）。为了让 FLUX 生成的图像在 $1024$ 到 $2048$ 像素的黄金生成区间内，我们可以通过调节 $S$ 来反推最合适的物理尺寸。

### 2. 实例计算（以 3x2 网格，目标视频 832x480，窄白边 G=16 为例）
我们希望生成一个 3x2 的故事板，裁切后的 Panel 完美等比例缩放至 $832 \times 480$。
- **若设 $S = 0.75$**：
  - $W_{board} = 0.75 \times 832 \times 3 + (3 - 1) \times 16 = 1872 + 32 = 1904$ 像素。
  - $H_{board} = 0.75 \times 480 \times 2 + (2 - 1) \times 16 = 720 + 16 = 736$ 像素。
  - ➔ 生成尺寸设为 **1904x736**。
- **若设 $S = 0.8$**：
  - $W_{board} = 0.8 \times 832 \times 3 + (3 - 1) \times 16 = 1996.8 + 32 \approx 2028$ 像素。
  - $H_{board} = 0.8 \times 480 \times 2 + (2 - 1) \times 16 = 768 + 16 = 784$ 像素。
  - ➔ 生成尺寸设为 **2028x784**。

通过该公式，我们保证了**生成的每一格画面在切片后，其宽高比与目标视频完全一致，无需任何二次变形拉伸**。

### 3. 高分辨率优势：免去超分直接作为首帧条件 (High-Resolution Advantage)
在 `experiments/probe/flux_klein_storyboard/run_cinematic_large.py` 实验中，我们验证了 FLUX.2 Klein 在大分辨率（如 1536x1024 或 2048x1024）下的生成能力。大分辨率故事板在 Montage 管线中具有极大的工程 and 质量优势：
- **免去超分步骤**：例如，在 2048x1024 的故事板中，2x2 网格切片后的每个 Panel 物理尺寸为 1024x512。这已经是一个非常高清的电影级宽屏画面，**可以直接作为 Wan2.1 或 LTX-2.3 视频生成模型的首帧条件**，完全不需要额外的 Super-Resolution (超分) 步骤，既节省了推理时间，又避免了超分模型带来的画质失真。
- **共享 Self-Attention 锁死特征**：在单次大图生成中，所有的 Panel 共享同一个 Latent 空间和 Self-Attention 机制，这使得角色特征（Identity）、服装、场景细节的锁定度比单张独立生成再拼接要高得多。

---

## 🎨 多风格支持 (Supported Styles)

本 Skill 提供了高自由度的多风格支持，所有风格模板和负向提示词均已模块化配置在 `styles.json` 中：

1. **电影写实风 (Cinematic Photorealistic)**：高电影感、胶片质感的写实剧照。
2. **日系动漫风 (Japanese Anime / Illustration)**：手绘数码插画，色彩鲜艳，吉卜力/新海诚质感。
3. **3D 动画/游戏 CG 风 (3D Render / Game CGI)**：皮克斯/迪士尼 3D 动画，或虚幻5引擎渲染的游戏 CG。
4. **赛博朋克科幻风 (Cyberpunk / Sci-Fi)**：未来主义、霓虹闪烁的高科技与低生活对比。
5. **水彩插画风 (Watercolor / Concept Art)**：写意水彩晕染，极强艺术感和氛围感。
6. **复古波普美漫风 (Classic Retro Comic / Pop Art)**：1970年代美漫印刷质感，带网点纸（halftone）纹理。
7. **国风水墨意境风 (Chinese Ink Painting / Shan Shui)**：传统水墨留白，墨色浓淡相宜，适合古风、武侠、禅意。
8. **极简线稿风 (Minimalist Line Art / Sketch)**：干净利落的黑白线条稿或手绘草图，适合早期概念创作。
9. **古典油画风 (Classical Oil Painting)**：古典艺术气息的油画风格，纹理丰富，笔触明显，光影戏剧化。
10. **像素艺术风 (Retro Pixel Art / 16-Bit)**：复古16位游戏像素艺术风格，具有独特的网格颗粒感和鲜明色彩。
11. **黏土定格动画风 (Claymation / Stop-Motion)**：可爱、温润的黏土定格动画风格，具有手作黏土的指纹质感。
12. **扁平矢量插画风 (Flat Vector Illustration)**：现代、简约的扁平化矢量插画风格，色彩干净，几何感强。

---

## 🛠️ 代码集成与使用指南 (Code Integration)

本 Skill 已完全代码化，封装在 `methods/MemStrata/src/memstrata/skills/storyboard_making` 目录下
（MemStrata 生产管线一律从这里调用；`src/montage/skills/storyboard_making` 是通用系统层的原始副本），包含：
- `styles.json`：风格模板与负向提示词配置。
- `storyboard_maker.py`：核心业务逻辑实现（尺寸计算、Prompt 组装、数学网格自动切片）。

### 1. 尺寸反推与 Prompt 组装
```python
from pathlib import Path
from memstrata.skills.storyboard_making import SHOT_SCOPE_WITHIN, StoryboardMaker

# 初始化 StoryboardMaker
maker = StoryboardMaker()

# 1. 根据目标视频尺寸（如 832x480）和 3x2 网格，反推故事板物理生成尺寸
width, height = maker.calculate_dimensions(
    video_w=832,
    video_h=480,
    cols=3,
    rows=2,
    gutter=16,
    scale_factor=0.75
)
print(f"Optimal Generation Size: {width}x{height}")  # 输出: 1904x736

# 2. 准备每一格的镜头描述
panel_descriptions = [
    "a young woman stands outside an old train station at sunset",
    "she turns back as wind blows through her coat",
    "she notices a mysterious suitcase on the platform",
    "she walks toward the suitcase carefully",
    "she reaches out her hand to touch the suitcase handle",
    "a train arrives with bright headlights cutting through thick fog"
]

# 3. 组装 Prompt：声明镜头范围 + 帧间时序 + 去 AI 味
prompt, negative_prompt = maker.format_prompt(
    style_key="cinematic",
    panel_descriptions=panel_descriptions,
    cols=3,
    rows=2,
    shot_scope=SHOT_SCOPE_WITHIN,          # 一个镜头内的关键帧；跨镜头用 SHOT_SCOPE_ACROSS
    panel_times_sec=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # 每格在镜头时间轴上的秒数
    raw_film=True,                          # 默认；去 AI 味 + 胶片锚点
)
```

### 2. 故事板生成与数学网格自动切片
```python
# 4. 调用 FLUX.2 Klein 生成故事板图像（保存到 storyboard.png）
# ... 执行生成逻辑 ...

# 5. 切片：定位白边 → 裁掉残留白边 → 强制统一到视频分辨率
sliced_paths = maker.slice_storyboard(
    image_path="storyboard.png",
    cols=3,
    rows=2,
    gutter=16,
    out_dir="output/sliced_panels",
    trim_white=True,            # 默认；剪掉每格残留白边
    target_size=(832, 480),     # 省略则统一到各格公共最小尺寸
)
# sliced_paths 将包含: panel_01.png, ..., panel_06.png，全部同分辨率，可直接作为多关键帧条件
```

---

## ⚠️ 核心注意事项 (Precautions)

1. **排版词汇与画风的权衡**：
   - **不建议绝对禁用排版词汇**，而是根据实际排版需求进行判定。
   - 当需要极强的空间布局控制（如严格的 3x2 或 2x2 网格）时，使用 `storyboard`, `panels` 或 `grid` 能够非常稳定地激活模型的排版先验，帮助画出整齐的白边。
   - 但必须注意：一旦使用了这些词汇，必须在 Prompt 结尾显式加入强力的**反向风格约束**（如 `"Style: high-quality anime keyframes, no drawings, no outlines."` 或 `"Style: photorealistic film stills, no drawings, no outlines."`），以此来对冲漫画书先验，保证输出依然是目标风格（如纯正的动漫风、电影风或 3D CG），而不是手绘或漫画风。
2. **强制负向约束**：必须在 Prompt 结尾加入强力的反向约束词，明确拒绝手绘、卡通和描边。
   - **注意**：MemStrata 的 FLUX 后端（`flux_klein_backend` / `flux_persistent_server`）**没有负向提示通道**
     ——klein 是 guidance 蒸馏模型，无 CFG 可用。`format_prompt` 返回的 `negative_prompt` 在这条路上会被丢弃，
     因此所有"不要漫画/不要描边/不要页码"的约束都必须写进**正向** Prompt（`_GEOMETRY_TEXT` 与各风格模板的
     `Style:` 行已经这么做）。
3. **生产环境首选 FP8 精度**：
   - 在实际生产和常驻服务中，**默认且首选 `flux.2-klein-9b-kv-fp8` 精度**。
   - FP8 能够在保持与 BF16 几乎完全一致的特征锁死度（MAE 误差极小）的前提下，**降低 50% 的显存开销（仅需 ~9.1G VRAM）**，并显著提升推理速度，是极速生产的首选。
   - **BF16 精度仅作为特定环境下的备选/调试方案**（例如当特定的 `diffusers` 单文件加载器在某些容器环境中出现兼容性 bug 时，可临时回退到 BF16 以保证 100% 跑通）。
4. **单线程串行生成**：由于 `diffusers` 库的 Pipeline 实例在执行 `__call__` 时是非线程安全的，在批量生成故事板时**必须采用单线程串行**，单张生成仅需约 4-7 秒，安全且高效。
