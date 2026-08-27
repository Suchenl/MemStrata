# Storyboard Making Skill

> **Core definition**: this skill specifies how, within the upstream project long-video
> production pipeline, to use the ultra-fast large model `FLUX.2-klein-9b-kv` to
> directly generate a highly consistent, cleanly laid-out multi-shot combined
> storyboard (Storyboard/Grid) via a single Text-to-Image (T2I) inference pass,
> and then slice it with high precision automatically.
> **Applicable scenarios**: upstream project is a general long-video production system that
> supports many video styles — film, anime, advertising, game CG, sci-fi, art-house,
> Chinese traditional style, and more. The design principles and the dimension
> derivation formula in this skill apply to all visual styles, not just realistic
> cinematic film.

---

## 🔑 Core mechanism and design principles

### 1. Narrow white gutters (physical separation)
- **Conclusion**: seamless (no-gutter) collages are artistically strong, but the
  picture elements are highly intermingled and the boundaries are blurred, making
  automated slicing extremely hard in engineering terms.
- **Rule**: in real production, **you must require the model to generate narrow
  white gutters**. High-contrast pure-white dividing lines not only make the
  picture clear and organized, they also let downstream OpenCV/PIL scripts achieve
  100% precise automatic slicing via contour detection or even-division projection.

### 2. The golden layout grid (3x2 or 2x2 grid)
- **Conclusion**: long strips (e.g. a 1x4 horizontal film strip) or ultra-high-density
  grids (e.g. 4x4 / 16 cells) tend to trigger the model's "comic-book prior" during
  generation, causing layout collapse, ghost gutters, or severe semantic repetition.
- **Rule**:
  - First choice: **3x2 grid (6 keyframes)** — the perfect balance between narrative
    length and picture detail, best suited to carry the shot sequence of a full chunk.
  - Second choice: **2x2 grid (4 keyframes)** — suited to short shot sequences, with
    higher per-cell resolution.
  - **Never use a 4x4 or higher-density grid.**

### 3. Consistent panel aspect ratio (cell resolution and aspect ratio)
- **Pain point**: without constraints, when generating a multi-cell storyboard the
  model produces cells with inconsistent resolution and aspect ratio (some cells
  are stretched or squeezed, which deforms them after slicing).
- **Rule**:
  - You must explicitly add strong layout and geometry constraints to the prompt:
    `"Each panel must have an identical aspect ratio and resolution, forming a perfectly symmetrical grid."`
    (i.e., every panel must have an identical aspect ratio and resolution, forming a perfectly symmetrical grid).
  - **The physical resolution and grid ratio must be mathematically aligned.**
  - **Instructing the model is not enough on its own**: in practice FLUX's gutter
    positions drift by a few pixels. So after slicing, `slice_storyboard()` will
    ① trim the residual gutter left on each cell, and ② force all cells to be
    normalized to the same resolution (to the video size if `target_size` is given,
    otherwise to the common minimum size of the cells). Multi-keyframe video calls
    require all reference images to be the same resolution, so this step is a hard
    constraint, not a beautification.

### 4. Shot scope must be declared
The same 2x2 grid can be **consecutive keyframes within a single shot** or the
**storyboard of several separate shots**; the two impose exactly opposite
requirements on the model, and without a declaration the model can only guess:

| `shot_scope` | Meaning | Changes **allowed** between frames | Changes **forbidden** between frames |
|---|---|---|---|
| `SHOT_SCOPE_WITHIN` | Keyframes within a single shot | Character pose, action progression | Camera position, focal length, composition, shot size, background, lighting |
| `SHOT_SCOPE_ACROSS` | Storyboard across several shots | Camera position, composition, shot size (like film cuts) | Character identity and face, costume, props, scene |

Use `WITHIN` for intra-segment mapping; use `ACROSS` for inter-segment
(cross-shot). The MLLM **must** provide this field when planning the image.

### 5. Explicit inter-panel timing must be given to the model
`panel_times_sec` passes the number of seconds of each cell on that shot's
timeline, and the prompt expands this into per-cell increments
(`Panel 2 (+1.7s after Panel 1)`) plus an overall timing constraint, so that "the
amount of change between frames" is tied to the real time interval: a small
interval advances the pose only a little, a large interval is a clearly later
moment. If not passed, timing is not mentioned at all (nothing is fabricated).

### 6. De-AI film anchors
`raw_film=True` (the default) strips words like `photorealistic / hyperrealistic /
highly detailed / 8k / flawless skin / studio lighting` that push the picture
toward a plastic CG feel, and appends a separate film paragraph
(`raw photo, un-retouched, shot on 35mm film, Fujifilm Superia, natural skin
texture, visible pores, slight film grain`). The wording is shared with the video
prompt path via `memstrata.lib.prompt_standardizer`'s `AI_BUZZWORDS` /
`FILM_ANCHORS`, so the two do not drift.

---

## 📐 Dimension reverse-engineering and proportional scaling formula

In the upstream project production pipeline, **the storyboard's physical generation size
must be reverse-derived from the target video's size**, to ensure that each sliced
panel can serve directly, proportionally (no stretch, no crop), as the first-frame
condition of a video-generation model (such as Wan2.1-I2V, LTX-2.3).

### 1. Core formula
Let the single-frame size of the target video be $W_{video} \times H_{video}$
(e.g. $832 \times 480$ or $720 \times 480$).
Let the required narrow-gutter width be $G$ (Gutters, typically set to $16$ to $32$
pixels).
Let the grid layout be $C$ columns (Columns) $\times$ $R$ rows (Rows) (e.g. in a
3x2 grid, $C=3, R=2$).

For each sliced panel to scale perfectly proportionally to the video size, the
storyboard's overall physical generation size $W_{board} \times H_{board}$ must
satisfy:

$$W_{board} = S \times W_{video} \times C + (C - 1) \times G$$
$$H_{board} = S \times H_{video} \times R + (R - 1) \times G$$

where $S$ is the scale factor. To keep the FLUX-generated image within the golden
generation range of $1024$ to $2048$ pixels, you can adjust $S$ to reverse-derive
the most appropriate physical size.

### 2. Worked example (3x2 grid, target video 832x480, narrow gutter G=16)
We want to generate a 3x2 storyboard whose cropped panels scale perfectly
proportionally to $832 \times 480$.
- **If $S = 0.75$**:
  - $W_{board} = 0.75 \times 832 \times 3 + (3 - 1) \times 16 = 1872 + 32 = 1904$ pixels.
  - $H_{board} = 0.75 \times 480 \times 2 + (2 - 1) \times 16 = 720 + 16 = 736$ pixels.
  - ➔ Set the generation size to **1904x736**.
- **If $S = 0.8$**:
  - $W_{board} = 0.8 \times 832 \times 3 + (3 - 1) \times 16 = 1996.8 + 32 \approx 2028$ pixels.
  - $H_{board} = 0.8 \times 480 \times 2 + (2 - 1) \times 16 = 768 + 16 = 784$ pixels.
  - ➔ Set the generation size to **2028x784**.

Through this formula, we guarantee that **after slicing, every generated cell has
exactly the same aspect ratio as the target video, with no secondary deformation
or stretching required**.

### 3. High-resolution advantage: skip super-resolution and use directly as the first-frame condition
In the `experiments/probe/flux_klein_storyboard/run_cinematic_large.py` experiment
we validated FLUX.2 Klein's generation capability at large resolutions (such as
1536x1024 or 2048x1024). A large-resolution storyboard has major engineering and
quality advantages in the upstream project pipeline:
- **Skips the super-resolution step**: for example, in a 2048x1024 storyboard, each
  panel of a 2x2 grid has a physical size of 1024x512 after slicing. This is
  already a very high-definition cinematic widescreen picture, and it **can serve
  directly as the first-frame condition of a Wan2.1 or LTX-2.3 video-generation
  model**, with no extra Super-Resolution step needed at all — saving inference
  time while avoiding the quality distortion introduced by super-resolution models.
- **Shared Self-Attention locks features**: in a single large-image generation, all
  panels share the same latent space and Self-Attention mechanism, which makes the
  lock-in of character identity, costume, and scene detail far higher than
  generating single images separately and then stitching them.

---

## 🎨 Supported styles

This skill provides highly flexible multi-style support; all style templates and
negative prompts are modularly configured in `styles.json`:

1. **Cinematic Photorealistic**: highly cinematic, film-textured realistic stills.
2. **Japanese Anime / Illustration**: hand-drawn digital illustration, vivid colors, Ghibli / Shinkai texture.
3. **3D Render / Game CGI**: Pixar/Disney 3D animation, or Unreal Engine 5 rendered game CG.
4. **Cyberpunk / Sci-Fi**: futuristic, neon-lit high-tech vs. low-life contrast.
5. **Watercolor / Concept Art**: expressive watercolor washes, strong artistic and atmospheric feel.
6. **Classic Retro Comic / Pop Art**: 1970s American-comic print texture with halftone dot-screen grain.
7. **Chinese Ink Painting / Shan Shui**: traditional ink wash with negative space, graded ink tones — suited to classical, wuxia, and Zen moods.
8. **Minimalist Line Art / Sketch**: clean black-and-white line art or hand-drawn sketches, suited to early concept work.
9. **Classical Oil Painting**: classical oil-painting style with rich texture, visible brushstrokes, and dramatic light and shadow.
10. **Retro Pixel Art / 16-Bit**: retro 16-bit game pixel-art style with a distinctive grid grain and vivid colors.
11. **Claymation / Stop-Motion**: cute, warm clay stop-motion style with the fingerprint texture of handmade clay.
12. **Flat Vector Illustration**: modern, minimalist flat vector illustration with clean colors and a strong geometric feel.

---

## 🛠️ Code integration and usage guide

This skill is fully implemented in code, packaged under the
`this repository/src/memstrata/skills/storyboard_making` directory (the MemStrata
production pipeline always calls it from here; `the upstream source tree/skills/storyboard_making`
is the original copy at the general system layer), containing:
- `styles.json`: style templates and negative-prompt configuration.
- `storyboard_maker.py`: the core business-logic implementation (dimension
  computation, prompt assembly, mathematical grid auto-slicing).

### 1. Dimension reverse-engineering and prompt assembly
```python
from pathlib import Path
from memstrata.skills.storyboard_making import SHOT_SCOPE_WITHIN, StoryboardMaker

# Initialize the StoryboardMaker
maker = StoryboardMaker()

# 1. From the target video size (e.g. 832x480) and a 3x2 grid, reverse-derive the storyboard's physical generation size
width, height = maker.calculate_dimensions(
    video_w=832,
    video_h=480,
    cols=3,
    rows=2,
    gutter=16,
    scale_factor=0.75
)
print(f"Optimal Generation Size: {width}x{height}")  # Output: 1904x736

# 2. Prepare the shot description for each cell
panel_descriptions = [
    "a young woman stands outside an old train station at sunset",
    "she turns back as wind blows through her coat",
    "she notices a mysterious suitcase on the platform",
    "she walks toward the suitcase carefully",
    "she reaches out her hand to touch the suitcase handle",
    "a train arrives with bright headlights cutting through thick fog"
]

# 3. Assemble the prompt: declare shot scope + inter-panel timing + de-AI film anchors
prompt, negative_prompt = maker.format_prompt(
    style_key="cinematic",
    panel_descriptions=panel_descriptions,
    cols=3,
    rows=2,
    shot_scope=SHOT_SCOPE_WITHIN,          # keyframes within a single shot; use SHOT_SCOPE_ACROSS for cross-shot
    panel_times_sec=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # seconds of each cell on the shot timeline
    raw_film=True,                          # default; de-AI + film anchors
)
```

### 2. Storyboard generation and mathematical grid auto-slicing
```python
# 4. Call FLUX.2 Klein to generate the storyboard image (saved to storyboard.png)
# ... run the generation logic ...

# 5. Slicing: locate gutters → trim residual gutters → force-normalize to video resolution
sliced_paths = maker.slice_storyboard(
    image_path="storyboard.png",
    cols=3,
    rows=2,
    gutter=16,
    out_dir="output/sliced_panels",
    trim_white=True,            # default; trim residual gutter on each cell
    target_size=(832, 480),     # omit to normalize to the common minimum size of the cells
)
# sliced_paths will contain: panel_01.png, ..., panel_06.png, all at the same resolution, usable directly as multi-keyframe conditions
```

---

## ⚠️ Core precautions

1. **Trade-off between layout vocabulary and art style**:
   - **We do not recommend absolutely banning layout vocabulary**; instead, decide
     based on the actual layout need.
   - When you need very strong spatial-layout control (such as a strict 3x2 or 2x2
     grid), using `storyboard`, `panels`, or `grid` very reliably activates the
     model's layout prior and helps draw neat gutters.
   - But note: once you use these words, you must explicitly add a strong
     **counter-style constraint** at the end of the prompt (such as
     `"Style: high-quality anime keyframes, no drawings, no outlines."` or
     `"Style: photorealistic film stills, no drawings, no outlines."`) to offset
     the comic-book prior and ensure the output is still the target style (such as
     pure anime, film, or 3D CG), not a hand-drawn or comic style.
2. **Mandatory negative constraints**: you must add strong counter-constraint words
   at the end of the prompt, explicitly rejecting hand-drawing, cartoon, and outlines.
   - **Note**: MemStrata's FLUX backend (`flux_klein_backend` /
     `flux_persistent_server`) **has no negative-prompt channel** — klein is a
     guidance-distilled model with no CFG available. The `negative_prompt` returned
     by `format_prompt` is discarded on this path, so all "no comic / no outlines /
     no page numbers" constraints must be written into the **positive** prompt
     (`_GEOMETRY_TEXT` and the `Style:` line of each style template already do this).
3. **FP8 precision is preferred in production**:
   - In real production and resident services, **`flux.2-klein-9b-kv-fp8` precision
     is the default and preferred choice**.
   - FP8 can, while keeping feature lock-in almost identical to BF16 (extremely
     small MAE error), **cut VRAM overhead by 50% (only ~9.1G VRAM needed)** and
     significantly speed up inference, making it the top choice for ultra-fast
     production.
   - **BF16 precision is only a fallback/debug option for specific environments**
     (for example, when a specific `diffusers` single-file loader has a
     compatibility bug in certain container environments, you can temporarily fall
     back to BF16 to guarantee a 100% successful run).
4. **Single-threaded serial generation**: because the `diffusers` library's
   Pipeline instance is not thread-safe when executing `__call__`, batch storyboard
   generation **must use single-threaded serial execution**; a single generation
   takes only about 4-7 seconds, which is safe and efficient.
