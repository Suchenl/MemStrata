# Motion Stability Skill

> **Core definition**: turn "the picture keeps shaking" from a subjective feeling
> into a comparable number, used to compare the frame-to-frame micro-jitter of
> different generators, LoRAs, and step-count configurations.
> **Typical uses**: decide whether a given video-generation route is production-ready;
> A/B compare a distilled model against a full-step model; regression monitoring.

---

## 🔑 Why displacement is not enough

**Displacement cannot distinguish "camera movement" from "jitter":**
- A slow pan has large displacement but no shake at all;
- A locked-off shot that trembles slightly every frame has near-zero displacement but is very unpleasant to watch.

The real discriminative signal is the **second derivative of motion
(acceleration):**
- A real camera (or a real film scan) accelerates smoothly, with acceleration far smaller than velocity;
- A generator that re-"decides" the composition frame by frame produces a random walk whose acceleration is comparable to or even larger than its velocity — what the viewer reads is "the picture keeps shaking".

Per-frame displacement is obtained via **phase correlation** on Hann-windowed
grayscale image pairs, tracking the global dominant displacement and ignoring
local subject motion.

---

## 📏 Three metrics and the verdict

| Metric | Meaning |
|---|---|
| `speed_px` | Median magnitude of per-frame displacement (pixels) |
| `accel_px` | Median magnitude of the **change** in per-frame displacement (pixels) — the jitter signal itself |
| `ratio` | `accel / speed`; ≥1 means the motion is dominated by per-frame noise |

The verdict order **looks at `ratio` first**: being noise-dominated is the worst
case, even when the absolute magnitude is small — the eye follows inconsistency,
not magnitude.

| verdict | condition |
|---|---|
| `noise-dominated` | `ratio >= 1.0` |
| `jittery` | `accel_px > 3 x 0.05px` (3× the real-film floor) |
| `steady` | everything else |

---

## 🚨 The most important point: big pixel diff ≠ a defect

A large frame-to-frame pixel difference is very likely just **large motion
itself**. That is exactly why this skill does not use displacement but instead
uses `accel` and `ratio`, and also why **you must always report `speed` and
`accel` together**.

Measured example (`seg_023`): displacement of 1.906px, the largest of all
material, yet `ratio = 0.70` → judged `jittery` rather than noise-dominated,
because its motion really is genuine picture movement. The counter-example is
`seg_013`: displacement of only 0.347px (the picture is nearly static), but
`accel` is as high as 0.448px, `ratio = 1.29` → this is the case of "barely
moving yet constantly trembling".

**The only defensible way to compare**: compare `accel` between two arms of the
same shot, with the same prompt, and with similar amounts of motion.
Comparing absolute values across different material is meaningless.

---

## 📌 Measured reference (2026-07-28)

All measured with this skill; commands are at the end. `cuts` = the number of
detected content-discontinuity pairs; a real long film is stitched together, so
you must use `within_shot=True` to take a cut-free window.

### Generated material (Track B story 0001, same film)

| Material | speed px | accel px | ratio | verdict | cuts |
|---|---|---|---|---|---|
| A14B distill4step + morphic + dual keyframes | 0.293 | 0.194 | 0.66 | jittery | 18 |
| A14B distill4step + dual keyframes, no LoRA | 0.821 | 0.560 | 0.68 | jittery | 18 |
| A14B distill4step + morphic + single first frame | 0.135 | 0.079 | 0.59 | **steady** | 0 |
| Turbo-5B seg_013 (same shot & prompt as above) | 0.347 | **0.448** | **1.29** | noise-dominated | 0 |
| Turbo-5B seg_005 | 0.942 | 0.919 | 0.98 | jittery | 0 |
| Turbo-5B seg_016 | 1.105 | 0.772 | 0.70 | jittery | 0 |
| Turbo-5B seg_023 | 1.906 | 1.327 | 0.70 | jittery | 0 |
| A14B distill4step + 5-frame motion prefix (continuation probe P5) | 0.239 | 0.211 | 0.88 | jittery | 0 |

**Usable conclusion**: seg_013 and the A14B dual-keyframe arm have similar amounts
of motion (0.347 vs 0.293), and in that case Turbo's per-frame acceleration is
2.3× that of A14B — at equal motion, Turbo is clearly shakier. This is a
same-shot, same-prompt comparison, so it holds.

### The skill's first real-world win: tracing Turbo's jitter to resolution as the root cause

Using the same FLUX keyframe, the same prompt, the same seed 2026, and the same
resident service, changing only the render resolution:

| Render resolution | speed px | accel px | ratio | verdict | Generation time |
|---|---|---|---|---|---|
| **704x1280 (checkpoint native)** | 1.858 | **0.170** | **0.09** | jittery | 25.4 s |
| 480x832 (original Track B setting) | 0.435 | **0.490** | **1.13** | noise-dominated | 14.6 s |

At native resolution the picture has 4.3× the motion, yet the per-frame
acceleration is 2.9× lower and the ratio differs by 12.5×. The jitter comes from
deviating from the checkpoint's native resolution, not from the model itself. The
fix is native rendering + a single downsampling pass at delivery; see
`this repository/configs/video_gen/wan22_ti2v5b_turbo.toml`.

Note that the 704x1280 row is still tagged `jittery`: absolute accel rises with
the amount of motion, and its motion was large to begin with. Only by looking at
`ratio = 0.09` do you see it is clean motion — **this is exactly what "big pixel
diff ≠ a defect" looks like in a real decision**, and why this skill insists that
speed and accel must always be reported together.

The evidence is in `experiments/results/probe/turbo_resolution_jitter/`.

**By-product**: each of the two dual-keyframe arms detected 18 content
discontinuities, while the single-first-frame arm detected 0. This is because
those two arms were required to morph from seg_013's composition to seg_014's
composition within 81 frames (cross-segment interpolation), so the picture jumps
in chunks.

### Real films (cut-free windows)

| Material | Window | speed px | accel px | ratio | verdict | cuts |
|---|---|---|---|---|---|---|
| Casablanca (fixed-camera old film) | 61 frames | 0.952 | 0.095 | 0.59 | jittery | 4 |
| Big Buck Bunny (CGI) | 155 frames | 0.145 | 0.109 | 0.58 | steady | 1 |
| **American Beauty (handheld live-action)** | 91 frames | 0.373 | **0.546** | **1.44** | **noise-dominated** | 4 |

**This row is the most important calibration fact for this skill**: a real movie,
even in a completely cut-free window, can measure a ratio of 1.44 — "worse" than
the Turbo clip (1.29) that originally triggered this whole metric. Real handheld
cinematography is itself a small-amplitude random walk. So the **verdict is not an
absolute true/false or good/bad gate**; it can only be used for A/B ranking under
identical conditions.

### A confound ruled out: film grain

Suspecting that phase correlation might read grain/encoding noise as per-frame
displacement, we first applied a σ=2 Gaussian blur to all material and then
measured: American Beauty 0.546→0.439, Casablanca 0.095→0.072, Turbo seg_013
0.448→0.472. **Both the magnitude and the ordering are unchanged, so grain is not
the cause**, which is why this skill does no denoising preprocessing.

---

## 🛠️ Usage

```python
from memstrata.skills.mesure_video_jitter import measure_jitter, compare

report = measure_jitter("outputs/seg_013.mp4")
print(report.speed_px, report.accel_px, report.ratio, report.verdict, report.cuts_detected)

# A stitched long film must be restricted to a cut-free window, otherwise it is not comparable to a single generated segment
film = measure_jitter("0001_American_Beauty.mp4", within_shot=True)
print(film.window, film.accel_px)

for r in compare(["a.mp4", "b.mp4"]):
    print(r.as_line())     # Unreadable clips are skipped with a warning and do not drag down the whole batch
```

Command line (this is exactly what these measurements used):

```bash
PYTHONPATH=this repository/src python3 -m memstrata.skills.mesure_video_jitter \
  <clip.mp4> ... [--within-shot] [--max-frames 200]
```

The numerical kernel and the video I/O are separated, for easy testing and reuse:

```python
from memstrata.skills.mesure_video_jitter import translations_from_frames, jitter_from_translations

shifts = translations_from_frames(my_gray_frames)   # per-frame global displacement
report = jitter_from_translations(shifts)
```

---

## ⚠️ Precautions

1. **Within-shot vs. across-shot**: the pixel difference at a shot transition is
   inherently large. `cuts_detected` always reports the number of detected content
   discontinuities; when you see a non-zero value, use `within_shot=True` to
   restrict to the longest cut-free window and report the count together with
   `window`. By default it does **not** restrict automatically, to avoid quietly
   producing numbers from a truncated window. Measurement finding: the median is
   itself robust to a single cut (one cut barely changes the result); what really
   ruins the statistics is dense fast cuts — in that case this skill directly
   errors out and refuses to produce a number, rather than giving a fake one.
2. **Static segments**: when completely static, `speed_px = 0` and `ratio` is
   `inf`; in that case look only at the absolute value of `accel_px`.
3. **At least 3 frames**: acceleration needs two displacement differences; too few
   frames errors out directly instead of returning a suspicious 0.
4. **Re-encoding does not change the conclusion**: when the jitter comes from the
   generator, the raw segment file and the stitched video measure the same
   magnitude; if the two differ greatly, check the stitching/encoding stage first.
