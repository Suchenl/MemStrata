# generation_routing — Generation-Path Router (role R2b)

Decide **how to seed the next video segment**. A deterministic rule layer restricts the
*feasible* modes (hard physical constraints); the MLLM (`generation_router`, thinking off)
picks among them, and the choice is re-validated against feasibility with a safe fallback.

## Modes

| mode | seeding | when | Helios call |
|---|---|---|---|
| `continue_ar` | continue from prior video window | same scene, no cut, all referenced entities on prev last frame | `video=[style_anchor + ~73 recent frames]`, `history_sizes=[16,2,1]`, `keep_first_frame=True` |
| `reanchor_lastframe` | prev last frame as i2v anchor + new prompt | same place, new beat, no entity to introduce | `image=<prev_last_frame>` |
| `recompose_partial` | paste returning crop onto prev last frame, then i2v | scene mostly same, must inject ONE returning asset | `image=<composited_last_frame>` |
| `recompose_keyframe` | fresh FLUX keyframe from memory crops (default: native FLUX multi-image; legacy R3→R4→FLUX collage via `MEMSTRATA_KEYFRAME_MODE=collage`) | scene cut / new location / time jump / returning asset absent from prev frame / chunk 0 | `image=<flux_keyframe>` |

## Hard constraints (rule layer, not the model)

- No prior segment (chunk 0) ⇒ `recompose_keyframe` only.
- Scene cut (`continue_vs_cut=="cut"`) ⇒ rolling window is broken ⇒ `recompose_keyframe` only.
- A referenced entity not on the prev last frame ⇒ `continue_ar` is infeasible.

`onscreen_entities` is derived cheaply from the **previous chunk's decompose observations**
(who was grounded in the prior segment) — no extra vision call.

## Critical: `continue_ar` must use `video=`, not `image=`

The Helios pipeline treats `image` and `video` as **mutually exclusive**. `continue_ar`
feeds `video = [style_anchor_frame] + [recent frames]`; Helios keeps frame 0 as the always-on
style anchor (`keep_first_frame=True`) and samples `[16,2,1]` long/mid/short history latents
from the rest (≈ a 73-pixel-frame rolling window). Passing only the last frame as `image=`
cannot preserve motion continuity with the prior segment.

## API

```python
from memstrata.skills.generation_routing import GenerationRouter
d = GenerationRouter().route(
    prompt=g_n, chunk_id=i, referenced_entities=[...], onscreen_entities=[...],
    has_prev_segment=i > 0, continue_vs_cut="continue", scene_return=False, prev_summary="...")
d.mode        # GenMode
d.to_dict()   # {mode, reason, recompose_asset_ids, continuity, feasible, source}
```
