# memstrata.skills

Self-contained, reusable capabilities for the MemStrata production loop. Each
subpackage has its own `README.md` + `registry.toml`, mirrors the
`montage.skills` layout, and avoids importing `montage` internals (helpers are
vendored or rewired to `memstrata.*`).

## Present

| skill | source | deps | used by |
|---|---|---|---|
| `decomposition` | MemStrata Decompose step (moved from `steps/decompose.py`) | `memstrata.bank/encoders/mllm` | pipeline Decompose; emits Observations for curate |
| `memory_update` | MemStrata Stratified Update/落库 step (moved from `steps/curate.py`) | `memstrata.bank/encoders/lib/mllm` | pipeline Curate; admits Observations into the stratified bank |
| `crop_acquisition` | vendored `vmem_bench` S5 propose_and_pick perception | torch, transformers>=5.9 (vendored), SAM3/GDINO/DINOv3 | decompose `Cropper` (production, clean masked crops); persistent server |
| `entity_grounding` | mirror of `vmem_bench` S5 `vlm_grounding` | `memstrata.mllm` (Qwen) | decompose `Cropper` (lightweight VLM tight-box) |
| `layout_anchor_processing` | adapted from `montage.skills` + local `crop2image` | PIL, `memstrata.mllm` | R3 layout, R4 crop→region, FLUX I2I keyframe |
| `focus_segmentation` | verbatim `montage.skills` | numpy only | clean-crop extraction, decompose focus/QA |
| `embedding_deduplication` | `montage.skills`, import rewired to `memstrata.encoders.base` | none (stdlib) | curate/R9 crop dedup, non-redundant selection |

## montage.skills copy candidates (not yet vendored)

- `letterbox_detection` — 0 montage-core deps; clean if a preprocessing step is needed.
- `storyboard_making` — 0 montage-core deps.
- `shot_boundary_detection` — 3 montage-core deps (errors / model_weights / media.probe) + bundled TransNetV2 weights; heavier.
- `long_video_decomposition` — 6 montage-core deps (schemas / stage / serialization / ...); largest, and now superseded by MemStrata's own `decomposition` skill — do not vendor.

Copy rule of thumb: prefer skills with 0 `montage.*` imports (verbatim) or a
single trivially-rewireable helper import (like dedup → `memstrata.encoders.base`).
