# layout_anchor_processing (+ Crop2Image)

Vendored/adapted from `montage.skills.layout_anchor_processing`. The renderers are
verbatim (PIL-only, no `montage` dependency); the planner is rewired to
`memstrata.mllm.MllmRoleRunner` so it honours the `layout_planner` (R3) contract, and
a new `crop2image` module adds the R4 crop→region + collage step.

## What it does

```
screenplay ──R3(MLLM layout)──▶ elements [{label, box_2d, shape}]
                                   │
                    ColorBlockProcessor / LineArtProcessor
                                   │  color-block anchor (FLUX) / line-art (Qwen-Image-Edit)
retrieved real crops ──R4(vision MLLM: which crop → which region)──▶ assignments
                                   │  composite_crops (paste crops onto anchor)
                                   ▼
                             Crop2Image collage ──FLUX.2 Klein I2I──▶ coherent keyframe
```

## Public API

- `LayoutPlanner(runner=None).plan_layout(screenplay) -> [element dicts]` — role **R3**.
- `ColorBlockProcessor` / `LineArtProcessor` — render `LayoutElement`s to an anchor image.
- `assign_crops_to_regions(elements, crops, runner=, use_mllm=) -> assignments` — role **R4**
  (vision MLLM decides placement; deterministic label/kind fallback when offline).
- `composite_crops(elements, assignments, crops, width=, height=, base=anchor)` — paste crops.
- `crop2image_canvas(elements, crops, width=, height=, anchor=, runner=)` — R4 + composite in one.
- `CropRef(asset_id, name, kind, image_path, representation_id=None)` — a retrieved-crop candidate.

## Notes

- One multimodal MLLM (Qwen3.5-9B) plays both R3 (text) and R4 (vision). See
  `docs/method/mllm_roles.md`.
- FLUX I2I fusion of the collage is a generation-backend concern and runs in the
  a Python that has `black-forest-labs/FLUX.2-klein-9b-kv` (not `flux2`,
  whose huggingface_hub is broken). `composite_crops` only produces the input canvas.
- Coordinates are normalized `[0,1000]`, `box_2d = [ymin,xmin,ymax,xmax]`.
