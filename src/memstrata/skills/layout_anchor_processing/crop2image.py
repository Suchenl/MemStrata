"""Crop2Image: R4 crop->region assignment + collage compositing.

This is the core of the Crop2Image keyframe path:

    R3 layout (color-block regions)  +  retrieved real crops
        --R4 (vision MLLM)-->  which crop fills which region
        --composite-->         collage canvas (real crops pasted into regions,
                               unassigned regions keep their color-block prior)
        --FLUX I2I (elsewhere)--> a single coherent photorealistic keyframe

Two pieces live here (both dependency-light: PIL only):
  * ``assign_crops_to_regions`` — the R4 role call via ``MllmRoleRunner`` (vision),
    with a deterministic label/kind fallback so the loop still runs offline or when
    the VL server is unavailable. The placement decision is the MLLM's, per the
    R4 contract in ``memstrata.mllm.roles`` (never hard-coded by an operator).
  * ``composite_crops`` — paste each assigned crop into its layout box on a canvas.

The FLUX I2I fusion step itself is a generation backend concern (it needs a GPU
server) and lives outside this module; ``composite_crops`` produces the canvas it
consumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from memstrata.skills.layout_anchor_processing.base import BaseLayoutProcessor, LayoutElement
from memstrata.mllm.roles import get_role
from memstrata.mllm.runner import MllmRoleRunner

logger = logging.getLogger(__name__)

R4_ROLE = "crop_region_assigner"

# Character-ish shapes/kinds that should prefer a 'human' box.
_PERSON_KINDS = {"character", "person", "human", "actor"}
_PERSON_SHAPES = {"human", "person", "actor", "speaker", "ellipse_and_rect"}

R4_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "description": "One entry per crop that should be placed. Omit crops that fit no region.",
            "items": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "asset id of the crop to place"},
                    "box_index": {"type": "integer", "description": "index into the provided layout regions"},
                    "representation_id": {"type": ["string", "null"], "description": "chosen variant, or null"},
                    "reasoning": {"type": "string"},
                },
                "required": ["asset_id", "box_index"],
                "additionalProperties": True,
            },
        }
    },
    "required": ["assignments"],
}


@dataclass(slots=True)
class CropRef:
    """A retrieved real-crop candidate for one asset."""

    asset_id: str
    name: str
    kind: str
    image_path: str
    representation_id: Optional[str] = None


def _build_prompt(elements: List[Dict[str, Any]], crops: List[CropRef]) -> str:
    lines = [
        "You are a composition director. A color-block layout has been planned; each region",
        "has an index, a semantic label, a shape, and a normalized box [ymin,xmin,ymax,xmax] in",
        "[0,1000]. You are given real reference crops of named entities (attached images, in the",
        "same order as listed). Decide which crop should be placed into which region so the final",
        "scene matches the layout and each entity lands in a plausible location. Prefer 'human'",
        "regions for people. Return JSON per the schema; omit a crop if no region fits.",
        "",
        "Layout regions:",
    ]
    for i, e in enumerate(elements):
        lines.append(f"  [{i}] label={e.get('label')!r} shape={e.get('shape')!r} box={e.get('box_2d')}")
    lines.append("")
    lines.append("Reference crops (image order):")
    for j, c in enumerate(crops):
        lines.append(f"  image#{j}: asset_id={c.asset_id!r} name={c.name!r} kind={c.kind!r}")
    return "\n".join(lines)


def _fallback_assign(elements: List[Dict[str, Any]], crops: List[CropRef]) -> List[Dict[str, Any]]:
    """Deterministic placement when the MLLM is unavailable.

    Strategy: exact/substring label<->name match first; then people to unused 'human'
    boxes; then remaining crops to remaining boxes in order. Never reuse a box.
    """
    used: set[int] = set()
    out: List[Dict[str, Any]] = []

    def _take(idx: int, c: CropRef) -> None:
        used.add(idx)
        out.append({"asset_id": c.asset_id, "box_index": idx,
                    "representation_id": c.representation_id, "reasoning": "fallback"})

    def _labels() -> list[tuple[int, str, str]]:
        return [(i, str(e.get("label", "")).lower(), str(e.get("shape", "")).lower())
                for i, e in enumerate(elements)]

    remaining = list(crops)
    # 1) name<->label match
    for c in list(remaining):
        nm = c.name.lower().strip()
        hit = next((i for i, lbl, _ in _labels() if i not in used and nm and (nm in lbl or lbl in nm)), None)
        if hit is not None:
            _take(hit, c); remaining.remove(c)
    # 2) people -> human boxes
    for c in list(remaining):
        if c.kind.lower() in _PERSON_KINDS:
            hit = next((i for i, _, shp in _labels() if i not in used and shp in _PERSON_SHAPES), None)
            if hit is not None:
                _take(hit, c); remaining.remove(c)
    # 3) leftovers -> any free box in order
    for c in list(remaining):
        hit = next((i for i, _, _ in _labels() if i not in used), None)
        if hit is None:
            break
        _take(hit, c); remaining.remove(c)
    return out


def assign_crops_to_regions(
    elements: List[Dict[str, Any]],
    crops: List[CropRef],
    *,
    runner: MllmRoleRunner | None = None,
    use_mllm: bool = True,
) -> List[Dict[str, Any]]:
    """R4: decide crop->region placement. Returns a list of assignment dicts.

    Falls back to :func:`_fallback_assign` on any MLLM error (keeps the loop running).
    """
    if not crops or not elements:
        return []
    get_role(R4_ROLE)  # fail fast if registry drifts
    if not use_mllm:
        return _fallback_assign(elements, crops)
    runner = runner or MllmRoleRunner()
    try:
        instruction = _build_prompt(elements, crops)
        images = [c.image_path for c in crops]
        result = runner.run(R4_ROLE, instruction=instruction, images=images, schema=R4_SCHEMA)
        assignments = list(result.get("assignments", []))
        # sanity: drop out-of-range / duplicate boxes
        seen: set[int] = set()
        clean: List[Dict[str, Any]] = []
        for a in assignments:
            bi = a.get("box_index")
            if isinstance(bi, int) and 0 <= bi < len(elements) and bi not in seen:
                seen.add(bi)
                clean.append(a)
        if clean:
            return clean
        logger.warning("R4 returned no usable assignments; using fallback.")
    except Exception as e:  # noqa: BLE001
        logger.warning("R4 MLLM assignment failed (%s); using fallback.", e)
    return _fallback_assign(elements, crops)


def composite_crops(
    elements: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    crops: List[CropRef],
    *,
    width: int,
    height: int,
    base: Image.Image | None = None,
    normalized_range: tuple[float, float] | None = (0, 1000),
    margin: float = 0.04,
) -> Image.Image:
    """Paste each assigned crop into its layout box on a canvas.

    ``base`` is the starting canvas — pass the color-block anchor so unassigned
    regions keep their color-block prior (recommended for FLUX I2I); if None, a
    black canvas is used. Crops are resized to fit their box (with a small inset
    margin) preserving aspect ratio and centered.
    """
    proc = BaseLayoutProcessor(default_width=width, default_height=height)
    canvas = base.convert("RGB").resize((width, height)) if base is not None else Image.new("RGB", (width, height), (0, 0, 0))
    by_asset = {c.asset_id: c for c in crops}

    for a in assignments:
        c = by_asset.get(a.get("asset_id", ""))
        idx = a.get("box_index")
        if c is None or not isinstance(idx, int) or not (0 <= idx < len(elements)):
            continue
        elem = LayoutElement.from_dict(elements[idx])
        xmin, ymin, xmax, ymax = proc.scale_coordinates(elem.box_2d, width, height, normalized_range=normalized_range)
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 1 or bh <= 1:
            continue
        # inset margin so crops don't touch region edges
        mx, my = int(bw * margin), int(bh * margin)
        tw, th = max(1, bw - 2 * mx), max(1, bh - 2 * my)
        try:
            crop_img = Image.open(c.image_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            logger.warning("could not open crop %s (%s); skipping", c.image_path, e)
            continue
        # fit preserving aspect ratio, centered in the box
        crop_img.thumbnail((tw, th), Image.LANCZOS)
        px = xmin + mx + (tw - crop_img.width) // 2
        py = ymin + my + (th - crop_img.height) // 2
        canvas.paste(crop_img, (px, py))
    return canvas


def crop2image_canvas(
    screenplay_elements: List[Dict[str, Any]],
    crops: List[CropRef],
    *,
    width: int,
    height: int,
    anchor: Image.Image | None = None,
    runner: MllmRoleRunner | None = None,
    use_mllm: bool = True,
) -> tuple[Image.Image, List[Dict[str, Any]]]:
    """Convenience: run R4 assignment then composite onto the anchor.

    Returns ``(collage_canvas, assignments)``.
    """
    assignments = assign_crops_to_regions(screenplay_elements, crops, runner=runner, use_mllm=use_mllm)
    canvas = composite_crops(screenplay_elements, assignments, crops, width=width, height=height, base=anchor)
    return canvas, assignments


__all__ = [
    "CropRef",
    "R4_ROLE",
    "R4_SCHEMA",
    "assign_crops_to_regions",
    "composite_crops",
    "crop2image_canvas",
]
