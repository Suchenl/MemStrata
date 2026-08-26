"""Type-constrained discovery: propose supported-type entities nobody asked for.

The requested path (``orchestrator.acquire_entity_crop``) answers "where is *this* named
entity?". This module answers the complementary question the paper's Evidence Acquisition
poses: "what else in this frame is worth remembering?" — while staying *type-constrained*,
which is what separates controlled memory expansion from unrestricted open-vocabulary
accumulation.

Concretely, for each requested asset type we run SAM3 concept segmentation over that type's
concept vocabulary, keep proposals that pass the deterministic mask/geometry/QA gates, drop
proposals that overlap a region already acquired for a *named* entity, and return the
survivors as ``DiscoveredEntity`` rows. Naming is deliberately NOT attempted here: the
curate step decides identity via reconciliation (χ), which is reproducible; a perception
module guessing names is not.

Locations get their own concept vocabulary here. The requested path has none (its
``_concepts_for`` returns no concepts for ``location``), so before discovery a location
could only ever be found through a GroundingDINO phrase — which is why scene identities were
systematically under-served.

Heavy model objects (SAM3 / DINOv3) are passed in already-constructed, so importing this
module needs neither torch nor transformers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from memstrata.skills.crop_acquisition.crop_io import materialize_crop
from memstrata.skills.crop_acquisition.crop_qa import audit_crop, bbox_area_fraction
from memstrata.skills.crop_acquisition.geometry import (
    bbox_iou,
    dedup_by_iou,
    mask_to_bbox_norm,
)
from memstrata.skills.crop_acquisition.mask_quality import assess_mask_quality

# Concept vocabulary per asset type. Kept small and generic on purpose: these are the
# *types* memory supports, not an open vocabulary of things a scene might contain.
DISCOVERY_CONCEPTS: dict[str, tuple[str, ...]] = {
    "character": ("person", "animal"),
    "prop": ("object",),
    # The requested path cannot propose locations at all; a whole-scene concept set is
    # what makes the location stratum reachable by perception.
    "location": ("room", "building", "landscape"),
}

# A discovered region must clear these to be worth a memory slot. Deliberately stricter
# than the requested path: an unrequested proposal has no name backing it, so a weak one
# is pure noise, whereas a weak *requested* crop at least has a known identity.
_MIN_MASK_FILL = 0.30
_MIN_SIDE_PX = 24
_MIN_AREA_FRACTION = 0.004
_IOU_DEDUP = 0.6
_MAX_PER_KIND = 3


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())[:24] or "region"


def discover_entities(
    frame_path: str | Path,
    *,
    kinds: tuple[str, ...],
    out_dir: str | Path,
    segmenter: Any,
    embedder: Any | None = None,
    exclude_bboxes: list[list[int]] | None = None,
    max_per_kind: int = _MAX_PER_KIND,
    min_mask_fill: float = _MIN_MASK_FILL,
    min_side_px: int = _MIN_SIDE_PX,
    min_area_fraction: float = _MIN_AREA_FRACTION,
    iou_threshold: float = _IOU_DEDUP,
) -> list[dict[str, Any]]:
    """Propose type-constrained crops from one frame.

    Returns ``[{kind, crop_path, bbox, score, mask_quality}]``, at most ``max_per_kind``
    per type, ranked by segmenter score. ``exclude_bboxes`` are regions already acquired
    for named entities; anything overlapping them is skipped so discovery only ever
    *adds* information.
    """
    from PIL import Image

    frame_path = Path(frame_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if segmenter is None:
        return []

    pil = Image.open(frame_path).convert("RGB")
    width, height = pil.size
    excluded = [b for b in (exclude_bboxes or []) if b and len(b) == 4]

    discovered: list[dict[str, Any]] = []
    for kind in kinds:
        concepts = DISCOVERY_CONCEPTS.get(str(kind), ())
        if not concepts:
            continue
        raw: list[dict[str, Any]] = []
        by_concept = segmenter.segment_multi(frame_path, list(concepts))
        for concept, instances in by_concept.items():
            for ordinal, (bbox_px, score, mask) in enumerate(instances):
                if mask is None:
                    continue
                bbox_norm = mask_to_bbox_norm(mask)
                if bbox_norm is None or len(bbox_norm) != 4:
                    continue
                x0, y0, x1, y1 = (int(v) for v in bbox_px)
                if (x1 - x0) < min_side_px or (y1 - y0) < min_side_px:
                    continue
                if bbox_area_fraction(bbox_norm) < min_area_fraction:
                    continue
                quality = assess_mask_quality(mask)
                if not quality.ok:
                    continue
                # ``location`` is a whole-scene stratum, so a high mask-fill requirement
                # (which suits a compact subject) would reject exactly the wide regions
                # we are looking for.
                if kind != "location":
                    fill = _mask_fill_ratio(mask, bbox_norm, height, width)
                    if fill < min_mask_fill:
                        continue
                if any(bbox_iou(bbox_norm, other) >= iou_threshold for other in excluded):
                    continue  # already explained by a requested crop
                crop_path = out_dir / f"disc_{kind}_{_slug(concept)}_{ordinal}.png"
                crop_path = materialize_crop(
                    frame=frame_path, bbox_norm=bbox_norm, out_path=crop_path, mask=mask
                )
                raw.append({
                    "kind": str(kind),
                    "crop_path": str(crop_path),
                    "bbox": bbox_norm,
                    "score": float(score),
                    "concept": concept,
                    "mask_quality": quality.to_dict(),
                })

        for cand in dedup_by_iou(raw, iou_threshold=iou_threshold, bbox_key="bbox")[:max_per_kind]:
            qa = audit_crop(crop=Path(cand["crop_path"]), bbox_norm=cand["bbox"], kind=cand["kind"])
            if not qa.accepted:
                continue
            cand["qa"] = qa.to_dict()
            discovered.append(cand)
            excluded.append(cand["bbox"])  # never report the same region twice

    if embedder is not None and discovered:
        try:
            vectors = embedder.embed_batch([Path(c["crop_path"]) for c in discovered])
            if len(vectors) == len(discovered):
                for cand, vec in zip(discovered, vectors):
                    cand["embedding"] = vec
        except Exception:  # noqa: BLE001 - embeddings are an optimization, not required
            pass
    return discovered


def _mask_fill_ratio(mask: object, bbox_norm: list[int], height: int, width: int) -> float:
    import numpy as np

    y0, x0, y1, x1 = bbox_norm
    top = max(0, min(height - 1, round(y0 / 1000 * height)))
    left = max(0, min(width - 1, round(x0 / 1000 * width)))
    bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
    right = max(left + 1, min(width, round(x1 / 1000 * width)))
    array = np.asarray(mask, dtype=bool)[top:bottom, left:right]
    if array.size == 0:
        return 0.0
    return float(array.mean())


__all__ = ["DISCOVERY_CONCEPTS", "discover_entities"]
