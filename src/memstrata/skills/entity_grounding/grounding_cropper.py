"""VLM-grounded decompose cropper (production R7).

Mirrors the ``vmem_bench`` S5 ``QwenImageGrounder`` crop-acquisition logic
(``s5_entities_visual_crop_acquisition/vlm_grounding.py``) but lives in the production
``memstrata`` package (zero-import of ``vmem_bench`` per design_philosophy.md) and
runs on the unified Qwen3.5-9B already serving R1-R4 via ``MllmRoleRunner``.

Unlike the layout-rectangle stand-in, this asks the VLM to *locate* the named entity in a
generated frame and returns a TIGHT box (+positive point). Decompose then crops that box,
so the observation stored in the bank is a clean, entity-isolated crop — not a background-
heavy region slice. SAM3 mask-refine and identity-consistency audit are the documented
follow-ons (still PARTIAL upstream); this is the targeted-grounding core.

Plugs into ``RoleAwareDecomposer`` as the ``Cropper`` protocol:
``crop(segment_video, entity, *, segment_id) -> str | None``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memstrata.mllm.runner import MllmRoleRunner

if TYPE_CHECKING:  # avoid import cycle at runtime
    from memstrata.steps.decompose import NamedEntity

logger = logging.getLogger(__name__)

# Mirror of the S5 grounding schema (we only *require* the three core fields; the extra
# visibility/confidence/reason are accepted but optional so validation stays lenient).
GROUNDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["usable", "bbox_norm", "point_norm"],
    "properties": {
        "usable": {"type": "boolean"},
        "bbox_norm": {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 4},
        "point_norm": {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 2},
        "visibility": {"type": "string"},
        "confidence": {"type": "string"},
        "reason": {"type": "string"},
    },
}

_PROMPT = (
    "Locate the named entity in THIS image only. Return a TIGHT axis-aligned box covering "
    "the entity body (not the whole frame).\n"
    "Coordinate system: integers 0-1000, bbox_norm=[xmin,ymin,xmax,ymax], point_norm=[x,y] "
    "must lie INSIDE the box.\n"
    "If the entity is absent/too occluded: usable=false and empty arrays.\n"
    "name={name}\nkind={kind}\ndescription={description}"
)


def _area(box: list[int]) -> float:
    y0, x0, y1, x1 = box
    return max(0, y1 - y0) * max(0, x1 - x0) / 1_000_000.0


def _valid_yxyx(box: list[int]) -> bool:
    y0, x0, y1, x1 = box
    return 0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000


def canonicalize_box_point(bbox: list[int], point: list[int]) -> tuple[list[int], list[int]] | None:
    """Return canonical [ymin,xmin,ymax,xmax] + [y,x], or None. Accepts yxyx or xyxy
    (Qwen-VL often emits xyxy). Faithful mirror of the S5 helper."""

    def xyxy_to_yxyx(v: list[int]) -> list[int]:
        x0, y0, x1, y1 = v
        return [y0, x0, y1, x1]

    def inside(box: list[int], pt: list[int]) -> bool:
        if len(pt) != 2:
            return False
        y, x = pt
        y0, x0, y1, x1 = box
        return y0 <= y <= y1 and x0 <= x <= x1

    if len(bbox) != 4:
        return None
    candidates = [list(bbox), xyxy_to_yxyx(bbox)]
    points = [list(point), [point[1], point[0]]] if len(point) == 2 else [[500, 500]]
    best: tuple[list[int], list[int]] | None = None
    best_key = (-1.0, False, -1.0)
    for box in candidates:
        if not _valid_yxyx(box):
            continue
        for pt in points:
            hit = inside(box, pt)
            key = (1.0 if hit else 0.0, _area(box) >= 0.02, _area(box))
            if key > best_key:
                best_key = key
                center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
                best = (box, pt if hit else center)
    if best is None or _area(best[0]) < 0.01:
        return None
    return best


class VlmGroundingCropper:
    """Crop each named entity out of a generated frame using VLM (R7) grounding."""

    def __init__(
        self,
        runner: MllmRoleRunner | None = None,
        *,
        work_dir: str | Path = "/tmp/memstrata_observations",
        frame_pos: float = 0.8,
        pad: float = 0.04,
        min_side: int = 16,
    ) -> None:
        self.runner = runner or MllmRoleRunner()
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.frame_pos = min(max(float(frame_pos), 0.0), 1.0)
        self.pad = max(0.0, float(pad))
        self.min_side = int(min_side)

    def _sample_frame(self, segment_video: str, out_path: Path):
        try:
            import imageio.v3 as iio
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            logger.warning("VlmGroundingCropper: imageio/PIL unavailable (%s)", exc)
            return None
        try:
            frames = iio.imread(segment_video, index=None)  # (T, H, W, 3)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VlmGroundingCropper: cannot read %s (%s)", segment_video, exc)
            return None
        if frames is None or len(frames) == 0:
            return None
        idx = min(len(frames) - 1, int(round(self.frame_pos * (len(frames) - 1))))
        img = Image.fromarray(frames[idx]).convert("RGB")
        img.save(out_path)
        return img

    def crop(self, segment_video: str, entity: "NamedEntity", *, segment_id: int) -> str | None:
        out_dir = self.work_dir / f"segment_{segment_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_path = out_dir / f"frame_{entity.entity_id}.jpg"
        frame = self._sample_frame(segment_video, frame_path)
        if frame is None:
            return None
        prompt = _PROMPT.format(
            name=entity.name,
            kind=getattr(entity.kind, "value", entity.kind),
            description=getattr(entity, "description", "") or entity.name,
        )
        try:
            res = self.runner.run(
                "entity_detector", instruction=prompt,
                images=[str(frame_path)], schema=GROUNDING_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001 - grounding failure must not break the loop
            logger.warning("VlmGroundingCropper: R7 grounding failed for %s (%s)", entity.entity_id, exc)
            return None
        if not res.get("usable"):
            logger.info("VlmGroundingCropper: entity %s not usable in segment %s (%s)",
                        entity.entity_id, segment_id, res.get("reason", ""))
            return None
        bbox = [int(v) for v in res.get("bbox_norm", [])]
        point = [int(v) for v in res.get("point_norm", [])]
        canon = canonicalize_box_point(bbox, point)
        if canon is None:
            logger.info("VlmGroundingCropper: unusable bbox %s for %s", bbox, entity.entity_id)
            return None
        (ymin, xmin, ymax, xmax), _pt = canon
        w, h = frame.size
        # normalized [0,1000] -> pixels, with a small pad.
        px0 = int(xmin / 1000.0 * w)
        py0 = int(ymin / 1000.0 * h)
        px1 = int(xmax / 1000.0 * w)
        py1 = int(ymax / 1000.0 * h)
        if self.pad:
            dx = int((px1 - px0) * self.pad)
            dy = int((py1 - py0) * self.pad)
            px0, py0, px1, py1 = max(0, px0 - dx), max(0, py0 - dy), min(w, px1 + dx), min(h, py1 + dy)
        if px1 - px0 < self.min_side or py1 - py0 < self.min_side:
            logger.info("VlmGroundingCropper: box too small for %s: %s", entity.entity_id, (px0, py0, px1, py1))
            return None
        crop = frame.crop((px0, py0, px1, py1))
        out = out_dir / f"obs_{entity.entity_id}.png"
        crop.save(out)
        logger.info("VlmGroundingCropper: segment %s %s box=%s -> %s",
                    segment_id, entity.entity_id, (px0, py0, px1, py1), out)
        return str(out)


__all__ = ["VlmGroundingCropper", "canonicalize_box_point", "GROUNDING_SCHEMA"]
