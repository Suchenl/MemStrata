"""Deterministic decompose cropper: re-observe named entities from the *generated* segment.

The video backend continues each segment (i2v) from a scene keyframe that ``KeyframeComposer``
built by placing every selected entity's crop into an R3/R4 layout region. Because the
generation is anchored to that keyframe, each named entity stays in its assigned region.
This cropper therefore recovers a **fresh** observation of every named entity by cropping
its known region out of a late frame of the generated video — no extra detector/VLM.

Why this matters: those fresh crops (which show generation drift / new pose / new state)
are what ``MemoryUpdater`` ingests, so the stratified bank actually *grows* segment over
segment instead of re-reading the same seed crop. It plugs into ``RoleAwareDecomposer`` as
the ``cropper`` (the ``Cropper`` protocol: ``crop(segment_video, entity, *, segment_id)``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memstrata.skills.layout_anchor_processing import ColorBlockProcessor, LayoutElement

if TYPE_CHECKING:  # avoid import cycle at runtime
    from memstrata.steps.decompose import NamedEntity
    from memstrata.steps.keyframe import KeyframeComposer

logger = logging.getLogger(__name__)


class KeyframeRegionCropper:
    """Crop each named entity out of the generated video using the layout regions that
    ``KeyframeComposer`` computed for the same segment (shared via its ``_cache``)."""

    def __init__(
        self,
        composer: "KeyframeComposer",
        *,
        width: int = 640,
        height: int = 384,
        work_dir: str | Path = "/tmp/memstrata_observations",
        frame_pos: float = 0.8,
        margin: float = 0.06,
    ) -> None:
        self.composer = composer
        self.width = int(width)
        self.height = int(height)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Sample a late frame so the observation reflects generation drift, not the anchor.
        self.frame_pos = min(max(float(frame_pos), 0.0), 1.0)
        # Shrink the region slightly so we crop the entity, not the region seams.
        self.margin = max(0.0, float(margin))
        self._proc = ColorBlockProcessor(self.width, self.height)

    def _region_for(self, entity: "NamedEntity", record: dict[str, Any]) -> tuple[int, int, int, int] | None:
        elements = record.get("elements") or []
        assignments = record.get("assignments") or []
        target = str(entity.entity_id or "")
        box_index: int | None = None
        for a in assignments:
            if str(a.get("asset_id", "")) == target:
                bi = a.get("box_index")
                if isinstance(bi, int):
                    box_index = bi
                break
        if box_index is None or not (0 <= box_index < len(elements)):
            return None
        elem = LayoutElement.from_dict(elements[box_index])
        xmin, ymin, xmax, ymax = self._proc.scale_coordinates(
            elem.box_2d, self.width, self.height, normalized_range=(0, 1000)
        )
        if self.margin:
            dx = int((xmax - xmin) * self.margin)
            dy = int((ymax - ymin) * self.margin)
            xmin, ymin, xmax, ymax = xmin + dx, ymin + dy, xmax - dx, ymax - dy
        if xmax - xmin < 8 or ymax - ymin < 8:
            return None
        return xmin, ymin, xmax, ymax

    def _load_frame(self, segment_video: str):
        try:
            import imageio.v3 as iio
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            logger.warning("KeyframeRegionCropper: imageio/PIL unavailable (%s)", exc)
            return None
        try:
            frames = iio.imread(segment_video, index=None)  # (T, H, W, 3)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KeyframeRegionCropper: cannot read %s (%s)", segment_video, exc)
            return None
        if frames is None or len(frames) == 0:
            return None
        idx = min(len(frames) - 1, int(round(self.frame_pos * (len(frames) - 1))))
        img = Image.fromarray(frames[idx]).convert("RGB")
        # Layout coords are in (self.width, self.height); align the frame to it.
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height))
        return img

    def crop(self, segment_video: str, entity: "NamedEntity", *, segment_id: int) -> str | None:
        record = self.composer._cache.get(segment_id)
        if not record:
            logger.info("KeyframeRegionCropper: no keyframe record for segment %s; skip.", segment_id)
            return None
        box = self._region_for(entity, record)
        if box is None:
            logger.info("KeyframeRegionCropper: no region for %s in segment %s; skip.",
                        entity.entity_id, segment_id)
            return None
        frame = self._load_frame(segment_video)
        if frame is None:
            return None
        crop = frame.crop(box)
        out_dir = self.work_dir / f"segment_{segment_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"obs_{entity.entity_id}.png"
        crop.save(out)
        logger.info("KeyframeRegionCropper: segment %s %s region=%s -> %s",
                    segment_id, entity.entity_id, box, out)
        return str(out)


__all__ = ["KeyframeRegionCropper"]
