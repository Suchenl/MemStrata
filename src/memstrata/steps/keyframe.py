"""Crop2Image keyframe composition — the bridge between MemStrata's stratified memory
and a reference-conditioned video generator.

The generator (Helios / SVI) continues a segment from a single *scene keyframe*, but the
memory bank stores per-entity **crops**, not scenes. This module turns the crops that
``compose`` selected for a segment into one coherent photorealistic keyframe:

    prompt + selected crops
        --R3 (LayoutPlanner, MLLM)-->  color-block layout regions
        --R4 (assign, MLLM) + composite-->  collage (real crops pasted into regions)
        --FLUX.2 Klein I2I fuse-->          one coherent scene keyframe

The keyframe is then handed to the video backend as ``controls['composed_references']``
(a single fused image) instead of the raw crops, so Helios animates a real scene while
identities come from memory. R3/R4 run on the single multimodal Qwen3.5-9B; the FLUX
fusion runs on the vendored FLUX image backend (its own env/persistent server).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from memstrata.mllm.runner import MllmRoleRunner
from memstrata.skills.layout_anchor_processing import (
    ColorBlockProcessor,
    CropRef,
    LayoutElement,
    LayoutPlanner,
    crop2image_canvas,
)
from memstrata.steps.generate.schemas import MediaGenerationTask, MediaTaskType

logger = logging.getLogger(__name__)


class KeyframeComposer:
    """Compose selected memory crops into one fused scene keyframe (Crop2Image).

    Parameters
    ----------
    image_backend:
        A backend with ``generate(MediaGenerationTask) -> GenerationArtifact`` that does
        FLUX.2 Klein I2I (e.g. ``build_image_backend('flux.2-klein-9b-kv-...')``). It is
        fed the collage as a single ``composed_references`` image and returns the fused
        keyframe artifact (its ``object_uri`` is the keyframe path).
    runner:
        Shared ``MllmRoleRunner`` for R3/R4 (defaults to the unified Qwen3.5-9B).
    width / height:
        Keyframe resolution (match the video backend's, e.g. 640x384 for Helios).
    steps / seed:
        FLUX I2I controls.
    """

    def __init__(
        self,
        image_backend: Any = None,
        *,
        runner: MllmRoleRunner | None = None,
        width: int = 640,
        height: int = 384,
        steps: int = 4,
        seed: int = 2026,
        work_dir: str | Path | None = None,
        mllm_timeout: float = 300.0,
    ) -> None:
        self.image_backend = image_backend
        # Generous timeout: the first R3/R4 call after a cold Qwen load compiles the
        # structured-output grammar + thinking tokens and can exceed the 90s default.
        self.runner = runner or MllmRoleRunner(timeout=mllm_timeout)
        self.planner = LayoutPlanner(runner=self.runner)
        self.color_block = ColorBlockProcessor(width, height)
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed
        self.work_dir = Path(work_dir) if work_dir else Path("/tmp/memstrata_keyframes")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # segment_id -> keyframe record. Lets a keyframes-first phase (FLUX resident) precompute
        # keyframes, then a video phase (Helios resident, FLUX down) reuse them without a
        # second FLUX call — the two big models never need to co-reside on one card.
        self._cache: dict[int, dict[str, Any]] = {}

    def _crop_refs(self, references: list[dict[str, Any]]) -> list[CropRef]:
        crops: list[CropRef] = []
        for ref in references:
            img = ref.get("image")
            if not img or not Path(str(img)).is_file():
                continue
            crops.append(
                CropRef(
                    asset_id=str(ref.get("asset_id", "")),
                    name=str(ref.get("name", "")),
                    kind=str(ref.get("kind", "")),
                    image_path=str(img),
                    representation_id=ref.get("representation_id"),
                )
            )
        return crops

    def compose_keyframe(
        self,
        prompt: str,
        references: list[dict[str, Any]],
        *,
        segment_id: int,
        use_mllm: bool = True,
    ) -> dict[str, Any] | None:
        """Return a record for the fused keyframe, or ``None`` if it cannot be built.

        Record: ``{keyframe, collage, anchor, elements, assignments, n_crops}``.
        """
        cached = self._cache.get(segment_id)
        if cached and cached.get("keyframe") and Path(str(cached["keyframe"])).is_file():
            logger.info("KeyframeComposer: segment %s keyframe from cache -> %s",
                        segment_id, cached["keyframe"])
            return cached

        crops = self._crop_refs(references)
        segment_dir = self.work_dir / f"segment_{segment_id:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)

        # Cold start / all-first-appearances: no crops in memory yet, so there is nothing to
        # fuse and no identity to preserve *against*. Bootstrap the scene's first visual directly
        # from the prompt via FLUX text-to-image; decompose then banks real crops from the
        # generated video, and later appearances go through the Crop2Image fusion path below.
        if not crops:
            if self.image_backend is None:
                logger.warning("KeyframeComposer: no crops and no image backend; skipping keyframe.")
                return None
            task = MediaGenerationTask(
                task_id=f"kf_{segment_id:03d}",
                segment_id=f"kf_{segment_id:03d}",
                task_type=MediaTaskType.KEYFRAME,
                plan_version=1,
                model_name="flux_klein",
                prompt=prompt,
                controls={"width": self.width, "height": self.height,
                          "steps": self.steps, "seed": self.seed},
            )
            artifact = self.image_backend.generate(task)
            logger.info("KeyframeComposer: segment %s keyframe (flux_t2i bootstrap) -> %s",
                        segment_id, artifact.object_uri)
            record = {
                "keyframe": artifact.object_uri, "fused": "flux_t2i",
                "collage": None, "anchor": None, "elements": [], "assignments": [],
                "n_crops": 0,
            }
            self._cache[segment_id] = record
            return record

        # R3: layout regions from the prompt.
        elements = self.planner.plan_layout(prompt)
        anchor = self.color_block.render_anchor(
            [LayoutElement.from_dict(e) for e in elements], self.width, self.height
        )
        anchor_path = segment_dir / "anchor.png"
        anchor.save(anchor_path)

        # R4 + composite: crops pasted into their assigned regions over the anchor.
        collage, assignments = crop2image_canvas(
            elements,
            crops,
            width=self.width,
            height=self.height,
            anchor=anchor,
            runner=self.runner,
            use_mllm=use_mllm,
        )
        collage_path = segment_dir / "collage.png"
        collage.save(collage_path)

        # FLUX I2I fuse the collage into a coherent keyframe. Without a FLUX backend the
        # collage itself is used as the keyframe (still a valid crop-conditioned anchor;
        # the fusion just makes it photorealistically coherent).
        fused = "collage"
        keyframe = str(collage_path)
        if self.image_backend is not None:
            task = MediaGenerationTask(
                task_id=f"kf_{segment_id:03d}",
                segment_id=f"kf_{segment_id:03d}",
                task_type=MediaTaskType.KEYFRAME,
                plan_version=1,
                model_name="flux_klein",
                prompt=prompt,
                controls={
                    "composed_references": [{"image": str(collage_path)}],
                    "width": self.width,
                    "height": self.height,
                    "steps": self.steps,
                    "seed": self.seed,
                },
            )
            artifact = self.image_backend.generate(task)
            keyframe = artifact.object_uri
            fused = "flux_i2i"
        logger.info("KeyframeComposer: segment %s keyframe (%s) -> %s", segment_id, fused, keyframe)
        record = {
            "keyframe": keyframe,
            "fused": fused,
            "collage": str(collage_path),
            "anchor": str(anchor_path),
            "elements": elements,
            "assignments": assignments,
            "n_crops": len(crops),
        }
        self._cache[segment_id] = record
        return record


__all__ = ["KeyframeComposer"]
