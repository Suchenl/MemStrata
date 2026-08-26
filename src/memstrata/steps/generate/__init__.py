"""MemStrata Step-2 adapter: ComposedContext → MediaGenerationTask → GenerationArtifact.

controls['composed_references'], controls['continuation'], MediaGenerator.execute(task)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memstrata.bank import AssetBank
from memstrata.steps.compose import ComposedContext
from memstrata.steps.generate.executor import MediaGenerationBackend, MediaGenerator
from memstrata.steps.generate.materialize import composed_reference_images, reference_directives
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class GenerationResult:
    """Pipeline-facing result wrapping GenerationArtifact."""

    segment_id: int
    video_path: str | None
    meta: dict
    artifact: GenerationArtifact | None = None
    task: MediaGenerationTask | None = None


class NullGenerator:
    """No-op Step 2 (Track A / plan-only)."""

    def generate(
        self,
        enhanced_prompt: str,
        context: ComposedContext,
        *,
        segment_id: int,
        bank: AssetBank | None = None,
        controls: dict[str, Any] | None = None,
    ) -> GenerationResult:
        _ = enhanced_prompt, context, bank, controls
        return GenerationResult(segment_id=segment_id, video_path=None, meta={"mode": "null"})


class OracleGenerator:
    """Map segment_id → pre-known video path without building a Task."""

    def __init__(self, oracle_paths: dict[int, str] | None = None) -> None:
        self.oracle_paths = oracle_paths or {}

    def generate(
        self,
        enhanced_prompt: str,
        context: ComposedContext,
        *,
        segment_id: int,
        bank: AssetBank | None = None,
        controls: dict[str, Any] | None = None,
    ) -> GenerationResult:
        _ = enhanced_prompt, context, bank, controls
        path = self.oracle_paths.get(segment_id)
        return GenerationResult(segment_id=segment_id, video_path=path, meta={"mode": "oracle_map"})


class MediaTaskGenerator:
    """Build MediaGenerationTask from Composed Context, then MediaGenerator.execute."""

    def __init__(
        self,
        backend: MediaGenerationBackend,
        *,
        bank: AssetBank,
        model_name: str = "oracle",
        plan_version: int = 1,
        default_controls: dict[str, Any] | None = None,
        log_dir: str | Path | None = None,
        keyframe_composer: Any = None,
    ) -> None:
        self.bank = bank
        self.model_name = model_name
        self.plan_version = plan_version
        self.default_controls = dict(default_controls or {})
        self.executor = MediaGenerator(backend)
        # Optional Crop2Image keyframe step: turns the selected crops into a single fused
        # scene keyframe (R3->R4->FLUX) that the video backend continues from. When set,
        # ``composed_references`` becomes that one keyframe; raw crops move to ``source_crops``.
        self.keyframe_composer = keyframe_composer
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prev_artifact: GenerationArtifact | None = None
        self.history: list[dict[str, Any]] = []
        # The current scene's style/identity anchor (the last FLUX-composed keyframe). continue_ar
        # reuses it as Helios's always-kept first frame so identity persists across the window.
        self.scene_anchor: str | None = None

    def build_task(
        self,
        prompt: str,
        context: ComposedContext,
        *,
        segment_id: int,
        controls: dict[str, Any] | None = None,
        continue_from_previous: bool = True,
    ) -> MediaGenerationTask:
        merged = dict(self.default_controls)
        if controls:
            merged.update(controls)

        # R2b generation mode decides seeding. continue_ar / reanchor_lastframe reuse the
        # prior segment (no fresh FLUX keyframe); recompose_* build a keyframe from memory.
        gen_mode = str(merged.get("gen_mode", "recompose_keyframe"))
        needs_keyframe = gen_mode not in {"continue_ar", "reanchor_lastframe"}

        if needs_keyframe:
            references = composed_reference_images(context, self.bank)
            if references:
                merged["composed_references"] = references
            # Crop2Image: fuse the selected crops into one scene keyframe and hand THAT to the
            # video backend (raw crops preserved under source_crops for logging). This runs even
            # when ``references`` is empty — a cold-start / all-first-appearance segment has no crops
            # yet, so the composer bootstraps the keyframe via FLUX text-to-image from the prompt
            # (seed comment: "the first shot's keyframe bootstraps each entity's visual").
            if self.keyframe_composer is not None:
                kf = self.keyframe_composer.compose_keyframe(
                    prompt, references, segment_id=segment_id
                )
                if kf and kf.get("keyframe"):
                    if references:
                        merged["source_crops"] = references
                    merged["keyframe_record"] = kf
                    merged["composed_references"] = [{
                        "asset_id": "__keyframe__",
                        "kind": "keyframe",
                        "name": "scene_keyframe",
                        "role": "keyframe",
                        "image": kf["keyframe"],
                    }]
                    # This keyframe becomes the scene's style anchor for later continue_ar.
                    self.scene_anchor = kf["keyframe"]
        else:
            # Continuation modes: seed Helios from the prior segment; pass the scene anchor so
            # continue_ar keeps identity. history_video falls back to prev_artifact in the backend.
            if self.prev_artifact is not None and self.prev_artifact.object_uri:
                merged.setdefault("history_video", self.prev_artifact.object_uri)
            if self.scene_anchor:
                merged.setdefault("style_anchor", self.scene_anchor)

        if continue_from_previous and self.prev_artifact is not None:
            transition = str(merged.get("transition", "continue")).lower()
            if transition in {"continue", "continuation", "continuity"}:
                merged["continuation"] = {
                    "source_video": self.prev_artifact.object_uri,
                    "source_artifact_id": self.prev_artifact.artifact_id,
                }

        rep_ids: list[str] = []
        for aid in context.asset_ids:
            rep_ids.extend(context.representation_ids.get(aid, []))

        return MediaGenerationTask(
            task_id=_new_id("task"),
            segment_id=str(segment_id),
            task_type=MediaTaskType.VIDEO_SEGMENT,
            plan_version=self.plan_version,
            model_name=str(merged.pop("model_name", self.model_name)),
            prompt=prompt,
            input_representation_ids=rep_ids,
            reference_directives=reference_directives(context, self.bank),
            controls=merged,
        )

    def generate(
        self,
        enhanced_prompt: str,
        context: ComposedContext,
        *,
        segment_id: int,
        bank: AssetBank | None = None,
        controls: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if bank is not None:
            self.bank = bank
        task = self.build_task(
            enhanced_prompt,
            context,
            segment_id=segment_id,
            controls=controls,
        )
        artifact = self.executor.execute(task)
        self.prev_artifact = artifact

        record = {
            "segment_id": segment_id,
            "task": task.to_dict(),
            "artifact": artifact.to_dict(),
            "composed_asset_ids": list(context.asset_ids),
            "composed_functions": dict(context.functions),
            "exclusions": list(context.exclusions),
        }
        self.history.append(record)
        if self.log_dir is not None:
            segment_dir = self.log_dir / f"segment_{segment_id:03d}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            (segment_dir / "media_generation_task.json").write_text(
                json.dumps(task.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (segment_dir / "generation_artifact.json").write_text(
                json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (segment_dir / "composed_context.json").write_text(
                json.dumps({
                    "asset_ids": context.asset_ids,
                    "representation_ids": context.representation_ids,
                    "functions": context.functions,
                    "requirements": context.requirements,
                    "exclusions": context.exclusions,
                    "enhanced_prompt": context.enhanced_prompt,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return GenerationResult(
            segment_id=segment_id,
            video_path=artifact.object_uri,
            meta={
                "mode": "media_task",
                "model_name": artifact.model_name,
                "degradation_notes": list(artifact.degradation_notes),
                "composed_references": list(task.controls.get("composed_references") or []),
                "has_continuation": "continuation" in task.controls,
            },
            artifact=artifact,
            task=task,
        )


Generator = MediaTaskGenerator | NullGenerator | OracleGenerator
