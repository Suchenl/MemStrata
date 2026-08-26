"""Media generation task and archive artifact contracts (MemStrata Step 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MediaTaskType(str, Enum):
    VIDEO_SEGMENT = "video_segment"
    REFERENCE_IMAGE = "reference_image"
    KEYFRAME = "keyframe"
    IMAGE_EDIT = "image_edit"


@dataclass(slots=True)
class MediaGenerationTask:
    task_id: str
    segment_id: str
    task_type: MediaTaskType
    plan_version: int
    model_name: str
    prompt: str
    input_representation_ids: list[str] = field(default_factory=list)
    reference_directives: list[dict[str, Any]] = field(default_factory=list)
    controls: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "segment_id": self.segment_id,
            "task_type": self.task_type.value,
            "plan_version": self.plan_version,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "input_representation_ids": list(self.input_representation_ids),
            "reference_directives": list(self.reference_directives),
            "controls": dict(self.controls),
        }


@dataclass(slots=True)
class GenerationArtifact:
    artifact_id: str
    task_id: str
    segment_id: str
    media_type: str
    object_hash: str
    object_uri: str
    model_name: str
    degradation_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "segment_id": self.segment_id,
            "media_type": self.media_type,
            "object_hash": self.object_hash,
            "object_uri": self.object_uri,
            "model_name": self.model_name,
            "degradation_notes": list(self.degradation_notes),
            "created_at": self.created_at,
        }
