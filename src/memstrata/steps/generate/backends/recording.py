"""Recording backend: captures MediaGenerationTask payloads for protocol tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from memstrata.steps.generate.schemas import GenerationArtifact, MediaGenerationTask, MediaTaskType


def _new_id(prefix: str) -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RecordingBackend:
    """Does not call a real model; writes a tiny placeholder and stores every task."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tasks: list[MediaGenerationTask] = []
        self.artifacts: list[GenerationArtifact] = []

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        self.tasks.append(task)
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("RecordingBackend smoke path expects video_segment")
        out = self.output_dir / f"{task.segment_id}.mp4"
        # Minimal ISO BMFF-ish placeholder so callers see a file path.
        payload = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + task.prompt.encode()[:64]
        out.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()[:16]
        artifact = GenerationArtifact(
            artifact_id=_new_id("artifact"),
            task_id=task.task_id,
            segment_id=task.segment_id,
            media_type="video",
            object_hash=digest,
            object_uri=str(out.resolve()),
            model_name=task.model_name or "recording",
            degradation_notes=["recording_backend_placeholder"],
        )
        self.artifacts.append(artifact)
        return artifact
