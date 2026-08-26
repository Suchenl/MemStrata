"""Oracle backend: ffmpeg-slice a frozen source segment (self-contained)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from memstrata.steps.generate.schemas import GenerationArtifact, MediaGenerationTask, MediaTaskType


def _new_id(prefix: str) -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for segment in iter(lambda: f.read(1 << 20), b""):
            h.update(segment)
    return h.hexdigest()[:16]


def _ffmpeg_split(source: Path, output: Path, start_sec: float, duration_sec: float) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}", "-t", f"{duration_sec:.6f}",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-movflags", "+faststart",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output.is_file() and output.stat().st_size > 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


class OracleBackend:
    """Reveal a frozen source segment as the generated segment.

    Reads time range from ``task.controls``:
      source_video, source_start_sec, source_end_sec | duration_sec
    """

    def __init__(self, output_dir: str | Path, *, model_name: str = "oracle") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("OracleBackend only supports video_segment tasks")
        controls = task.controls
        source = Path(str(controls.get("source_video", "")))
        if not source.is_file():
            raise FileNotFoundError(f"oracle source_video missing: {source}")

        start = float(controls.get("source_start_sec", 0.0))
        if "source_end_sec" in controls:
            duration = max(0.01, float(controls["source_end_sec"]) - start)
        else:
            duration = float(controls.get("duration_sec", 2.0))

        out = self.output_dir / f"{task.segment_id}.mp4"
        notes: list[str] = []
        if not _ffmpeg_split(source, out, start, duration):
            shutil.copy2(source, out)
            notes.append("oracle_full_copy_no_ffmpeg")

        return GenerationArtifact(
            artifact_id=_new_id("artifact"),
            task_id=task.task_id,
            segment_id=task.segment_id,
            media_type="video",
            object_hash=_file_digest(out),
            object_uri=str(out.resolve()),
            model_name=self.model_name,
            degradation_notes=notes,
        )
