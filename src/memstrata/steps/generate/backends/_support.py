"""Local I/O helpers for video backends (no external package imports)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from memstrata.lib.weights import hf_cache_dir, repo_root, weights_root
from memstrata.steps.generate.schemas import GenerationArtifact


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for segment in iter(lambda: f.read(1 << 20), b""):
            h.update(segment)
    return h.hexdigest()


@dataclass(slots=True)
class RunContext:
    """Minimal workspace: temp + object archive under output_dir."""

    workspace_path: Path

    @classmethod
    def create(cls, output_dir: str | Path) -> RunContext:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "runs").mkdir(exist_ok=True)
        (root / "objects" / "sha256").mkdir(parents=True, exist_ok=True)
        return cls(workspace_path=root)


class ArtifactStore:
    """Archive generated media under the run workspace (sha256 content-addressed)."""

    def __init__(self, context: RunContext | None = None) -> None:
        self.context = context

    def import_object(self, context: RunContext, source: Path) -> tuple[str, Path]:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source}")
        digest = file_sha256(source)
        suffix = source.suffix.lower() or ".mp4"
        destination = context.workspace_path / "objects" / "sha256" / digest[:2] / f"{digest}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        return digest, destination


def write_placeholder_mp4(path: Path, *, seconds: float = 1.0) -> list[str]:
    """Write a tiny H.264 clip so CPU smoke can concat without a generator.

    Prefers ffmpeg / imageio-ffmpeg. Falls back to a few ISO-BMFF bytes so the
    path still exists when no encoder is available.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    ff = None
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg")
    if ff:
        duration = max(0.25, float(seconds))
        try:
            subprocess.run(
                [
                    ff, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=black:s=320x240:d={duration:g}:r=8",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(path),
                ],
                check=True, capture_output=True, timeout=30,
            )
            if path.is_file() and path.stat().st_size > 0:
                notes.append("placeholder_mp4")
                return notes
        except Exception:
            pass
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isomplaceholder")
    notes.append("placeholder_mp4_bytes")
    return notes


def resolve_model_reference(model: str) -> str:
    token = "${PUBLIC_MODELS_ROOT}"
    if model.startswith(token):
        override = os.environ.get("PUBLIC_MODELS_ROOT")
        root = Path(override).expanduser().resolve() if override else Path(
            "${PUBLIC_MODELS_ROOT}"
        )
        return str(root / model[len(token):].lstrip("/"))
    return os.path.expandvars(os.path.expanduser(model))


def default_models_config_dir() -> Path:
    """Prefer MemStrata-local configs; fall back to repo model_configs."""
    memstrata_root = Path(__file__).resolve().parents[5]
    local = memstrata_root / "configs"
    if (local / "video_gen").is_dir():
        return local
    return repo_root() / "models" / "model_configs"


def make_artifact(
    *,
    task_id: str,
    segment_id: str,
    object_hash: str,
    object_uri: str,
    model_name: str,
    degradation_notes: list[str] | None = None,
) -> GenerationArtifact:
    return GenerationArtifact(
        artifact_id=new_id("artifact"),
        task_id=task_id,
        segment_id=segment_id,
        media_type="video",
        object_hash=object_hash,
        object_uri=object_uri,
        model_name=model_name,
        degradation_notes=list(degradation_notes or []),
    )


__all__ = [
    "ArtifactStore",
    "RunContext",
    "default_models_config_dir",
    "file_sha256",
    "hf_cache_dir",
    "make_artifact",
    "new_id",
    "repo_root",
    "resolve_model_reference",
    "weights_root",
    "write_placeholder_mp4",
]
