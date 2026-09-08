"""Read media metadata through ffprobe; load reference crops for model feed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Masked crops may be stored as RGBA PNG; composite onto white before model feed.
MODEL_FEED_BACKGROUND: tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_sec: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    format_name: str | None


@dataclass(frozen=True, slots=True)
class FrameSample:
    """One relative-position frame request.

    ``basis="count"`` preserves the namer's historical ``round(N * p)`` rule;
    ``basis="last"`` preserves the cropper's ``round((N - 1) * p)`` rule.
    """

    position: float
    output: Path
    basis: Literal["count", "last"] = "last"


def load_crop_rgb_for_model(
    path: Path | str,
    *,
    background: tuple[int, int, int] = MODEL_FEED_BACKGROUND,
) -> Any:
    """Load a crop as RGB, compositing RGBA onto ``background`` (default white)."""
    from PIL import Image

    image = Image.open(path)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*background, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def _video_binary(name: str, env_name: str) -> str | None:
    configured = os.environ.get(env_name, "").strip()
    if configured:
        return configured
    return shutil.which(name)


def _video_shape_and_count(video: Path | str) -> tuple[int, int, int] | None:
    """Read coded dimensions and container frame count without decoding the stream."""
    ffprobe = _video_binary("ffprobe", "FFPROBE_BIN")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        stream = json.loads(completed.stdout)["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        frame_count = int(stream["nb_frames"])
        if width > 0 and height > 0 and frame_count > 0:
            return width, height, frame_count
    except (KeyError, IndexError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        pass
    return None


def _sample_index(sample: FrameSample, frame_count: int) -> int:
    position = min(max(float(sample.position), 0.0), 1.0)
    scale = frame_count if sample.basis == "count" else frame_count - 1
    return min(frame_count - 1, max(0, round(scale * position)))


def _selective_rgb_frames(
    video: Path | str,
    *,
    width: int,
    height: int,
    indices: list[int],
) -> dict[int, Any] | None:
    """Decode only requested frame indices, stopping after the last requested frame."""
    ffmpeg = _video_binary("ffmpeg", "FFMPEG_BIN")
    if ffmpeg is None or not indices:
        return None
    unique = sorted(set(indices))
    expression = "+".join(f"eq(n\\,{index})" for index in unique)
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                f"select={expression}",
                "-vsync",
                "0",
                "-frames:v",
                str(len(unique)),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        frame_bytes = width * height * 3
        if len(completed.stdout) != frame_bytes * len(unique):
            return None
        from PIL import Image

        return {
            index: Image.frombytes(
                "RGB",
                (width, height),
                completed.stdout[offset * frame_bytes : (offset + 1) * frame_bytes],
            )
            for offset, index in enumerate(unique)
        }
    except (OSError, subprocess.SubprocessError):
        return None


def materialize_video_frames(
    video: Path | str,
    samples: list[FrameSample],
) -> list[Path]:
    """Materialize requested frames with one selective decode.

    Falls back to the historical full ``imageio`` decode when stream metadata or ffmpeg
    selection is unavailable. The fallback remains best-effort for minimal installations.
    """
    if not samples:
        return []
    metadata = _video_shape_and_count(video)
    images: dict[int, Any] | None = None
    indices: list[int] = []
    if metadata is not None:
        width, height, frame_count = metadata
        indices = [_sample_index(sample, frame_count) for sample in samples]
        images = _selective_rgb_frames(
            video,
            width=width,
            height=height,
            indices=indices,
        )
    if images is None:
        try:
            import imageio.v3 as iio
            from PIL import Image

            frames = iio.imread(str(video), index=None)
            if frames is None or len(frames) == 0:
                return []
            frame_count = len(frames)
            indices = [_sample_index(sample, frame_count) for sample in samples]
            images = {
                index: Image.fromarray(frames[index]).convert("RGB")
                for index in set(indices)
            }
        except Exception:  # noqa: BLE001 - optional dependency or unreadable video
            return []

    saved: list[Path] = []
    for sample, index in zip(samples, indices):
        try:
            sample.output.parent.mkdir(parents=True, exist_ok=True)
            images[index].save(sample.output)
            saved.append(sample.output)
        except Exception:  # noqa: BLE001 - one failed output must not discard the others
            continue
    return saved


def sample_video_frames(
    video: Path | str,
    out_dir: Path | str,
    *,
    count: int = 3,
    prefix: str = "frame",
) -> list[str]:
    """Write up to ``count`` evenly spaced frames of ``video`` as PNG; return their paths.

    Several views of the same segment are what let a namer recognise an entity that is only
    legible in one of them. Best-effort: returns ``[]`` rather than raising, because a caller
    that cannot sample frames must degrade to its non-visual path, not fail the segment.
    """
    if count <= 0:
        return []
    out_root = Path(out_dir)
    # Interior positions: the very first/last frames of a generated clip are the most likely
    # to be a fade or a duplicated boundary frame.
    samples = [
        FrameSample(position, out_root / f"{prefix}_{order}.png", basis="count")
        for order, position in enumerate(even_frame_positions(count))
    ]
    return [str(path) for path in materialize_video_frames(video, samples)]


def even_frame_positions(count: int) -> list[float]:
    if count == 1:
        return [0.5]
    step = 1.0 / (count + 1)
    return [step * (i + 1) for i in range(count)]


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def probe_media(path: Path) -> MediaInfo:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Media file does not exist: {source}")
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return _probe_media_with_cv2(source)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "ffprobe failed") from exc
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream found: {source}")
    duration = payload.get("format", {}).get("duration") or video.get("duration")
    if duration is None:
        raise RuntimeError(f"Cannot determine media duration: {source}")
    return MediaInfo(
        duration_sec=float(duration),
        width=video.get("width"),
        height=video.get("height"),
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
        format_name=payload.get("format", {}).get("format_name"),
    )


def _probe_media_with_cv2(source: Path) -> MediaInfo:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("ffprobe is not installed or not on PATH") from exc

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open media file: {source}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or None
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
    finally:
        capture.release()

    if not fps or frame_count <= 0:
        raise RuntimeError(f"Cannot determine media duration: {source}")
    return MediaInfo(
        duration_sec=frame_count / fps,
        width=width,
        height=height,
        fps=fps,
        has_audio=False,
        format_name=source.suffix.lstrip(".") or None,
    )
