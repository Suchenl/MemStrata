"""Read media metadata through ffprobe; load reference crops for model feed."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    try:
        import imageio.v3 as iio
        from PIL import Image
    except Exception:  # noqa: BLE001 - optional at import time, absent in no-GPU smokes
        return []
    try:
        frames = iio.imread(str(video), index=None)  # (T, H, W, 3)
    except Exception:  # noqa: BLE001 - unreadable/absent video
        return []
    total = len(frames)
    if total == 0:
        return []
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    # Interior positions: the very first/last frames of a generated clip are the most likely
    # to be a fade or a duplicated boundary frame.
    picks = sorted({min(total - 1, max(0, round(total * p))) for p in _frame_positions(count)})
    paths: list[str] = []
    for order, index in enumerate(picks):
        out = out_root / f"{prefix}_{order}.png"
        try:
            Image.fromarray(frames[index]).convert("RGB").save(out)
        except Exception:  # noqa: BLE001
            continue
        paths.append(str(out))
    return paths


def _frame_positions(count: int) -> list[float]:
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
