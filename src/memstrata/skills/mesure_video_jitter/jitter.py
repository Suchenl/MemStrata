"""Measure per-frame micro-jitter in a generated clip.

Displacement alone cannot tell a deliberate camera move from shake: a slow pan has large
displacement and no jitter at all, while a locked-off shot that trembles has almost no
displacement. The distinguishing signal is the *second* derivative of the motion. A real camera —
or a real film scan — accelerates smoothly, so acceleration stays well below speed. A generator
that re-decides the framing every frame produces a random walk whose acceleration rivals or
exceeds its speed, which is exactly what a viewer reads as "the picture keeps shaking".

Per-frame translation comes from phase correlation on a Hann-windowed grayscale pair, which tracks
the dominant global shift and ignores local subject motion.

Measured reference points are in this skill's README. The one to keep in mind here: a cut-free
window of real handheld film (*American Beauty*) measures ratio 1.44, worse than the Turbo clip that
prompted this tool. The metric ranks arms of the same shot; it does not certify realism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

#: Median per-frame acceleration measured on real film; the floor we compare generators against.
REAL_FILM_ACCEL_PX = 0.05

#: Above this multiple of the film floor, jitter is visible on a normal screen. Calibrated so the
#: A14B distill+morphic arm (0.19 px, ~4x the floor, still perceptible) is not called steady.
JITTER_ACCEL_MULTIPLE = 3.0

#: At or above this acceleration-to-speed ratio the motion is mostly frame-to-frame noise.
NOISE_DOMINATED_RATIO = 1.0

#: A frame pair whose mean absolute difference exceeds this multiple of the clip's median is a cut.
CUT_DIFF_MULTIPLE = 4.0

#: Never call a barely-changing pair a cut, however quiet the clip around it is (0-255 scale).
CUT_DIFF_FLOOR = 8.0

#: Shortest window worth measuring; below this the medians are not meaningful.
MIN_WINDOW_FRAMES = 24

VERDICT_STEADY = "steady"
VERDICT_JITTERY = "jittery"
VERDICT_NOISE_DOMINATED = "noise-dominated"


@dataclass(frozen=True)
class JitterReport:
    """Motion-stability summary for one clip."""

    frames: int
    speed_px: float
    accel_px: float
    ratio: float
    verdict: str
    source: str | None = None
    window: tuple[int, int] | None = None
    cuts_detected: int = 0

    def as_line(self) -> str:
        name = Path(self.source).name if self.source else "clip"
        cuts = f" cuts={self.cuts_detected}" if self.cuts_detected else ""
        span = f" window={self.window[0]}-{self.window[1]}" if self.window else ""
        return (
            f"{name:44s} frames={self.frames:4d} speed={self.speed_px:6.3f}px "
            f"accel={self.accel_px:6.3f}px ratio={self.ratio:5.2f} {self.verdict}{cuts}{span}"
        )


def classify(accel_px: float, ratio: float) -> str:
    """Turn the two numbers into a verdict.

    Ratio is checked first: motion dominated by frame-to-frame change reads as instability even
    when the absolute amplitude is small, because the eye tracks the inconsistency rather than the
    size.

    **This verdict is not an absolute quality gate.** Genuinely handheld footage is itself a
    small-amplitude random walk: a cut-free 91-frame window of *American Beauty* measures
    accel 0.55 px, ratio 1.44, i.e. ``noise-dominated``, while the locked-off *Casablanca* window
    measures 0.10 px / 0.59. Use the verdict to compare arms of the *same* shot and prompt, and
    always report ``speed_px`` next to ``accel_px`` so a clip that simply moves more is not
    mistaken for a clip that shakes.
    """
    if ratio >= NOISE_DOMINATED_RATIO:
        return VERDICT_NOISE_DOMINATED
    if accel_px > JITTER_ACCEL_MULTIPLE * REAL_FILM_ACCEL_PX:
        return VERDICT_JITTERY
    return VERDICT_STEADY


def translations_from_frames(frames: Iterable["object"]) -> "object":
    """Per-frame global translation for a sequence of grayscale float32 arrays."""
    import cv2
    import numpy as np

    prev = None
    window = None
    shifts: list[tuple[float, float]] = []
    for frame in frames:
        gray = np.asarray(frame, dtype=np.float32)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            if window is None:
                # A Hann window kills the frame-edge discontinuity that would otherwise dominate
                # the correlation peak.
                window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(prev, gray, window)
            shifts.append((dx, dy))
        prev = gray
    return np.asarray(shifts, dtype=np.float64)


def jitter_from_translations(shifts: Sequence[tuple[float, float]], *, source: str | None = None) -> JitterReport:
    """Summarise a translation sequence. Needs at least 3 frames (2 shifts) to have an accel."""
    import numpy as np

    array = np.asarray(shifts, dtype=np.float64).reshape(-1, 2)
    if len(array) < 2:
        raise ValueError(
            f"need at least 3 frames (2 translations) to measure acceleration, got {len(array) + 1}"
        )
    speed = np.linalg.norm(array, axis=1)
    accel = np.linalg.norm(np.diff(array, axis=0), axis=1)
    med_speed = float(np.median(speed))
    med_accel = float(np.median(accel))
    # A perfectly static clip has no motion to be noisy relative to; report the raw amplitude.
    ratio = med_accel / med_speed if med_speed > 1e-6 else float("inf")
    return JitterReport(
        frames=len(array) + 1,
        speed_px=med_speed,
        accel_px=med_accel,
        ratio=ratio,
        verdict=classify(med_accel, ratio),
        source=source,
    )


def cut_indices(frames: Sequence["object"]) -> list[int]:
    """Indices ``i`` where the pair ``(i, i+1)`` is a shot cut.

    Mean absolute frame difference, thresholded against the clip's own median. A hard cut moves
    every pixel at once, which a jitter measurement would read as one enormous translation and a
    matching acceleration spike — that single spike is enough to make real film look worse than a
    shaky generator, so cuts must be excluded rather than averaged over.
    """
    import numpy as np

    if len(frames) < 3:
        return []
    diffs = np.asarray(
        [
            float(np.mean(np.abs(np.float32(frames[i + 1]) - np.float32(frames[i]))))
            for i in range(len(frames) - 1)
        ]
    )
    threshold = max(CUT_DIFF_MULTIPLE * float(np.median(diffs)), CUT_DIFF_FLOOR)
    return [int(i) for i in np.flatnonzero(diffs > threshold)]


def longest_cut_free_span(frames: Sequence["object"]) -> tuple[int, int]:
    """Longest ``[start, end)`` run of frames containing no cut."""
    cuts = cut_indices(frames)
    if not cuts:
        return 0, len(frames)
    # A cut at pair i ends the run at frame i and starts the next at frame i+1.
    bounds = [0] + [c + 1 for c in cuts] + [len(frames)]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    return max(spans, key=lambda span: span[1] - span[0])


def read_frames(video: str | Path, *, max_frames: int = 200) -> list["object"]:
    """Decode up to ``max_frames`` grayscale frames from a clip."""
    import cv2

    path = Path(video)
    if not path.is_file():
        raise FileNotFoundError(f"no such clip: {path}")
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"decoded no frames from {path}; is it a readable video?")
    return frames


def measure_jitter(
    video: str | Path, *, max_frames: int = 200, within_shot: bool = False
) -> JitterReport:
    """Measure motion stability of a clip on disk.

    Content jumps are always counted and reported as ``cuts_detected`` — a stitched film measured
    whole, or a generated clip that morphs abruptly, both show up there. ``within_shot`` then
    restricts the measurement to the longest jump-free window. It is *off* by default so a number
    is never silently taken from a truncated window; turn it on when comparing a segment against a
    full movie, and quote the ``window`` alongside the number.
    """
    frames = read_frames(video, max_frames=max_frames)
    cuts = cut_indices(frames)
    window = None
    if within_shot and cuts:
        start, end = longest_cut_free_span(frames)
        if end - start < MIN_WINDOW_FRAMES:
            raise ValueError(
                f"{Path(video).name}: {len(cuts)} content jumps leave no cut-free window of "
                f"{MIN_WINDOW_FRAMES}+ frames (longest is {end - start}); "
                "measure a single shot instead"
            )
        window = (start, end)
        frames = list(frames)[start:end]
    shifts = translations_from_frames(frames)
    report = jitter_from_translations(shifts, source=str(video))
    return JitterReport(
        frames=report.frames, speed_px=report.speed_px, accel_px=report.accel_px,
        ratio=report.ratio, verdict=report.verdict, source=report.source,
        window=window, cuts_detected=len(cuts),
    )


def compare(
    videos: Sequence[str | Path], *, max_frames: int = 200, within_shot: bool = False
) -> list[JitterReport]:
    """Measure several clips, skipping unreadable ones so one bad path cannot lose the batch."""
    reports = []
    for video in videos:
        try:
            report = measure_jitter(video, max_frames=max_frames, within_shot=within_shot)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping %s: %s", video, exc)
            continue
        if report.cuts_detected and not within_shot:
            logger.warning(
                "%s spans %d content jumps; pass within_shot=True to measure one shot",
                Path(video).name, report.cuts_detected,
            )
        reports.append(report)
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument(
        "--within-shot", action="store_true",
        help="restrict to the longest jump-free window (default: measure the whole clip)",
    )
    args = parser.parse_args(argv)
    for report in compare(args.clips, max_frames=args.max_frames, within_shot=args.within_shot):
        print(report.as_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
