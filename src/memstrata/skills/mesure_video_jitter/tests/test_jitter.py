"""Tests for the video-jitter skill.

The numeric core is exercised with synthetic textures shifted by known amounts, so phase
correlation is really run but no video file is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from memstrata.skills.mesure_video_jitter import (
    VERDICT_JITTERY,
    VERDICT_NOISE_DOMINATED,
    VERDICT_STEADY,
    classify,
    compare,
    jitter_from_translations,
    measure_jitter,
    translations_from_frames,
)


def _texture(width: int = 160, height: int = 120, seed: int = 0) -> np.ndarray:
    """A blurred noise field: enough structure for phase correlation to lock onto."""
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.random((height, width)).astype(np.float32) * 255.0
    return cv2.GaussianBlur(base, (0, 0), 2.0)


def _shifted(texture: np.ndarray, dx: float, dy: float) -> np.ndarray:
    import cv2

    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        texture, matrix, (texture.shape[1], texture.shape[0]), borderMode=cv2.BORDER_REFLECT
    )


def _clip(offsets: list[tuple[float, float]]) -> list[np.ndarray]:
    texture = _texture()
    return [_shifted(texture, dx, dy) for dx, dy in offsets]


def test_smooth_pan_is_steady() -> None:
    """Constant-velocity pan: large speed, near-zero acceleration."""
    offsets = [(i * 2.0, 0.0) for i in range(12)]
    report = jitter_from_translations(translations_from_frames(_clip(offsets)))
    assert report.speed_px > 1.5
    assert report.accel_px < 0.15
    assert report.ratio < 0.2
    assert report.verdict == VERDICT_STEADY


def test_random_walk_is_noise_dominated() -> None:
    """Framing re-decided every frame: acceleration rivals speed."""
    rng = np.random.default_rng(7)
    offsets = [(float(x), float(y)) for x, y in rng.normal(0, 1.2, size=(14, 2))]
    report = jitter_from_translations(translations_from_frames(_clip(offsets)))
    assert report.ratio >= 1.0
    assert report.verdict == VERDICT_NOISE_DOMINATED


def test_pan_with_shake_separates_from_clean_pan() -> None:
    """A pan plus per-frame tremor keeps the speed but gains acceleration."""
    rng = np.random.default_rng(3)
    clean = [(i * 2.0, 0.0) for i in range(12)]
    shaky = [(i * 2.0 + rng.normal(0, 0.6), rng.normal(0, 0.6)) for i in range(12)]
    clean_report = jitter_from_translations(translations_from_frames(_clip(clean)))
    shaky_report = jitter_from_translations(translations_from_frames(_clip(shaky)))
    assert shaky_report.accel_px > 3 * clean_report.accel_px
    assert shaky_report.speed_px == pytest.approx(clean_report.speed_px, rel=0.6)


def test_frame_count_is_reported() -> None:
    report = jitter_from_translations([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)])
    assert report.frames == 4


def test_too_short_clip_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 3 frames"):
        jitter_from_translations([(1.0, 0.0)])


def test_static_clip_reports_infinite_ratio_not_a_crash() -> None:
    """No motion means no denominator; the amplitude is what matters."""
    report = jitter_from_translations([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
    assert report.speed_px == 0.0
    assert report.ratio == float("inf")
    assert report.verdict == VERDICT_NOISE_DOMINATED


def test_verdict_thresholds() -> None:
    assert classify(accel_px=0.03, ratio=0.1) == VERDICT_STEADY
    assert classify(accel_px=0.19, ratio=0.66) == VERDICT_JITTERY  # A14B distill + morphic
    assert classify(accel_px=0.45, ratio=1.29) == VERDICT_NOISE_DOMINATED  # Turbo 5B
    # Small amplitude but noise-dominated is still the worse verdict.
    assert classify(accel_px=0.01, ratio=1.4) == VERDICT_NOISE_DOMINATED


def test_missing_clip_is_reported_not_swallowed(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        measure_jitter(tmp_path / "nope.mp4")


def test_compare_skips_unreadable_clips(tmp_path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert compare([tmp_path / "missing.mp4", broken]) == []


def _write_clip(path, frames) -> bool:
    """Write frames as an mp4; returns False when this OpenCV build has no writer."""
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(cv2.cvtColor(np.uint8(np.clip(frame, 0, 255)), cv2.COLOR_GRAY2BGR))
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def test_cut_is_detected_and_excluded(tmp_path) -> None:
    """A hard cut must not be averaged into the jitter statistics."""
    from memstrata.skills.mesure_video_jitter import cut_indices, longest_cut_free_span

    shot_a = [_shifted(_texture(seed=1), i * 0.4, 0.0) for i in range(30)]
    shot_b = [_shifted(_texture(seed=99), i * 0.4, 0.0) for i in range(40)]
    frames = shot_a + shot_b

    cuts = cut_indices(frames)
    assert cuts == [29], f"expected one cut at the seam, got {cuts}"
    assert longest_cut_free_span(frames) == (30, 70)


def test_single_cut_is_reported_but_medians_survive_it(tmp_path) -> None:
    """One cut among many frames is already absorbed by the median; it is still reported."""
    shot_a = [_shifted(_texture(seed=1), i * 0.4, 0.0) for i in range(30)]
    shot_b = [_shifted(_texture(seed=99), i * 0.4, 0.0) for i in range(40)]
    clip = tmp_path / "two_shots.mp4"
    if not _write_clip(clip, shot_a + shot_b):
        pytest.skip("no mp4 writer in this OpenCV build")

    within = measure_jitter(clip, within_shot=True)
    across = measure_jitter(clip, within_shot=False)
    assert within.cuts_detected == 1
    assert within.window == (30, 70)
    # Cuts are always counted; only the window differs, and the default keeps the whole clip.
    assert across.cuts_detected == 1 and across.window is None
    assert within.accel_px == pytest.approx(across.accel_px, rel=0.3)


def test_rapid_cutting_refuses_rather_than_reporting_a_bogus_number(tmp_path) -> None:
    """Cut every few frames: no window is long enough, so the measurement must decline."""
    frames = []
    for shot in range(10):
        texture = _texture(seed=shot)
        frames += [_shifted(texture, i * 0.4, 0.0) for i in range(6)]
    clip = tmp_path / "rapid_cuts.mp4"
    if not _write_clip(clip, frames):
        pytest.skip("no mp4 writer in this OpenCV build")

    with pytest.raises(ValueError, match="no cut-free window"):
        measure_jitter(clip, within_shot=True)
    # Insisting on the whole timeline still returns a number, with the jump count attached.
    whole = measure_jitter(clip, within_shot=False)
    assert whole.cuts_detected >= 8 and whole.window is None
    assert compare([clip], within_shot=True) == []


def test_continuous_clip_reports_no_window(tmp_path) -> None:
    frames = [_shifted(_texture(), i * 0.4, 0.0) for i in range(40)]
    clip = tmp_path / "one_shot.mp4"
    if not _write_clip(clip, frames):
        pytest.skip("no mp4 writer in this OpenCV build")
    report = measure_jitter(clip)
    assert report.cuts_detected == 0
    assert report.window is None


def test_report_line_is_human_readable() -> None:
    report = jitter_from_translations([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)], source="/tmp/a.mp4")
    line = report.as_line()
    assert "a.mp4" in line and "speed=" in line and "accel=" in line and "ratio=" in line
