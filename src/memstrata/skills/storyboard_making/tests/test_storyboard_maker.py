"""Tests for the storyboard skill's prompt contract and panel geometry guarantees."""

from __future__ import annotations

import pytest
from PIL import Image

from memstrata.skills.storyboard_making import (
    SHOT_SCOPE_ACROSS,
    SHOT_SCOPE_WITHIN,
    StoryboardMaker,
)
from memstrata.skills.storyboard_making.storyboard_maker import (
    _normalize_panel_sizes,
    _trim_white_border,
)


@pytest.fixture
def maker() -> StoryboardMaker:
    return StoryboardMaker()


def _panels(count: int) -> list[str]:
    return [f"the keeper climbs step {i}" for i in range(1, count + 1)]


def test_dimensions_slice_back_to_video_aspect(maker: StoryboardMaker) -> None:
    width, height = maker.calculate_dimensions(
        video_w=832, video_h=480, cols=3, rows=2, gutter=16, scale_factor=0.75
    )
    assert (width, height) == (1904, 736)
    panel_w = (width - 2 * 16) / 3
    panel_h = (height - 1 * 16) / 2
    assert panel_w / panel_h == pytest.approx(832 / 480, abs=1e-3)


def test_within_shot_scope_forbids_a_cut(maker: StoryboardMaker) -> None:
    prompt, _ = maker.format_prompt(
        "cinematic", _panels(4), cols=2, rows=2, shot_scope=SHOT_SCOPE_WITHIN
    )
    assert "ONE SINGLE CONTINUOUS SHOT" in prompt
    assert "Never cut to another angle" in prompt


def test_across_shots_scope_allows_a_cut_but_pins_identity(maker: StoryboardMaker) -> None:
    prompt, _ = maker.format_prompt(
        "cinematic", _panels(4), cols=2, rows=2, shot_scope=SHOT_SCOPE_ACROSS
    )
    assert "SEPARATE SHOTS" in prompt
    assert "SHOULD differ from panel to panel" in prompt
    assert "identity and face" in prompt


def test_unknown_shot_scope_is_rejected(maker: StoryboardMaker) -> None:
    with pytest.raises(ValueError, match="shot_scope"):
        maker.format_prompt("cinematic", _panels(4), cols=2, rows=2, shot_scope="whatever")


def test_geometry_constraints_are_always_stated(maker: StoryboardMaker) -> None:
    prompt, negative = maker.format_prompt("anime", _panels(4), cols=2, rows=2)
    assert "exactly 2 columns by 2 rows" in prompt
    assert "identical width, height and aspect ratio" in prompt
    assert "pure-white gutters" in prompt and "pure-white margin" in prompt
    assert "page numbers" in negative


def test_panel_times_become_explicit_deltas(maker: StoryboardMaker) -> None:
    prompt, _ = maker.format_prompt(
        "cinematic", _panels(3), cols=3, rows=1, panel_times_sec=[0.0, 1.5, 5.0]
    )
    assert "Panel 1 (first keyframe, t=0s)" in prompt
    assert "Panel 2 (+1.5s after Panel 1)" in prompt
    assert "Panel 3 (+3.5s after Panel 2)" in prompt
    assert "1.5s, 3.5s apart and span 5.0s" in prompt


def test_timing_omitted_when_not_supplied(maker: StoryboardMaker) -> None:
    prompt, _ = maker.format_prompt("cinematic", _panels(3), cols=3, rows=1)
    assert "Timing (mandatory)" not in prompt
    assert "Panel 1: " in prompt


def test_mismatched_timeline_length_is_ignored_not_fatal(maker: StoryboardMaker) -> None:
    prompt, _ = maker.format_prompt(
        "cinematic", _panels(3), cols=3, rows=1, panel_times_sec=[0.0, 1.0]
    )
    assert "Timing (mandatory)" not in prompt


def test_backwards_timeline_is_rejected(maker: StoryboardMaker) -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        maker.format_prompt(
            "cinematic", _panels(3), cols=3, rows=1, panel_times_sec=[0.0, 3.0, 1.0]
        )


def test_raw_film_strips_ai_buzzwords_and_adds_film_anchors(maker: StoryboardMaker) -> None:
    plain, _ = maker.format_prompt("cinematic", _panels(4), cols=2, rows=2, raw_film=False)
    filmic, _ = maker.format_prompt("cinematic", _panels(4), cols=2, rows=2, raw_film=True)
    assert "photorealistic" in plain
    assert "photorealistic" not in filmic
    assert "shot on 35mm film" in filmic
    assert "visible pores" in filmic and "slight film grain" in filmic


def test_panel_count_mismatch_is_padded_or_truncated(maker: StoryboardMaker) -> None:
    """Lenient by inheritance: the grid is filled, but callers should size it themselves."""
    short, _ = maker.format_prompt("cinematic", _panels(3), cols=2, rows=2)
    assert "Panel 4:" in short
    long, _ = maker.format_prompt("cinematic", _panels(5), cols=2, rows=2)
    assert "Panel 5:" not in long


def _sheet(cols: int, rows: int, panel: tuple[int, int], gutter: int, margin: int) -> Image.Image:
    """A synthetic storyboard: coloured panels on white, with gutters and an outer margin."""
    pw, ph = panel
    width = margin * 2 + cols * pw + (cols - 1) * gutter
    height = margin * 2 + rows * ph + (rows - 1) * gutter
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    for r in range(rows):
        for c in range(cols):
            shade = 30 + 40 * (r * cols + c)
            x = margin + c * (pw + gutter)
            y = margin + r * (ph + gutter)
            sheet.paste(Image.new("RGB", (pw, ph), (shade, shade, shade)), (x, y))
    return sheet


def test_slicing_trims_white_and_recovers_the_panels(tmp_path, maker: StoryboardMaker) -> None:
    sheet = _sheet(cols=2, rows=2, panel=(200, 120), gutter=16, margin=12)
    path = tmp_path / "storyboard.png"
    sheet.save(path)

    panels = maker.slice_storyboard(path, cols=2, rows=2, gutter=16, out_dir=tmp_path / "out")

    assert len(panels) == 4
    images = [Image.open(p) for p in panels]
    assert {img.size for img in images} == {(200, 120)}
    # No white survives the trim, and the panels stay distinguishable.
    shades = [img.convert("L").getpixel((100, 60)) for img in images]
    assert all(shade < 200 for shade in shades)
    assert len(set(shades)) == 4


def test_panels_share_one_resolution_even_when_gutters_drift(
    tmp_path, maker: StoryboardMaker
) -> None:
    sheet = _sheet(cols=3, rows=1, panel=(180, 100), gutter=20, margin=9)
    # Nudge one gutter so the mathematical grid no longer matches the real layout.
    sheet.paste(Image.new("RGB", (20, 100), (255, 255, 255)), (9 + 180 - 5, 9))
    path = tmp_path / "drifted.png"
    sheet.save(path)

    panels = maker.slice_storyboard(path, cols=3, rows=1, gutter=20, out_dir=tmp_path / "out")

    sizes = {Image.open(p).size for p in panels}
    assert len(sizes) == 1, f"panels must share one resolution, got {sizes}"


def test_slicing_can_target_the_video_resolution(tmp_path, maker: StoryboardMaker) -> None:
    sheet = _sheet(cols=2, rows=2, panel=(208, 120), gutter=16, margin=10)
    path = tmp_path / "storyboard.png"
    sheet.save(path)

    panels = maker.slice_storyboard(
        path, cols=2, rows=2, gutter=16, out_dir=tmp_path / "out", target_size=(832, 480)
    )

    assert {Image.open(p).size for p in panels} == {(832, 480)}


def test_trim_keeps_a_bright_but_real_image(tmp_path) -> None:
    """An overexposed sky is bright, not blank: trimming must not eat it."""
    image = Image.new("RGB", (100, 60), (250, 250, 250))
    image.paste(Image.new("RGB", (100, 20), (40, 40, 40)), (0, 40))
    trimmed = _trim_white_border(image)
    # The white rows are shaved to the max allowance, never past the dark content.
    assert trimmed.size[1] >= 20
    assert trimmed.convert("L").getpixel((50, trimmed.size[1] - 1)) < 100


def test_trim_refuses_to_empty_an_all_white_panel() -> None:
    blank = Image.new("RGB", (60, 40), (255, 255, 255))
    assert _trim_white_border(blank).size == (60, 40)


def test_normalize_crops_to_aspect_before_resizing() -> None:
    wide = Image.new("RGB", (400, 100), (10, 10, 10))
    tall = Image.new("RGB", (100, 400), (20, 20, 20))
    out = _normalize_panel_sizes([wide, tall], target_size=(200, 100))
    assert [img.size for img in out] == [(200, 100), (200, 100)]
