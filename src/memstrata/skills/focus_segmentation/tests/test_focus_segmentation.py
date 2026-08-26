"""Tests for in-focus vs defocused-plane segmentation (reblur gradient ratio)."""

import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from memstrata.skills.focus_segmentation import (
    FocusMap,
    focus_map,
    overlay,
    defocus_fill,
)
from memstrata.skills.focus_segmentation.segmenter import _gaussian_blur_gray


def _sharp_texture(h: int, w: int, *, seed: int = 0) -> np.ndarray:
    """High-frequency random texture (lots of fine detail) as an RGB uint8 image."""

    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(h, w)).astype(np.float64)
    return np.repeat(gray[:, :, None], 3, axis=2)


class FocusMapTest(TestCase):
    def test_sharp_texture_is_in_focus_blurred_is_not(self) -> None:
        sharp = _sharp_texture(120, 120)
        fmap_sharp = focus_map(sharp)
        # Re-blurring a sharp texture makes the in-focus plane collapse.
        blurred = np.repeat(
            _gaussian_blur_gray(sharp[:, :, 0], 3.0)[:, :, None], 3, axis=2
        )
        fmap_blur = focus_map(blurred)

        self.assertGreater(fmap_sharp.in_focus_ratio, 0.5)
        self.assertLess(fmap_blur.in_focus_ratio, 0.1)
        self.assertGreater(fmap_sharp.in_focus_ratio, fmap_blur.in_focus_ratio)

    def test_flat_region_is_not_in_focus(self) -> None:
        flat = np.full((80, 80, 3), 128.0)
        fmap = focus_map(flat)
        # No gradient energy -> uninformative -> excluded from the in-focus plane.
        self.assertLess(fmap.in_focus_ratio, 0.02)

    def test_region_split_sharp_left_blurred_right(self) -> None:
        # Compose one image: sharp texture on the left half, defocused on the right.
        h, w = 120, 240
        sharp = _sharp_texture(h, w, seed=1)[:, :, 0]
        blurred_right = _gaussian_blur_gray(sharp, 3.0)
        composed = sharp.copy()
        composed[:, w // 2:] = blurred_right[:, w // 2:]
        img = np.repeat(composed[:, :, None], 3, axis=2)

        fmap = focus_map(img)
        left = fmap.region_in_focus((0, 0, w // 2, h))
        right = fmap.region_in_focus((w // 2, 0, w, h))
        self.assertGreater(left, 0.5)
        self.assertLess(right, 0.15)
        # Background-outside-left-box (== the blurred right) should read defocused.
        self.assertLess(fmap.background_in_focus_ratio([[0, 0, w // 2, h]]), 0.15)

    def test_sharp_share_and_defocus_share(self) -> None:
        h, w = 120, 240
        sharp = _sharp_texture(h, w, seed=2)[:, :, 0]
        composed = sharp.copy()
        composed[:, w // 2:] = _gaussian_blur_gray(sharp, 3.0)[:, w // 2:]
        img = np.repeat(composed[:, :, None], 3, axis=2)
        fmap = focus_map(img)

        left_mask = np.zeros((h, w), dtype=bool)
        left_mask[:, : w // 2] = True
        right_mask = np.zeros((h, w), dtype=bool)
        right_mask[:, w // 2:] = True
        # Sharp half: structured pixels are mostly in focus; blurred half: mostly defocused.
        self.assertGreater(fmap.mask_sharp_share(left_mask), 0.8)
        self.assertLess(fmap.mask_sharp_share(right_mask), 0.3)
        # Background (the blurred right) defocus share is high.
        self.assertGreater(fmap.background_defocus_share([[0, 0, w // 2, h]]), 0.6)

    def test_defocus_fill_removes_blurred_keeps_flat(self) -> None:
        h, w = 100, 100
        sharp = _sharp_texture(h, w, seed=3)
        blurred = np.repeat(_gaussian_blur_gray(sharp[:, :, 0], 3.0)[:, :, None], 3, axis=2)
        fmap = focus_map(blurred)
        filled = defocus_fill(blurred, fmap, (0, 0, w, h))
        # Defocused-structure pixels became white; the array still has the right shape.
        self.assertEqual(filled.shape, blurred.shape)
        self.assertGreaterEqual(int((filled == 255).all(axis=2).sum()), int(fmap.defocused.sum()) - 1)

    def test_accepts_path_and_writes_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            src = Path(tmp) / "tex.png"
            Image.fromarray(_sharp_texture(64, 64).astype(np.uint8)).save(src)
            fmap = focus_map(src)
            self.assertIsInstance(fmap, FocusMap)
            out = overlay(src, fmap, Path(tmp) / "ov.png")
            self.assertTrue(out.exists())
