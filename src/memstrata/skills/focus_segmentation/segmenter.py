"""Explicit in-focus vs out-of-focus (defocused) plane segmentation from a single image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REBLUR_SIGMA = 2.0
R_TAU = 1.7
GRAD_FLOOR = 14.0
WINDOW = 8


@dataclass(slots=True)
class FocusMap:
    """Per-pixel focus segmentation of one image."""

    sharpness: np.ndarray
    in_focus: np.ndarray
    defocused: np.ndarray
    in_focus_ratio: float
    r_tau: float
    grad_floor: float

    def region_in_focus(self, box: tuple[float, float, float, float]) -> float:
        sub = self._box(self.in_focus, box)
        return float(sub.mean()) if sub.size else 0.0

    def background_in_focus_ratio(self, boxes: list[list[float]]) -> float:
        bg = self._background_mask(boxes)
        if not bool(bg.any()):
            return self.in_focus_ratio
        return float(self.in_focus[bg].mean())

    def mask_sharp_share(self, mask: np.ndarray) -> float:
        infocus = int((self.in_focus & mask).sum())
        defoc = int((self.defocused & mask).sum())
        return infocus / (infocus + defoc) if (infocus + defoc) else 0.0

    def background_defocus_share(self, boxes: list[list[float]]) -> float:
        bg = self._background_mask(boxes)
        infocus = int((self.in_focus & bg).sum())
        defoc = int((self.defocused & bg).sum())
        return defoc / (infocus + defoc) if (infocus + defoc) else 0.0

    def _box(self, arr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        h, w = arr.shape
        left, top, right, bottom = box
        x0, y0 = max(0, int(left)), max(0, int(top))
        x1, y1 = min(w, int(right)), min(h, int(bottom))
        if x1 <= x0 or y1 <= y0:
            return arr[0:0, 0:0]
        return arr[y0:y1, x0:x1]

    def _background_mask(self, boxes: list[list[float]]) -> np.ndarray:
        h, w = self.in_focus.shape
        bg = np.ones((h, w), dtype=bool)
        for left, top, right, bottom in boxes:
            x0, y0 = max(0, int(left)), max(0, int(top))
            x1, y1 = min(w, int(right)), min(h, int(bottom))
            if x1 > x0 and y1 > y0:
                bg[y0:y1, x0:x1] = False
        return bg


def _integral_box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    h, w = x.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(x, axis=0), axis=1)
    ys, xs = np.arange(h), np.arange(w)
    y0 = np.clip(ys - radius, 0, h)[:, None]
    y1 = np.clip(ys + radius + 1, 0, h)[:, None]
    x0 = np.clip(xs - radius, 0, w)[None, :]
    x1 = np.clip(xs + radius + 1, 0, w)[None, :]
    total = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    count = np.maximum((y1 - y0) * (x1 - x0), 1)
    return total / count


def _grad_energy(gray: np.ndarray, window: int) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return _integral_box_mean(gx * gx + gy * gy, window)


def _gaussian_blur_gray(gray: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return gray
    radius = max(1, int(round(3 * sigma)))
    ax = np.arange(-radius, radius + 1)
    k = np.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    blurred = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, gray)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, blurred)


def focus_map(
    image: np.ndarray | str | Path,
    *,
    window: int = WINDOW,
    reblur_sigma: float = REBLUR_SIGMA,
    r_tau: float = R_TAU,
    grad_floor: float = GRAD_FLOOR,
) -> FocusMap:
    rgb = _as_rgb(image)
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    g0 = _grad_energy(gray, window)
    g1 = _grad_energy(_gaussian_blur_gray(gray, reblur_sigma), window)
    ratio = np.sqrt((g0 + 1e-6) / (g1 + 1e-6))
    structured = g0 >= grad_floor
    in_focus = structured & (ratio >= r_tau)
    defocused = structured & (ratio < r_tau)
    return FocusMap(
        sharpness=ratio,
        in_focus=in_focus,
        defocused=defocused,
        in_focus_ratio=float(in_focus.mean()),
        r_tau=r_tau,
        grad_floor=grad_floor,
    )


def _as_rgb(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        return arr[:, :, :3].astype(np.float64)
    from PIL import Image

    return np.asarray(Image.open(Path(image)).convert("RGB"), dtype=np.float64)


def overlay(image: np.ndarray | str | Path, fmap: FocusMap, out_path: str | Path) -> Path:
    from PIL import Image

    rgb = _as_rgb(image).astype(np.float64)
    out = rgb.copy()
    flat = ~(fmap.in_focus | fmap.defocused)
    out[flat] = out[flat] * 0.6 + 110.0 * 0.4
    out[fmap.defocused] = out[fmap.defocused] * 0.3 + np.array([150.0, 0.0, 0.0]) * 0.7
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path)
    return Path(out_path)


def defocus_fill(
    crop: np.ndarray | str | Path,
    fmap: FocusMap,
    box: tuple[float, float, float, float],
    *,
    subject_mask: np.ndarray | None = None,
    fill: int = 255,
) -> np.ndarray:
    rgb = _as_rgb(crop).astype(np.uint8).copy()
    ch, cw = rgb.shape[:2]
    left, top = int(box[0]), int(box[1])
    h, w = fmap.defocused.shape
    y0, x0 = max(0, top), max(0, left)
    defoc_box = fmap.defocused[y0:y0 + ch, x0:x0 + cw]
    pad = np.zeros((ch, cw), dtype=bool)
    pad[: defoc_box.shape[0], : defoc_box.shape[1]] = defoc_box
    remove = pad
    if subject_mask is not None:
        remove = remove & subject_mask[:ch, :cw]
    rgb[remove] = fill
    return rgb
