"""Deterministic crop-quality gate for the memory bank (WHO-before-WHERE, layer ①).

Mirror of the bench-side ``vmem_bench...crop_qa`` dark/low-information check,
kept as an independent copy per the ``memstrata`` ↔ ``vmem_bench`` zero-import
rule. A near-black, near-flat crop carries no usable identity signal and must never
become a representation (design_philosophy.md §1 axiom 4, §2 gate ①).

Pure-PIL (no numpy) so it stays dependency-light on the ingest hot path. Measures
luminance over the *masked entity pixels* (alpha>0) when present, so a dark
silhouette composited onto white is not mistaken for a bright crop.
"""

from __future__ import annotations

from pathlib import Path

# Deliberately conservative: only kills near-black AND near-flat crops. A dimly-lit
# but visible subject keeps edge/facial contrast (higher std) and is NOT rejected,
# protecting intra-entity diversity (axiom 5).
DARK_MEAN_MAX = 26.0  # mean luminance 0-255 over entity pixels
DARK_STD_MAX = 16.0   # luminance std 0-255 over entity pixels

# Symmetric to the dark gate: a near-white AND near-flat crop is blown-out (a window,
# a lamp flare, an over-lit wall) and carries no identity signal either. Same
# conservative posture — high mean floor + low std ceiling — so a bright-but-textured
# subject (which still has edge/facial contrast, hence higher std) is NOT rejected.
OVEREXP_MEAN_MIN = 232.0  # mean luminance 0-255 over entity pixels
OVEREXP_STD_MAX = 12.0    # luminance std 0-255 over entity pixels

# Downscale cap: luminance statistics are stable on a thumbnail and this keeps the
# gate cheap on large crops. NEAREST keeps the alpha mask crisp.
_MAX_SIDE = 96


def entity_luminance_stats(crop: str | Path) -> tuple[float, float] | None:
    """Return ``(mean, std)`` luminance over entity pixels, or ``None`` if unreadable.

    ``None`` means "cannot assess" (unreadable / empty) — callers must treat it as
    *pass* (do not reject), matching how the VLM classifier degrades to ``unknown``.
    """
    try:
        from PIL import Image

        image = Image.open(crop)
        image.load()
    except Exception:  # noqa: BLE001 - unreadable/corrupt/non-image → cannot assess
        return None

    try:
        image.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.NEAREST)
    except Exception:  # noqa: BLE001 - thumbnailing is best-effort
        pass

    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        raw = image.convert("RGBA").tobytes()
        lums = [
            0.299 * raw[i] + 0.587 * raw[i + 1] + 0.114 * raw[i + 2]
            for i in range(0, len(raw), 4)
            if raw[i + 3] > 0
        ]
    else:
        raw = image.convert("RGB").tobytes()
        lums = [
            0.299 * raw[i] + 0.587 * raw[i + 1] + 0.114 * raw[i + 2]
            for i in range(0, len(raw), 3)
        ]

    if len(lums) < 4:
        return None
    mean = sum(lums) / len(lums)
    var = sum((x - mean) ** 2 for x in lums) / len(lums)
    return mean, var ** 0.5


def is_dark_low_information(
    crop: str | Path,
    *,
    mean_max: float = DARK_MEAN_MAX,
    std_max: float = DARK_STD_MAX,
) -> bool:
    """True iff the crop is near-black AND near-flat (no usable identity signal).

    Unreadable crops return ``False`` (cannot assess → do not reject).
    """
    stats = entity_luminance_stats(crop)
    if stats is None:
        return False
    mean, std = stats
    return mean < mean_max and std < std_max


def is_overexposed_low_information(
    crop: str | Path,
    *,
    mean_min: float = OVEREXP_MEAN_MIN,
    std_max: float = OVEREXP_STD_MAX,
) -> bool:
    """True iff the crop is near-white AND near-flat (blown-out, no identity signal).

    Unreadable crops return ``False`` (cannot assess → do not reject), matching the
    dark gate's "pass when in doubt" semantics.
    """
    stats = entity_luminance_stats(crop)
    if stats is None:
        return False
    mean, std = stats
    return mean > mean_min and std < std_max


__all__ = [
    "DARK_MEAN_MAX",
    "DARK_STD_MAX",
    "OVEREXP_MEAN_MIN",
    "OVEREXP_STD_MAX",
    "entity_luminance_stats",
    "is_dark_low_information",
    "is_overexposed_low_information",
]
