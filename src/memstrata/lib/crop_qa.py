"""Deterministic, production-only quality checks for already-isolated crops.

This intentionally mirrors only the geometry/sharpness semantics of the Bench
quality gate.  It does not import Bench code or encode gold coverage rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from memstrata.lib.media import load_crop_rgb_for_model


@dataclass(frozen=True, slots=True)
class CropQualityReport:
    accepted: bool
    reasons: list[str]
    width: int
    height: int
    sharpness: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_crop(
    crop: Path | str,
    *,
    min_side: int = 16,
    min_sharpness: float = 1.0,
) -> CropQualityReport:
    """Reject unreadable, tiny, and nearly uniform crops without semantic inference."""
    try:
        rgb = load_crop_rgb_for_model(crop)
        width, height = rgb.size
        gray = np.asarray(rgb.convert("L"), dtype="float32")
    except Exception as exc:  # noqa: BLE001 - external crop is a trust boundary
        return CropQualityReport(
            accepted=False,
            reasons=[f"unreadable_crop:{type(exc).__name__}"],
            width=0,
            height=0,
            sharpness=0.0,
        )

    reasons: list[str] = []
    if min(width, height) < min_side:
        reasons.append("crop_too_small")
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharpness = float(laplacian.var()) if laplacian.size else 0.0
    if sharpness <= min_sharpness:
        reasons.append("low_sharpness")
    return CropQualityReport(
        accepted=not reasons,
        reasons=reasons,
        width=width,
        height=height,
        sharpness=sharpness,
    )


__all__ = ["CropQualityReport", "audit_crop"]
