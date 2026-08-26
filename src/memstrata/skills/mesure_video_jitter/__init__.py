"""Motion Stability Skill.

Quantifies per-frame micro-jitter in generated clips so "the picture keeps shaking" becomes a
number that can be compared across generators, LoRAs and step counts.
"""

from __future__ import annotations

from .jitter import (
    CUT_DIFF_MULTIPLE,
    REAL_FILM_ACCEL_PX,
    VERDICT_JITTERY,
    VERDICT_NOISE_DOMINATED,
    VERDICT_STEADY,
    JitterReport,
    classify,
    compare,
    cut_indices,
    longest_cut_free_span,
    jitter_from_translations,
    measure_jitter,
    translations_from_frames,
)

__all__ = [
    "CUT_DIFF_MULTIPLE",
    "JitterReport",
    "REAL_FILM_ACCEL_PX",
    "VERDICT_JITTERY",
    "VERDICT_NOISE_DOMINATED",
    "VERDICT_STEADY",
    "classify",
    "compare",
    "cut_indices",
    "longest_cut_free_span",
    "jitter_from_translations",
    "measure_jitter",
    "translations_from_frames",
]
