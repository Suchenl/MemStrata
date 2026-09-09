"""Explicit read-side policy for legacy and adaptive composition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompositionPolicy:
    """Composition controls.

    Adaptive behavior is opt-in.  The legacy defaults intentionally preserve the
    historical per-asset selection followed by deterministic trimming.
    """

    adaptive_location_enabled: bool = False
    global_rep_budget: int | None = None
    location_read_max_refs: int = 4
    location_extra_budget_share: float = 0.50
    location_high_confidence_margin: float = 0.20
    location_min_marginal_gain: float = 0.08
    location_diversity_distance: float = 0.18
    location_coverage_stop: float = 0.90

    def normalised(self) -> CompositionPolicy:
        return CompositionPolicy(
            adaptive_location_enabled=bool(self.adaptive_location_enabled),
            global_rep_budget=(
                None
                if self.global_rep_budget is None
                else max(1, int(self.global_rep_budget))
            ),
            location_read_max_refs=max(1, int(self.location_read_max_refs)),
            location_extra_budget_share=min(
                1.0, max(0.0, float(self.location_extra_budget_share))
            ),
            location_high_confidence_margin=min(
                1.0, max(0.0, float(self.location_high_confidence_margin))
            ),
            location_min_marginal_gain=min(
                1.0, max(0.0, float(self.location_min_marginal_gain))
            ),
            location_diversity_distance=min(
                2.0, max(0.0, float(self.location_diversity_distance))
            ),
            location_coverage_stop=min(
                1.0, max(0.0, float(self.location_coverage_stop))
            ),
        )


__all__ = ["CompositionPolicy"]
