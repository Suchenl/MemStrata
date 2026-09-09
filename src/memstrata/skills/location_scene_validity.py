"""Deterministic, evidence-driven admission for location scene references.

Location evidence answers a different question from WHO/WHAT identity visibility:
does this crop contain enough stable place context to condition a later scene?  The
gate therefore never reads or rewrites ``identity_visible``.  It consumes optional
evidence produced by existing detector/place-encoder passes and fails closed to
``quarantine`` when that evidence is absent.

Candidate production is deliberately outside this module.  A referring bbox, expanded
context crop, or whole-frame scene plate is only a *candidate* until this predicate
grants ``scene_reference_eligible``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol


class SceneValidityStatus(str, Enum):
    ACCEPT = "accept"
    QUARANTINE = "quarantine"
    REJECT = "reject"


def _optional_float(value: Any, *, minimum: float = 0.0, maximum: float = 1.0) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return min(maximum, max(minimum, parsed))


def _optional_int(value: Any, *, minimum: int = 0) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(minimum, parsed)


@dataclass(frozen=True, slots=True)
class LocationSceneEvidence:
    """Model-agnostic evidence attached by an upstream detector/place encoder.

    ``foreground_*`` must describe dynamic foreground subjects such as people,
    animals, or body parts, not the mask of the requested location itself.
    ``place_support_count`` and ``place_observation_count`` support any sampled-frame
    count; the write path does not assume three frames or a fixed number of views.
    """

    foreground_max_coverage: float | None = None
    foreground_union_coverage: float | None = None
    crop_area_fraction: float | None = None
    place_support_count: int | None = None
    place_observation_count: int | None = None
    shot_size: str = "unknown"
    crop_quality_accepted: bool | None = None
    foreground_source: str = ""
    place_source: str = ""

    @classmethod
    def from_annotations(cls, annotations: Mapping[str, Any]) -> LocationSceneEvidence:
        raw = annotations.get("scene_validity_evidence")
        if not isinstance(raw, Mapping):
            acquisition = annotations.get("crop_acquisition")
            raw = (
                acquisition.get("scene_validity_evidence")
                if isinstance(acquisition, Mapping)
                else None
            )
        evidence = raw if isinstance(raw, Mapping) else {}

        attrs = annotations.get("crop_attributes")
        attrs = attrs if isinstance(attrs, Mapping) else {}
        crop_quality = annotations.get("crop_quality")
        crop_quality = crop_quality if isinstance(crop_quality, Mapping) else {}

        area = _optional_float(evidence.get("crop_area_fraction"))
        if area is None:
            bbox = annotations.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    y0, x0, y1, x1 = (float(value) for value in bbox)
                    area = min(1.0, max(0.0, (y1 - y0) * (x1 - x0) / 1_000_000.0))
                except (TypeError, ValueError):
                    area = None

        quality: bool | None = None
        if "crop_quality_accepted" in evidence:
            quality = bool(evidence.get("crop_quality_accepted"))
        elif "accepted" in crop_quality:
            quality = bool(crop_quality.get("accepted"))

        return cls(
            foreground_max_coverage=_optional_float(
                evidence.get("foreground_max_coverage")
            ),
            foreground_union_coverage=_optional_float(
                evidence.get("foreground_union_coverage")
            ),
            crop_area_fraction=area,
            place_support_count=_optional_int(evidence.get("place_support_count")),
            place_observation_count=_optional_int(
                evidence.get("place_observation_count")
            ),
            shot_size=str(evidence.get("shot_size") or attrs.get("shot_size") or "unknown"),
            crop_quality_accepted=quality,
            foreground_source=str(evidence.get("foreground_source") or ""),
            place_source=str(evidence.get("place_source") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocationSceneEvidenceProvider(Protocol):
    """Pluggable producer implemented by cached detector/place-encoder adapters."""

    def collect(
        self,
        *,
        crop_path: str,
        frame_paths: list[str],
        bbox_norm: list[int] | None,
    ) -> LocationSceneEvidence: ...


@dataclass(frozen=True, slots=True)
class LocationSceneValidityPolicy:
    min_crop_area_fraction: float = 0.50
    max_foreground_accept: float = 0.40
    max_foreground_union_accept: float = 0.55
    foreground_hard_reject: float = 0.60
    foreground_union_hard_reject: float = 0.70
    min_place_support_count: int = 2
    min_place_support_ratio: float = 0.60


@dataclass(frozen=True, slots=True)
class LocationSceneDecision:
    status: SceneValidityStatus
    scene_reference_eligible: bool
    reasons: tuple[str, ...]
    evidence: LocationSceneEvidence

    def to_annotations(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "scene_reference_eligible": self.scene_reference_eligible,
            "reasons": list(self.reasons),
            "evidence": self.evidence.to_dict(),
        }


def evaluate_location_scene(
    evidence: LocationSceneEvidence,
    *,
    policy: LocationSceneValidityPolicy | None = None,
) -> LocationSceneDecision:
    """Return a deterministic scene-reference decision without identity semantics."""

    pol = policy or LocationSceneValidityPolicy()
    if evidence.crop_quality_accepted is False:
        return LocationSceneDecision(
            SceneValidityStatus.REJECT,
            False,
            ("crop_quality_rejected",),
            evidence,
        )

    fg_max = evidence.foreground_max_coverage
    fg_union = evidence.foreground_union_coverage
    if fg_max is None and fg_union is None:
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("missing_foreground_evidence",),
            evidence,
        )

    if (
        fg_max is not None
        and fg_max >= pol.foreground_hard_reject
    ) or (
        fg_union is not None
        and fg_union >= pol.foreground_union_hard_reject
    ):
        return LocationSceneDecision(
            SceneValidityStatus.REJECT,
            False,
            ("foreground_dominant",),
            evidence,
        )

    if (
        fg_max is not None
        and fg_max >= pol.max_foreground_accept
    ) or (
        fg_union is not None
        and fg_union >= pol.max_foreground_union_accept
    ):
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("foreground_ambiguous",),
            evidence,
        )

    area_support = (
        evidence.crop_area_fraction is not None
        and evidence.crop_area_fraction >= pol.min_crop_area_fraction
    )
    wide_support = evidence.shot_size == "wide"
    place_support = False
    if (
        evidence.place_support_count is not None
        and evidence.place_observation_count is not None
        and evidence.place_observation_count > 0
    ):
        ratio = evidence.place_support_count / evidence.place_observation_count
        place_support = (
            evidence.place_support_count >= pol.min_place_support_count
            and ratio >= pol.min_place_support_ratio
        )

    if not (area_support or wide_support or place_support):
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("insufficient_scene_support",),
            evidence,
        )

    reasons = ["low_foreground"]
    if area_support:
        reasons.append("broad_context")
    if wide_support:
        reasons.append("wide_shot")
    if place_support:
        reasons.append("temporal_place_support")
    return LocationSceneDecision(
        SceneValidityStatus.ACCEPT,
        True,
        tuple(reasons),
        evidence,
    )


__all__ = [
    "LocationSceneDecision",
    "LocationSceneEvidence",
    "LocationSceneEvidenceProvider",
    "LocationSceneValidityPolicy",
    "SceneValidityStatus",
    "evaluate_location_scene",
]
