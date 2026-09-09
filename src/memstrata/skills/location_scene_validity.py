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

SCENE_EVIDENCE_SCHEMA_VERSION = "memstrata.location_scene_evidence.v1"
FOREGROUND_DETECTION_SCHEMA = "memstrata.foreground_detection.v1"
COVERAGE_COMPLETE = "complete_estimate"
COVERAGE_LOWER_BOUND = "lower_bound"
COVERAGE_MISSING = "missing"

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
class ForegroundDetectionEvidence:
    """Validated candidate-local detector record retained for policy audit."""

    bbox: tuple[float, float, float, float]
    score: float
    label: str
    category: str
    candidate_coverage: float


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
    content_area_fraction: float | None = None
    coverage_semantics: str = COVERAGE_MISSING
    coverage_geometry: str = ""
    place_support_count: int | None = None
    place_observation_count: int | None = None
    temporal_visual_support_count: int | None = None
    temporal_visual_observation_count: int | None = None
    temporal_visual_consistency: float | None = None
    temporal_visual_status: str = "missing"
    shot_size: str = "unknown"
    candidate_local_scene_support: str = "unknown"
    candidate_local_scene_support_version: str = ""
    foreground_critical_max_coverage: float | None = None
    foreground_critical_union_coverage: float | None = None
    foreground_max_score: float | None = None
    foreground_detections: tuple[ForegroundDetectionEvidence, ...] = ()
    foreground_detection_schema: str = ""
    crop_quality_accepted: bool | None = None
    foreground_source: str = ""
    place_source: str = ""
    temporal_source: str = ""
    schema_version: str = ""
    schema_compatible: bool = True

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

        schema_version = str(evidence.get("schema_version") or "")
        schema_compatible = (
            not schema_version
            or schema_version == SCENE_EVIDENCE_SCHEMA_VERSION
        )
        foreground_max = _optional_float(
            evidence.get("foreground_max_coverage")
        )
        foreground_union = _optional_float(
            evidence.get("foreground_union_coverage")
        )
        coverage_semantics = str(evidence.get("coverage_semantics") or "")
        if not coverage_semantics:
            coverage_semantics = (
                COVERAGE_COMPLETE
                if foreground_max is not None or foreground_union is not None
                else COVERAGE_MISSING
            )
        if coverage_semantics not in {
            COVERAGE_COMPLETE,
            COVERAGE_LOWER_BOUND,
            COVERAGE_MISSING,
        }:
            coverage_semantics = COVERAGE_MISSING

        attribute_shot_size = str(attrs.get("shot_size") or "unknown")
        shot_size = str(
            evidence.get("shot_size") or attribute_shot_size
        )
        candidate_local_scene_support = str(
            evidence.get("candidate_local_scene_support") or "unknown"
        )
        candidate_local_scene_support_version = str(
            evidence.get("candidate_local_scene_support_version") or ""
        )
        if (
            candidate_local_scene_support == "unknown"
            and attribute_shot_size == "wide"
        ):
            # Deterministic adapter over an attribute already produced by the
            # existing batched crop-attribute call. No free-text parsing and no
            # second VLM request are introduced.
            candidate_local_scene_support = "wide_shot"
            candidate_local_scene_support_version = (
                "crop_attributes.shot_size.v1"
            )

        return cls(
            foreground_max_coverage=foreground_max,
            foreground_union_coverage=foreground_union,
            crop_area_fraction=area,
            content_area_fraction=_optional_float(
                evidence.get("content_area_fraction")
            ),
            coverage_semantics=coverage_semantics,
            coverage_geometry=str(evidence.get("coverage_geometry") or ""),
            place_support_count=_optional_int(evidence.get("place_support_count")),
            place_observation_count=_optional_int(
                evidence.get("place_observation_count")
            ),
            temporal_visual_support_count=_optional_int(
                evidence.get("temporal_visual_support_count")
            ),
            temporal_visual_observation_count=_optional_int(
                evidence.get("temporal_visual_observation_count")
            ),
            temporal_visual_consistency=_optional_float(
                evidence.get("temporal_visual_consistency"),
                minimum=-1.0,
            ),
            temporal_visual_status=str(
                evidence.get("temporal_visual_status") or "missing"
            ),
            shot_size=shot_size,
            candidate_local_scene_support=candidate_local_scene_support,
            candidate_local_scene_support_version=(
                candidate_local_scene_support_version
            ),
            foreground_critical_max_coverage=_optional_float(
                evidence.get("foreground_critical_max_coverage")
            ),
            foreground_critical_union_coverage=_optional_float(
                evidence.get("foreground_critical_union_coverage")
            ),
            foreground_max_score=_optional_float(
                evidence.get("foreground_max_score")
            ),
            foreground_detections=_parse_foreground_detections(evidence),
            foreground_detection_schema=str(
                evidence.get("foreground_detection_schema") or ""
            ),
            crop_quality_accepted=quality,
            foreground_source=str(evidence.get("foreground_source") or ""),
            place_source=str(evidence.get("place_source") or ""),
            temporal_source=str(evidence.get("temporal_source") or ""),
            schema_version=schema_version,
            schema_compatible=schema_compatible,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocationSceneEvidenceProvider(Protocol):
    """Pluggable server-side producer for a shared sampled-frame candidate set."""

    def collect_candidates(
        self,
        *,
        frame_paths: list[str],
        candidates: list[Mapping[str, Any]],
        lower_bound_bboxes: Mapping[Any, list[list[int]]] | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class LocationSceneValidityPolicy:
    min_crop_area_fraction: float = 0.50
    min_scene_context_fraction: float = 0.20
    strong_scene_context_fraction: float = 0.90
    foreground_review_threshold: float = 0.15
    foreground_union_review_threshold: float = 0.25
    foreground_significant_threshold: float = 0.35
    foreground_union_significant_threshold: float = 0.40
    max_foreground_accept: float = 0.40
    max_foreground_union_accept: float = 0.55
    foreground_hard_reject: float = 0.60
    foreground_union_hard_reject: float = 0.70
    min_place_support_count: int = 2
    min_place_support_ratio: float = 0.60
    subject_review_detection_schema: str = FOREGROUND_DETECTION_SCHEMA
    subject_review_min_candidate_coverage: float = 0.10
    subject_review_score_thresholds: tuple[tuple[str, float], ...] = (
        ("animal", 0.50),
        ("bird", 0.50),
        ("person", 0.60),
        ("human_body", 0.60),
        ("body_part", 0.60),
    )


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
    if not evidence.schema_compatible:
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("scene_evidence_schema_mismatch",),
            evidence,
        )
    if evidence.crop_quality_accepted is False:
        return LocationSceneDecision(
            SceneValidityStatus.REJECT,
            False,
            ("crop_quality_rejected",),
            evidence,
        )

    fg_max = evidence.foreground_max_coverage
    fg_union = evidence.foreground_union_coverage
    if (
        evidence.coverage_semantics == COVERAGE_MISSING
        or (fg_max is None and fg_union is None)
    ):
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("missing_foreground_evidence",),
            evidence,
        )

    detailed_foreground_compatible = (
        evidence.foreground_detection_schema == FOREGROUND_DETECTION_SCHEMA
    )
    critical_max = (
        evidence.foreground_critical_max_coverage
        if (
            detailed_foreground_compatible
            and evidence.foreground_critical_max_coverage is not None
        )
        else fg_max
    )
    critical_union = (
        evidence.foreground_critical_union_coverage
        if (
            detailed_foreground_compatible
            and evidence.foreground_critical_union_coverage is not None
        )
        else fg_union
    )
    foreground_dominant = (
        critical_max is not None
        and critical_max >= pol.foreground_hard_reject
    ) or (
        critical_union is not None
        and critical_union >= pol.foreground_union_hard_reject
    )
    if foreground_dominant:
        return LocationSceneDecision(
            SceneValidityStatus.REJECT,
            False,
            ("foreground_dominant",),
            evidence,
        )

    foreground_ambiguous = (
        critical_max is not None
        and critical_max >= pol.max_foreground_accept
    ) or (
        critical_union is not None
        and critical_union >= pol.max_foreground_union_accept
    )
    if foreground_ambiguous:
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("foreground_ambiguous",),
            evidence,
        )

    if evidence.coverage_semantics == COVERAGE_LOWER_BOUND:
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("foreground_lower_bound_only",),
            evidence,
        )

    effective_area = (
        evidence.content_area_fraction
        if evidence.content_area_fraction is not None
        else evidence.crop_area_fraction
    )
    if (
        effective_area is None
        or effective_area < pol.min_scene_context_fraction
    ):
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("insufficient_candidate_scene_context",),
            evidence,
        )

    foreground_significant = (
        critical_max is not None
        and critical_max >= pol.foreground_significant_threshold
    ) or (
        critical_union is not None
        and critical_union >= pol.foreground_union_significant_threshold
    )
    if foreground_significant:
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("significant_foreground_subject",),
            evidence,
        )

    place_support = _has_place_specific_support(evidence, pol)
    if (
        _has_high_confidence_subject_alert(evidence, pol)
        and not place_support
    ):
        # Wide/whole-frame geometry and generic temporal DINO consistency do
        # not establish that the candidate is place-dominant. Quarantine lets
        # acquisition prefer another plate without rejecting scenes that
        # legitimately contain a small dynamic subject.
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("high_confidence_dynamic_subject_review",),
            evidence,
        )

    foreground_needs_local_support = (
        critical_max is not None
        and critical_max >= pol.foreground_review_threshold
    ) or (
        critical_union is not None
        and critical_union >= pol.foreground_union_review_threshold
    )
    strong_context = effective_area >= pol.strong_scene_context_fraction
    structured_context = (
        evidence.candidate_local_scene_support == "wide_shot"
        and evidence.candidate_local_scene_support_version
        == "crop_attributes.shot_size.v1"
    )
    if (
        foreground_needs_local_support
        and not strong_context
        and not structured_context
    ):
        return LocationSceneDecision(
            SceneValidityStatus.QUARANTINE,
            False,
            ("foreground_requires_candidate_scene_support",),
            evidence,
        )

    area_support = (
        effective_area is not None
        and effective_area >= pol.min_crop_area_fraction
    )
    wide_support = evidence.shot_size == "wide"
    temporal_support = False
    if (
        evidence.temporal_visual_status == "available"
        and evidence.temporal_visual_support_count is not None
        and evidence.temporal_visual_observation_count is not None
        and evidence.temporal_visual_observation_count > 0
    ):
        temporal_ratio = (
            evidence.temporal_visual_support_count
            / evidence.temporal_visual_observation_count
        )
        temporal_support = (
            evidence.temporal_visual_support_count >= pol.min_place_support_count
            and temporal_ratio >= pol.min_place_support_ratio
        )

    if not (area_support or wide_support or place_support or temporal_support):
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
    if temporal_support:
        reasons.append("temporal_visual_support")
    return LocationSceneDecision(
        SceneValidityStatus.ACCEPT,
        True,
        tuple(reasons),
        evidence,
    )


def _parse_foreground_detections(
    evidence: Mapping[str, Any],
) -> tuple[ForegroundDetectionEvidence, ...]:
    if evidence.get("foreground_detection_schema") != FOREGROUND_DETECTION_SCHEMA:
        return ()
    raw_rows = evidence.get("foreground_detections")
    if not isinstance(raw_rows, (list, tuple)):
        return ()
    parsed: list[ForegroundDetectionEvidence] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        bbox = raw.get("bbox")
        score = _optional_float(raw.get("score"))
        coverage = _optional_float(raw.get("candidate_coverage"))
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or score is None
            or coverage is None
        ):
            continue
        try:
            parsed_bbox = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        parsed.append(
            ForegroundDetectionEvidence(
                bbox=parsed_bbox,
                score=score,
                label=str(raw.get("label") or "")[:80],
                category=str(raw.get("category") or "subject_unknown"),
                candidate_coverage=coverage,
            )
        )
    return tuple(parsed)


def _has_place_specific_support(
    evidence: LocationSceneEvidence,
    policy: LocationSceneValidityPolicy,
) -> bool:
    if (
        evidence.place_support_count is None
        or evidence.place_observation_count is None
        or evidence.place_observation_count <= 0
        or not evidence.place_source
    ):
        return False
    ratio = evidence.place_support_count / evidence.place_observation_count
    return (
        evidence.place_support_count >= policy.min_place_support_count
        and ratio >= policy.min_place_support_ratio
    )


def _has_high_confidence_subject_alert(
    evidence: LocationSceneEvidence,
    policy: LocationSceneValidityPolicy,
) -> bool:
    if (
        evidence.foreground_detection_schema
        != policy.subject_review_detection_schema
    ):
        return False
    thresholds = dict(policy.subject_review_score_thresholds)
    for detection in evidence.foreground_detections:
        score_threshold = thresholds.get(detection.category)
        if score_threshold is None:
            continue
        if (
            detection.score >= score_threshold
            and detection.candidate_coverage
            >= policy.subject_review_min_candidate_coverage
        ):
            return True
    return False


__all__ = [
    "COVERAGE_COMPLETE",
    "COVERAGE_LOWER_BOUND",
    "COVERAGE_MISSING",
    "FOREGROUND_DETECTION_SCHEMA",
    "ForegroundDetectionEvidence",
    "LocationSceneDecision",
    "LocationSceneEvidence",
    "LocationSceneEvidenceProvider",
    "LocationSceneValidityPolicy",
    "SceneValidityStatus",
    "SCENE_EVIDENCE_SCHEMA_VERSION",
    "evaluate_location_scene",
]
