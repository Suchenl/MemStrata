"""Prompt-only routing over causal, scene-valid location representatives."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any, Mapping, Sequence

from memstrata.bank import Asset, AssetRepresentation
from memstrata.lib.dedup import cosine_or_none, text_similarity
from memstrata.skills.composition.policy import CompositionPolicy
from memstrata.skills.memory_update.location_coreset import (
    CORESET_KEY,
    LocationStrata,
    location_strata,
    scene_reference_eligible,
)


_HINT_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "environment": {
        "indoor": ("indoor", "inside", "interior", "室内", "内部"),
        "outdoor": ("outdoor", "outside", "exterior", "室外", "户外"),
    },
    "elevation": {
        "overhead": ("overhead", "top-down", "bird's-eye", "aerial", "俯拍", "鸟瞰", "航拍"),
        "high": ("high angle", "高机位"),
        "low": ("low angle", "仰拍", "低机位"),
        "eye": ("eye level", "eye-level", "平视"),
    },
    "scale": {
        "establishing": ("establishing shot", "全景", "建立镜头"),
        "wide": ("wide shot", "wide view", "远景", "广角"),
        "detail": ("detail shot", "insert shot", "extreme close-up", "细节", "大特写"),
        "medium": ("medium shot", "中景"),
    },
    "lighting": {
        "night": ("night", "nighttime", "after dark", "夜晚", "夜间", "夜景"),
        "day": ("daylight", "daytime", "白天", "日间"),
        "artificial": ("artificial light", "neon", "灯光", "霓虹"),
        "overcast": ("overcast", "阴天"),
    },
}


@dataclass(frozen=True, slots=True)
class LocationReadQuery:
    raw_prompt: str
    planner_hints: Mapping[str, Any] = field(default_factory=dict)
    as_of_segment_id: int | None = None


@dataclass(slots=True)
class LocationCandidate:
    asset_id: str
    rep: AssetRepresentation
    cluster_id: str
    strata: LocationStrata
    score: float
    posterior: float
    quality: float
    canonical: bool
    support_count: int
    score_parts: dict[str, float]


@dataclass(slots=True)
class LocationRanking:
    candidates: list[LocationCandidate]
    hints: dict[str, str]
    hint_source: str
    margin: float
    high_confidence: bool
    fallback: str
    dropped: dict[str, int]


def _normalise_hint_value(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def extract_location_hints(
    raw_prompt: str,
    planner_hints: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    """Extract only generic camera/environment intent; never inspect target media."""

    prompt = str(raw_prompt or "").lower()
    hints: dict[str, str] = {}
    for axis, values in _HINT_TERMS.items():
        for value, terms in values.items():
            if any(term in prompt for term in terms):
                hints[axis] = value
                break
    source = "raw_prompt" if hints else "none"
    for axis in _HINT_TERMS:
        value = _normalise_hint_value((planner_hints or {}).get(axis))
        if value and value not in {"unknown", "none", "null"}:
            hints[axis] = value
            source = "planner" if source == "none" else "planner+raw_prompt"
    return hints, source


def _quality(rep: AssetRepresentation) -> float:
    try:
        return min(1.0, max(0.0, float(rep.annotations.get("quality", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _description(rep: AssetRepresentation) -> str:
    attrs = rep.annotations.get("crop_attributes")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    return str(
        rep.annotations.get("observation_description")
        or attrs.get("description")
        or ""
    )


def _embedding(rep: AssetRepresentation) -> list[float] | None:
    raw = rep.annotations.get("embedding")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    return values or None


def _distance(left: AssetRepresentation, right: AssetRepresentation) -> float | None:
    a, b = _embedding(left), _embedding(right)
    if a is None or b is None:
        return None
    score = cosine_or_none(a, b)
    return None if score is None else max(0.0, min(2.0, 1.0 - float(score)))


def _axis_value(strata: LocationStrata, axis: str) -> str:
    return str(getattr(strata, axis, "unknown") or "unknown").replace("-", "_")


def _hint_score(hints: Mapping[str, str], strata: LocationStrata) -> float:
    if not hints:
        return 0.0
    scores: list[float] = []
    for axis, wanted in hints.items():
        actual = _axis_value(strata, axis)
        if actual == "unknown":
            scores.append(0.25)
        elif actual == wanted:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def _cluster_rows(asset: Asset) -> tuple[list[dict[str, Any]], str]:
    state = asset.metadata.get(CORESET_KEY)
    if not isinstance(state, Mapping):
        return [], ""
    rows = [row for row in state.get("clusters", []) if isinstance(row, dict)]
    return rows, str(state.get("canonical_cluster_id") or "")


def rank_location_candidates(
    asset: Asset,
    query: LocationReadQuery,
    *,
    policy: CompositionPolicy,
) -> LocationRanking:
    """Return causal representatives ranked from raw prompt/planner intent only."""

    dropped = {"deprecated": 0, "future": 0, "ineligible": 0, "non_representative": 0}
    active: list[AssetRepresentation] = []
    for rep in asset.representations:
        if rep.deprecated:
            dropped["deprecated"] += 1
            continue
        if query.as_of_segment_id is not None and rep.origin_segment_id >= query.as_of_segment_id:
            dropped["future"] += 1
            continue
        if not scene_reference_eligible(rep):
            dropped["ineligible"] += 1
            continue
        active.append(rep)

    rows, canonical_cluster = _cluster_rows(asset)
    support_by_cluster: dict[str, int] = {}
    representative_ids: set[str] = set()
    for row in rows:
        cluster_id = str(row.get("cluster_id") or "")
        representative_id = str(row.get("representative_rep_id") or "")
        if cluster_id and representative_id:
            support_by_cluster[cluster_id] = max(1, int(row.get("support_count", 1) or 1))
            representative_ids.add(representative_id)
    if representative_ids:
        chosen_active = [rep for rep in active if rep.representation_id in representative_ids]
        dropped["non_representative"] += len(active) - len(chosen_active)
        active = chosen_active

    hints, hint_source = extract_location_hints(query.raw_prompt, query.planner_hints)
    provisional: list[tuple[AssetRepresentation, str, LocationStrata, float, dict[str, float]]] = []
    max_support = max(support_by_cluster.values(), default=1)
    for index, rep in enumerate(active):
        cluster_id = str(rep.annotations.get("location_cluster_id") or f"legacy-{index:04d}")
        strata = location_strata(rep)
        hint = _hint_score(hints, strata)
        text = text_similarity(query.raw_prompt, _description(rep))
        quality = _quality(rep)
        canonical = cluster_id == canonical_cluster
        support = support_by_cluster.get(cluster_id, 1) / max_support
        score = (
            0.45 * hint
            + 0.20 * text
            + 0.20 * quality
            + 0.10 * float(canonical and not hints)
            + 0.05 * support
        )
        provisional.append(
            (
                rep,
                cluster_id,
                strata,
                score,
                {
                    "hint": round(hint, 6),
                    "text": round(text, 6),
                    "quality": round(quality, 6),
                    "canonical": float(canonical),
                    "support": round(support, 6),
                },
            )
        )

    if not provisional:
        return LocationRanking([], hints, hint_source, 0.0, False, "no_eligible_rep", dropped)

    # A diffuse posterior is intentional when the prompt carries no view evidence.
    logits = [exp(4.0 * row[3]) for row in provisional]
    total = sum(logits) or 1.0
    candidates: list[LocationCandidate] = []
    for row, posterior in zip(provisional, logits, strict=True):
        rep, cluster_id, strata, score, parts = row
        candidates.append(
            LocationCandidate(
                asset_id=asset.asset_id,
                rep=rep,
                cluster_id=cluster_id,
                strata=strata,
                score=score,
                posterior=posterior / total,
                quality=_quality(rep),
                canonical=cluster_id == canonical_cluster,
                support_count=support_by_cluster.get(cluster_id, 1),
                score_parts=parts,
            )
        )
    if hints:
        candidates.sort(
            key=lambda candidate: (
                -candidate.posterior,
                -candidate.score,
                -int(candidate.canonical),
                -candidate.rep.origin_segment_id,
                candidate.rep.representation_id,
            )
        )
    else:
        candidates.sort(
            key=lambda candidate: (
                -int(candidate.canonical),
                -candidate.posterior,
                -candidate.quality,
                candidate.rep.representation_id,
            )
        )
    margin = (
        candidates[0].posterior - candidates[1].posterior
        if len(candidates) > 1
        else 1.0
    )
    high_confidence = bool(hints) and margin >= policy.location_high_confidence_margin
    fallback = "hint_ranked" if hints else "canonical_then_diverse"
    return LocationRanking(
        candidates,
        hints,
        hint_source,
        margin,
        high_confidence,
        fallback,
        dropped,
    )


def location_coverage(
    selected: Sequence[LocationCandidate],
) -> float:
    return min(1.0, sum(candidate.posterior for candidate in selected))


def location_marginal_gain(
    candidate: LocationCandidate,
    selected: Sequence[LocationCandidate],
    *,
    ranking: LocationRanking,
    policy: CompositionPolicy,
) -> float:
    if not selected:
        return candidate.posterior
    distances = [
        distance
        for distance in (_distance(candidate.rep, previous.rep) for previous in selected)
        if distance is not None
    ]
    if distances:
        diversity = min(distances)
    else:
        distinct_strata = all(candidate.strata != previous.strata for previous in selected)
        diversity = policy.location_diversity_distance if distinct_strata else 0.0
    if diversity < policy.location_diversity_distance:
        return 0.0
    ambiguity = 1.0 - min(1.0, max(0.0, ranking.margin))
    gain = candidate.posterior * min(1.0, diversity) * (0.5 + 0.5 * ambiguity)
    if ranking.high_confidence:
        gain *= 0.25
    return gain


__all__ = [
    "LocationCandidate",
    "LocationRanking",
    "LocationReadQuery",
    "extract_location_hints",
    "location_coverage",
    "location_marginal_gain",
    "rank_location_candidates",
]
