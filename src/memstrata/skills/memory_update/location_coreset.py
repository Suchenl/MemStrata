"""Adaptive, scene-valid coreset maintenance for location representations.

The number of live representatives follows observed location strata.  ``storage_cap``
is only a guardrail against noisy metadata or embeddings; it is not a target K.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p, sqrt
from typing import Any, Mapping, Sequence

from memstrata.bank import Asset, AssetRepresentation
from memstrata.lib.dedup import cosine_or_none


CORESET_KEY = "location_coreset_v1"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LocationCoresetPolicy:
    enabled: bool = False
    storage_cap: int = 12
    cluster_join_distance: float = 0.22
    cluster_merge_distance: float = 0.10
    representative_replace_margin: float = 0.02


@dataclass(frozen=True, slots=True)
class LocationStrata:
    environment: str = UNKNOWN
    elevation: str = UNKNOWN
    scale: str = UNKNOWN
    lighting: str = UNKNOWN
    zone_signature: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "elevation": self.elevation,
            "scale": self.scale,
            "lighting": self.lighting,
            "zone_signature": list(self.zone_signature),
        }


def scene_reference_eligible(rep: AssetRepresentation) -> bool:
    """Read the authoritative tri-state gate, with a legacy aspect fallback."""

    explicit = rep.annotations.get("scene_reference_eligible")
    if explicit is not None:
        return explicit is True
    supports = any(str(value).lower() == "scene_reference" for value in rep.reference_aspects)
    excludes = any(str(value).lower() == "scene_reference" for value in rep.excluded_aspects)
    return supports and not excludes


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _known(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text and text not in {"none", "null", "n/a"} else UNKNOWN


def location_strata(rep: AssetRepresentation) -> LocationStrata:
    """Extract factorised location metadata without reusing object-centric angles."""

    annotations = rep.annotations
    attrs = _mapping(annotations.get("crop_attributes"))
    relation = _mapping(
        annotations.get("location_relation_metadata")
        or annotations.get("location_metadata")
    )
    zones = relation.get("zone_signature") or relation.get("zones") or ()
    if isinstance(zones, str):
        zones = (zones,)
    if not isinstance(zones, (list, tuple)):
        zones = ()
    return LocationStrata(
        environment=_known(
            relation.get("environment")
            or relation.get("indoor_outdoor")
            or annotations.get("environment")
        ),
        elevation=_known(
            relation.get("camera_elevation")
            or relation.get("elevation")
            or annotations.get("camera_elevation")
        ),
        scale=_known(
            relation.get("scene_scale")
            or relation.get("scale")
            or attrs.get("shot_size")
            or annotations.get("shot_size")
        ),
        lighting=_known(
            relation.get("lighting")
            or attrs.get("lighting")
            or annotations.get("lighting")
        ),
        zone_signature=tuple(sorted({_known(value) for value in zones if _known(value) != UNKNOWN})),
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


def _encoder_route(rep: AssetRepresentation) -> str:
    return str(rep.annotations.get("encoder_route") or "")


def _quality(rep: AssetRepresentation) -> float:
    try:
        return min(1.0, max(0.0, float(rep.annotations.get("quality", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _compatible(left: LocationStrata, right: LocationStrata) -> bool:
    for mine, theirs in (
        (left.environment, right.environment),
        (left.elevation, right.elevation),
        (left.scale, right.scale),
        (left.lighting, right.lighting),
    ):
        if mine != UNKNOWN and theirs != UNKNOWN and mine != theirs:
            return False
    if left.zone_signature and right.zone_signature:
        if set(left.zone_signature).isdisjoint(right.zone_signature):
            return False
    return True


def _distance(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None:
        return None
    score = cosine_or_none(left, right)
    return None if score is None else max(0.0, min(2.0, 1.0 - float(score)))


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0.0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _updated_centroid(
    centroid: Sequence[float] | None,
    count: int,
    vector: Sequence[float] | None,
) -> list[float] | None:
    if vector is None:
        return list(centroid) if centroid is not None else None
    if centroid is None or len(centroid) != len(vector) or count <= 0:
        return _normalise(vector)
    return _normalise(
        [
            (float(centroid[index]) * count + float(vector[index])) / (count + 1)
            for index in range(len(vector))
        ]
    )


def _state(asset: Asset) -> dict[str, Any]:
    raw = asset.metadata.get(CORESET_KEY)
    if isinstance(raw, dict) and int(raw.get("schema_version", 0) or 0) == 1:
        raw.setdefault("clusters", [])
        raw.setdefault("next_cluster_index", 0)
        raw.setdefault("cap_hit_count", 0)
        return raw
    state = {
        "schema_version": 1,
        "canonical_cluster_id": None,
        "next_cluster_index": 0,
        "cap_hit_count": 0,
        "clusters": [],
    }
    asset.metadata[CORESET_KEY] = state
    return state


def _rep_by_id(asset: Asset) -> dict[str, AssetRepresentation]:
    return {rep.representation_id: rep for rep in asset.representations}


def _new_cluster(
    state: dict[str, Any],
    rep: AssetRepresentation,
    strata: LocationStrata,
) -> dict[str, Any]:
    index = int(state.get("next_cluster_index", 0) or 0)
    state["next_cluster_index"] = index + 1
    cluster_id = f"location-cluster-{index:04d}"
    vector = _embedding(rep)
    cluster = {
        "cluster_id": cluster_id,
        "representative_rep_id": rep.representation_id,
        "centroid": _normalise(vector) if vector is not None else None,
        "encoder_route": _encoder_route(rep),
        "support_count": 1,
        "first_segment_id": int(rep.origin_segment_id),
        "last_segment_id": int(rep.origin_segment_id),
        "quality_ema": _quality(rep),
        "strata": strata.to_dict(),
        "strata_histogram": {_strata_key(strata): 1},
    }
    state["clusters"].append(cluster)
    rep.annotations["location_cluster_id"] = cluster_id
    rep.annotations["location_cluster_assignment"] = {
        "reason": "new_cluster",
        "distance": None,
    }
    return cluster


def _cluster_strata(cluster: Mapping[str, Any]) -> LocationStrata:
    raw = _mapping(cluster.get("strata"))
    zones = raw.get("zone_signature") or ()
    return LocationStrata(
        environment=_known(raw.get("environment")),
        elevation=_known(raw.get("elevation")),
        scale=_known(raw.get("scale")),
        lighting=_known(raw.get("lighting")),
        zone_signature=tuple(str(value) for value in zones) if isinstance(zones, list) else (),
    )


def _strata_key(strata: LocationStrata) -> str:
    zones = ",".join(strata.zone_signature)
    return "|".join(
        (
            strata.environment,
            strata.elevation,
            strata.scale,
            strata.lighting,
            zones,
        )
    )


def _choose_cluster(
    state: Mapping[str, Any],
    rep: AssetRepresentation,
    strata: LocationStrata,
    policy: LocationCoresetPolicy,
) -> tuple[dict[str, Any] | None, float | None]:
    vector = _embedding(rep)
    route = _encoder_route(rep)
    best: tuple[float, str, dict[str, Any]] | None = None
    metadata_only: list[dict[str, Any]] = []
    for raw in state.get("clusters", []):
        if not isinstance(raw, dict) or not _compatible(strata, _cluster_strata(raw)):
            continue
        cluster_route = str(raw.get("encoder_route") or "")
        if route and cluster_route and route != cluster_route:
            continue
        distance = _distance(vector, raw.get("centroid"))
        if distance is None:
            metadata_only.append(raw)
            continue
        key = (distance, str(raw.get("cluster_id") or ""), raw)
        if best is None or key[:2] < best[:2]:
            best = key
    if best is not None and best[0] <= policy.cluster_join_distance:
        return best[2], best[0]
    # Missing embeddings must not create one all-unknown cluster per observation.
    if vector is None and metadata_only:
        metadata_only.sort(key=lambda row: str(row.get("cluster_id") or ""))
        return metadata_only[0], None
    return None, None


def _prototype_utility(
    rep: AssetRepresentation,
    centroid: Sequence[float] | None,
) -> float:
    distance = _distance(_embedding(rep), centroid)
    proximity = 0.0 if distance is None else 1.0 - min(1.0, distance)
    return 0.75 * proximity + 0.25 * _quality(rep)


def _absorb(
    asset: Asset,
    cluster: dict[str, Any],
    rep: AssetRepresentation,
    distance: float | None,
    policy: LocationCoresetPolicy,
) -> bool:
    reps = _rep_by_id(asset)
    old_id = str(cluster.get("representative_rep_id") or "")
    old = reps.get(old_id)
    count = max(1, int(cluster.get("support_count", 1) or 1))
    centroid = _updated_centroid(cluster.get("centroid"), count, _embedding(rep))
    cluster["centroid"] = centroid
    cluster["support_count"] = count + 1
    cluster["last_segment_id"] = max(
        int(cluster.get("last_segment_id", rep.origin_segment_id)),
        int(rep.origin_segment_id),
    )
    cluster["quality_ema"] = round(
        0.9 * float(cluster.get("quality_ema", 0.0) or 0.0) + 0.1 * _quality(rep),
        6,
    )
    histogram = cluster.get("strata_histogram")
    if not isinstance(histogram, dict):
        histogram = {}
        cluster["strata_histogram"] = histogram
    key = _strata_key(location_strata(rep))
    histogram[key] = int(histogram.get(key, 0) or 0) + 1
    cluster_id = str(cluster.get("cluster_id") or "")
    rep.annotations["location_cluster_id"] = cluster_id
    rep.annotations["location_cluster_assignment"] = {
        "reason": "compatible_cluster",
        "distance": round(distance, 6) if distance is not None else None,
    }
    old_utility = _prototype_utility(old, centroid) if old is not None else -1.0
    new_utility = _prototype_utility(rep, centroid)
    if old is None or new_utility >= old_utility + policy.representative_replace_margin:
        cluster["representative_rep_id"] = rep.representation_id
        if old is not None and old is not rep:
            old.deprecated = True
            old.deprecated_by = "location_cluster_representative_update"
            old.annotations["replaced_by"] = rep.representation_id
        rep.annotations["location_representative"] = True
        return True
    rep.deprecated = True
    rep.deprecated_by = "absorbed_by_location_cluster"
    rep.annotations["location_representative"] = False
    rep.annotations["represented_by"] = old_id
    return False


def _merge_clusters(
    asset: Asset,
    state: dict[str, Any],
    policy: LocationCoresetPolicy,
) -> None:
    reps = _rep_by_id(asset)
    while True:
        clusters = [row for row in state["clusters"] if isinstance(row, dict)]
        best: tuple[float, str, str, dict[str, Any], dict[str, Any]] | None = None
        for index, left in enumerate(clusters):
            for right in clusters[index + 1 :]:
                if not _compatible(_cluster_strata(left), _cluster_strata(right)):
                    continue
                if str(left.get("encoder_route") or "") != str(right.get("encoder_route") or ""):
                    continue
                distance = _distance(left.get("centroid"), right.get("centroid"))
                if distance is None or distance > policy.cluster_merge_distance:
                    continue
                key = (
                    distance,
                    str(left.get("cluster_id") or ""),
                    str(right.get("cluster_id") or ""),
                    left,
                    right,
                )
                if best is None or key[:3] < best[:3]:
                    best = key
        if best is None:
            return
        _, _, _, left, right = best
        left_count = max(1, int(left.get("support_count", 1) or 1))
        right_count = max(1, int(right.get("support_count", 1) or 1))
        right_centroid = right.get("centroid")
        centroid = left.get("centroid")
        if centroid is not None and right_centroid is not None and len(centroid) == len(right_centroid):
            centroid = _normalise(
                [
                    (float(centroid[i]) * left_count + float(right_centroid[i]) * right_count)
                    / (left_count + right_count)
                    for i in range(len(centroid))
                ]
            )
        elif centroid is None:
            centroid = right_centroid
        left["centroid"] = centroid
        left["support_count"] = left_count + right_count
        left["first_segment_id"] = min(
            int(left.get("first_segment_id", 0)),
            int(right.get("first_segment_id", 0)),
        )
        left["last_segment_id"] = max(
            int(left.get("last_segment_id", 0)),
            int(right.get("last_segment_id", 0)),
        )
        left["quality_ema"] = round(
            (
                float(left.get("quality_ema", 0.0)) * left_count
                + float(right.get("quality_ema", 0.0)) * right_count
            )
            / (left_count + right_count),
            6,
        )
        left_histogram = left.get("strata_histogram")
        if not isinstance(left_histogram, dict):
            left_histogram = {}
            left["strata_histogram"] = left_histogram
        right_histogram = right.get("strata_histogram")
        if isinstance(right_histogram, dict):
            for key, value in right_histogram.items():
                left_histogram[str(key)] = (
                    int(left_histogram.get(str(key), 0) or 0) + int(value or 0)
                )
        left_rep = reps.get(str(left.get("representative_rep_id") or ""))
        right_rep = reps.get(str(right.get("representative_rep_id") or ""))
        if right_rep is not None and (
            left_rep is None
            or _prototype_utility(right_rep, centroid) > _prototype_utility(left_rep, centroid)
        ):
            if left_rep is not None:
                left_rep.deprecated = True
                left_rep.deprecated_by = "location_cluster_merge"
                left_rep.annotations["replaced_by"] = right_rep.representation_id
            left["representative_rep_id"] = right_rep.representation_id
            right_rep.deprecated = False
            right_rep.deprecated_by = ""
        elif right_rep is not None:
            right_rep.deprecated = True
            right_rep.deprecated_by = "location_cluster_merge"
            if left_rep is not None:
                right_rep.annotations["represented_by"] = left_rep.representation_id
        old_cluster_id = str(right.get("cluster_id") or "")
        new_cluster_id = str(left.get("cluster_id") or "")
        for rep in asset.representations:
            if rep.annotations.get("location_cluster_id") == old_cluster_id:
                rep.annotations["location_cluster_id"] = new_cluster_id
        state["clusters"].remove(right)


def _refresh_canonical(state: dict[str, Any]) -> None:
    clusters = [row for row in state.get("clusters", []) if isinstance(row, dict)]
    if not clusters:
        state["canonical_cluster_id"] = None
        return
    chosen = max(
        clusters,
        key=lambda row: (
            log1p(max(0, int(row.get("support_count", 0) or 0))),
            float(row.get("quality_ema", 0.0) or 0.0),
            -int(row.get("first_segment_id", 0) or 0),
            str(row.get("cluster_id") or ""),
        ),
    )
    state["canonical_cluster_id"] = chosen.get("cluster_id")


def _enforce_cap(
    asset: Asset,
    state: dict[str, Any],
    policy: LocationCoresetPolicy,
) -> None:
    cap = max(1, int(policy.storage_cap))
    _merge_clusters(asset, state, policy)
    if len(state["clusters"]) <= cap:
        _refresh_canonical(state)
        return
    state["cap_hit_count"] = int(state.get("cap_hit_count", 0) or 0) + 1
    _refresh_canonical(state)
    canonical = str(state.get("canonical_cluster_id") or "")
    reps = _rep_by_id(asset)
    while len(state["clusters"]) > cap and len(state["clusters"]) > 1:
        candidates = [
            row
            for row in state["clusters"]
            if str(row.get("cluster_id") or "") != canonical
        ]
        if not candidates:
            break
        evicted = min(
            candidates,
            key=lambda row: (
                log1p(max(0, int(row.get("support_count", 0) or 0))),
                float(row.get("quality_ema", 0.0) or 0.0),
                int(row.get("last_segment_id", 0) or 0),
                str(row.get("cluster_id") or ""),
            ),
        )
        rep = reps.get(str(evicted.get("representative_rep_id") or ""))
        if rep is not None:
            rep.deprecated = True
            rep.deprecated_by = "location_storage_cap_guardrail"
            rep.annotations["location_cap_eviction"] = True
        state["clusters"].remove(evicted)
    _refresh_canonical(state)


def update_location_coreset(
    asset: Asset,
    new_rep: AssetRepresentation,
    *,
    policy: LocationCoresetPolicy,
) -> bool:
    """Attach one location observation and maintain adaptive live representatives."""

    if not policy.enabled:
        raise ValueError("location coreset policy must be enabled")
    # Quarantined evidence remains auditable but can neither form nor consume a live cluster.
    if not scene_reference_eligible(new_rep):
        asset.representations.append(new_rep)
        return True

    state = _state(asset)
    # Lazily bootstrap legacy eligible reps so enabling the explicit policy needs no migration.
    for rep in list(asset.representations):
        if rep.deprecated or not scene_reference_eligible(rep):
            continue
        cluster_ids = {
            str(row.get("cluster_id") or "")
            for row in state["clusters"]
            if isinstance(row, dict)
        }
        if rep.annotations.get("location_cluster_id") in cluster_ids:
            continue
        strata = location_strata(rep)
        cluster, distance = _choose_cluster(state, rep, strata, policy)
        if cluster is None:
            _new_cluster(state, rep, strata)
        else:
            _absorb(asset, cluster, rep, distance, policy)

    strata = location_strata(new_rep)
    cluster, distance = _choose_cluster(state, new_rep, strata, policy)
    asset.representations.append(new_rep)
    if cluster is None:
        _new_cluster(state, new_rep, strata)
        represented = True
    else:
        represented = _absorb(asset, cluster, new_rep, distance, policy)
    _enforce_cap(asset, state, policy)
    # The append, cluster support update, representative replacement, or cap audit
    # always mutates persisted bank state even when the newcomer is absorbed/evicted.
    _ = represented
    return True


__all__ = [
    "CORESET_KEY",
    "LocationCoresetPolicy",
    "LocationStrata",
    "location_strata",
    "scene_reference_eligible",
    "update_location_coreset",
]
