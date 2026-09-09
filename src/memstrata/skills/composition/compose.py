"""Step 1 (tail): model-free composition — q_n → Composed Context C_n via O(1) lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memstrata.skills.intent_understanding.interpreter import CompositionRequest, FUNCTION_BY_TYPE
from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRepresentation,
    AssetType,
    NON_USABLE,
    RelationType,
    SpatialAngle,
    StateAngle,
)
from memstrata.skills.composition.location_router import (
    LocationCandidate,
    LocationRanking,
    LocationReadQuery,
    location_coverage,
    location_marginal_gain,
    rank_location_candidates,
)
from memstrata.skills.composition.policy import CompositionPolicy


@dataclass(slots=True)
class ComposedContext:
    """Composed Context C_n for one segment (generation conditioning)."""

    asset_ids: list[str]
    representation_ids: dict[str, list[str]]  # asset_id → chosen reps
    functions: dict[str, str]
    requirements: dict[str, str]
    exclusions: list[str]
    enhanced_prompt: str = ""
    intent_resolution_source: str = "recency"
    # Legacy fields kept for older ActiveComposer callers / tests.
    selected: list[str] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    excluded_representations: list[str] = field(default_factory=list)
    selection_trace: dict[str, Any] = field(default_factory=dict)
    representation_scores: dict[str, float] = field(default_factory=dict)


def is_usable(asset: Asset) -> bool:
    return asset.status not in NON_USABLE


def select_reps(
    asset: Asset,
    *,
    function: str,
    max_reps: int = 1,
    preferred_rep_id: str | None = None,
    preferred_spatial: SpatialAngle | None = None,
    preferred_state: StateAngle | None = None,
    preferred_count: int | None = None,
    as_of_segment_id: int | None = None,
) -> list[str]:
    """Pick causal, non-deprecated reps; prefer matching count, then spatial/state angles."""
    if preferred_rep_id:
        for rep in asset.representations:
            if (
                rep.representation_id == preferred_rep_id
                and not rep.deprecated
                and (as_of_segment_id is None or rep.origin_segment_id < as_of_segment_id)
            ):
                return [preferred_rep_id]

    active = [
        rep
        for rep in asset.representations
        if not rep.deprecated and (as_of_segment_id is None or rep.origin_segment_id < as_of_segment_id)
    ]
    if not active:
        return []

    def excludes(rep: AssetRepresentation) -> bool:
        return any(a.lower() == function.lower() for a in rep.excluded_aspects)

    candidates = [rep for rep in active if not excludes(rep)] or active
    aspect_hits = [
        rep
        for rep in candidates
        if any(a.lower() == function.lower() for a in rep.reference_aspects)
    ]
    pool = aspect_hits if aspect_hits else candidates

    # Eq(10): explicit id is handled above; otherwise prefer count, then state, then view,
    # then recency. Quality is a write-path/admission concern, not a read priority.
    #
    # Count outranks the angles because it is the one dimension a viewer counts rather than
    # infers: a prompt asking for "the last two floats" is visibly wrong when handed the crop
    # showing three, whereas a front/side mismatch reads as a different shot of the same thing.
    wanted_state = preferred_state if preferred_state != StateAngle.UNKNOWN else None
    wanted_spatial = preferred_spatial if preferred_spatial != SpatialAngle.UNKNOWN else None
    wanted_count = preferred_count if preferred_count and preferred_count > 0 else None
    ordered = sorted(
        pool,
        key=lambda r: (
            int(wanted_count is not None and r.count == wanted_count),
            int(wanted_state is not None and r.state_angle == wanted_state),
            int(wanted_spatial is not None and r.spatial_angle == wanted_spatial),
            r.origin_segment_id,
        ),
    )
    chosen = ordered[-max_reps:] if max_reps > 0 else ordered
    return [rep.representation_id for rep in chosen]


def select_reps_for_function(
    asset: Asset,
    conditioning_function: str,
    *,
    max_reps: int = 1,
) -> list[str]:
    return select_reps(asset, function=conditioning_function, max_reps=max_reps)


# Backward-compatible aliases.
select_representations = select_reps
select_representations_for_function = select_reps_for_function


def usable_representation_ids(asset: Asset) -> list[str]:
    return [rep.representation_id for rep in asset.representations if not rep.deprecated]


def _expand_relations(
    bank: AssetBank,
    selected_ids: list[str],
    *,
    max_hops: int,
    allowed_types: tuple[RelationType, ...] = (RelationType.PART_OF, RelationType.LOCATED_IN),
    as_of_segment_id: int | None = None,
) -> list[str]:
    """Bounded, deterministic expansion for structural continuity evidence."""
    if max_hops <= 0:
        return []
    allowed = set(allowed_types)
    seen = set(selected_ids)
    frontier = list(selected_ids)
    expanded: list[str] = []
    for _ in range(max_hops):
        next_frontier: list[str] = []
        for asset_id in frontier:
            asset = bank.get_asset(asset_id)
            if asset is None:
                continue
            for relation in asset.relations:
                if relation.relation_type not in allowed or relation.target_asset_id in seen:
                    continue
                relation_origin = relation.attributes.get("origin_segment_id")
                if as_of_segment_id is not None and relation_origin is not None:
                    try:
                        if int(relation_origin) >= as_of_segment_id:
                            continue
                    except (TypeError, ValueError):
                        pass
                target = bank.get_asset(relation.target_asset_id)
                if target is None or not is_usable(target):
                    continue
                seen.add(target.asset_id)
                expanded.append(target.asset_id)
                next_frontier.append(target.asset_id)
        frontier = next_frontier
        if not frontier:
            break
    return expanded


def _apply_context_budget(
    representation_ids: dict[str, list[str]],
    *,
    primary_ids: list[str],
    budget: int | None,
) -> None:
    """Trim the composed context to ``budget`` total reps (axiom 6: minimal-sufficient).

    Deterministic drop order, in place: (1) relation-expanded continuity reps first,
    (2) then extra reps of named assets beyond their first, (3) never the last rep of
    a *named* asset — losing a requested identity is worse than slightly exceeding the
    budget. ``None``/non-positive budget is a no-op (back-compat).
    """
    if budget is None or budget <= 0:
        return
    total = sum(len(v) for v in representation_ids.values())
    if total <= budget:
        return

    primary = set(primary_ids)
    # Drop expanded-asset reps first (whole assets), oldest-added last kept.
    for aid in reversed(list(representation_ids)):
        if total <= budget:
            break
        if aid in primary:
            continue
        total -= len(representation_ids[aid])
        representation_ids[aid] = []
    # Then trim named assets' extra reps down toward one each.
    for aid in reversed(primary_ids):
        if total <= budget:
            break
        reps = representation_ids.get(aid, [])
        while len(reps) > 1 and total > budget:
            reps.pop()
            total -= 1


def _compose_legacy(
    bank: AssetBank,
    request: CompositionRequest,
    *,
    as_of_segment_id: int | None = None,
) -> ComposedContext:
    """Dereference q_n against A_n — no similarity search, no extra model call."""
    asset_ids: list[str] = []
    representation_ids: dict[str, list[str]] = {}
    functions: dict[str, str] = {}
    requirements: dict[str, str] = {}
    excluded: list[str] = []

    for ref in request.references:
        asset = bank.get_asset(ref.asset_id)
        if asset is None or not is_usable(asset):
            continue
        function = ref.function or FUNCTION_BY_TYPE.get(asset.kind, "identity_anchor")
        reps = select_reps(
            asset,
            function=function,
            max_reps=max(1, int(request.max_reps_per_asset)),
            preferred_rep_id=ref.representation_id,
            preferred_spatial=ref.preferred_spatial,
            preferred_state=ref.preferred_state,
            preferred_count=ref.preferred_count,
            as_of_segment_id=as_of_segment_id,
        )
        asset_ids.append(asset.asset_id)
        representation_ids[asset.asset_id] = reps
        functions[asset.asset_id] = function
        requirements[asset.asset_id] = ref.requirement
        for rep in asset.representations:
            if rep.deprecated:
                excluded.append(rep.representation_id)

    primary_ids = list(asset_ids)  # named refs rank above relation-expanded continuity

    expanded = _expand_relations(
        bank,
        asset_ids,
        max_hops=max(0, int(request.relation_hops)),
        allowed_types=tuple(request.relation_types),
    )
    for asset_id in expanded:
        asset = bank.get_asset(asset_id)
        if asset is None:
            continue
        function = FUNCTION_BY_TYPE.get(asset.kind, "identity_anchor")
        asset_ids.append(asset_id)
        representation_ids[asset_id] = select_reps(
            asset,
            function=function,
            as_of_segment_id=as_of_segment_id,
        )
        functions[asset_id] = function
        requirements[asset_id] = "continuity"
        for rep in asset.representations:
            if rep.deprecated:
                excluded.append(rep.representation_id)

    _apply_context_budget(
        representation_ids,
        primary_ids=primary_ids,
        budget=request.context_rep_budget,
    )

    excluded = sorted(set(excluded))
    return ComposedContext(
        asset_ids=asset_ids,
        representation_ids=representation_ids,
        functions=functions,
        requirements=requirements,
        exclusions=excluded,
        enhanced_prompt=request.enhanced_prompt,
        intent_resolution_source=request.intent_resolution_source,
        selected=list(asset_ids),
        expanded=expanded,
        forbidden=sorted(aid for aid, a in bank.assets.items() if a.status in NON_USABLE),
        excluded_representations=excluded,
    )


@dataclass(slots=True)
class _AssetCandidateBundle:
    asset: Asset
    function: str
    requirement: str
    primary: bool
    must_include: bool
    order: int
    candidates: list[AssetRepresentation]
    location_ranking: LocationRanking | None = None
    chosen: list[AssetRepresentation] = field(default_factory=list)
    stop_reason: str = ""


def _active_rep_map(asset: Asset) -> dict[str, AssetRepresentation]:
    return {
        rep.representation_id: rep
        for rep in asset.representations
        if not rep.deprecated
    }


def _adaptive_candidate_bundle(
    asset: Asset,
    *,
    function: str,
    requirement: str,
    primary: bool,
    must_include: bool,
    order: int,
    preferred_rep_id: str | None,
    preferred_spatial: SpatialAngle | None,
    preferred_state: StateAngle | None,
    preferred_count: int | None,
    raw_prompt: str,
    planner_hints: dict[str, Any],
    as_of_segment_id: int | None,
    request: CompositionRequest,
    policy: CompositionPolicy,
) -> _AssetCandidateBundle:
    if asset.kind is AssetType.LOCATION:
        ranking = rank_location_candidates(
            asset,
            LocationReadQuery(
                raw_prompt=raw_prompt,
                planner_hints=planner_hints,
                as_of_segment_id=as_of_segment_id,
            ),
            policy=policy,
        )
        candidates = [candidate.rep for candidate in ranking.candidates]
        if preferred_rep_id:
            original_order = {
                rep.representation_id: index for index, rep in enumerate(candidates)
            }
            candidates.sort(
                key=lambda rep: (
                    rep.representation_id != preferred_rep_id,
                    original_order[rep.representation_id],
                )
            )
        return _AssetCandidateBundle(
            asset=asset,
            function=function,
            requirement=requirement,
            primary=primary,
            must_include=must_include,
            order=order,
            candidates=candidates,
            location_ranking=ranking,
        )

    rep_map = _active_rep_map(asset)
    rep_ids = select_reps(
        asset,
        function=function,
        max_reps=max(1, int(request.max_reps_per_asset)),
        preferred_rep_id=preferred_rep_id,
        preferred_spatial=preferred_spatial,
        preferred_state=preferred_state,
        preferred_count=preferred_count,
        as_of_segment_id=as_of_segment_id,
    )
    return _AssetCandidateBundle(
        asset=asset,
        function=function,
        requirement=requirement,
        primary=primary,
        must_include=must_include,
        order=order,
        # ``select_reps`` returns the chosen suffix in ascending sort order; the
        # globally allocated first slot must take its strongest (last) element.
        candidates=[rep_map[rep_id] for rep_id in reversed(rep_ids) if rep_id in rep_map],
    )


def _location_candidate(
    bundle: _AssetCandidateBundle,
    rep: AssetRepresentation,
) -> LocationCandidate | None:
    ranking = bundle.location_ranking
    if ranking is None:
        return None
    return next(
        (candidate for candidate in ranking.candidates if candidate.rep is rep),
        None,
    )


def _next_gain(
    bundle: _AssetCandidateBundle,
    *,
    policy: CompositionPolicy,
) -> tuple[float, AssetRepresentation] | None:
    remaining = [rep for rep in bundle.candidates if rep not in bundle.chosen]
    if not remaining:
        bundle.stop_reason = bundle.stop_reason or "candidates_exhausted"
        return None
    if bundle.asset.kind is not AssetType.LOCATION:
        index = len(bundle.chosen)
        return 1.0 / (index + 1), remaining[0]

    ranking = bundle.location_ranking
    assert ranking is not None
    selected = [
        candidate
        for rep in bundle.chosen
        if (candidate := _location_candidate(bundle, rep)) is not None
    ]
    if len(selected) >= policy.location_read_max_refs:
        bundle.stop_reason = "location_read_cap"
        return None
    if selected and location_coverage(selected) >= policy.location_coverage_stop:
        bundle.stop_reason = "coverage_stop"
        return None
    scored = [
        (
            location_marginal_gain(
                candidate,
                selected,
                ranking=ranking,
                policy=policy,
            ),
            candidate.rep,
        )
        for candidate in ranking.candidates
        if candidate.rep in remaining
    ]
    if not scored:
        bundle.stop_reason = "candidates_exhausted"
        return None
    scored.sort(key=lambda row: (-row[0], row[1].representation_id))
    gain, rep = scored[0]
    if selected and gain < policy.location_min_marginal_gain:
        bundle.stop_reason = "marginal_gain_stop"
        return None
    return gain, rep


def _compose_adaptive(
    bank: AssetBank,
    request: CompositionRequest,
    *,
    as_of_segment_id: int | None,
    raw_prompt: str,
    policy: CompositionPolicy,
) -> ComposedContext:
    budget = request.context_rep_budget
    if budget is None or budget <= 0:
        budget = policy.global_rep_budget or 16
    budget = max(1, int(budget))

    bundles: list[_AssetCandidateBundle] = []
    functions: dict[str, str] = {}
    requirements: dict[str, str] = {}
    excluded: list[str] = []
    primary_ids: list[str] = []

    for ref in request.references:
        asset = bank.get_asset(ref.asset_id)
        if asset is None or not is_usable(asset):
            continue
        function = ref.function or FUNCTION_BY_TYPE.get(asset.kind, "identity_anchor")
        primary_ids.append(asset.asset_id)
        functions[asset.asset_id] = function
        requirements[asset.asset_id] = ref.requirement
        bundles.append(
            _adaptive_candidate_bundle(
                asset,
                function=function,
                requirement=ref.requirement,
                primary=True,
                must_include=bool(ref.must_include),
                order=len(bundles),
                preferred_rep_id=ref.representation_id,
                preferred_spatial=ref.preferred_spatial,
                preferred_state=ref.preferred_state,
                preferred_count=ref.preferred_count,
                raw_prompt=raw_prompt,
                planner_hints=dict(ref.location_hints),
                as_of_segment_id=as_of_segment_id,
                request=request,
                policy=policy,
            )
        )
        excluded.extend(
            rep.representation_id for rep in asset.representations if rep.deprecated
        )

    expanded = _expand_relations(
        bank,
        primary_ids,
        max_hops=max(0, int(request.relation_hops)),
        allowed_types=tuple(request.relation_types),
        as_of_segment_id=as_of_segment_id,
    )
    for asset_id in expanded:
        asset = bank.get_asset(asset_id)
        if asset is None:
            continue
        function = FUNCTION_BY_TYPE.get(asset.kind, "identity_anchor")
        functions[asset_id] = function
        requirements[asset_id] = "continuity"
        bundles.append(
            _adaptive_candidate_bundle(
                asset,
                function=function,
                requirement="continuity",
                primary=False,
                must_include=False,
                order=len(bundles),
                preferred_rep_id=None,
                preferred_spatial=None,
                preferred_state=None,
                preferred_count=None,
                raw_prompt=raw_prompt,
                planner_hints={},
                as_of_segment_id=as_of_segment_id,
                request=request,
                policy=policy,
            )
        )
        excluded.extend(
            rep.representation_id for rep in asset.representations if rep.deprecated
        )

    trace: dict[str, Any] = {
        "policy": "adaptive_location_v1",
        "as_of_segment_id": as_of_segment_id,
        "budget": budget,
        "budget_infeasible": False,
        "dropped_primary_asset_ids": [],
        "allocation": [],
        "assets": {},
    }
    scores: dict[str, float] = {}

    # Feasible named requests get one representation each before any extras.  If
    # infeasible, keep the hard global budget and make the deterministic loss explicit.
    reservable = [bundle for bundle in bundles if bundle.primary and bundle.candidates]
    reservation_order = sorted(
        reservable,
        key=lambda bundle: (
            -int(bundle.must_include),
            int(bundle.asset.kind is AssetType.LOCATION),
            bundle.order,
        ),
    )
    if len(reservation_order) > budget:
        trace["budget_infeasible"] = True
        trace["dropped_primary_asset_ids"] = [
            bundle.asset.asset_id for bundle in reservation_order[budget:]
        ]
        reservation_order = reservation_order[:budget]
    for bundle in reservation_order:
        candidate = _next_gain(bundle, policy=policy)
        if candidate is None:
            continue
        gain, rep = candidate
        bundle.chosen.append(rep)
        scores[rep.representation_id] = round(gain, 6)
        trace["allocation"].append(
            {
                "asset_id": bundle.asset.asset_id,
                "representation_id": rep.representation_id,
                "phase": "primary_reservation",
                "marginal_gain": round(gain, 6),
            }
        )

    used = sum(len(bundle.chosen) for bundle in bundles)
    location_extra_limit = int(budget * policy.location_extra_budget_share)
    location_extras = 0
    while used < budget:
        options: list[tuple[float, int, str, _AssetCandidateBundle, AssetRepresentation]] = []
        for bundle in bundles:
            if (
                bundle.primary
                and bundle.candidates
                and not bundle.chosen
            ):
                # This primary was dropped by an infeasible reservation; extras cannot
                # silently reinsert it ahead of the recorded deterministic decision.
                continue
            candidate = _next_gain(bundle, policy=policy)
            if candidate is None:
                continue
            gain, rep = candidate
            is_location_extra = (
                bundle.asset.kind is AssetType.LOCATION and bool(bundle.chosen)
            )
            if is_location_extra and location_extras >= location_extra_limit:
                bundle.stop_reason = "location_extra_share_guard"
                continue
            # Relation-expanded evidence has no minimum guarantee and loses ties.
            effective_gain = gain if bundle.primary else gain * 0.5
            options.append(
                (
                    effective_gain,
                    bundle.order,
                    rep.representation_id,
                    bundle,
                    rep,
                )
            )
        if not options:
            break
        options.sort(key=lambda row: (-row[0], row[1], row[2]))
        gain, _, _, bundle, rep = options[0]
        if gain < policy.location_min_marginal_gain:
            break
        was_extra = bundle.asset.kind is AssetType.LOCATION and bool(bundle.chosen)
        bundle.chosen.append(rep)
        if was_extra:
            location_extras += 1
        used += 1
        scores[rep.representation_id] = round(gain, 6)
        trace["allocation"].append(
            {
                "asset_id": bundle.asset.asset_id,
                "representation_id": rep.representation_id,
                "phase": "global_marginal",
                "marginal_gain": round(gain, 6),
            }
        )

    representation_ids = {
        bundle.asset.asset_id: [rep.representation_id for rep in bundle.chosen]
        for bundle in bundles
    }
    for bundle in bundles:
        ranking = bundle.location_ranking
        location_candidates = (
            [
                {
                    "representation_id": candidate.rep.representation_id,
                    "cluster_id": candidate.cluster_id,
                    "score": round(candidate.score, 6),
                    "posterior": round(candidate.posterior, 6),
                    "score_parts": candidate.score_parts,
                }
                for candidate in ranking.candidates
            ]
            if ranking is not None
            else []
        )
        coreset = bundle.asset.metadata.get("location_coreset_v1")
        trace["assets"][bundle.asset.asset_id] = {
            "kind": bundle.asset.kind.value,
            "primary": bundle.primary,
            "must_include": bundle.must_include,
            "candidate_rep_ids": [rep.representation_id for rep in bundle.candidates],
            "candidates": location_candidates,
            "chosen_rep_ids": representation_ids[bundle.asset.asset_id],
            "margin": round(ranking.margin, 6) if ranking is not None else None,
            "hint_source": ranking.hint_source if ranking is not None else "legacy",
            "hints": ranking.hints if ranking is not None else {},
            "fallback": ranking.fallback if ranking is not None else "legacy_rank",
            "dropped": ranking.dropped if ranking is not None else {},
            "storage_cap_hit_count": (
                int(coreset.get("cap_hit_count", 0) or 0)
                if isinstance(coreset, dict)
                else 0
            ),
            "stop_reason": bundle.stop_reason or (
                "budget_full" if used >= budget else "not_selected"
            ),
        }
    trace["used_budget"] = used
    trace["unused_budget"] = max(0, budget - used)
    trace["location_extra_count"] = location_extras
    trace["location_extra_limit"] = location_extra_limit

    asset_ids = [bundle.asset.asset_id for bundle in bundles]
    return ComposedContext(
        asset_ids=asset_ids,
        representation_ids=representation_ids,
        functions=functions,
        requirements=requirements,
        exclusions=sorted(set(excluded)),
        enhanced_prompt=request.enhanced_prompt,
        intent_resolution_source=request.intent_resolution_source,
        selected=list(asset_ids),
        expanded=expanded,
        forbidden=sorted(aid for aid, asset in bank.assets.items() if asset.status in NON_USABLE),
        excluded_representations=sorted(set(excluded)),
        selection_trace=trace,
        representation_scores=scores,
    )


def compose(
    bank: AssetBank,
    request: CompositionRequest,
    *,
    as_of_segment_id: int | None = None,
    raw_prompt: str = "",
    policy: CompositionPolicy | None = None,
) -> ComposedContext:
    """Compose legacy context or an adaptive globally-budgeted location context."""

    read_policy = (policy or CompositionPolicy()).normalised()
    if not read_policy.adaptive_location_enabled:
        return _compose_legacy(
            bank,
            request,
            as_of_segment_id=as_of_segment_id,
        )
    return _compose_adaptive(
        bank,
        request,
        as_of_segment_id=as_of_segment_id,
        raw_prompt=raw_prompt,
        policy=read_policy,
    )


class ActiveComposer:
    """Thin wrapper kept for scripts that still construct ActiveComposer(bank)."""

    def __init__(self, asset_space: AssetBank, **_: object) -> None:
        self.asset_space = asset_space

    def compose(self, selected_ids, *, forbidden_ids=()) -> ComposedContext:
        from memstrata.skills.intent_understanding.interpreter import AssetReference, CompositionRequest

        _ = forbidden_ids
        refs = [AssetReference(asset_id=str(aid)) for aid in selected_ids]
        # Fill function/requirement from bank state (segment_id unknown → continuity if any prior rep).
        for ref in refs:
            asset = self.asset_space.get_asset(ref.asset_id)
            if asset is None:
                continue
            ref.function = FUNCTION_BY_TYPE.get(asset.kind, "identity_anchor")
            ref.requirement = (
                "continuity"
                if any(r.origin_segment_id >= 0 for r in asset.representations)
                else "introduce"
            )
        return compose(self.asset_space, CompositionRequest(references=refs))
