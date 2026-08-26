"""Step 1 (tail): model-free composition — q_n → Composed Context C_n via O(1) lookup."""

from __future__ import annotations

from dataclasses import dataclass, field

from memstrata.skills.intent_understanding.interpreter import CompositionRequest, FUNCTION_BY_TYPE
from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRepresentation,
    NON_USABLE,
    RelationType,
    SpatialAngle,
    StateAngle,
)


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
    as_of_segment_id: int | None = None,
) -> list[str]:
    """Pick causal, non-deprecated reps; prefer matching spatial/state angles."""
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

    # Eq(10): explicit id is handled above; otherwise prefer state, then view,
    # then recency. Quality is a write-path/admission concern, not a read priority.
    wanted_state = preferred_state if preferred_state != StateAngle.UNKNOWN else None
    wanted_spatial = preferred_spatial if preferred_spatial != SpatialAngle.UNKNOWN else None
    ordered = sorted(
        pool,
        key=lambda r: (
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


def compose(
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
