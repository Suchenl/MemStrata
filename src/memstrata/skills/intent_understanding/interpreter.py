"""Intent Understanding skill (paper Stage 3: Intent Interpretation).

Turn an external generation intent ``g_n`` plus the addressable asset space ``A_n``
into a structured composition request ``q_n`` (+ a lightly enhanced prompt ``p̃_n``).
This is the *read-path front-end*: it only decides which already-addressable asset
ids belong in ``q_n`` and with what angle/state preference — it never dereferences
representations (that is ``skills.composition.compose``) and never invents ids
outside ``A_n``.

Three resolution paths (FAST is the default; the others are opt-in):
  * FAST (default, model-free): name/alias recall via ``skills.memory_retrieval.name_match``;
    when the prompt carries no resolvable name it falls back to deterministic
    description-overlap matching, and only then to recency.
  * SLOW: an MLLM resolver (``MllmIntentResolver`` over the shared MLLM transport)
    reasons over the bank listing. It still cannot invent ids outside A_n. It runs
    either as the primary path (``mode="slow"``) or, preferred for read-path budget, as
    a *fast→slow cascade* (``slow_on_miss=True``): fast name+description matching runs
    first at no model cost, and the resolver is spent only when both miss — resolving
    aliases / coreference / cross-lingual references (e.g. a prompt saying ``大兔子``
    against a stored ``large white rabbit``) that surface-form matching cannot bridge.
  * PLAN (``mode="plan"``): one bounded call returns a typed ``IntentPlanV1`` — not just
    *which* entities, but the required appearance state, the entities that must NOT appear,
    and the generation route (see ``.plan``). Needed where an id list is not expressive
    enough: appearance changes that must persist across beats, and negative constraints
    (a destroyed prop, a look-alike distinguished from this beat's subject). Falls back to
    FAST whenever the plan is unusable, so enabling it cannot produce an empty read path.

Kept separate from ``skills.composition`` (the Compose / dereference stage) so the
paper's Interpret (Stage 3) and Compose (Stage 4) map to distinct skills. Name/alias
matching itself is a *memory-retrieval* mechanism and lives in
``skills.memory_retrieval.name_match``; this skill consumes it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetType,
    NON_USABLE,
    RelationType,
    SpatialAngle,
    StateAngle,
)
from memstrata.skills.intent_understanding.plan import (
    ROUTE_T2V,
    PlanProducer,
    parse_plan,
    resolve_plan,
)
from memstrata.skills.memory_retrieval.name_match import (
    NameHit as _NameHit,
    match_cache as _match_cache,
    name_hits as _name_hits,
    unique_asset_ids as _unique_asset_ids,
)

INTENT_MODE_FAST = "fast"
INTENT_MODE_SLOW = "slow"
# PLAN: one bounded call returns a typed IntentPlanV1 (references + required state + forbidden
# entities + generation route) instead of a bare id list. Opt-in, because FAST is model-free and
# is what the no-generation Track A protocol budgets for.
INTENT_MODE_PLAN = "plan"
INTENT_MODES = frozenset({INTENT_MODE_FAST, INTENT_MODE_SLOW, INTENT_MODE_PLAN})

# Type-default conditioning role (paper Table: Main Function).
FUNCTION_BY_TYPE: dict[AssetType, str] = {
    AssetType.CHARACTER: "identity_anchor",
    AssetType.PROP: "object_continuity",
    AssetType.LOCATION: "scene_reference",
}


@dataclass(slots=True)
class AssetReference:
    """One resolvable reference inside q_n (must name an id already in A_n)."""

    asset_id: str
    representation_id: str | None = None
    function: str = "identity_anchor"
    requirement: str = "continuity"  # introduce | continuity
    preferred_spatial: SpatialAngle | None = None
    preferred_state: StateAngle | None = None


@dataclass(slots=True)
class CompositionRequest:
    """Structured composition request q_n."""

    references: list[AssetReference] = field(default_factory=list)
    enhanced_prompt: str = ""
    requested_mode: str = INTENT_MODE_FAST
    used_mode: str = INTENT_MODE_FAST
    fallback_reason: str = ""
    intent_resolution_source: str = "recency"  # name | description | recency | mllm | plan | miss
    # Plan-driven read path (mode="plan") only; empty elsewhere so FAST/SLOW stay byte-identical.
    # ``route`` is the generation route the planner chose for this beat (see plan.ROUTES);
    # ``forbidden_asset_ids`` are entities that must NOT be conditioned on (a destroyed prop, a
    # look-alike explicitly distinguished from this beat's subject).
    route: str = ""
    forbidden_asset_ids: tuple[str, ...] = ()
    plan_unresolved_names: tuple[str, ...] = ()
    # Efficient-composition budget (design_philosophy.md axiom 6): how many reps a
    # named asset may contribute, and an optional hard cap on total reps across the
    # whole composed context. ``None`` budget disables the cap (back-compat).
    max_reps_per_asset: int = 1
    context_rep_budget: int | None = None
    relation_hops: int = 0
    relation_types: tuple[RelationType, ...] = (
        RelationType.PART_OF,
        RelationType.LOCATED_IN,
    )


class IntentResolver(Protocol):
    def resolve(self, prompt: str, candidates: list[dict[str, Any]]) -> list[str]:
        """Return selected asset_ids already present in the bank. Empty → caller falls back."""


def _requirement(asset: Asset, segment_id: int) -> str:
    if any(rep.origin_segment_id < segment_id for rep in asset.representations):
        return "continuity"
    return "introduce"


# Minimal generic-English stopword set so description overlap keys on content words, not glue.
_DESC_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "into", "from", "onto", "over", "under",
    "then", "they", "them", "there", "here", "which", "while", "when", "where", "what",
    "who", "whom", "his", "her", "its", "their", "our", "your", "are", "was", "were",
    "has", "have", "had", "not", "but", "out", "off", "own", "some", "more", "most",
    "very", "such", "than", "too", "also", "just", "now", "still", "back", "down", "the",
    "character", "object", "place", "entity", "appears", "appear", "scene", "segment",
})


def _content_tokens(text: str) -> set[str]:
    """CJK-aware match units for overlap scoring (search-engine style, model-free).

    The old version treated a whole run of CJK as one token (``re.findall`` over
    ``\\u4e00-\\u9fff`` then ``len>=3``), so "红裙女孩" became a single token and only ever
    matched a byte-identical run — Chinese descriptions essentially never overlapped
    (observed ``description=0/52`` on Track A). We switch to the standard CJK full-text
    recipe (WPS / Lucene CJK analyzer):

    * Latin/digit runs -> whole words (len>=3, non-stopword), a normal word index.
    * CJK runs -> overlapping character **bigrams** ("红裙女孩" -> 红裙, 裙女, 女孩); a lone
      CJK char is kept as a unigram. Bigram overlap is a set intersection (fast) and gives
      fuzzy substring-like recall without needing a Chinese word segmenter.
    """
    s = str(text or "").casefold()
    units: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", s):
        if len(word) >= 3 and word not in _DESC_STOPWORDS:
            units.add(word)
    for run in re.findall(r"[\u4e00-\u9fff]+", s):
        if len(run) == 1:
            units.add(run)
            continue
        for i in range(len(run) - 1):
            units.add(run[i:i + 2])
    return units


def _description_match(prompt: str, candidates: list[Asset], *, cap: int = 5,
                       min_overlap: int = 1) -> list[str]:
    """Deterministic description-overlap matcher (Regime-B semantic path, model-free).

    When the prompt carries no resolvable *name* (e.g. description-only Regime B), MemStrata
    still does matching-based active composition — it just switches the match key from name to
    stored asset *description* (fairness decision D1). Ranked by content-token overlap with a
    deterministic (-overlap, -recency, asset_id) tiebreak; empty → caller falls back to recency."""
    ptoks = _content_tokens(prompt)
    if not ptoks:
        return []
    scored: list[tuple[int, int, str]] = []
    for asset in candidates:
        atoks = _content_tokens(asset.d)
        overlap = len(ptoks & atoks)
        if overlap >= min_overlap:
            recency = max((rep.origin_segment_id for rep in asset.representations), default=-1)
            scored.append((overlap, recency, asset.asset_id))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [aid for _, _, aid in scored[:cap]]


def _angle_preferences(prompt: str) -> tuple[SpatialAngle | None, StateAngle | None]:
    """Extract explicit visual constraints without a read-path model call."""
    lowered = prompt.casefold()
    spatial_terms = (
        (SpatialAngle.FRONT, ("front view", "frontal", "facing camera", "正面", "前视", "正对镜头")),
        (SpatialAngle.SIDE, ("side view", "profile", "侧面", "侧视")),
        (SpatialAngle.BACK, ("back view", "rear view", "背面", "后视")),
        (SpatialAngle.TOP, ("top view", "overhead", "俯视", "顶视")),
    )
    state_terms = (
        (StateAngle.DAMAGED, ("damaged", "destroyed", "broken", "损坏", "破损", "死亡")),
        (StateAngle.CHANGED, ("changed", "transformed", "disguised", "变化", "改变", "伪装")),
        (StateAngle.DEFAULT, ("default state", "original appearance", "原本状态", "默认状态")),
    )
    spatial = next((value for value, terms in spatial_terms if any(term in lowered for term in terms)), None)
    state = next((value for value, terms in state_terms if any(term in lowered for term in terms)), None)
    return spatial, state


_CLAUSE_BOUNDARIES = "\n\r,.;:!?，。；：！？、"


def _local_prompt_window(prompt: str, *, start: int, end: int, radius: int = 48) -> str:
    """Return the clause around one entity mention, capped to a small neighborhood."""
    left = max([prompt.rfind(ch, 0, start) for ch in _CLAUSE_BOUNDARIES] + [-1]) + 1
    right_candidates = [idx for ch in _CLAUSE_BOUNDARIES if (idx := prompt.find(ch, end)) >= 0]
    right = min(right_candidates) if right_candidates else len(prompt)
    if right - left > radius * 2 + (end - start):
        left = max(left, start - radius)
        right = min(right, end + radius)
    return prompt[left:right]


def _angle_preferences_by_asset(prompt: str, hits: list[_NameHit], *, radius: int = 80) -> dict[str, tuple[SpatialAngle | None, StateAngle | None]]:
    """Bind explicit view/state cues to the *nearest* entity mention.

    The window for each mention runs from the previous mention's end to the next
    mention's start (capped by ``radius``), NOT by clause/comma boundaries. This keeps
    an entity's trailing attributes attached ("Hero back view, damaged." → BACK+DAMAGED)
    while still splitting cleanly when a *different* entity follows ("侧面拍灯塔，Elias
    正对镜头" → 灯塔 SIDE / Elias FRONT), because the adjacent mention bounds the window.
    """
    prefs: dict[str, tuple[SpatialAngle | None, StateAngle | None]] = {}
    ordered = sorted(hits, key=lambda h: h.start)
    for i, hit in enumerate(ordered):
        prev_end = ordered[i - 1].end if i > 0 else 0
        next_start = ordered[i + 1].start if i + 1 < len(ordered) else len(prompt)
        left = max(prev_end, hit.start - radius)
        right = min(next_start, hit.end + radius)
        spatial, state = _angle_preferences(prompt[left:right])
        old_spatial, old_state = prefs.get(hit.asset_id, (None, None))
        prefs[hit.asset_id] = (old_spatial or spatial, old_state or state)
    return prefs


class IntentInterpreter:
    """Paper Stage 3: turn external intent g_n into q_n + p̃_n.

    Composition itself is model-free (see ``skills.composition.compose``). The optional
    resolver only chooses which *already-addressable* asset ids belong in q_n.
    """

    def __init__(
        self,
        bank: AssetBank,
        resolver: IntentResolver | None = None,
        *,
        plan_producer: PlanProducer | None = None,
        mode: str = INTENT_MODE_FAST,
        slow_on_miss: bool = False,
        disable_name_anchor: bool = False,
        recency_cap: int = 5,
        max_reps_per_asset: int = 1,
        context_rep_budget: int | None = None,
    ) -> None:
        if mode not in INTENT_MODES:
            raise ValueError(f"unknown intent mode {mode!r}; expected one of {sorted(INTENT_MODES)}")
        self.bank = bank
        self.resolver = resolver
        # Only consulted in mode="plan"; absent producer degrades to FAST rather than failing.
        self.plan_producer = plan_producer
        self.mode = mode
        # Fast→slow cascade: when FAST name+description matching both miss, spend ONE
        # bounded resolver call instead of collapsing to an empty selection. Keeps the
        # read path model-free on the common (hit) case; only aliases/coreference pay.
        self.slow_on_miss = bool(slow_on_miss)
        # Ablation switch: when True, stable name/alias anchoring is removed from the read
        # path; without identifiers to resolve the intent, selection degrades to the K
        # most-recently-active assets (paper ablation "- name-anchored identity"). A
        # faithful appearance-only variant would instead rank by a visual embedder; this
        # deterministic recency proxy keeps the ablation model-free on the gold protocol.
        self.disable_name_anchor = disable_name_anchor
        self.recency_cap = recency_cap
        # Read-side context budget (design_philosophy.md axiom 6). Default 1 rep/asset keeps
        # the historical minimal-sufficient behaviour; the Track A adapter raises it so a
        # matched entity can contribute several stored angles/states (equal-budget with the
        # retrieval baselines that fill the benchmark ceiling). ``context_rep_budget`` is an
        # optional hard cap on total reps across the composed context (a ceiling, not a target).
        self.max_reps_per_asset = max(1, int(max_reps_per_asset))
        self.context_rep_budget = context_rep_budget

    def _mllm_select(self, prompt: str, candidates: list[Asset]) -> list[str]:
        """One resolver call over the bank listing; returns ids that exist in A_n only."""
        summaries = [
            {"id": a.asset_id, "name": a.name, "kind": a.kind.value, "description": a.d}
            for a in candidates
        ]
        raw = self.resolver.resolve(prompt, summaries)
        known = {a.asset_id for a in candidates}
        return list(dict.fromkeys(aid for aid in raw if aid in known))

    def interpret(self, prompt: str, *, segment_id: int = 0) -> tuple[CompositionRequest, int]:
        """Return (q_n, model_calls). Never invents ids outside A_n."""
        candidates = [a for a in self.bank.assets.values() if a.status not in NON_USABLE]
        cache = _match_cache(self.bank, candidates)
        model_calls = 0
        selected_ids: list[str] = []
        name_hits: list[_NameHit] = []
        used_mode = self.mode
        fallback_reason = ""
        intent_resolution_source = "recency"
        state_by_id: dict[str, StateAngle] = {}
        forbidden_ids: tuple[str, ...] = ()
        route = ""
        unresolved_names: tuple[str, ...] = ()
        # A committed plan owns the selection outcome, including a deliberate *empty* one
        # (route=t2v). Without this flag the FAST fallback below would treat "no references"
        # as a miss and re-populate the selection, silently overriding the planner.
        plan_committed = False

        if self.mode == INTENT_MODE_PLAN and candidates:
            if self.plan_producer is None:
                used_mode = INTENT_MODE_FAST
                fallback_reason = "plan_producer_unavailable"
            else:
                plan = None
                try:
                    model_calls += 1
                    plan = parse_plan(
                        self.plan_producer.make_plan(
                            prompt,
                            [
                                {"name": a.name, "kind": a.kind.value, "description": a.d}
                                for a in candidates
                            ],
                        )
                    )
                except Exception as exc:
                    fallback_reason = f"plan_error:{type(exc).__name__}"
                if plan is None:
                    used_mode = INTENT_MODE_FAST
                    fallback_reason = fallback_reason or "plan_empty"
                else:
                    resolved = resolve_plan(
                        plan,
                        self.bank,
                        lambda name: _unique_asset_ids(_name_hits(name, cache)),
                    )
                    route = resolved.route
                    forbidden_ids = tuple(resolved.forbidden_ids)
                    unresolved_names = tuple(resolved.unresolved_names)
                    if resolved.selected_ids or route == ROUTE_T2V:
                        selected_ids = resolved.selected_ids
                        state_by_id = resolved.state_by_id
                        intent_resolution_source = "plan"
                        plan_committed = True
                    else:
                        # The plan only named entities the bank cannot resolve (e.g. a first
                        # sighting). Let FAST try surface matching rather than emit nothing.
                        used_mode = INTENT_MODE_FAST
                        fallback_reason = "plan_unresolved"

        if self.mode == INTENT_MODE_SLOW and candidates:
            if self.resolver is None:
                used_mode = INTENT_MODE_FAST
                fallback_reason = "resolver_unavailable"
            else:
                try:
                    model_calls += 1
                    selected_ids = self._mllm_select(prompt, candidates)
                    if not selected_ids:
                        used_mode = INTENT_MODE_FAST
                        fallback_reason = "resolver_empty"
                    else:
                        intent_resolution_source = "mllm"
                except Exception as exc:
                    selected_ids = []
                    used_mode = INTENT_MODE_FAST
                    fallback_reason = f"resolver_error:{type(exc).__name__}"

        if self.disable_name_anchor:
            # No name anchoring: keep the most-recently-active assets (recency proxy).
            def _recency_key(a: Asset) -> int:
                return max((rep.origin_segment_id for rep in a.representations), default=-1)

            ranked = sorted(candidates, key=_recency_key, reverse=True)
            selected_ids = [a.asset_id for a in ranked[: self.recency_cap]]
            intent_resolution_source = "recency"
        elif not plan_committed and (self.mode != INTENT_MODE_SLOW or not selected_ids):
            name_hits = _name_hits(prompt, cache)
            selected_ids = _unique_asset_ids(name_hits)
            if selected_ids:
                intent_resolution_source = "name"
            else:
                # Regime-agnostic: when the prompt has no resolvable name (description-only
                # Regime B), match on stored descriptions instead of collapsing to recency, so
                # matching-based active composition still holds (decision D1). In Regime A this
                # rarely triggers because name matching already resolves the intent.
                selected_ids = _description_match(prompt, candidates, cap=self.recency_cap)
                if selected_ids:
                    intent_resolution_source = "description"
                elif self.slow_on_miss and self.resolver is not None and candidates:
                    # fast→slow cascade: name AND description both missed. Spend one bounded
                    # resolver call to bridge aliases / coreference / cross-lingual references
                    # to ids already in A_n, instead of returning an empty selection.
                    try:
                        model_calls += 1
                        selected_ids = self._mllm_select(prompt, candidates)
                    except Exception as exc:
                        selected_ids = []
                        fallback_reason = f"resolver_error:{type(exc).__name__}"
                    if selected_ids:
                        intent_resolution_source = "mllm"
                        used_mode = INTENT_MODE_SLOW
                    else:
                        # name + description + slow resolver all missed: nothing is selected.
                        # Label it a genuine miss, not "recency" — the name-anchored FAST path
                        # never ranks by recency (that only happens under disable_name_anchor),
                        # so the old "recency" label overcounted an empty selection.
                        intent_resolution_source = "miss"
                else:
                    intent_resolution_source = "miss"

        if not name_hits:
            name_hits = _name_hits(prompt, cache)
        local_angle_prefs = _angle_preferences_by_asset(prompt, name_hits)
        global_preferred_spatial, global_preferred_state = _angle_preferences(prompt)
        refs: list[AssetReference] = []
        for asset_id in selected_ids:
            if asset_id in forbidden_ids:
                continue
            asset = self.bank.get_asset(asset_id)
            if asset is None:
                continue
            preferred_spatial, preferred_state = local_angle_prefs.get(asset_id, (None, None))
            if (
                preferred_spatial is None
                and preferred_state is None
                and intent_resolution_source != "name"
                and len(selected_ids) == 1
            ):
                preferred_spatial, preferred_state = global_preferred_spatial, global_preferred_state
            # An explicit plan state beats the surface-cue heuristics: the planner has read the
            # whole beat (and carries state persistence across beats), the regex cues have not.
            plan_state = state_by_id.get(asset_id)
            if plan_state is not None:
                preferred_state = plan_state
            refs.append(
                AssetReference(
                    asset_id=asset_id,
                    function=FUNCTION_BY_TYPE.get(asset.kind, "identity_anchor"),
                    requirement=_requirement(asset, segment_id),
                    preferred_spatial=preferred_spatial,
                    preferred_state=preferred_state,
                )
            )

        # p̃_n: lightweight deterministic enrichment (no extra model call).
        # ponytail: full MLLM rewrite of the generation prompt is optional; bank facts
        # appended here keep the enhanced prompt grounded in A_n without a second API hop.
        enrich_bits = []
        for ref in refs:
            asset = self.bank.get_asset(ref.asset_id)
            if asset is None:
                continue
            desc = asset.d
            if desc:
                enrich_bits.append(f"{asset.name}: {desc}" if asset.name else desc)
        enhanced = prompt if not enrich_bits else f"{prompt}\n\n[asset cues] " + "; ".join(enrich_bits)

        return CompositionRequest(
            references=refs,
            enhanced_prompt=enhanced,
            requested_mode=self.mode,
            used_mode=used_mode,
            fallback_reason=fallback_reason,
            intent_resolution_source=intent_resolution_source,
            max_reps_per_asset=self.max_reps_per_asset,
            context_rep_budget=self.context_rep_budget,
            route=route,
            forbidden_asset_ids=forbidden_ids,
            plan_unresolved_names=unresolved_names,
        ), model_calls


class MllmIntentResolver:
    """Thin OpenAI-compatible resolver used as the paper's multimodal intent model."""

    def __init__(self, planner: Any) -> None:
        # Reuse existing MllmPlanner.select_assets transport without baking HTTP here.
        self._planner = planner

    def resolve(self, prompt: str, candidates: list[dict[str, Any]]) -> list[str]:
        return list(self._planner.select_assets(prompt, candidates) or [])
