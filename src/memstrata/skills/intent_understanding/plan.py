"""IntentPlanV1 — the plan-driven read path (forward-compatible slice of the unified design).

The FAST/SLOW paths answer only "which stored ids does this prompt mention?". Track B's hard
cases need three things surface matching cannot express:

  * **state requirement** — ``persist_state`` / ``state_change`` need "Ana, but the *wet* look",
    not any stored crop of Ana;
  * **negative constraint** — ``deprecation_avoidance`` / ``false_friend`` need "and definitely
    NOT the lantern that was smashed / NOT the look-alike twin";
  * **route** — whether this beat should be conditioned on a composed keyframe, continue the
    previous shot, or be generated from text alone.

So this module models the read side as a *plan*: one bounded model call turns the beat prompt +
the addressable asset space into a typed ``IntentPlanV1``, which the interpreter then lowers into
the existing ``CompositionRequest``. Identity stays name-authoritative — the plan may only *name*
entities, and unresolvable names are reported rather than invented, so the plan can never
fabricate an id outside ``A_n``.

This is deliberately the small half of ``docs/method/unified_video_memory_pipeline_DESIGN.md``:
the plan contract, without the instance-cache / lazy-materialisation migration. Landing the
schema now means the later migration only has to change the *execution* side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from memstrata.bank import AssetBank, StateAngle

ROUTE_T2V = "t2v"
ROUTE_I2V_COMPOSED = "i2v_composed"
ROUTE_I2V_CONTINUE = "i2v_continue"
ROUTES = frozenset({ROUTE_T2V, ROUTE_I2V_COMPOSED, ROUTE_I2V_CONTINUE})

# Reported when the plan names an entity the bank cannot resolve. That is legitimate on a
# first sighting (nothing stored yet), so it is surfaced as telemetry rather than an error.
_NO_STATE = {"", "unknown", "any", "none", "null"}


@dataclass(slots=True)
class PlanReference:
    """One entity the beat wants on screen, named (never id'd) by the planner."""

    name: str
    state_required: StateAngle | None = None
    must_include: bool = True


@dataclass(slots=True)
class IntentPlanV1:
    """Typed read-side plan for one beat."""

    route: str = ROUTE_I2V_COMPOSED
    references: list[PlanReference] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class ResolvedPlan:
    """``IntentPlanV1`` lowered onto the bank: names became ids, or were reported unresolved."""

    route: str
    selected_ids: list[str] = field(default_factory=list)
    state_by_id: dict[str, StateAngle] = field(default_factory=dict)
    forbidden_ids: list[str] = field(default_factory=list)
    unresolved_names: list[str] = field(default_factory=list)
    reason: str = ""


class PlanProducer(Protocol):
    def make_plan(self, prompt: str, assets: list[dict[str, Any]]) -> dict[str, Any]:
        """Return a raw IntentPlanV1-shaped dict. Empty/raising => caller falls back to FAST."""


# Strict structured-output schema (the shared MLLM transport enforces it, so the parser below
# only has to normalise values, never repair malformed JSON).
PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": sorted(ROUTES)},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "state_required": {"type": "string"},
                    "must_include": {"type": "boolean"},
                },
                "required": ["name", "state_required", "must_include"],
                "additionalProperties": False,
            },
        },
        "forbidden": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["route", "references", "forbidden", "reason"],
    "additionalProperties": False,
}

PLAN_INSTRUCTION = """You are the read-side planner of a video memory system. For ONE shot of a
story, decide what visual memory the generator needs.

Shot prompt:
{user_prompt}

Stored entities (the ONLY entities you may name; use their exact stored names):
{assets_list}

Return JSON:
- "references": entities that must be VISIBLE in this shot. For each:
    - "name": the exact stored name.
    - "state_required": which stored appearance is needed. Use "default" for the normal look,
      "changed" if the shot describes an altered appearance (wet, transformed, disguised, aged,
      dirty, new outfit), "damaged" if broken/burnt/torn, "unknown" if the shot does not care.
      IMPORTANT: if an earlier shot changed this entity's appearance and the story has not
      reverted it, the changed appearance PERSISTS — keep asking for "changed".
    - "must_include": true if the shot is about this entity, false if it is incidental.
  Only list entities actually on screen in THIS shot. Do not list an entity merely mentioned
  in dialogue or memory.
- "forbidden": entities that must NOT appear: destroyed, removed, discarded, left behind, or a
  look-alike explicitly distinguished from the one in this shot.
- "route": "i2v_composed" if any stored entity must be visible (reference images will be
  composed); "i2v_continue" if this shot directly continues the previous shot with no new
  identity requirement; "t2v" if no stored entity is needed at all (a brand-new scene).
- "reason": one short sentence.
"""


def _state(raw: Any) -> StateAngle | None:
    text = str(raw or "").strip().lower()
    if text in _NO_STATE:
        return None
    try:
        return StateAngle(text)
    except ValueError:
        return None


def parse_plan(payload: dict[str, Any] | None) -> IntentPlanV1 | None:
    """Normalise a raw planner dict into ``IntentPlanV1``; ``None`` when unusable."""
    if not isinstance(payload, dict):
        return None
    route = str(payload.get("route") or "").strip().lower()
    if route not in ROUTES:
        # An unknown route is not fatal: presence of references still implies conditioning.
        route = ROUTE_I2V_COMPOSED if payload.get("references") else ROUTE_T2V
    references: list[PlanReference] = []
    for item in payload.get("references") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        references.append(
            PlanReference(
                name=name,
                state_required=_state(item.get("state_required")),
                must_include=bool(item.get("must_include", True)),
            )
        )
    forbidden = [str(x).strip() for x in (payload.get("forbidden") or []) if str(x).strip()]
    if not references and not forbidden and route == ROUTE_I2V_COMPOSED:
        # Claims to need references but named none => unusable, let the caller fall back.
        return None
    return IntentPlanV1(
        route=route,
        references=references,
        forbidden=forbidden,
        reason=str(payload.get("reason") or "").strip(),
    )


def resolve_plan(plan: IntentPlanV1, bank: AssetBank, name_to_ids) -> ResolvedPlan:
    """Lower a plan onto the bank.

    ``name_to_ids`` maps one name string to bank asset ids (the caller passes the shared
    CJK-aware name matcher, so plan names resolve exactly like prompt names do). Names that
    resolve to nothing are reported, never invented: a plan is not allowed to widen ``A_n``.
    Forbidden ids win over selected ids -- a negative constraint must not be overridable by a
    positive one, otherwise ``false_friend`` / ``deprecation_avoidance`` would silently leak.
    """
    forbidden_ids: list[str] = []
    for name in plan.forbidden:
        forbidden_ids.extend(name_to_ids(name))
    forbidden = list(dict.fromkeys(forbidden_ids))

    selected: list[str] = []
    state_by_id: dict[str, StateAngle] = {}
    unresolved: list[str] = []
    for ref in plan.references:
        ids = [aid for aid in name_to_ids(ref.name) if aid not in forbidden]
        if not ids:
            unresolved.append(ref.name)
            continue
        for aid in ids:
            if aid not in selected and bank.get_asset(aid) is not None:
                selected.append(aid)
                if ref.state_required is not None:
                    state_by_id[aid] = ref.state_required
    return ResolvedPlan(
        route=plan.route,
        selected_ids=selected,
        state_by_id=state_by_id,
        forbidden_ids=forbidden,
        unresolved_names=unresolved,
        reason=plan.reason,
    )


class MllmPlanProducer:
    """Adapter over the shared MLLM transport (mirrors ``MllmIntentResolver``)."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def make_plan(self, prompt: str, assets: list[dict[str, Any]]) -> dict[str, Any]:
        return dict(self._planner.make_intent_plan(prompt, assets) or {})
