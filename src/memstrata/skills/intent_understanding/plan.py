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
    count_required: int | None = None
    must_include: bool = True


@dataclass(slots=True)
class IntentPlanV1:
    """Typed read-side plan for one beat."""

    route: str = ROUTE_I2V_COMPOSED
    references: list[PlanReference] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    # Entities this beat removes from the story for good. ``forbidden`` is per-beat; this is the
    # subset that must also hold in every later beat, so it is applied to the bank's lifecycle
    # instead of just this beat's selection.
    retired: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class ResolvedPlan:
    """``IntentPlanV1`` lowered onto the bank: names became ids, or were reported unresolved."""

    route: str
    selected_ids: list[str] = field(default_factory=list)
    state_by_id: dict[str, StateAngle] = field(default_factory=dict)
    count_by_id: dict[str, int] = field(default_factory=dict)
    forbidden_ids: list[str] = field(default_factory=list)
    retired_ids: list[str] = field(default_factory=list)
    # Ids the plan both asked for and ruled out; resolved in favour of the reference and reported
    # here so a planner that keeps contradicting itself is visible rather than silently patched.
    self_conflicts: list[str] = field(default_factory=list)
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
                    "count_required": {"type": "integer"},
                    "must_include": {"type": "boolean"},
                },
                "required": ["name", "state_required", "count_required", "must_include"],
                "additionalProperties": False,
            },
        },
        "forbidden": {"type": "array", "items": {"type": "string"}},
        "retired": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["route", "references", "forbidden", "retired", "reason"],
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
    - "count_required": how many of this entity the shot shows, but ONLY when TWO OR MORE are
      visible together and the story states or implies the number ("the three floats", "the last
      two"). Track the running count across shots: if an earlier shot established three and this
      shot removes one, ask for 2. Use 0 for a single individual and whenever the shot gives no
      number — 0 is the normal answer, so do not fill in 1.
    - "must_include": true if the shot is about this entity, false if it is incidental.
  Only list entities actually on screen in THIS shot. Do not list an entity merely mentioned
  in dialogue or memory. Be COMPLETE: if the shot prompt names a stored entity and that entity
  is on screen, it must appear here. Sweep the prompt for the characters acting, the place they
  are in, and the objects they touch or look at — an on-screen entity you leave out will be
  missing from the shot's reference images entirely.
- "forbidden": entities that must NOT appear in THIS shot: destroyed, removed, discarded, left
  behind, or a look-alike explicitly distinguished from the one in this shot. State absence here
  rather than implying it by silence: if the shot NAMES a stored entity that must stay off screen
  (spoken of but not shown, a place the character has never been), list it in "forbidden".
- "retired": entities that leave the story PERMANENTLY as of this shot — destroyed, burnt, sunk,
  shattered, eaten, dead, thrown overboard. This is about every LATER shot, not this one: an
  entity may be visible here (it is being destroyed on screen) and still belong in "retired", so
  list it in "references" too if it is visible. A merely absent, hidden, set-aside or left-behind
  entity is NOT retired. If only some of a group is destroyed and others remain, do NOT retire it
  — give the surviving number in "count_required" instead. Once retired it will never be offered
  again, so do not guess.
- "route": "i2v_composed" if any stored entity must be visible (reference images will be
  composed); "i2v_continue" if this shot directly continues the previous shot with no new
  identity requirement; "t2v" if no stored entity is needed at all (a brand-new scene).
- "reason": one short sentence.
"""


def _count(raw: Any) -> int | None:
    """Normalise a planner count; anything below two means "no count constraint".

    Absent, zero, negative and non-numeric are all "not stated". So is *one*: asked for a count,
    the planner reliably answers 1 for any single individual (measured on a live 9B planner across
    every beat, including ones whose prompt states no number at all), so a 1 cannot be told apart
    from that default. Since one instance also carries no group information, dropping it costs
    nothing and stops a noisy 1 from being read as "one of these survived".
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 2 else None


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
    stated = route in ROUTES
    decided = payload.get("references") or payload.get("forbidden") or payload.get("retired")
    if not (stated or decided):
        # A planner that stated *nothing* did not decide anything: a dropped call, a timeout or
        # unparseable JSON all arrive here as ``{}``. Without this the route defaulted to t2v
        # below and the read path committed that as a *deliberate* empty context, so one failed
        # call silently cost a segment all of its references (measured: segment 5 of the v8
        # lighthouse run, whose named entities were both in the bank and resolvable). Refusing
        # the payload sends the caller to the model-free FAST path instead.
        return None
    if not stated:
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
                count_required=_count(item.get("count_required")),
                must_include=bool(item.get("must_include", True)),
            )
        )
    forbidden = [str(x).strip() for x in (payload.get("forbidden") or []) if str(x).strip()]
    retired = [str(x).strip() for x in (payload.get("retired") or []) if str(x).strip()]
    if not references and not forbidden and route == ROUTE_I2V_COMPOSED:
        # Claims to need references but named none => unusable, let the caller fall back.
        return None
    return IntentPlanV1(
        route=route,
        references=references,
        forbidden=forbidden,
        retired=retired,
        reason=str(payload.get("reason") or "").strip(),
    )


def resolve_plan(plan: IntentPlanV1, bank: AssetBank, name_to_ids) -> ResolvedPlan:
    """Lower a plan onto the bank.

    ``name_to_ids`` maps one name string to bank asset ids (the caller passes the shared
    CJK-aware name matcher, so plan names resolve exactly like prompt names do). Names that
    resolve to nothing are reported, never invented: a plan is not allowed to widen ``A_n``.
    Forbidden ids win over selected ids -- a negative constraint must not be overridable by a
    positive one, otherwise ``false_friend`` / ``deprecation_avoidance`` would silently leak. The
    single exception is an entity the same beat also retires, where the ban describes the entity's
    future rather than this shot; see the comment on that branch.
    """
    referenced_ids: dict[str, int | None] = {}
    for ref in plan.references:
        for aid in name_to_ids(ref.name):
            referenced_ids.setdefault(aid, ref.count_required)

    retiring_ids: dict[str, None] = {}  # ordered set: retirement order must be reproducible
    for name in plan.retired:
        for aid in name_to_ids(name):
            retiring_ids.setdefault(aid, None)

    self_conflicts: list[str] = []
    forbidden_ids: list[str] = []
    for name in plan.forbidden:
        for aid in name_to_ids(name):
            # A ban normally wins over a reference (see the docstring). The exception is a beat
            # that also *retires* the entity: there the planner is describing a destruction it can
            # see, and it bans the entity because it will be gone — measured on the beats where a
            # lantern shatters and where one of three floats smashes. Honouring the ban there
            # strips the reference from the one shot that shows the event, so on this narrow
            # overlap the reference wins and the ban is deferred to the retirement below.
            if aid in referenced_ids and aid in retiring_ids:
                if aid not in self_conflicts:
                    self_conflicts.append(aid)
                continue
            forbidden_ids.append(aid)
    forbidden = list(dict.fromkeys(forbidden_ids))

    retired_ids: list[str] = []
    for aid in retiring_ids:
        # Still wanted in some number => some instances survive, so the record does not die.
        # Being visible while it is destroyed does not save it: only a surviving count does.
        if referenced_ids.get(aid):
            if aid not in self_conflicts:
                self_conflicts.append(aid)
            continue
        if aid not in retired_ids:
            retired_ids.append(aid)

    selected: list[str] = []
    state_by_id: dict[str, StateAngle] = {}
    count_by_id: dict[str, int] = {}
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
                if ref.count_required is not None:
                    count_by_id[aid] = ref.count_required
    # The planner sometimes lists must-include references and still stamps the beat ``t2v``
    # (observed on "Mara crouches ... studying the anemones", where all three names resolved).
    # The references are the concrete claim and the route is a label, so let the references
    # correct it rather than shipping a record that contradicts itself.
    route = ROUTE_I2V_COMPOSED if (selected and plan.route == ROUTE_T2V) else plan.route
    return ResolvedPlan(
        route=route,
        selected_ids=selected,
        state_by_id=state_by_id,
        count_by_id=count_by_id,
        forbidden_ids=forbidden,
        retired_ids=retired_ids,
        self_conflicts=self_conflicts,
        unresolved_names=unresolved,
        reason=plan.reason,
    )


class MllmPlanProducer:
    """Adapter over the shared MLLM transport (mirrors ``MllmIntentResolver``)."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def make_plan(self, prompt: str, assets: list[dict[str, Any]]) -> dict[str, Any]:
        return dict(self._planner.make_intent_plan(prompt, assets) or {})
