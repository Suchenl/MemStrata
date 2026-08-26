"""Plan-driven read path (IntentPlanV1) contract tests.

Covers the three things a bare id list cannot express -- required appearance state, negative
constraints, and the generation route -- plus the fallback discipline that keeps enabling PLAN
from ever producing an empty read path, and the guard that a deliberate ``t2v`` plan is not
silently re-populated by the FAST path.
"""

from __future__ import annotations

from memstrata.bank import Asset, AssetBank, AssetType, LifecycleStatus, StateAngle
from memstrata.skills.intent_understanding.plan import (
    ROUTE_I2V_COMPOSED,
    ROUTE_T2V,
    IntentPlanV1,
    PlanReference,
    parse_plan,
    resolve_plan,
)
from memstrata.steps.intent import IntentInterpreter


class _Planner:
    """Stand-in for the MLLM plan producer."""

    def __init__(self, payload: dict | None = None, *, fail: bool = False) -> None:
        self.payload = payload or {}
        self.fail = fail
        self.calls = 0

    def make_plan(self, prompt: str, assets: list[dict]) -> dict:
        del prompt, assets
        self.calls += 1
        if self.fail:
            raise RuntimeError("planner unavailable")
        return dict(self.payload)


class _Resolver:
    def __init__(self, output: list[str] | None = None) -> None:
        self.output = output or []
        self.calls = 0

    def resolve(self, prompt: str, candidates: list[dict]) -> list[str]:
        del prompt, candidates
        self.calls += 1
        return list(self.output)


def _bank() -> AssetBank:
    bank = AssetBank()
    bank.add_asset(
        Asset(
            asset_id="char_ana",
            kind=AssetType.CHARACTER,
            name="Ana",
            status=LifecycleStatus.REUSABLE,
            description="lighthouse keeper in a yellow raincoat",
        )
    )
    bank.add_asset(
        Asset(
            asset_id="prop_lantern",
            kind=AssetType.PROP,
            name="brass lantern",
            status=LifecycleStatus.REUSABLE,
            description="small brass lantern with a glass pane",
        )
    )
    return bank


def _plan_payload(**over) -> dict:
    payload = {
        "route": ROUTE_I2V_COMPOSED,
        "references": [{"name": "Ana", "state_required": "changed", "must_include": True}],
        "forbidden": [],
        "reason": "Ana is soaked after the storm",
    }
    payload.update(over)
    return payload


# --- parse_plan ---------------------------------------------------------------------------


def test_parse_plan_normalises_state_and_keeps_route() -> None:
    plan = parse_plan(_plan_payload())
    assert plan is not None
    assert plan.route == ROUTE_I2V_COMPOSED
    assert [(r.name, r.state_required) for r in plan.references] == [("Ana", StateAngle.CHANGED)]


def test_parse_plan_treats_unknown_state_as_no_constraint() -> None:
    plan = parse_plan(
        _plan_payload(references=[{"name": "Ana", "state_required": "unknown", "must_include": True}])
    )
    assert plan is not None
    assert plan.references[0].state_required is None


def test_parse_plan_rejects_reference_route_with_no_references() -> None:
    # Claims conditioning but named nothing => unusable, so the caller falls back to FAST
    # rather than emitting a reference-conditioned beat with no references.
    assert parse_plan(_plan_payload(references=[])) is None


def test_parse_plan_accepts_t2v_with_no_references() -> None:
    plan = parse_plan(_plan_payload(route=ROUTE_T2V, references=[]))
    assert plan is not None
    assert plan.route == ROUTE_T2V


def test_parse_plan_rejects_non_dict() -> None:
    assert parse_plan(None) is None


# --- resolve_plan -------------------------------------------------------------------------


def test_resolve_plan_forbidden_wins_over_selected() -> None:
    """A negative constraint must not be overridable by a positive one.

    Otherwise false_friend / deprecation_avoidance would silently leak the very entity the
    plan asked to keep off screen.
    """
    bank = _bank()
    plan = IntentPlanV1(
        route=ROUTE_I2V_COMPOSED,
        references=[PlanReference(name="brass lantern")],
        forbidden=["brass lantern"],
    )
    resolved = resolve_plan(plan, bank, lambda name: ["prop_lantern"] if "lantern" in name else [])
    assert resolved.selected_ids == []
    assert resolved.forbidden_ids == ["prop_lantern"]


def test_resolve_plan_reports_unresolvable_names_instead_of_inventing_ids() -> None:
    bank = _bank()
    plan = IntentPlanV1(references=[PlanReference(name="a stranger never stored")])
    resolved = resolve_plan(plan, bank, lambda name: [])
    assert resolved.selected_ids == []
    assert resolved.unresolved_names == ["a stranger never stored"]


# --- interpreter, mode="plan" -------------------------------------------------------------


def test_plan_mode_sets_state_route_and_costs_one_call() -> None:
    planner = _Planner(_plan_payload())
    request, model_calls = IntentInterpreter(
        _bank(), plan_producer=planner, mode="plan"
    ).interpret("Ana steps out into the downpour", segment_id=3)
    assert [ref.asset_id for ref in request.references] == ["char_ana"]
    # The whole point: the plan pins WHICH appearance, not just which entity.
    assert request.references[0].preferred_state == StateAngle.CHANGED
    assert request.route == ROUTE_I2V_COMPOSED
    assert request.intent_resolution_source == "plan"
    assert model_calls == 1
    assert planner.calls == 1


def test_plan_mode_excludes_forbidden_entity_from_references() -> None:
    planner = _Planner(
        _plan_payload(
            references=[
                {"name": "Ana", "state_required": "default", "must_include": True},
                {"name": "brass lantern", "state_required": "default", "must_include": False},
            ],
            forbidden=["brass lantern"],
        )
    )
    request, _ = IntentInterpreter(
        _bank(), plan_producer=planner, mode="plan"
    ).interpret("Ana returns; the lantern she smashed is gone", segment_id=5)
    assert [ref.asset_id for ref in request.references] == ["char_ana"]
    assert "prop_lantern" in request.forbidden_asset_ids


def test_plan_mode_t2v_commits_an_empty_selection() -> None:
    """A deliberate t2v plan must survive: the FAST fallback must not re-populate it.

    ``Ana`` is a literal name hit in this prompt, so a missing commit guard would let FAST
    override the planner and condition a brand-new scene on stale references.
    """
    planner = _Planner(_plan_payload(route=ROUTE_T2V, references=[]))
    request, model_calls = IntentInterpreter(
        _bank(), plan_producer=planner, mode="plan"
    ).interpret("A wide establishing shot of a village Ana has never visited", segment_id=7)
    assert request.references == []
    assert request.route == ROUTE_T2V
    assert request.intent_resolution_source == "plan"
    assert model_calls == 1


def test_plan_mode_falls_back_to_fast_when_producer_missing() -> None:
    request, model_calls = IntentInterpreter(_bank(), mode="plan").interpret(
        "Ana walks the pier", segment_id=1
    )
    assert [ref.asset_id for ref in request.references] == ["char_ana"]
    assert request.requested_mode == "plan"
    assert request.used_mode == "fast"
    assert request.fallback_reason == "plan_producer_unavailable"
    assert model_calls == 0


def test_plan_mode_falls_back_to_fast_on_producer_error() -> None:
    planner = _Planner(fail=True)
    request, model_calls = IntentInterpreter(
        _bank(), plan_producer=planner, mode="plan"
    ).interpret("Ana walks the pier", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_ana"]
    assert request.used_mode == "fast"
    assert request.fallback_reason == "plan_error:RuntimeError"
    assert model_calls == 1


def test_plan_mode_falls_back_to_fast_when_plan_names_are_unresolvable() -> None:
    planner = _Planner(
        _plan_payload(
            references=[{"name": "someone not stored", "state_required": "default", "must_include": True}]
        )
    )
    request, _ = IntentInterpreter(
        _bank(), plan_producer=planner, mode="plan"
    ).interpret("Ana walks the pier", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_ana"]
    assert request.used_mode == "fast"
    assert request.fallback_reason == "plan_unresolved"


# --- back-compat --------------------------------------------------------------------------


def test_pipeline_defaults_to_fast_and_env_flag_selects_plan(monkeypatch) -> None:
    """The operational contract: plan mode is opt-in via MEMSTRATA_INTENT_MODE.

    Track A budgets for a model-free read path, so an unset flag must stay FAST.
    """
    from memstrata.pipeline import MemStrata

    monkeypatch.delenv("MEMSTRATA_INTENT_MODE", raising=False)
    assert MemStrata(_bank()).interpreter.mode == "fast"

    monkeypatch.setenv("MEMSTRATA_INTENT_MODE", "plan")
    mem = MemStrata(_bank())
    assert mem.interpreter.mode == "plan"
    assert mem.interpreter.plan_producer is not None

    # An explicit argument still wins over the environment.
    assert MemStrata(_bank(), intent_mode="fast").interpreter.mode == "fast"


def test_fast_and_slow_paths_carry_no_plan_fields() -> None:
    """Enabling the plan layer must not perturb the model-free path Track A budgets for."""
    planner = _Planner(_plan_payload())
    fast, fast_calls = IntentInterpreter(
        _bank(), plan_producer=planner, mode="fast"
    ).interpret("Ana walks the pier", segment_id=1)
    assert [ref.asset_id for ref in fast.references] == ["char_ana"]
    assert fast.route == ""
    assert fast.forbidden_asset_ids == ()
    assert fast.plan_unresolved_names == ()
    assert fast_calls == 0
    assert planner.calls == 0  # a plan producer is never consulted outside mode="plan"

    slow, slow_calls = IntentInterpreter(
        _bank(), resolver=_Resolver(["char_ana"]), plan_producer=planner, mode="slow"
    ).interpret("she walks the pier", segment_id=1)
    assert [ref.asset_id for ref in slow.references] == ["char_ana"]
    assert slow.route == ""
    assert slow_calls == 1
    assert planner.calls == 0
