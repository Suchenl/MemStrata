"""A failed planner call must not read as a deliberate "show nothing" plan.

Measured on the v8 run of ``0001_lighthouse_keeper``: segment 5 ("Elias looks out the window at
the darkening sea") shipped zero references even though replaying that beat against the bank it
had at the time returns ``Elias`` + ``lighthouse`` and resolves both. The planner call itself had
come back empty, and ``{}`` used to parse into a plan whose route *defaulted* to t2v — which the
read path then committed as an intentional empty context, skipping the model-free FAST match that
would have found both names.

The deliberate empty is still honoured when the planner actually states it, so a beat that names
an entity it must not show (spoken of, never seen) is not repopulated behind the plan's back.
"""

from __future__ import annotations

from memstrata.bank import Asset, AssetBank, AssetType, LifecycleStatus
from memstrata.skills.intent_understanding.interpreter import (
    INTENT_MODE_PLAN,
    IntentInterpreter,
)
from memstrata.skills.intent_understanding.plan import (
    ROUTE_I2V_COMPOSED,
    ROUTE_T2V,
    parse_plan,
    resolve_plan,
)

CHAR = AssetType.CHARACTER
LOC = AssetType.LOCATION


class _Plan:
    """A plan producer that returns one canned payload, like a frozen 9B response."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def make_plan(self, prompt: str, assets: list[dict]) -> dict:
        self.calls += 1
        return self.payload


def _bank() -> AssetBank:
    bank = AssetBank()
    bank.add_asset(Asset("Elias", CHAR, "Elias", LifecycleStatus.REUSABLE))
    bank.add_asset(Asset("rocky shore", LOC, "rocky shore", LifecycleStatus.REUSABLE))
    return bank


def test_an_empty_planner_response_falls_back_to_name_matching() -> None:
    bank = _bank()
    producer = _Plan({})
    interpreter = IntentInterpreter(bank, mode=INTENT_MODE_PLAN, plan_producer=producer)

    request, _ = interpreter.interpret("Elias looks out at the darkening sea.")

    assert producer.calls == 1
    assert [ref.asset_id for ref in request.references] == ["Elias"]
    assert request.used_mode == "fast"


def test_a_response_carrying_only_prose_is_also_a_failure() -> None:
    """A planner that explains itself but decides nothing has still decided nothing."""
    bank = _bank()
    producer = _Plan({"reason": "the shot continues from the previous one"})
    interpreter = IntentInterpreter(bank, mode=INTENT_MODE_PLAN, plan_producer=producer)

    request, _ = interpreter.interpret("Elias looks out at the darkening sea.")

    assert [ref.asset_id for ref in request.references] == ["Elias"]


def test_an_explicitly_stated_t2v_plan_keeps_its_empty_context() -> None:
    """The planner is allowed to say "show nothing" — silence is what we distrust."""
    bank = _bank()
    producer = _Plan({"route": "t2v", "references": [], "reason": "Elias is only spoken of"})
    interpreter = IntentInterpreter(bank, mode=INTENT_MODE_PLAN, plan_producer=producer)

    request, _ = interpreter.interpret("Elias is only spoken of, never shown.")

    assert [ref.asset_id for ref in request.references] == []
    assert request.intent_resolution_source == "plan"


def test_an_unresolvable_plan_still_falls_back() -> None:
    bank = _bank()
    producer = _Plan(
        {"route": "i2v_composed", "references": [{"name": "the stranger"}], "reason": ""}
    )
    interpreter = IntentInterpreter(bank, mode=INTENT_MODE_PLAN, plan_producer=producer)

    request, _ = interpreter.interpret("Elias meets the stranger on the rocky shore.")

    ids = [ref.asset_id for ref in request.references]
    assert "Elias" in ids and "rocky shore" in ids


def test_the_fallback_invents_nothing_for_a_beat_that_names_nothing_banked() -> None:
    bank = _bank()
    producer = _Plan({})
    interpreter = IntentInterpreter(bank, mode=INTENT_MODE_PLAN, plan_producer=producer)

    request, _ = interpreter.interpret("A wide empty sky above open water.")

    assert [ref.asset_id for ref in request.references] == []


def test_references_correct_a_contradicting_t2v_route() -> None:
    bank = _bank()
    plan = parse_plan(
        {
            "route": "t2v",
            "references": [{"name": "Elias", "must_include": True}],
            "reason": "",
        }
    )
    assert plan.route == ROUTE_T2V

    resolved = resolve_plan(plan, bank, lambda name: ["Elias"] if name == "Elias" else [])

    assert resolved.selected_ids == ["Elias"]
    assert resolved.route == ROUTE_I2V_COMPOSED


def test_an_empty_selection_leaves_the_t2v_route_alone() -> None:
    plan = parse_plan({"route": "t2v", "references": [], "reason": ""})
    resolved = resolve_plan(plan, _bank(), lambda name: [])
    assert resolved.route == ROUTE_T2V
