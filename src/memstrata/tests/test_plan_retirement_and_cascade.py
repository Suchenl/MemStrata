"""Two remaining hard-case families that a per-beat plan alone cannot cover.

``deprecation_avoidance`` / ``false_friend``: a plan's ``forbidden`` list expires with its beat,
but a lantern that shatters is gone for the rest of the film — and a later prompt that simply
does not mention it gives the planner nothing to re-derive that from. The plan therefore reports
permanent removals separately, and the pipeline turns those into a bank lifecycle transition that
every later beat inherits through ``compose``'s intrinsic usability gate.

``reference_indirect`` / ``temporal_reference``: a beat can name an entity in words no surface
match can reach ("the boat we left at the pier" against a record named Petrel). The fast→slow
cascade is what keeps that beat from composing nothing, so production must actually have it wired
— it was defaulted off, which made the whole slow path dead code in every real run.
"""

from __future__ import annotations

from memstrata.bank import Asset, AssetBank, AssetType, LifecycleStatus
from memstrata.pipeline import MemStrata
from memstrata.skills.intent_understanding.plan import ROUTE_I2V_COMPOSED, parse_plan, resolve_plan
from memstrata.steps.intent import IntentInterpreter


class _Planner:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def make_plan(self, prompt: str, assets: list[dict]) -> dict:
        del prompt, assets
        self.calls += 1
        return dict(self.payload)


class _Resolver:
    def __init__(self, output: list[str]) -> None:
        self.output = output
        self.calls = 0

    def resolve(self, prompt: str, candidates: list[dict]) -> list[str]:
        del prompt, candidates
        self.calls += 1
        return list(self.output)


def _bank() -> AssetBank:
    bank = AssetBank()
    bank.add_asset(
        Asset("prop_lantern", AssetType.PROP, "brass lantern", LifecycleStatus.REUSABLE,
              description="small brass lantern with a glass pane")
    )
    bank.add_asset(
        Asset("boat_petrel", AssetType.PROP, "Petrel", LifecycleStatus.REUSABLE,
              description="an indigo fishing boat")
    )
    bank.add_asset(
        Asset("prop_float", AssetType.PROP, "glass float", LifecycleStatus.REUSABLE,
              description="green blown-glass fishing floats")
    )
    return bank


def _payload(**over) -> dict:
    payload = {
        "route": ROUTE_I2V_COMPOSED,
        "references": [{"name": "Petrel", "state_required": "unknown", "must_include": True}],
        "forbidden": [],
        "retired": [],
        "reason": "",
    }
    payload.update(over)
    return payload


# ------------------------------------------------------------------- permanent retirement


_IDS = {"brass lantern": ["prop_lantern"], "Petrel": ["boat_petrel"], "glass float": ["prop_float"]}


def test_retirement_does_not_ban_the_entity_in_its_own_beat() -> None:
    """The shot that shatters the lantern is the shot that must show the lantern."""
    plan = parse_plan(
        _payload(
            references=[{"name": "brass lantern", "state_required": "damaged", "must_include": True}],
            retired=["brass lantern"],
        )
    )
    assert plan is not None
    resolved = resolve_plan(plan, _bank(), lambda name: _IDS.get(name, []))
    assert resolved.selected_ids == ["prop_lantern"]
    assert resolved.forbidden_ids == []
    assert resolved.retired_ids == ["prop_lantern"]


def test_a_plain_ban_still_wins_over_a_reference() -> None:
    """Without a retirement the contradiction is about presence, and the ban stays authoritative."""
    plan = parse_plan(
        _payload(
            references=[{"name": "Petrel", "state_required": "unknown", "must_include": True}],
            forbidden=["Petrel"],
        )
    )
    assert plan is not None
    resolved = resolve_plan(plan, _bank(), lambda name: _IDS.get(name, []))
    assert resolved.selected_ids == []
    assert resolved.forbidden_ids == ["boat_petrel"]


def test_a_surviving_count_blocks_retirement() -> None:
    """One of three floats smashing must not delete the record holding the other two."""
    plan = parse_plan(
        _payload(
            references=[
                {"name": "glass float", "state_required": "unknown", "count_required": 2,
                 "must_include": True}
            ],
            forbidden=["glass float"],
            retired=["glass float"],
        )
    )
    assert plan is not None
    resolved = resolve_plan(plan, _bank(), lambda name: _IDS.get(name, []))
    assert resolved.selected_ids == ["prop_float"]
    assert resolved.retired_ids == []
    assert resolved.count_by_id == {"prop_float": 2}
    assert resolved.self_conflicts == ["prop_float"]


def test_retirement_resolves_to_ids_separately_from_the_beat_ban() -> None:
    plan = parse_plan(
        _payload(
            references=[{"name": "Mara", "state_required": "unknown", "must_include": True}],
            forbidden=["Petrel"],
            retired=["brass lantern"],
        )
    )
    assert plan is not None
    resolved = resolve_plan(plan, _bank(), lambda name: _IDS.get(name, []))
    assert resolved.retired_ids == ["prop_lantern"]
    assert resolved.forbidden_ids == ["boat_petrel"]


def test_a_merely_forbidden_entity_is_not_retired() -> None:
    """A look-alike ruled out for one shot must stay available for every later shot."""
    plan = parse_plan(_payload(forbidden=["brass lantern"]))
    assert plan is not None
    resolved = resolve_plan(plan, _bank(), lambda name: ["prop_lantern"] if "lantern" in name else [])
    assert resolved.forbidden_ids == ["prop_lantern"]
    assert resolved.retired_ids == []


def test_the_pipeline_deprecates_a_retired_asset() -> None:
    bank = _bank()
    mem = MemStrata(
        bank=bank,
        intent_mode="plan",
        plan_producer=_Planner(_payload(retired=["brass lantern"])),
    )
    mem.step1_compose("the lantern shatters on the rocks", segment_id=3)
    assert bank.assets["prop_lantern"].status is LifecycleStatus.DEPRECATED


def test_a_retired_asset_stays_out_of_later_beats_without_being_re_forbidden() -> None:
    """The whole point: beat 4 never mentions the lantern, and it still cannot come back."""
    bank = _bank()
    planner = _Planner(_payload(retired=["brass lantern"]))
    mem = MemStrata(bank=bank, intent_mode="plan", plan_producer=planner)
    mem.step1_compose("the lantern shatters on the rocks", segment_id=3)

    planner.payload = _payload(
        references=[
            {"name": "brass lantern", "state_required": "unknown", "must_include": True},
            {"name": "Petrel", "state_required": "unknown", "must_include": True},
        ]
    )
    _, context, _ = mem.step1_compose("she looks for something to light the way", segment_id=4)
    assert "prop_lantern" not in context.asset_ids


def test_a_beat_with_no_retirement_leaves_the_bank_alone() -> None:
    bank = _bank()
    mem = MemStrata(bank=bank, intent_mode="plan", plan_producer=_Planner(_payload()))
    mem.step1_compose("the Petrel rocks at anchor", segment_id=1)
    assert bank.assets["prop_lantern"].status is LifecycleStatus.REUSABLE


# ------------------------------------------------------------------------ fast->slow cascade


def _indirect_interpreter(bank: AssetBank, resolver: _Resolver, **over) -> IntentInterpreter:
    return IntentInterpreter(bank, resolver=resolver, **over)


def test_an_indirect_reference_reaches_the_slow_resolver() -> None:
    resolver = _Resolver(["boat_petrel"])
    interpreter = _indirect_interpreter(_bank(), resolver, slow_on_miss=True)
    request, _ = interpreter.interpret("the vessel they left at the pier", segment_id=2)
    assert resolver.calls == 1
    assert [ref.asset_id for ref in request.references] == ["boat_petrel"]


def test_a_direct_name_never_pays_for_the_slow_resolver() -> None:
    resolver = _Resolver(["boat_petrel"])
    interpreter = _indirect_interpreter(_bank(), resolver, slow_on_miss=True)
    interpreter.interpret("the Petrel rocks at anchor", segment_id=2)
    assert resolver.calls == 0


def test_production_wires_the_cascade_on_by_default() -> None:
    """It was off, so the slow path never ran in a real run despite being implemented."""
    mem = MemStrata(bank=_bank(), resolver=_Resolver([]))
    assert mem.interpreter.slow_on_miss is True
