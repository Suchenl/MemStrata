"""The read path must not go empty on a name that is still matchable in the bank.

A measured 18-segment English run had three segments whose composed context came back
empty while the named entity sat in the bank as a usable asset with causally valid reps
(segment 11 named ``Elias`` at 6 reps, segment 14 named ``Mara`` at 2). The trigger was
never reproduced from that run's artifacts, so these tests pin the *failure mode* rather
than one trigger: whenever the primary path returns nothing, a deterministic rematch has
to recover it, and the legitimate empties must stay empty.

A stale term index is used as the stand-in trigger because it is the one mechanism that
can be induced deterministically: the shared snapshot is keyed on ``bank.version``, so a
mutation that does not advance the version leaves the read path matching against an index
that predates the asset.
"""

from __future__ import annotations

from memstrata.bank import Asset, AssetBank, AssetType, LifecycleStatus
from memstrata.skills.intent_understanding import INTENT_MODE_PLAN
from memstrata.steps.intent import IntentInterpreter


def _asset(
    asset_id: str,
    name: str,
    kind: AssetType = AssetType.CHARACTER,
    description: str = "",
) -> Asset:
    # Descriptions deliberately do not echo the name: description-overlap matching is a
    # legitimate second path, and if it can answer these prompts the name path is never the
    # one under test.
    return Asset(
        asset_id=asset_id,
        kind=kind,
        name=name,
        status=LifecycleStatus.REUSABLE,
        description=description or "a weathered figure in oilskins",
    )


def _bank_with_stale_index() -> AssetBank:
    """A bank whose cached term index predates ``Mara``.

    ``interpret`` populates the shared index at the bank's current version; inserting
    straight into ``assets`` afterwards skips ``add_asset``'s ``touch()``, so the version
    still matches and the stale index is served on the next read.
    """
    bank = AssetBank()
    bank.add_asset(_asset("Elias", "Elias"))
    IntentInterpreter(bank).interpret("Elias climbs the stairs", segment_id=1)
    bank.assets["Mara"] = _asset("Mara", "Mara", description="a lean fisher girl")
    return bank


def test_a_name_missed_by_a_stale_index_is_still_recovered() -> None:
    bank = _bank_with_stale_index()
    request, _ = IntentInterpreter(bank).interpret(
        "Mara crouches by a tidal pool", segment_id=14
    )
    assert [ref.asset_id for ref in request.references] == ["Mara"]


def test_the_recovery_is_labelled_so_it_cannot_pass_as_a_normal_hit() -> None:
    bank = _bank_with_stale_index()
    request, _ = IntentInterpreter(bank).interpret(
        "Mara crouches by a tidal pool", segment_id=14
    )
    assert request.intent_resolution_source == "name_recovered"


def test_the_recovery_costs_no_model_call() -> None:
    bank = _bank_with_stale_index()
    _, model_calls = IntentInterpreter(bank).interpret(
        "Mara crouches by a tidal pool", segment_id=14
    )
    assert model_calls == 0


def test_a_healthy_read_is_untouched_and_still_labelled_name() -> None:
    bank = AssetBank()
    bank.add_asset(_asset("Elias", "Elias"))
    request, _ = IntentInterpreter(bank).interpret("Elias lights the lamp", segment_id=2)
    assert [ref.asset_id for ref in request.references] == ["Elias"]
    assert request.intent_resolution_source == "name"


def test_a_prompt_naming_nobody_stays_a_genuine_miss() -> None:
    bank = AssetBank()
    bank.add_asset(_asset("Elias", "Elias"))
    request, _ = IntentInterpreter(bank).interpret("waves break on the rocks", segment_id=3)
    assert request.references == []
    assert request.intent_resolution_source == "miss"


class _EmptyT2VPlan:
    """A committed plan that deliberately wants no references (cold open, pure t2v)."""

    def make_plan(self, prompt: str, candidates: list[dict]) -> dict:
        del prompt, candidates
        return {"references": [], "forbidden": [], "route": "t2v"}


def test_a_committed_t2v_plan_keeps_its_deliberate_empty() -> None:
    bank = AssetBank()
    bank.add_asset(_asset("Elias", "Elias"))
    request, _ = IntentInterpreter(
        bank, plan_producer=_EmptyT2VPlan(), mode=INTENT_MODE_PLAN
    ).interpret("Elias is only spoken of, never shown", segment_id=4)
    assert request.references == []
    assert request.route == "t2v"


class _ForbidPlan:
    def make_plan(self, prompt: str, candidates: list[dict]) -> dict:
        del prompt, candidates
        return {"references": [], "forbidden": ["Elias"], "route": ""}


def test_an_entity_the_plan_forbids_is_not_recovered_behind_the_plan() -> None:
    bank = AssetBank()
    bank.add_asset(_asset("Elias", "Elias"))
    request, _ = IntentInterpreter(
        bank, plan_producer=_ForbidPlan(), mode=INTENT_MODE_PLAN
    ).interpret("a memorial plaque engraved with Elias", segment_id=5)
    assert [ref.asset_id for ref in request.references] == []


def test_the_name_anchor_ablation_is_not_repaired_back_into_name_matching() -> None:
    bank = AssetBank()
    bank.add_asset(_asset("Elias", "Elias"))
    request, _ = IntentInterpreter(bank, disable_name_anchor=True).interpret(
        "Elias lights the lamp", segment_id=6
    )
    assert request.intent_resolution_source == "recency"
