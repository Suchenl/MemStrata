"""The Track B hard-case families that had no deterministic guard until now.

Each group below fixes one way a measured 87-segment run lost an identity or answered with the
wrong evidence:

* ``plural``    — the same prop banked twice as `glass floats` (segment 13) and `glass float`
  (segment 22), so a prompt using either spelling reached only half the evidence.
* ``lookalike`` — χ merges on embedding distance alone, so two characters an encoder cannot
  separate collapse into one record and the read side can never tell them apart again, even
  though their names were distinct in every prompt.
* ``count``     — nothing in the bank recorded *how many*, so "the last two floats" was served
  the crop showing three.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    StateAngle,
)
from memstrata.bank.schema import surface_key
from memstrata.encoders import HashEmbedding
from memstrata.skills.composition.compose import select_reps
from memstrata.skills.decomposition import SOURCE_DISCOVERED, Observation
from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer
from memstrata.skills.intent_understanding.plan import parse_plan, resolve_plan
from memstrata.skills.memory_update.curator import MemoryPolicy, MemoryUpdater

_CHAR = AssetType.CHARACTER
_PROP = AssetType.PROP
_BETA = 0.50


class _OneRow:
    """Namer transport returning a single canned entity row."""

    def __init__(self, row: dict) -> None:
        self.row = row

    def run(self, role: str, *, instruction: str, images, schema):
        del role, images, schema
        if "naming auditor" in instruction:
            return {"resolved": []}
        return {"entities": [self.row]}


def _updater(bank: AssetBank, **over) -> MemoryUpdater:
    base = dict(name="hard", reconcile_threshold=_BETA, identity_shortcircuit_margin=0.15)
    base.update(over)
    return MemoryUpdater(bank, HashEmbedding(), policy=MemoryPolicy(**base))


def _bank(*named: tuple[str, AssetType]) -> AssetBank:
    bank = AssetBank()
    for name, kind in named:
        bank.add_asset(Asset(name, kind, name, LifecycleStatus.REUSABLE))
    return bank


# --------------------------------------------------------------------------- plural folding


def test_a_plural_spelling_keys_the_same_identity() -> None:
    assert surface_key("glass floats") == surface_key("glass float")


def test_only_the_head_noun_is_singularized() -> None:
    assert surface_key("Mrs Hobbs lantern") == "mrs hobbs lantern"


def test_words_that_merely_end_in_s_survive() -> None:
    for word in ("glass", "lens", "iris", "canvas", "atlas", "compass", "news"):
        assert surface_key(word) == word.lower(), word


def test_irregular_plurals_fold_where_the_rule_is_safe() -> None:
    assert surface_key("supply boxes") == "supply box"
    assert surface_key("oil dishes") == "oil dish"
    assert surface_key("lifebuoies") == "lifebuoy"


def test_cjk_names_are_untouched() -> None:
    assert surface_key("大兔子") == "大兔子"


def test_the_plural_spelling_resolves_to_the_existing_record() -> None:
    bank = _bank(("glass floats", _PROP))
    upd = _updater(bank)
    resolved = upd._resolve_asset(entity_id="glass float", name="glass float", kind=_PROP)
    assert len(bank.assets) == 1
    assert resolved is bank.assets["glass floats"]


# ------------------------------------------------------------------- lookalike disambiguation


def _sharp_crop(path: Path) -> str:
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)).save(path)
    return str(path)


def _with_rep(bank: AssetBank, name: str, crop: str) -> AssetBank:
    bank.assets[name].representations.append(
        AssetRepresentation(
            representation_id=f"{name}@s000", asset_id=name, object_uri=crop, origin_segment_id=0
        )
    )
    return bank


def _discovered(crop: str, name: str) -> Observation:
    return Observation(
        observation_id="o_disc",
        kind=_CHAR,
        name=name,
        image_path=crop,
        description="a figure in oilskins",
        source=SOURCE_DISCOVERED,
    )


def _merge_at(upd: MemoryUpdater, chi: float) -> None:
    upd.identity_score = lambda o, a, _c=chi: (_c, _c, _c)  # type: ignore[assignment]


def test_a_named_lookalike_is_not_absorbed_by_its_twin(tmp_path) -> None:
    """Two records exist and the crop names one of them: χ must not move it to the other."""
    crop = _sharp_crop(tmp_path / "c.png")
    bank = _with_rep(_bank(("Mara", _CHAR), ("Sara", _CHAR)), "Mara", crop)
    upd = _updater(bank)
    _merge_at(upd, 0.99)  # visually indistinguishable from Mara
    upd.curate_observations([_discovered(crop, "Sara")], segment_id=1)
    assert [r.asset_id for r in bank.assets["Sara"].representations] == ["Sara"]
    assert len(bank.assets["Mara"].representations) == 1


def test_an_unnamed_lookalike_still_merges_visually(tmp_path) -> None:
    """The rule only lets a *resolvable* name win; discovery by description is unchanged."""
    crop = _sharp_crop(tmp_path / "c.png")
    bank = _with_rep(_bank(("Mara", _CHAR)), "Mara", crop)
    upd = _updater(bank)
    _merge_at(upd, 0.99)
    upd.curate_observations([_discovered(crop, "a woman in oilskins")], segment_id=1)
    assert len(bank.assets) == 1


# ------------------------------------------------------------------------------ count memory


def _asset_with_counts(*counts: int) -> Asset:
    asset = Asset("glass float", _PROP, "glass float", LifecycleStatus.REUSABLE)
    for index, count in enumerate(counts):
        asset.representations.append(
            AssetRepresentation(
                representation_id=f"r{index}",
                asset_id="glass float",
                object_uri=f"/tmp/r{index}.png",
                origin_segment_id=index,
                count=count,
            )
        )
    return asset


def test_the_requested_count_outranks_recency() -> None:
    asset = _asset_with_counts(2, 3)  # the 3-float crop is the more recent one
    assert select_reps(asset, function="prop_reference", preferred_count=2) == ["r0"]


def test_no_requested_count_keeps_the_recency_default() -> None:
    asset = _asset_with_counts(2, 3)
    assert select_reps(asset, function="prop_reference") == ["r1"]


def test_an_unmatchable_count_still_returns_evidence() -> None:
    """A count is a preference, never a filter: a shot must not lose its reference over it."""
    asset = _asset_with_counts(2, 3)
    assert select_reps(asset, function="prop_reference", preferred_count=7) == ["r1"]


def test_count_outranks_state_when_both_are_requested() -> None:
    asset = _asset_with_counts(2, 3)
    asset.representations[1].state_angle = StateAngle.CHANGED
    chosen = select_reps(
        asset,
        function="prop_reference",
        preferred_count=2,
        preferred_state=StateAngle.CHANGED,
    )
    assert chosen == ["r0"]


def test_a_planned_count_reaches_the_resolved_plan() -> None:
    bank = _bank(("glass float", _PROP))
    plan = parse_plan({
        "route": "i2v_composed",
        "references": [
            {
                "name": "glass float",
                "state_required": "unknown",
                "count_required": 2,
                "must_include": True,
            }
        ],
        "forbidden": [],
        "reason": "two floats remain",
    })
    assert plan is not None
    resolved = resolve_plan(plan, bank, lambda name: ["glass float"])
    assert resolved.count_by_id == {"glass float": 2}


def test_a_count_of_one_is_treated_as_unstated() -> None:
    """A live 9B planner answers 1 for any single entity, so a 1 cannot mean "one survived"."""
    plan = parse_plan({
        "route": "i2v_composed",
        "references": [{"name": "Petrel", "state_required": "unknown", "count_required": 1}],
        "forbidden": [],
        "reason": "",
    })
    assert plan is not None
    assert plan.references[0].count_required is None


def test_a_namer_count_of_one_is_not_stored() -> None:
    namer = VlmEntityDecomposer(runner=_OneRow({
        "kind": "prop", "label": "brass lantern", "state_modifier": "", "count": 1,
        "category": "lantern", "description": "small brass lantern",
    }))
    entities = namer.propose(frames=["f0.png"], prompt="the lantern burns on the sill")
    assert [e.count for e in entities] == [0]


def test_a_namer_group_count_is_stored() -> None:
    namer = VlmEntityDecomposer(runner=_OneRow({
        "kind": "prop", "label": "glass floats", "state_modifier": "", "count": 3,
        "category": "glass float", "description": "green blown-glass floats",
    }))
    entities = namer.propose(frames=["f0.png"], prompt="three floats on the rail")
    assert [e.count for e in entities] == [3]


def test_an_absent_or_zero_count_is_reported_as_unstated() -> None:
    for raw in (0, -1, "", None, "many"):
        plan = parse_plan({
            "route": "i2v_composed",
            "references": [
                {"name": "x", "state_required": "unknown", "count_required": raw}
            ],
            "forbidden": [],
            "reason": "",
        })
        assert plan is not None
        assert plan.references[0].count_required is None, raw


def test_the_count_survives_a_bank_round_trip(tmp_path) -> None:
    bank = AssetBank()
    bank.add_asset(_asset_with_counts(3))
    path = tmp_path / "bank.json"
    bank.save(path)
    restored = AssetBank.load(path)
    assert restored.assets["glass float"].representations[0].count == 3
