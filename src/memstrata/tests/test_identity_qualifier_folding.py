"""One identity must not split just because a shot drops or adds a qualifier.

A measured 20-segment English run banked Mara twice — `young Mara` (segments 11, 17) and
`Mara` (12-16) — and the boat twice as `indigo Petrel` and `Petrel`. Segment 12 read empty
as a direct consequence: its prompt says "Mara jumps down from the Petrel" while the bank
held only the qualified spellings, so surface matching found nothing and the write side
opened a second record for each identity.

Also covers the alias-promotion gate: a χ merge that only just cleared β_τ must not turn
the incoming name into a permanent retrieval alias. The unguarded writeback made 7 of the
8 aliases in that same run wrong (`Lena` answering to "old man", `ship's log` to "pen").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from memstrata.bank import Asset, AssetBank, AssetRepresentation, AssetType, LifecycleStatus
from memstrata.encoders import HashEmbedding
from memstrata.skills.decomposition import SOURCE_DISCOVERED, Observation
from memstrata.skills.intent_understanding import IntentInterpreter
from memstrata.skills.memory_update.curator import MemoryPolicy, MemoryUpdater

_CHAR = AssetType.CHARACTER
_PROP = AssetType.PROP
_BETA = 0.50
_SHORT = 0.15


def _updater(bank: AssetBank, **over) -> MemoryUpdater:
    base = dict(name="fold", reconcile_threshold=_BETA, identity_shortcircuit_margin=_SHORT)
    base.update(over)
    return MemoryUpdater(bank, HashEmbedding(), policy=MemoryPolicy(**base))


def _bank(*named: tuple[str, AssetType]) -> AssetBank:
    bank = AssetBank()
    for name, kind in named:
        bank.add_asset(Asset(name, kind, name, LifecycleStatus.REUSABLE))
    return bank


def _resolve(upd: MemoryUpdater, name: str, kind: AssetType = _CHAR) -> Asset:
    """The write-side namer shape: the prompt's own wording is both id and name."""
    return upd._resolve_asset(entity_id=name, name=name, kind=kind)


def _aliases(asset: Asset) -> list[str]:
    return list(asset.metadata.get("aliases") or [])


def test_the_bare_head_name_folds_onto_a_qualified_asset() -> None:
    bank = _bank(("young Mara", _CHAR))
    resolved = _resolve(_updater(bank), "Mara")
    assert len(bank.assets) == 1
    assert resolved is bank.assets["young Mara"]


def test_the_unqualified_form_becomes_the_identity() -> None:
    bank = _bank(("young Mara", _CHAR))
    resolved = _resolve(_updater(bank), "Mara")
    assert resolved.name == "Mara"
    assert "young Mara" in _aliases(resolved)


def test_a_qualifier_folds_onto_the_bare_head_asset() -> None:
    bank = _bank(("Mara", _CHAR))
    resolved = _resolve(_updater(bank), "young Mara")
    assert len(bank.assets) == 1
    assert resolved.name == "Mara"
    assert "young Mara" in _aliases(resolved)


def test_the_boat_from_that_run_stops_splitting() -> None:
    bank = _bank(("indigo Petrel", _PROP))
    resolved = _resolve(_updater(bank), "Petrel", _PROP)
    assert len(bank.assets) == 1
    assert resolved.name == "Petrel"


def test_both_spellings_retrieve_one_identity_after_folding() -> None:
    bank = _bank(("young Mara", _CHAR))
    _resolve(_updater(bank), "Mara")
    interpreter = IntentInterpreter(bank)
    bare, _ = interpreter.interpret("Mara ties the hawser", segment_id=12)
    qualified, _ = interpreter.interpret("young Mara ties the hawser", segment_id=12)
    bare_ids = [ref.asset_id for ref in bare.references]
    qualified_ids = [ref.asset_id for ref in qualified.references]
    assert bare_ids == qualified_ids != []


def test_an_ambiguous_head_noun_is_not_folded() -> None:
    """Two props sharing a head noun make a bare mention genuinely ambiguous."""
    bank = _bank(("red door", _PROP), ("blue door", _PROP))
    resolved = _resolve(_updater(bank), "door", _PROP)
    assert len(bank.assets) == 3
    assert resolved.name == "door"
    assert _aliases(bank.assets["red door"]) == []


def test_folding_stays_within_a_kind() -> None:
    bank = _bank(("indigo Petrel", _PROP))
    resolved = _resolve(_updater(bank), "Petrel", _CHAR)
    assert len(bank.assets) == 2
    assert bank.assets["indigo Petrel"].name == "indigo Petrel"
    assert resolved.kind is _CHAR


def test_unrelated_names_do_not_fold() -> None:
    bank = _bank(("Mara", _CHAR))
    _resolve(_updater(bank), "Lena")
    assert len(bank.assets) == 2


def test_a_shared_leading_word_is_not_a_qualifier_match() -> None:
    """Qualifiers sit in front of the head, so only a shared *tail* means one identity."""
    bank = _bank(("brass railing", _PROP))
    _resolve(_updater(bank), "brass lantern", _PROP)
    assert len(bank.assets) == 2


def _sharp_crop(path: Path) -> str:
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)).save(path)
    return str(path)


def _bank_with_ref(crop: str) -> AssetBank:
    bank = AssetBank()
    asset = Asset("Lena", _CHAR, "Lena", LifecycleStatus.REUSABLE)
    asset.representations.append(
        AssetRepresentation(
            representation_id="Lena@s000", asset_id="Lena", object_uri=crop, origin_segment_id=0
        )
    )
    bank.add_asset(asset)
    return bank


def _discovered(crop: str, name: str) -> Observation:
    return Observation(
        observation_id="o_disc",
        kind=_CHAR,
        name=name,
        image_path=crop,
        description="a figure on the quay",
        source=SOURCE_DISCOVERED,
    )


def _merge_at(upd: MemoryUpdater, chi: float) -> None:
    upd.identity_score = lambda o, a, _c=chi: (_c, _c, _c)  # type: ignore[assignment]


def test_a_barely_merged_name_does_not_become_an_alias(tmp_path) -> None:
    crop = _sharp_crop(tmp_path / "c.png")
    bank = _bank_with_ref(crop)
    upd = _updater(bank)
    _merge_at(upd, _BETA + 0.01)
    upd.curate_observations([_discovered(crop, "old man")], segment_id=1)
    assert _aliases(bank.assets["Lena"]) == []


def test_a_confidently_merged_name_becomes_an_alias(tmp_path) -> None:
    crop = _sharp_crop(tmp_path / "c.png")
    bank = _bank_with_ref(crop)
    upd = _updater(bank)
    _merge_at(upd, _BETA + _SHORT + 0.01)
    upd.curate_observations([_discovered(crop, "the twin sister")], segment_id=1)
    assert _aliases(bank.assets["Lena"]) == ["the twin sister"]


def test_a_barely_merged_observation_still_joins_the_asset(tmp_path) -> None:
    """The gate withholds the *name*, not the evidence: the rep still merges."""
    crop = _sharp_crop(tmp_path / "c.png")
    bank = _bank_with_ref(crop)
    upd = _updater(bank)
    _merge_at(upd, _BETA + 0.01)
    upd.curate_observations([_discovered(crop, "old man")], segment_id=1)
    assert len(bank.assets) == 1
