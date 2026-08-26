"""Head-noun folding must also cover the write-side namer's own discoveries.

The namer leaves ``entity_id`` unset, so its entities take the discovered branch and χ decides
identity. χ cannot merge a wide shot of a room with a close-up of the same room, so a bare "room"
opened a second record next to "lantern room" (and "shore" next to "rocky shore") in the measured
v8 bank, even though the named path already folds exactly this pair.
"""

from __future__ import annotations

from memstrata.bank import Asset, AssetBank, AssetType, LifecycleStatus
from memstrata.encoders import HashEmbedding
from memstrata.skills.decomposition.decomposer import SOURCE_DISCOVERED, Observation
from memstrata.skills.memory_update.curator import MemoryPolicy, MemoryUpdater

LOC = AssetType.LOCATION
PROP = AssetType.PROP


def _updater(bank: AssetBank) -> MemoryUpdater:
    return MemoryUpdater(bank, HashEmbedding(), policy=MemoryPolicy(name="test"))


def _obs(name: str, kind: AssetType) -> Observation:
    return Observation(
        observation_id=f"{kind.value}_{name}_0_0",
        kind=kind,
        name=name,
        image_path="",
        entity_id=None,
        source=SOURCE_DISCOVERED,
    )


def test_a_bare_head_noun_folds_onto_the_qualified_record() -> None:
    bank = AssetBank()
    bank.add_asset(Asset("lantern room", LOC, "lantern room", LifecycleStatus.REUSABLE))

    asset = _updater(bank)._fold_or_create_discovered(_obs("room", LOC))

    assert len(bank.assets) == 1
    assert asset.asset_id == "lantern room"
    assert asset.name == "room"  # the unqualified spelling becomes canonical
    assert bank.find_by_name("lantern room", kind=LOC) is asset


def test_a_qualifier_folds_onto_the_bare_record() -> None:
    bank = AssetBank()
    bank.add_asset(Asset("shore", LOC, "shore", LifecycleStatus.REUSABLE))

    asset = _updater(bank)._fold_or_create_discovered(_obs("rocky shore", LOC))

    assert len(bank.assets) == 1
    assert bank.find_by_name("rocky shore", kind=LOC) is asset


def test_an_ambiguous_head_noun_still_opens_its_own_record() -> None:
    bank = AssetBank()
    bank.add_asset(Asset("red door", LOC, "red door", LifecycleStatus.REUSABLE))
    bank.add_asset(Asset("blue door", LOC, "blue door", LifecycleStatus.REUSABLE))

    asset = _updater(bank)._fold_or_create_discovered(_obs("door", LOC))

    assert len(bank.assets) == 3
    assert asset.asset_id not in {"red door", "blue door"}


def test_folding_does_not_cross_kinds() -> None:
    bank = AssetBank()
    bank.add_asset(Asset("rocky shore", LOC, "rocky shore", LifecycleStatus.REUSABLE))

    asset = _updater(bank)._fold_or_create_discovered(_obs("shore", PROP))

    assert len(bank.assets) == 2
    assert asset.kind is PROP


def test_unrelated_names_do_not_fold() -> None:
    bank = AssetBank()
    bank.add_asset(Asset("lantern room", LOC, "lantern room", LifecycleStatus.REUSABLE))

    asset = _updater(bank)._fold_or_create_discovered(_obs("night sea", LOC))

    assert len(bank.assets) == 2
    assert asset.name == "night sea"


def test_the_real_curate_path_folds_and_not_just_the_helper(tmp_path) -> None:
    """Drive ``curate_observations`` so the wiring, not only the helper, is covered."""
    bank = AssetBank()
    bank.add_asset(Asset("lantern room", LOC, "lantern room", LifecycleStatus.REUSABLE))
    crop = tmp_path / "crop_room.png"
    crop.write_bytes(b"")

    obs = _obs("room", LOC)
    obs.image_path = str(crop)
    touched = _updater(bank).curate_observations([obs], segment_id=1)

    assert len(bank.assets) == 1, "a re-worded room must not open a second record"
    assert touched == ["lantern room"]
