"""Surface-name variants of one term must resolve to one identity.

Measured on the Track B naming probe: the write-side namer anchors identity on the prompt's own
wording, so shot 0 writing "lantern room" and shot 1 writing "lantern-room" opened two location
assets for one place. Folding is exact-after-normalisation (case, separators, whitespace runs),
never fuzzy, so identity stays reproducible.
"""

from __future__ import annotations

from memstrata.bank import AssetType
from memstrata.bank.schema import Asset, AssetBank, LifecycleStatus, surface_key


def _bank_with(name: str, kind: AssetType = AssetType.LOCATION) -> AssetBank:
    bank = AssetBank()
    bank.add_asset(
        Asset(asset_id=name, kind=kind, name=name, status=LifecycleStatus.REUSABLE, description="")
    )
    return bank


def test_surface_key_folds_case_separators_and_whitespace() -> None:
    assert surface_key("lantern-room") == surface_key("Lantern Room") == "lantern room"
    assert surface_key("keeper's  log") == "keeper's log"
    assert surface_key("night_market") == surface_key("night market")
    assert surface_key("") == ""


def test_surface_key_leaves_cjk_untouched() -> None:
    assert surface_key("灯塔看守人") == "灯塔看守人"
    assert surface_key("灯塔看守人") != surface_key("灯塔")


def test_find_by_name_matches_a_separator_variant() -> None:
    bank = _bank_with("lantern room")
    found = bank.find_by_name("lantern-room", kind=AssetType.LOCATION)
    assert found is not None and found.asset_id == "lantern room"


def test_find_by_name_still_respects_kind() -> None:
    bank = _bank_with("lantern room")
    assert bank.find_by_name("lantern-room", kind=AssetType.PROP) is None


def test_alias_registration_ignores_a_mere_separator_variant() -> None:
    """A variant that folds onto the canonical name carries no new information."""
    bank = _bank_with("lantern room")
    assert bank.register_alias("lantern room", "Lantern-Room") is False
    assert bank.register_alias("lantern room", "the light chamber") is True
    assert bank.register_alias("lantern room", "The  Light_Chamber") is False


def test_name_anchored_resolution_folds_onto_the_existing_asset() -> None:
    from memstrata.skills.memory_update.curator import MemoryUpdater

    bank = _bank_with("lantern room")
    curator = MemoryUpdater(bank)
    asset = curator._resolve_asset(
        entity_id="lantern-room", name="lantern-room", kind=AssetType.LOCATION
    )
    assert asset.asset_id == "lantern room"
    assert len(bank.assets) == 1
    # A fold-equivalent variant is not recorded as an alias: the alias list is for genuinely
    # different surface forms ("the light chamber"), which is what the read path needs.
    assert bank.assets["lantern room"].metadata.get("aliases", []) == []


def test_a_genuinely_different_label_is_kept_as_an_alias_when_it_folds_in() -> None:
    from memstrata.skills.memory_update.curator import MemoryUpdater

    bank = _bank_with("lantern room")
    bank.register_alias("lantern room", "the light chamber")
    curator = MemoryUpdater(bank)
    asset = curator._resolve_asset(
        entity_id="The Light-Chamber", name="The Light-Chamber", kind=AssetType.LOCATION
    )
    assert asset.asset_id == "lantern room"
    assert len(bank.assets) == 1


def test_an_authoritative_id_never_falls_back_to_a_name_lookup() -> None:
    """Track A packets carry real asset ids next to generic perception labels; a name fallback
    there would merge two distinct entities that happen to share the label."""
    from memstrata.skills.memory_update.curator import MemoryUpdater

    bank = _bank_with("character", kind=AssetType.CHARACTER)
    curator = MemoryUpdater(bank)
    asset = curator._resolve_asset(
        entity_id="mem@c00007_r01", name="character", kind=AssetType.CHARACTER
    )
    assert asset.asset_id == "mem@c00007_r01"
    assert len(bank.assets) == 2
