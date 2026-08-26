"""Alias writeback when χ-reconciliation merges a discovered observation.

When a self-discovered observation is merged into an existing identity under a DIFFERENT
surface name, that name must be registered as an alias (axiom 3), so the name-anchored read
path can also retrieve the identity by the incoming name. Without it, a cross-segment prompt
using the other name silently misses the merged asset and the bank looks fragmented on read.
"""

from __future__ import annotations

from memstrata.bank import AssetBank, AssetType
from memstrata.encoders import HashEmbedding
from memstrata.skills.decomposition import SOURCE_DISCOVERED, Observation
from memstrata.skills.memory_update.curator import MemoryUpdater

_CHAR = AssetType.CHARACTER
_VEC = [1.0, 0.0, 0.0, 0.0]  # identical embeddings force χ visual_sim = 1.0 → merge


def _disc(obs_id: str, name: str, desc: str) -> Observation:
    return Observation(
        observation_id=obs_id,
        kind=_CHAR,
        name=name,
        image_path=f"/tmp/{obs_id}.png",
        source=SOURCE_DISCOVERED,
        embedding=list(_VEC),
        description=desc,
    )


def test_reconcile_merge_registers_incoming_name_as_alias() -> None:
    bank = AssetBank()
    cur = MemoryUpdater(bank, HashEmbedding())
    cur.curate_observations([_disc("d0", "the tall guard", "a tall guard in armor")], segment_id=0)
    assert len(bank.assets) == 1
    aid = next(iter(bank.assets))

    # Identical embedding + description → χ ≥ β_τ → merged, but under a different name.
    cur.curate_observations([_disc("d1", "armored sentry", "a tall guard in armor")], segment_id=1)
    assert len(bank.assets) == 1  # merged, not fragmented

    asset = bank.get_asset(aid)
    aliases = [str(a).lower() for a in (asset.metadata.get("aliases") or [])]
    assert "armored sentry" in aliases
    # The name-anchored read path now resolves the identity by the second name too.
    assert bank.find_by_name("armored sentry", kind=_CHAR) is asset


def test_reconcile_merge_same_name_registers_no_alias() -> None:
    bank = AssetBank()
    cur = MemoryUpdater(bank, HashEmbedding())
    cur.curate_observations([_disc("d0", "the tall guard", "a tall guard")], segment_id=0)
    cur.curate_observations([_disc("d1", "the tall guard", "a tall guard")], segment_id=1)
    assert len(bank.assets) == 1
    asset = bank.get_asset(next(iter(bank.assets)))
    assert not (asset.metadata.get("aliases") or [])
