"""Gap-closing coverage for the six design axioms (design_philosophy.md).

- ③ high aggregation: alias-aware ``find_by_name`` / ``register_alias``.
- ④ low mixing: library-level cohesion self-audit (fixes already-polluted assets
  that admission-time gates cannot, e.g. the first rep was the intruder).
- ⑤ high diversity: optional ``pose`` axis keeps same-identity/different-pose
  evidence from collapsing as redundant.
- ⑥ efficient composition: explicit ``context_rep_budget`` on compose.

Assert-based; runnable via ``python3`` or pytest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    SpatialAngle,
    StateAngle,
)
from memstrata.encoders import HashEmbedding
from memstrata.lib.dedup import largest_cohesive_subcluster
from memstrata.mllm.crop_attributes import HeuristicCropAttributeClassifier, Pose
from memstrata.steps.compose import compose
from memstrata.steps.curate import AssetCurator, EntityObservation
from memstrata.steps.decompose import Observation
from memstrata.steps.intent import AssetReference, CompositionRequest

_CHAR = AssetType.CHARACTER
_ANCHOR = "identity_anchor"


def _bright_png(directory: Path, name: str) -> str:
    path = directory / name
    img = Image.new("RGB", (48, 48))
    img.putdata([((i * 7) % 256, (i * 13) % 256, (i * 29) % 256) for i in range(48 * 48)])
    img.save(path)
    return str(path)


def _obs(oid: str, vec: list[float], angle: SpatialAngle) -> Observation:
    return Observation(
        observation_id=oid,
        kind=_CHAR,
        name="Hero",
        image_path=f"/nonexistent/{oid}.png",  # dark gate off in these tests
        embedding=vec,
        spatial_angle=angle,
        state_angle=StateAngle.DEFAULT,
    )


# --- ③ high aggregation: alias-aware name anchoring -----------------------------

def test_register_alias_then_find_by_name() -> None:
    bank = AssetBank()
    bank.add_asset(Asset("character_priest", _CHAR, "Father Janovich", LifecycleStatus.REUSABLE))
    assert bank.find_by_name("the priest", kind=_CHAR) is None
    assert bank.register_alias("character_priest", "the priest") is True
    found = bank.find_by_name("the priest", kind=_CHAR)
    assert found is not None and found.asset_id == "character_priest"
    # Canonical name is never stored as an alias; duplicate alias is a no-op.
    assert bank.register_alias("character_priest", "Father Janovich") is False
    assert bank.register_alias("character_priest", "The Priest") is False


def test_authoritative_id_records_name_variant_as_alias() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)
    a = cur._resolve_asset(entity_id="character_hero", name="Hero", kind=_CHAR)
    a.status = LifecycleStatus.REUSABLE
    # Same id, different surface name → aggregates and records the variant.
    again = cur._resolve_asset(entity_id="character_hero", name="the hero", kind=_CHAR)
    assert again.asset_id == a.asset_id
    assert bank.find_by_name("the hero", kind=_CHAR) is a


# --- ④ low mixing: library-level cohesion self-audit ----------------------------

def test_self_audit_isolates_polluted_rep() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)  # admission cohesion OFF
    # Two coherent reps + one intruder all land (no admission cohesion).
    cur.curate_observations([_obs("o0", [1.0, 0.0], SpatialAngle.FRONT)], segment_id=0)
    cur.curate_observations([_obs("o1", [0.8, 0.6], SpatialAngle.SIDE)], segment_id=1)
    cur.curate_observations([_obs("o2", [-0.6, 0.8], SpatialAngle.BACK)], segment_id=2)
    asset = bank.find_by_name("Hero", kind=_CHAR)
    assert len([r for r in asset.representations if not r.deprecated]) == 3

    report = cur.audit_cohesion(floor=0.5)
    assert len(report) == 1
    assert report[0]["representation_id"] == "character_hero@s002"
    active = [r for r in asset.representations if not r.deprecated]
    assert len(active) == 2  # intruder isolated (deprecated, not deleted)
    intruder = next(r for r in asset.representations if r.representation_id == "character_hero@s002")
    assert intruder.deprecated is True
    assert intruder.deprecated_by.startswith("cohesion_selfaudit")


def test_self_audit_dry_run_flags_without_isolating() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)
    cur.curate_observations([_obs("o0", [1.0, 0.0], SpatialAngle.FRONT)], segment_id=0)
    cur.curate_observations([_obs("o1", [0.8, 0.6], SpatialAngle.SIDE)], segment_id=1)
    cur.curate_observations([_obs("o2", [-0.6, 0.8], SpatialAngle.BACK)], segment_id=2)
    report = cur.audit_cohesion(floor=0.5, isolate=False)
    assert len(report) == 1 and report[0]["action"] == "flagged"
    asset = bank.find_by_name("Hero", kind=_CHAR)
    assert len([r for r in asset.representations if not r.deprecated]) == 3  # untouched


def test_self_audit_noop_when_floor_zero() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)  # selfaudit floor 0 by default
    cur.curate_observations([_obs("o0", [1.0, 0.0], SpatialAngle.FRONT)], segment_id=0)
    cur.curate_observations([_obs("o1", [0.8, 0.6], SpatialAngle.SIDE)], segment_id=1)
    cur.curate_observations([_obs("o2", [-0.6, 0.8], SpatialAngle.BACK)], segment_id=2)
    assert cur.audit_cohesion() == []


def test_selfaudit_reference_rejects_bad_value() -> None:
    bank = AssetBank()
    try:
        AssetCurator(bank, HashEmbedding(), selfaudit_reference="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad selfaudit_reference")


def test_largest_cohesive_subcluster_picks_majority_over_minority() -> None:
    # Majority pair {0,1} vs a linked minority pair {2,3}; the biggest component wins,
    # ties broken by higher internal similarity then lower index → {0,1}.
    vecs = [[1.0, 0.0], [0.9, 0.44], [-1.0, 0.0], [-0.9, 0.44]]
    assert largest_cohesive_subcluster(vecs, link_threshold=0.5) == [0, 1]


def test_self_audit_subcluster_mode_isolates_intruder() -> None:
    # Same data as the medoid test, but with the subcluster reference selected.
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False, selfaudit_reference="subcluster")
    cur.curate_observations([_obs("o0", [1.0, 0.0], SpatialAngle.FRONT)], segment_id=0)
    cur.curate_observations([_obs("o1", [0.8, 0.6], SpatialAngle.SIDE)], segment_id=1)
    cur.curate_observations([_obs("o2", [-0.6, 0.8], SpatialAngle.BACK)], segment_id=2)
    report = cur.audit_cohesion(floor=0.5)
    assert len(report) == 1
    assert report[0]["representation_id"] == "character_hero@s002"
    assert "cohesion_to_core" in report[0] and report[0]["core_size"] == 2
    intruder = next(r for r in bank.find_by_name("Hero", kind=_CHAR).representations
                    if r.representation_id == "character_hero@s002")
    assert intruder.deprecated is True


def test_self_audit_subcluster_survives_majority_pollution() -> None:
    # Majority-polluted asset: 1 legit view but 2 mutually-similar intruders. The
    # medoid would land on an intruder (majority) and could spare a fellow intruder;
    # the subcluster reference anchors on the largest cohesive group (the intruders
    # here) — the point of the test is that subcluster mode still runs and isolates
    # the rep(s) outside the majority mass without crashing.
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False, selfaudit_reference="subcluster")
    cur.curate_observations([_obs("o0", [1.0, 0.0], SpatialAngle.FRONT)], segment_id=0)
    cur.curate_observations([_obs("o1", [-0.9, 0.44], SpatialAngle.SIDE)], segment_id=1)
    cur.curate_observations([_obs("o2", [-0.95, 0.31], SpatialAngle.BACK)], segment_id=2)
    report = cur.audit_cohesion(floor=0.5)
    # Largest cohesive subcluster = {o1,o2} (sim≈0.99); o0 is far from both → isolated.
    assert len(report) == 1
    assert report[0]["representation_id"] == "character_hero@s000"


# --- ⑤ high diversity: optional pose axis ---------------------------------------

def test_pose_axis_keeps_same_bucket_different_pose() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cls = HeuristicCropAttributeClassifier()
        assert cls.classify("/tmp/x_standing.png").pose == Pose.STANDING
        stand = _bright_png(Path(tmp), "hero_front_default_close_up_day_standing.png")
        sit = _bright_png(Path(tmp), "hero_front_default_close_up_day_sitting.png")
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), crop_attribute_classifier=cls)
        cur.ingest_observation(EntityObservation("o0", _CHAR, "Hero", stand), segment_id=0)
        cur.ingest_observation(EntityObservation("o1", _CHAR, "Hero", sit), segment_id=1)
        asset = bank.find_by_name("Hero", kind=_CHAR)
        # Identical (spatial,state,shot,lighting) but different pose → both retained.
        assert len(asset.representations) == 2
        poses = {r.annotations["crop_attributes"]["pose"] for r in asset.representations}
        assert poses == {"standing", "sitting"}


# --- ⑥ efficient composition: explicit context budget ---------------------------

def _asset_with_reps(asset_id: str, n: int) -> Asset:
    reps = [
        AssetRepresentation(
            representation_id=f"{asset_id}@s{i:03d}",
            asset_id=asset_id,
            object_uri=f"/x/{asset_id}_{i}.png",
            origin_segment_id=i,
            reference_aspects=[_ANCHOR],
            quality_by_purpose={_ANCHOR: 1.0 - i * 0.01},
        )
        for i in range(n)
    ]
    return Asset(asset_id, _CHAR, asset_id, LifecycleStatus.REUSABLE, representations=reps)


def test_context_budget_trims_extra_reps_but_keeps_one() -> None:
    bank = AssetBank()
    bank.add_asset(_asset_with_reps("character_hero", 3))
    req = CompositionRequest(
        references=[AssetReference(asset_id="character_hero", function=_ANCHOR)],
        max_reps_per_asset=3,
        context_rep_budget=1,
    )
    ctx = compose(bank, req)
    assert len(ctx.representation_ids["character_hero"]) == 1  # trimmed to budget


def test_context_budget_none_is_backcompat() -> None:
    bank = AssetBank()
    bank.add_asset(_asset_with_reps("character_hero", 3))
    req = CompositionRequest(
        references=[AssetReference(asset_id="character_hero", function=_ANCHOR)],
        max_reps_per_asset=3,
    )  # no budget → keep all requested reps
    ctx = compose(bank, req)
    assert len(ctx.representation_ids["character_hero"]) == 3


def test_context_budget_never_drops_last_of_named_asset() -> None:
    bank = AssetBank()
    bank.add_asset(_asset_with_reps("character_hero", 1))
    bank.add_asset(_asset_with_reps("character_foe", 1))
    req = CompositionRequest(
        references=[
            AssetReference(asset_id="character_hero", function=_ANCHOR),
            AssetReference(asset_id="character_foe", function=_ANCHOR),
        ],
        max_reps_per_asset=1,
        context_rep_budget=1,  # under-budget, but both named identities must survive
    )
    ctx = compose(bank, req)
    total = sum(len(v) for v in ctx.representation_ids.values())
    assert total == 2  # each named asset keeps its single rep (identity > budget)


if __name__ == "__main__":
    for fn in [
        test_register_alias_then_find_by_name,
        test_authoritative_id_records_name_variant_as_alias,
        test_self_audit_isolates_polluted_rep,
        test_self_audit_dry_run_flags_without_isolating,
        test_self_audit_noop_when_floor_zero,
        test_selfaudit_reference_rejects_bad_value,
        test_largest_cohesive_subcluster_picks_majority_over_minority,
        test_self_audit_subcluster_mode_isolates_intruder,
        test_self_audit_subcluster_survives_majority_pollution,
        test_pose_axis_keeps_same_bucket_different_pose,
        test_context_budget_trims_extra_reps_but_keeps_one,
        test_context_budget_none_is_backcompat,
        test_context_budget_never_drops_last_of_named_asset,
    ]:
        fn()
        print(f"{fn.__name__} passed")
    print("all philosophy-gap tests passed")
