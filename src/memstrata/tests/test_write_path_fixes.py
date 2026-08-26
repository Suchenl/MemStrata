"""Write-path correctness: the bugs that made the "stratified" bank not stratified.

Each test pins one previously-broken behaviour so it cannot silently regress:

- P0-1 the production preset (and its classifiers) actually reach the components;
- P0-3a a full asset can still record new evidence, and a pixel-less seed rep does not
  permanently occupy one of the few per-asset slots;
- P1-i ``_apply_rep_selection`` reports "R_j changed" the right way round;
- the bank-wide budget retires with 留痕 instead of deleting;
- embeddings from different encoder routes are never compared by silent truncation;
- a same-name/different-type observation does not get merged into an existing identity;
- discovery (O_disc) + identity reconciliation (χ) + per-type budgets B_τ behave;
- ``stratification_report`` tells a wired run from an unwired one.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

from PIL import Image

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetType,
    LifecycleStatus,
    SpatialAngle,
    StateAngle,
)
from memstrata.encoders import HashEmbedding
from memstrata.lib.dedup import (
    compatible_stratum,
    cosine_or_none,
    select_attribute_diverse,
    text_similarity,
)
from memstrata.mllm.angle_classifier import HeuristicAngleClassifier
from memstrata.mllm.crop_attributes import HeuristicCropAttributeClassifier
from memstrata.pipeline import MemStrata, build_curator, build_decomposer
from memstrata.steps.curate import AssetCurator, MemoryPolicy, stratification_report
from memstrata.steps.decompose import (
    SOURCE_DISCOVERED,
    DiscoveredEntity,
    NamedEntity,
    Observation,
    RoleAwareDecomposer,
)

_CHAR = AssetType.CHARACTER
_PROP = AssetType.PROP


def _png(directory: Path, name: str) -> str:
    """A bright, unique-bytes PNG (passes the dark gate, distinct hash embedding)."""
    path = directory / name
    img = Image.new("RGB", (48, 48))
    seed = sum(name.encode())
    img.putdata([(((i + seed) * 7) % 256, ((i + seed) * 13) % 256, ((i + seed) * 29) % 256)
                 for i in range(48 * 48)])
    img.save(path)
    return str(path)


# --- helpers --------------------------------------------------------------------------

def test_cosine_or_none_refuses_mismatched_dimensions() -> None:
    assert cosine_or_none([1.0, 0.0], [1.0, 0.0]) == 1.0
    # Previously a bare zip() truncated and returned a meaningless score.
    assert cosine_or_none([1.0, 0.0], [1.0, 0.0, 0.0]) is None
    assert cosine_or_none([], [1.0]) is None


def test_compatible_stratum_treats_unknown_as_compatible() -> None:
    assert compatible_stratum(rep_bucket=("front", "default"), new_bucket=("front", "default"))
    assert not compatible_stratum(rep_bucket=("front", "default"), new_bucket=("back", "default"))
    # Unknown cannot prove the strata differ → stays compatible (offline default no-op).
    assert compatible_stratum(rep_bucket=("unknown", "unknown"), new_bucket=("back", "damaged"))


def test_text_similarity_is_symmetric_and_bounded() -> None:
    assert text_similarity("a bearded keeper in a wool coat", "") == 0.0
    assert text_similarity("bearded keeper wool coat", "bearded keeper wool coat") == 1.0
    mixed = text_similarity("bearded keeper wool coat", "bearded keeper leather boots")
    assert 0.0 < mixed < 1.0
    assert mixed == text_similarity("bearded keeper leather boots", "bearded keeper wool coat")
    # CJK is compared per character, so no segmenter is needed.
    assert text_similarity("灯塔看守人", "灯塔看守") > 0.5


def test_pin_keeps_reserved_index_in_diversity_selection() -> None:
    buckets = [("unknown",)] * 4
    # Without a pin the all-unknown bucket always yields the lowest index.
    assert select_attribute_diverse(bucket_keys=buckets, max_keep=1) == [0]
    assert select_attribute_diverse(bucket_keys=buckets, max_keep=1, pin=[3]) == [3]


# --- P0-3a: memory freeze + placeholder slot ------------------------------------------

def test_full_asset_still_records_new_evidence() -> None:
    """A budget-full asset must keep learning; it used to freeze forever."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), max_reps_per_asset=3,
                           attributes_when_angles_known=False)
        for segment in range(6):
            cur.curate_observations(
                [Observation(observation_id=f"o{segment}", kind=_CHAR, name="Hero",
                             image_path=_png(Path(tmp), f"hero_{segment}.png"))],
                segment_id=segment,
            )
        asset = bank.find_by_name("Hero", kind=_CHAR)
        live = [r for r in asset.representations if not r.deprecated]
        assert len(live) == 3  # budget respected
        # The newest segment is present — that is the whole point of the reserved slot.
        assert max(r.origin_segment_id for r in live) == 5


def test_pixelless_seed_rep_does_not_consume_budget() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), max_reps_per_asset=2,
                           attributes_when_angles_known=False)
        # Screenplay seeding: appearance text, no pixels yet.
        cur.ingest_packet({
            "segment_id": -1,
            "observations": [{"entity_id": "char_hero", "kind": "character", "name": "Hero",
                              "representation_id": "char_hero@seed", "crop_path": "",
                              "description": "a bearded keeper in a wool coat"}],
        })
        asset = bank.get_asset("char_hero")
        assert len([r for r in asset.representations if not r.deprecated]) == 1
        assert asset.description  # seeded description survives for text-keyed retrieval

        for segment in range(2):
            cur.curate_observations(
                [Observation(observation_id=f"o{segment}", kind=_CHAR, name="Hero",
                             entity_id="char_hero",
                             image_path=_png(Path(tmp), f"hero_{segment}.png"))],
                segment_id=segment,
            )
        live = [r for r in asset.representations if not r.deprecated]
        # Both real crops fit: the placeholder was retired, not counted.
        assert len(live) == 2
        assert all(r.object_uri for r in live)
        seed = next(r for r in asset.representations if r.representation_id == "char_hero@seed")
        assert seed.deprecated and seed.deprecated_by == "superseded_by_visual_evidence"


# --- P1-i: return-value contract ------------------------------------------------------

def test_apply_rep_selection_reports_change_correctly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        from memstrata.bank import AssetRepresentation

        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), max_reps_per_asset=1,
                           attributes_when_angles_known=False)
        asset = Asset("char_hero", _CHAR, "Hero", LifecycleStatus.REUSABLE)
        bank.add_asset(asset)
        first = AssetRepresentation("r0", "char_hero", _png(Path(tmp), "a.png"), origin_segment_id=0)
        assert cur._apply_rep_selection(asset, first) is True
        before = bank.version
        second = AssetRepresentation("r1", "char_hero", _png(Path(tmp), "b.png"), origin_segment_id=1)
        # Over budget: the newer rep wins the reserved slot, so R_j really did change.
        assert cur._apply_rep_selection(asset, second) is True
        assert [r.representation_id for r in asset.representations] == ["r1"]
        cur.bank.touch()
        assert bank.version > before


# --- bank-wide budget retires with 留痕 ------------------------------------------------

def test_global_budget_retires_with_traceable_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), max_total_representations=2,
                           attributes_when_angles_known=False)
        for segment in range(3):
            crop = _png(Path(tmp), f"c{segment}.png")
            cur.ingest_packet({
                "segment_id": segment,
                "observations": [
                    {"entity_id": "char_hero", "kind": "character", "name": "Hero",
                     "representation_id": f"hero_c{segment}", "crop_path": crop,
                     "quality": 1.0 - 0.1 * segment},
                    {"entity_id": "loc_forest", "kind": "location", "name": "Forest",
                     "representation_id": f"forest_c{segment}", "crop_path": crop,
                     "quality": 0.5},
                ],
            })
        live = [r for a in bank.assets.values() for r in a.representations if not r.deprecated]
        assert len(live) <= 2
        retired = [r for a in bank.assets.values() for r in a.representations
                   if r.deprecated_by == "global_budget"]
        # Evidence is isolated, not deleted: still on the record and reversible.
        assert retired
        assert all(r.annotations.get("admission") == "retired_global_budget" for r in retired)


# --- identity conflict ----------------------------------------------------------------

def test_same_id_different_type_is_not_silently_merged() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)
    bank.add_asset(Asset("character_guard", _CHAR, "Guard", LifecycleStatus.REUSABLE))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asset = cur._resolve_asset(entity_id="character_guard", name="Guard", kind=_PROP)
    assert asset.asset_id == "character_guard"
    assert any("identity conflict" in str(w.message) for w in caught)
    assert bank.get_asset("character_guard").metadata["identity_conflicts"]


# --- per-type budgets B_τ -------------------------------------------------------------

def test_per_type_budget_is_applied() -> None:
    policy = MemoryPolicy(max_reps_per_asset=5, max_reps_by_type={"character": 2, "prop": 1})
    cur = AssetCurator(AssetBank(), HashEmbedding(), policy=policy)
    assert cur.budget_for(_CHAR) == 2
    assert cur.budget_for(_PROP) == 1
    assert cur.budget_for(AssetType.LOCATION) == 5  # falls back to the shared default
    assert cur.redundancy_for(_CHAR) == policy.redundancy_threshold


# --- discovery (O_disc) + reconciliation (χ) -------------------------------------------

class _StubDiscoverer:
    def __init__(self, rows: list[DiscoveredEntity]) -> None:
        self.rows = rows
        self.calls = 0

    def discover(self, segment_video, *, segment_id, kinds):
        self.calls += 1
        return [r for r in self.rows if r.kind in kinds]


class _StubCropper:
    def __init__(self, crop: str, bbox: list[int]) -> None:
        self.payload = {"crop_path": crop, "bbox": bbox}

    def crop(self, segment_video, entity, *, segment_id):
        return self.payload


def test_discovery_is_off_without_a_discoverer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = _png(Path(tmp), "hero.png")
        dec = RoleAwareDecomposer(HashEmbedding(), cropper=_StubCropper(crop, [0, 0, 400, 400]))
        obs = dec.decompose(
            segment_id=0, segment_video="segment.mp4",
            named_entities=[NamedEntity(name="Hero", kind=_CHAR, entity_id="char_hero")],
        )
        assert [o.source for o in obs] == ["requested"]


def test_discovery_drops_regions_already_covered() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hero = _png(root, "hero.png")
        overlapping = DiscoveredEntity(kind=_CHAR, crop_path=_png(root, "dup.png"),
                                       bbox_norm=[0, 0, 400, 400])
        fresh = DiscoveredEntity(kind=_PROP, crop_path=_png(root, "lamp.png"),
                                 bbox_norm=[700, 700, 900, 900])
        dec = RoleAwareDecomposer(
            HashEmbedding(), cropper=_StubCropper(hero, [0, 0, 400, 400]),
            discoverer=_StubDiscoverer([overlapping, fresh]),
        )
        obs = dec.decompose(
            segment_id=0, segment_video="segment.mp4",
            named_entities=[NamedEntity(name="Hero", kind=_CHAR, entity_id="char_hero")],
        )
        # Requested first (callers index into it), then only the non-overlapping discovery.
        assert [o.source for o in obs] == ["requested", SOURCE_DISCOVERED]
        assert obs[1].kind is _PROP and obs[1].entity_id is None


def test_discovery_type_constraint_rejects_unsupported_kind() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dec = RoleAwareDecomposer(
            HashEmbedding(),
            discoverer=_StubDiscoverer([
                DiscoveredEntity(kind=_PROP, crop_path=_png(root, "lamp.png"))
            ]),
            discovery_kinds=(_CHAR,),  # props are not a discovery target here
        )
        obs = dec.decompose(segment_id=0, segment_video="segment.mp4", named_entities=[])
        assert obs == []


def test_reconciliation_merges_above_beta_and_forks_below() -> None:
    bank = AssetBank()
    cur = AssetCurator(
        bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
        attributes_when_angles_known=False,
        policy=MemoryPolicy(reconcile_threshold=0.5, reconcile_text_weight=1.0),
    )
    bank.add_asset(Asset("char_keeper", _CHAR, "Keeper", LifecycleStatus.REUSABLE,
                         description="a bearded keeper in a heavy wool coat"))

    near = Observation(observation_id="d0", kind=_CHAR, name="disc_0",
                       image_path="/nonexistent/d0.png", source=SOURCE_DISCOVERED,
                       description="a bearded keeper wearing a heavy wool coat")
    matched, meta = cur._reconcile_identity(near)
    assert matched is not None and matched.asset_id == "char_keeper"
    assert meta["decision"] == "merged" and meta["chi"] >= meta["threshold"]

    far = Observation(observation_id="d1", kind=_CHAR, name="disc_1",
                      image_path="/nonexistent/d1.png", source=SOURCE_DISCOVERED,
                      description="a rusty bicycle leaning on a fence")
    matched_far, meta_far = cur._reconcile_identity(far)
    assert matched_far is None and meta_far["decision"] == "new_asset"


def test_reconciliation_never_crosses_types() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False,
                       policy=MemoryPolicy(reconcile_threshold=0.1, reconcile_text_weight=1.0))
    bank.add_asset(Asset("prop_coat", _PROP, "Coat", LifecycleStatus.REUSABLE,
                         description="a heavy wool coat"))
    obs = Observation(observation_id="d0", kind=_CHAR, name="disc_0",
                      image_path="/nonexistent/d0.png", source=SOURCE_DISCOVERED,
                      description="a heavy wool coat")
    matched, meta = cur._reconcile_identity(obs)
    assert matched is None and meta["decision"] == "new_asset"


def test_discovered_observation_creates_provisional_asset() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)
    cur.curate_observations(
        [Observation(observation_id="d0", kind=_PROP, name="prop_disc_0",
                     image_path="/nonexistent/d0.png", source=SOURCE_DISCOVERED,
                     description="a brass lantern")],
        segment_id=3,
    )
    created = [a for a in bank.assets.values() if a.metadata.get("provisional")]
    assert len(created) == 1
    assert created[0].kind is _PROP
    assert created[0].description == "a brass lantern"


# --- P0-1: preset + classifiers actually reach the components --------------------------

def test_build_curator_carries_policy_and_classifier() -> None:
    policy = MemoryPolicy.production(cohesion_floor=0.0, max_total_representations=64)
    cur = build_curator(
        AssetBank(), policy=policy, embedder=HashEmbedding(),
        crop_attribute_classifier=HeuristicCropAttributeClassifier(),
    )
    assert cur.policy is policy
    assert cur.max_total_representations == 64
    assert cur.budget_for(_CHAR) == 6  # per-type production budget, not the flat default
    assert isinstance(cur.crop_attribute_classifier, HeuristicCropAttributeClassifier)
    dec = build_decomposer(policy=policy, embedder=HashEmbedding())
    assert dec.crop_quality_gate is True  # the preset knob that used to be dropped


def test_pipeline_warns_when_a_preset_would_be_silently_dropped() -> None:
    bank = AssetBank()
    hand_built = AssetCurator(bank, HashEmbedding())  # not built from the policy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MemStrata(bank=bank, policy=MemoryPolicy.production(), curator=hand_built,
                  embedder=HashEmbedding())
    assert any("is NOT applied to the supplied" in str(w.message) for w in caught)


def test_production_run_wires_classifier_into_curator_and_decomposer() -> None:
    """The end-to-end P0-1 check: real angles land in the bank via the production preset."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop = _png(root, "rabbit_front_default_close_up_day.png")
        policy = MemoryPolicy.production()
        bank = AssetBank()
        curator = build_curator(
            bank, policy=policy, embedder=HashEmbedding(),
            angle_classifier=HeuristicAngleClassifier(),
            crop_attribute_classifier=HeuristicCropAttributeClassifier(),
        )
        decomposer = build_decomposer(
            policy=policy, embedder=HashEmbedding(),
            angle_classifier=HeuristicAngleClassifier(),
        )
        mem = MemStrata.for_production(
            persist_path=root / "bank.json", policy=policy, bank=bank,
            curator=curator, decomposer=decomposer, embedder=HashEmbedding(),
            run_dir=root / "pipeline",
        )
        mem.run_segment(
            "A rabbit.", segment_id=0, skip_generate=True,
            named_entities=[NamedEntity(name="Rabbit", kind=_CHAR, entity_id="char_rabbit",
                                        crop_path=crop)],
        )
        rep = mem.bank.get_asset("char_rabbit").representations[0]
        assert rep.spatial_angle is SpatialAngle.FRONT
        assert rep.state_angle is StateAngle.DEFAULT
        # The diagnostic must show a stratified run, and the file must be on disk.
        report = mem.stratification()
        assert report["spatial_known_ratio"] == 1.0
        assert report["state_known_ratio"] == 1.0
        assert report["described_ratio"] == 1.0  # observation description written through
        assert (root / "pipeline" / "stratification.json").is_file()
        assert mem.segment_log[0]["stratification"]["spatial_known_ratio"] == 1.0
        assert mem.segment_log[0]["observation_sources"] == {"requested": 1}


# --- C item 1/2: crop-acquisition provenance survives + bbox-only review flag ----------

def test_crop_acquisition_meta_survives_into_annotations_and_flags_bbox_only() -> None:
    """A GDINO bbox-only crop keeps its provenance in the rep and is flagged for review."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), attributes_when_angles_known=False)
        acq = {
            "identity_sim": 0.72,
            "source": "grounding_dino",
            "source_detail": {"quality_profile": "bbox_high_recall_no_mask"},
        }
        cur.curate_observations(
            [Observation(observation_id="o0", kind=_CHAR, name="Hero",
                         image_path=_png(Path(tmp), "hero.png"),
                         angle_meta={"crop_acquisition": acq})],
            segment_id=0,
        )
        rep = bank.find_by_name("Hero", kind=_CHAR).representations[0]
        # Provenance lands intact (item 1): source + nested source_detail preserved.
        assert rep.annotations["crop_acquisition"]["source"] == "grounding_dino"
        assert (rep.annotations["crop_acquisition"]["source_detail"]["quality_profile"]
                == "bbox_high_recall_no_mask")
        # Review flag set, scoring untouched (item 2): annotation-only.
        assert rep.annotations.get("needs_review") is True
        assert rep.annotations.get("review_reason") == "gdino_bbox_only_no_mask"


def test_masked_sam3_crop_is_not_flagged_for_review() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), attributes_when_angles_known=False)
        acq = {"source": "sam3_concept",
               "source_detail": {"quality_profile": "masked_sam3_quality_gated"}}
        cur.curate_observations(
            [Observation(observation_id="o0", kind=_CHAR, name="Hero",
                         image_path=_png(Path(tmp), "hero.png"),
                         angle_meta={"crop_acquisition": acq})],
            segment_id=0,
        )
        rep = bank.find_by_name("Hero", kind=_CHAR).representations[0]
        assert rep.annotations["crop_acquisition"]["source"] == "sam3_concept"
        assert "needs_review" not in rep.annotations


def test_stratification_report_flags_an_unwired_run() -> None:
    """The null classifier must be visible as 0 known ratios, not look like success."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding())  # Null* classifiers
        cur.curate_observations(
            [Observation(observation_id="o0", kind=_CHAR, name="Hero",
                         image_path=_png(Path(tmp), "hero.png"))],
            segment_id=0,
        )
        report = stratification_report(bank)
        assert report["representations"] == 1
        assert report["spatial_known_ratio"] == 0.0
        assert report["state_known_ratio"] == 0.0
        assert report["temporal_known_ratio"] == 1.0  # temporal stratum IS populated
        assert report["angle_source_counts"].get("null") == 1
