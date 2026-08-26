"""Contracts for causal compose, intent-targeted observation, and crop QA."""

from __future__ import annotations

import tempfile
from pathlib import Path

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRelation,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    RelationType,
    SpatialAngle,
    StateAngle,
)
from memstrata.encoders import HashEmbedding
from memstrata.lib.crop_qa import audit_crop
from memstrata.pipeline import MemStrata
from memstrata.steps.compose import compose, select_reps
from memstrata.steps.intent import AssetReference, CompositionRequest
from memstrata.steps.curate import AssetCurator, EntityObservation
from memstrata.steps.decompose import NamedEntity, RoleAwareDecomposer
from memstrata.steps.intent import IntentInterpreter


class _Cropper:
    def __init__(self, crop: str) -> None:
        self.crop_path = crop
        self.targets: list[str] = []

    def crop(self, segment_video: str, entity: NamedEntity, *, segment_id: int) -> str | None:
        del segment_video, segment_id
        self.targets.append(entity.entity_id or entity.name)
        return self.crop_path


def test_intent_parses_explicit_view_and_state() -> None:
    bank = AssetBank(
        {
            "char_hero": Asset(
                asset_id="char_hero",
                kind=AssetType.CHARACTER,
                name="Hero",
                status=LifecycleStatus.REUSABLE,
            )
        }
    )
    request, calls = IntentInterpreter(bank).interpret("Hero back view, damaged.", segment_id=1)
    assert calls == 0
    assert request.references[0].preferred_spatial is SpatialAngle.BACK
    assert request.references[0].preferred_state is not None


def test_compose_excludes_current_and_future_evidence() -> None:
    asset = Asset(
        asset_id="char_hero",
        kind=AssetType.CHARACTER,
        name="Hero",
        status=LifecycleStatus.REUSABLE,
        representations=[
            AssetRepresentation("hero@s000", "char_hero", "/tmp/old.png", origin_segment_id=0),
            AssetRepresentation("hero@s001", "char_hero", "/tmp/current.png", origin_segment_id=1),
        ],
    )
    assert select_reps(asset, function="identity_anchor", as_of_segment_id=1) == ["hero@s000"]


def test_compose_expands_explicit_structural_relation() -> None:
    location = Asset("loc_forest", AssetType.LOCATION, "Forest", LifecycleStatus.REUSABLE)
    hero = Asset(
        "char_hero",
        AssetType.CHARACTER,
        "Hero",
        LifecycleStatus.REUSABLE,
        relations=[AssetRelation(RelationType.LOCATED_IN, "loc_forest")],
    )
    bank = AssetBank({"char_hero": hero, "loc_forest": location})
    context = compose(
        bank,
        CompositionRequest(
            references=[AssetReference(asset_id="char_hero")],
            relation_hops=1,
        ),
    )
    assert context.asset_ids == ["char_hero", "loc_forest"]
    assert context.expanded == ["loc_forest"]


def test_pipeline_derives_observation_targets_from_intent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = str(Path(tmp) / "hero.jpg")
        Path(crop).write_bytes(b"hero-different-crop")
        bank = AssetBank()
        curator = AssetCurator(bank, HashEmbedding())
        curator.ingest_observation(
            EntityObservation("hero@s000", AssetType.CHARACTER, "Hero", crop, entity_id="char_hero"),
            segment_id=0,
        )
        cropper = _Cropper(crop)
        mem = MemStrata(
            bank=bank,
            decomposer=RoleAwareDecomposer(HashEmbedding(), cropper=cropper),
            curator=curator,
        )
        result = mem.run_segment("Hero returns.", segment_id=1, segment_video="segment.mp4", skip_generate=True)
        assert cropper.targets == ["char_hero"]
        assert result.observations[0].entity_id == "char_hero"


def test_crop_qa_rejects_unreadable_external_input() -> None:
    report = audit_crop("/definitely/missing/crop.png")
    assert not report.accepted
    assert report.reasons[0].startswith("unreadable_crop:")


def test_curate_writes_explicit_relations_for_expansion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        hero_crop = str(Path(tmp) / "hero.jpg")
        forest_crop = str(Path(tmp) / "forest.jpg")
        Path(hero_crop).write_bytes(b"hero-crop-bytes")
        Path(forest_crop).write_bytes(b"forest-crop-bytes")

        bank = AssetBank()
        curator = AssetCurator(bank, HashEmbedding())
        curator.ingest_packet({
            "segment_id": 0,
            "observations": [
                {
                    "entity_id": "char_hero",
                    "kind": "character",
                    "name": "Hero",
                    "representation_id": "rep_hero_c0",
                    "crop_path": hero_crop,
                },
                {
                    "entity_id": "loc_forest",
                    "kind": "location",
                    "name": "Forest",
                    "representation_id": "rep_forest_c0",
                    "crop_path": forest_crop,
                },
            ],
            "relations": [
                {
                    "asset_id": "char_hero",
                    "relation_type": "located_in",
                    "target_asset_id": "loc_forest",
                },
                # Duplicate + unknown type must be ignored idempotently.
                {
                    "asset_id": "char_hero",
                    "relation_type": "located_in",
                    "target_asset_id": "loc_forest",
                },
                {
                    "asset_id": "char_hero",
                    "relation_type": "not_a_relation",
                    "target_asset_id": "loc_forest",
                },
            ],
        })

        hero = bank.get_asset("char_hero")
        assert hero is not None
        assert len(hero.relations) == 1
        assert hero.relations[0].relation_type is RelationType.LOCATED_IN

        context = compose(
            bank,
            CompositionRequest(
                references=[AssetReference(asset_id="char_hero")],
                relation_hops=1,
            ),
        )
        assert context.expanded == ["loc_forest"]


def test_ingest_packet_skips_malformed_row() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = str(Path(tmp) / "good.jpg")
        Path(good).write_bytes(b"good-crop-bytes")
        bank = AssetBank()
        curator = AssetCurator(bank, HashEmbedding())
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            touched = curator.ingest_packet({
                "segment_id": 0,
                "observations": [
                    {"entity_id": "bad", "kind": "not_a_kind", "name": "Bad",
                     "representation_id": "rep_bad", "crop_path": good},
                    {"entity_id": "char_hero", "kind": "character", "name": "Hero",
                     "representation_id": "rep_hero", "crop_path": good},
                ],
            })
        assert touched == ["char_hero"]
        assert bank.get_asset("bad") is None
        assert bank.get_asset("char_hero") is not None


def test_state_event_builds_replacement_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = str(Path(tmp) / "c.jpg")
        Path(crop).write_bytes(b"crop-bytes")
        bank = AssetBank()
        curator = AssetCurator(bank, HashEmbedding())
        curator.ingest_packet({
            "segment_id": 0,
            "observations": [
                {"entity_id": "prop_apple", "kind": "prop", "name": "Apple",
                 "representation_id": "rep_apple_c0", "crop_path": crop},
                {"entity_id": "prop_core", "kind": "prop", "name": "Apple core",
                 "representation_id": "rep_core_c1", "crop_path": crop},
            ],
        })
        curator._apply_state_events([
            {
                "event_id": "evt_eaten",
                "segment_id": 1,
                "deprecates": ["rep_apple_c0"],
                "replaced_by": "prop_core",
            }
        ])
        apple = bank.get_asset("prop_apple")
        core = bank.get_asset("prop_core")
        assert apple is not None and core is not None
        assert any(
            r.relation_type is RelationType.DEPRECATED_BY and r.target_asset_id == "prop_core"
            for r in apple.relations
        )
        assert any(
            r.relation_type is RelationType.REPLACES and r.target_asset_id == "prop_apple"
            for r in core.relations
        )


def test_global_budget_evicts_weakest_but_keeps_one_per_asset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = AssetBank()
        curator = AssetCurator(bank, HashEmbedding(), max_total_representations=2)
        for segment in range(3):
            crop = str(Path(tmp) / f"hero_{segment}.jpg")
            Path(crop).write_bytes(f"hero-{segment}".encode())
            curator.ingest_packet({
                "segment_id": segment,
                "observations": [
                    {"entity_id": "char_hero", "kind": "character", "name": "Hero",
                     "representation_id": f"rep_hero_c{segment}", "crop_path": crop,
                     "spatial_angle": ["front", "side", "back"][segment], "quality": 1.0 - 0.1 * segment},
                    {"entity_id": "loc_forest", "kind": "location", "name": "Forest",
                     "representation_id": f"rep_forest_c{segment}", "crop_path": crop,
                     "quality": 0.5},
                ],
            })
        live = [rep for asset in bank.assets.values() for rep in asset.representations if not rep.deprecated]
        assert len(live) <= 2
        # Every asset keeps at least one live representation.
        for asset in bank.assets.values():
            assert any(not rep.deprecated for rep in asset.representations)


def test_bank_save_load_roundtrip_with_relations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        hero = Asset("char_hero", AssetType.CHARACTER, "Hero", LifecycleStatus.REUSABLE,
                     relations=[AssetRelation(RelationType.LOCATED_IN, "loc_forest")])
        hero.representations.append(
            AssetRepresentation("hero@s0", "char_hero", "/tmp/h.png", origin_segment_id=0)
        )
        bank = AssetBank({"char_hero": hero})
        bank.touch()
        path = Path(tmp) / "bank.json"
        bank.save(path)
        restored = AssetBank.load(path)
        assert restored.version == bank.version
        got = restored.get_asset("char_hero")
        assert got is not None
        assert got.relations[0].relation_type is RelationType.LOCATED_IN
        assert got.representations[0].representation_id == "hero@s0"


def test_for_production_preset_enables_optin_knobs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        persist = str(Path(tmp) / "bank.json")
        mem = MemStrata.for_production(persist_path=persist, max_total_representations=64)
        assert mem.persist_path is not None
        assert mem.relation_hops == 1
        assert mem.decomposer.crop_quality_gate is True
        assert mem.curator.max_total_representations == 64
        # Persistence closes the loop: a segment run must write the bank to disk.
        mem.run_segment("Nothing named here.", segment_id=0, skip_generate=True)
        assert Path(persist).exists()


def test_attribute_classifier_skipped_when_angles_known() -> None:
    class _CountingClassifier:
        def __init__(self) -> None:
            self.calls = 0

        def classify(self, image_path: str, **_: object):  # type: ignore[no-untyped-def]
            from memstrata.mllm.crop_attributes import CropAttributePack

            self.calls += 1
            return CropAttributePack(source="counting")

    with tempfile.TemporaryDirectory() as tmp:
        crop = str(Path(tmp) / "hero.jpg")
        Path(crop).write_bytes(b"hero-crop-bytes")

        spy = _CountingClassifier()
        curator = AssetCurator(
            AssetBank(),
            HashEmbedding(),
            crop_attribute_classifier=spy,
            attributes_when_angles_known=False,
        )
        curator.ingest_observation(
            EntityObservation(
                "hero@s0",
                AssetType.CHARACTER,
                "Hero",
                crop,
                entity_id="char_hero",
                spatial_angle=SpatialAngle.FRONT,
                state_angle=StateAngle.DEFAULT,
            ),
            segment_id=0,
        )
        assert spy.calls == 0


if __name__ == "__main__":
    test_intent_parses_explicit_view_and_state()
    test_compose_excludes_current_and_future_evidence()
    test_compose_expands_explicit_structural_relation()
    test_pipeline_derives_observation_targets_from_intent()
    test_crop_qa_rejects_unreadable_external_input()
    test_curate_writes_explicit_relations_for_expansion()
    test_ingest_packet_skips_malformed_row()
    test_state_event_builds_replacement_chain()
    test_global_budget_evicts_weakest_but_keeps_one_per_asset()
    test_bank_save_load_roundtrip_with_relations()
    test_for_production_preset_enables_optin_knobs()
    test_attribute_classifier_skipped_when_angles_known()
    print("test_production_loop_contracts passed")
