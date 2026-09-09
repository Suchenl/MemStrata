"""Location-specific scene admission and conservative resolver seams."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from memstrata.bank import (
    AssetBank,
    AssetType,
    RelationType,
    SpatialAngle,
    StateAngle,
)
from memstrata.encoders import HashEmbedding
from memstrata.production.realized import build_realized_segment_pipeline
from memstrata.skills.crop_acquisition.orchestrator import acquire_entity_crop
from memstrata.skills.memory_update.location_resolver import (
    LocationResolutionAction,
    LocationResolutionEvidence,
    LocationSemanticRelation,
    propose_lexical_location_relation,
    propose_location_resolution,
)
from memstrata.steps.curate import AssetCurator, EntityObservation, MemoryPolicy


_LOCATION = AssetType.LOCATION
_CHARACTER = AssetType.CHARACTER
_SCENE_REFERENCE = "scene_reference"
_IDENTITY_ANCHOR = "identity_anchor"


def _crop(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    image = Image.new("RGB", (48, 48))
    seed = sum(name.encode("utf-8")) % 256
    image.putdata(
        [
            (
                (i * 7 + seed) % 256,
                (i * 13 + seed * 3) % 256,
                (i * 29 + seed * 5) % 256,
            )
            for i in range(48 * 48)
        ]
    )
    image.save(path)
    return str(path)


def _curator(
    *,
    scene_enabled: bool = True,
    resolver_shadow: bool = False,
    resolver_enabled: bool = False,
    embed_on_ingest: bool = False,
) -> AssetCurator:
    return AssetCurator(
        AssetBank(),
        HashEmbedding(),
        policy=MemoryPolicy(
            location_scene_validity_enabled=scene_enabled,
            location_resolver_enabled=resolver_enabled,
            location_resolver_shadow_enabled=resolver_shadow,
        ),
        embed_on_ingest=embed_on_ingest,
        dark_gate=False,
        attributes_when_angles_known=False,
    )


def _ingest(
    curator: AssetCurator,
    *,
    crop: str,
    kind: AssetType,
    name: str,
    segment_id: int,
    meta: dict,
    spatial_angle: SpatialAngle = SpatialAngle.FRONT,
    entity_id: str | None = None,
) -> str:
    return curator.ingest_observation(
        EntityObservation(
            f"{name}-{segment_id}",
            kind,
            name,
            crop,
            entity_id=entity_id,
            spatial_angle=spatial_angle,
            state_angle=StateAngle.DEFAULT,
            angle_meta=meta,
        ),
        segment_id=segment_id,
    )


def test_low_foreground_scene_is_eligible_without_identity_visibility(
    tmp_path: Path,
) -> None:
    curator = _curator()
    _ingest(
        curator,
        crop=_crop(tmp_path, "river.png"),
        kind=_LOCATION,
        name="River",
        segment_id=0,
        meta={
            "crop_attributes": {
                "identity_visible": False,
                "shot_size": "wide",
            },
            "scene_validity_evidence": {
                "foreground_max_coverage": 0.05,
                "foreground_union_coverage": 0.10,
                "crop_area_fraction": 0.80,
                # Deliberately not K=3: support is a ratio over any sample count.
                "place_support_count": 3,
                "place_observation_count": 5,
                "foreground_source": "cached_detector",
                "place_source": "cached_vpr",
            },
        },
    )

    rep = curator.bank.find_by_name("River", kind=_LOCATION).representations[0]
    assert rep.annotations["crop_attributes"]["identity_visible"] is False
    assert rep.annotations["scene_validity"]["status"] == "accept"
    assert rep.annotations["scene_reference_eligible"] is True
    assert _SCENE_REFERENCE in rep.reference_aspects


def test_foreground_dominant_location_is_rejected(tmp_path: Path) -> None:
    curator = _curator()
    _ingest(
        curator,
        crop=_crop(tmp_path, "legs.png"),
        kind=_LOCATION,
        name="Stage",
        segment_id=0,
        meta={
            "crop_attributes": {
                "identity_visible": True,
                "shot_size": "wide",
            },
            "scene_validity_evidence": {
                "foreground_max_coverage": 0.75,
                "foreground_union_coverage": 0.82,
                "crop_area_fraction": 0.90,
                "foreground_source": "cached_detector",
            },
        },
    )

    asset = curator.bank.find_by_name("Stage", kind=_LOCATION)
    assert asset is not None
    assert asset.representations == []


def test_uncertain_foreground_is_quarantined_but_retained(tmp_path: Path) -> None:
    curator = _curator()
    _ingest(
        curator,
        crop=_crop(tmp_path, "people_in_forest.png"),
        kind=_LOCATION,
        name="Forest",
        segment_id=0,
        meta={
            "scene_validity_evidence": {
                "foreground_max_coverage": 0.45,
                "foreground_union_coverage": 0.50,
                "crop_area_fraction": 0.85,
            }
        },
    )

    rep = curator.bank.find_by_name("Forest", kind=_LOCATION).representations[0]
    assert rep.annotations["scene_validity"]["status"] == "quarantine"
    assert rep.annotations["scene_reference_eligible"] is False
    assert _SCENE_REFERENCE not in rep.reference_aspects
    assert _SCENE_REFERENCE in rep.excluded_aspects


def test_scene_plate_candidate_without_evidence_is_quarantined(
    tmp_path: Path,
) -> None:
    frame = _crop(tmp_path, "whole_frame.png")
    acquired = acquire_entity_crop(
        frame,
        entity_name="Cinema",
        entity_kind="location",
        exemplar_vectors=[],
        existing_rep_vectors=[],
        out_dir=tmp_path / "acquired",
        location_scene_plate_candidates=True,
    )
    assert acquired is not None
    assert acquired["source"] == "whole_frame_location_candidate"
    assert acquired["scene_candidate_only"] is True

    curator = _curator()
    _ingest(
        curator,
        crop=acquired["crop_path"],
        kind=_LOCATION,
        name="Cinema",
        segment_id=0,
        meta={
            "bbox": acquired["bbox"],
            "crop_acquisition": {
                "source": acquired["source"],
                "scene_candidate_only": acquired["scene_candidate_only"],
            },
        },
    )

    rep = curator.bank.find_by_name("Cinema", kind=_LOCATION).representations[0]
    assert rep.annotations["scene_validity"]["status"] == "quarantine"
    assert rep.annotations["scene_validity"]["reasons"] == [
        "missing_foreground_evidence"
    ]
    assert _SCENE_REFERENCE not in rep.reference_aspects


def test_location_scene_gate_does_not_change_character_identity_gate(
    tmp_path: Path,
) -> None:
    curator = _curator()
    _ingest(
        curator,
        crop=_crop(tmp_path, "hero_back.png"),
        kind=_CHARACTER,
        name="Hero",
        segment_id=0,
        meta={
            "crop_attributes": {
                "identity_visible": False,
                "occlusion": "none",
            },
            "scene_validity_evidence": {
                "foreground_max_coverage": 0.0,
                "crop_area_fraction": 1.0,
            },
        },
    )

    rep = curator.bank.find_by_name("Hero", kind=_CHARACTER).representations[0]
    assert "scene_validity" not in rep.annotations
    assert rep.annotations["identity_anchor_eligible"] is False
    assert _IDENTITY_ANCHOR not in rep.reference_aspects


def test_default_policy_preserves_legacy_location_behavior(tmp_path: Path) -> None:
    curator = _curator(scene_enabled=False)
    _ingest(
        curator,
        crop=_crop(tmp_path, "legacy_location.png"),
        kind=_LOCATION,
        name="Meadow",
        segment_id=0,
        meta={
            "crop_attributes": {
                "identity_visible": True,
                "occlusion": "none",
            }
        },
    )

    rep = curator.bank.find_by_name("Meadow", kind=_LOCATION).representations[0]
    assert "scene_validity" not in rep.annotations
    assert "scene_reference_eligible" not in rep.annotations
    assert _SCENE_REFERENCE in rep.reference_aspects


def test_realized_builder_wires_opt_in_location_seams(tmp_path: Path) -> None:
    mem = build_realized_segment_pipeline(
        run_dir=tmp_path,
        profile="production",
        write_naming="perception",
        embedder_provider="hash",
        location_scene_validity_enabled=True,
        location_resolver_shadow_enabled=True,
        location_scene_plate_candidates=True,
    )
    assert mem.curator.location_scene_validity_enabled is True
    assert mem.curator.location_resolver_shadow_enabled is True
    assert (
        mem.decomposer.cropper.extra_acquire_kwargs[
            "location_scene_plate_candidates"
        ]
        is True
    )


def test_structural_location_relations_never_propose_identity_merge() -> None:
    for relation in (
        LocationSemanticRelation.PART_OF,
        LocationSemanticRelation.ADJACENT_TO,
        LocationSemanticRelation.INTERIOR_OF,
    ):
        proposal = propose_location_resolution(
            incoming_name="Forest Path",
            candidate_name="Forest",
            evidence=LocationResolutionEvidence(
                candidate_asset_id="location_forest",
                semantic_relation=relation,
                visual_similarity=0.99,
                temporally_continuous=True,
                independent_support=10,
            ),
        )
        assert proposal.action is LocationResolutionAction.RELATE
        assert proposal.relation_type == relation.value
        assert proposal.observed_alias is None


def test_unrelated_location_surfaces_do_not_infer_structural_edges() -> None:
    evidence = LocationResolutionEvidence(
        candidate_asset_id="candidate",
        temporally_continuous=True,
        independent_support=2,
    )
    for incoming, candidate in (
        ("Room", "Building"),
        ("Store", "Street"),
    ):
        assert (
            propose_lexical_location_relation(
                incoming_name=incoming,
                candidate_name=candidate,
                evidence=evidence,
            )
            is None
        )


def test_strict_synonym_requires_visual_and_causal_support() -> None:
    unsupported = propose_location_resolution(
        incoming_name="Riverside",
        candidate_name="Riverbank",
        evidence=LocationResolutionEvidence(
            candidate_asset_id="location_riverbank",
            semantic_relation=LocationSemanticRelation.STRICT_SYNONYM,
            visual_similarity=0.95,
        ),
    )
    supported = propose_location_resolution(
        incoming_name="Riverside",
        candidate_name="Riverbank",
        evidence=LocationResolutionEvidence(
            candidate_asset_id="location_riverbank",
            semantic_relation=LocationSemanticRelation.STRICT_SYNONYM,
            visual_similarity=0.95,
            independent_support=2,
        ),
    )
    assert unsupported.action is LocationResolutionAction.DEFER
    assert supported.action is LocationResolutionAction.MERGE_ALIAS
    assert supported.observed_alias == "Riverside"


def test_same_location_name_without_evidence_is_shadow_deferred(
    tmp_path: Path,
) -> None:
    curator = _curator(scene_enabled=False, resolver_shadow=True)
    first = _crop(tmp_path, "forest_first.png")
    second = _crop(tmp_path, "forest_second.png")
    _ingest(
        curator,
        crop=first,
        kind=_LOCATION,
        name="Forest",
        segment_id=0,
        meta={},
    )
    _ingest(
        curator,
        crop=second,
        kind=_LOCATION,
        name="Forest",
        segment_id=1,
        meta={},
        spatial_angle=SpatialAngle.SIDE,
    )

    asset = curator.bank.find_by_name("Forest", kind=_LOCATION)
    assert len(asset.representations) == 2
    proposal = asset.representations[-1].annotations["location_resolution_proposal"]
    assert proposal["action"] == "defer"
    assert proposal["reasons"] == ["same_name_not_identity_evidence"]
    assert proposal["shadow_only"] is True


def test_exact_location_duplicate_reuses_canonical_with_visual_continuity(
    tmp_path: Path,
) -> None:
    curator = _curator(
        scene_enabled=False,
        resolver_enabled=True,
        embed_on_ingest=True,
    )
    crop = _crop(tmp_path, "same_prison.png")
    first_id = _ingest(
        curator,
        crop=crop,
        kind=_LOCATION,
        name="Prison",
        segment_id=1,
        meta={},
        entity_id="generated-location-1",
    )
    second_id = _ingest(
        curator,
        crop=crop,
        kind=_LOCATION,
        name="prison",
        segment_id=2,
        meta={},
        entity_id="generated-location-2",
    )

    assert first_id == second_id == "generated-location-1"
    assert len(curator.bank.list_assets(kind=_LOCATION)) == 1
    proposal = curator.bank.get_asset(first_id).metadata[
        "location_resolution_proposals_v2"
    ][-1]
    assert proposal["action"] == "reuse"
    assert proposal["applied"] is True
    assert proposal["temporal_gap"] == 1


def test_same_location_name_stays_separate_when_temporally_distant(
    tmp_path: Path,
) -> None:
    curator = _curator(
        scene_enabled=False,
        resolver_enabled=True,
        embed_on_ingest=True,
    )
    crop = _crop(tmp_path, "same_forest.png")
    first_id = _ingest(
        curator,
        crop=crop,
        kind=_LOCATION,
        name="Forest",
        segment_id=1,
        meta={},
        entity_id="forest-a",
    )
    second_id = _ingest(
        curator,
        crop=crop,
        kind=_LOCATION,
        name="Forest",
        segment_id=20,
        meta={},
        entity_id="forest-b",
    )

    assert first_id != second_id
    assert len(curator.bank.list_assets(kind=_LOCATION)) == 2
    proposal = curator.bank.get_asset(second_id).representations[-1].annotations[
        "location_resolution_proposal"
    ]
    assert proposal["action"] == "defer"
    assert proposal["applied"] is False


def test_qualified_location_creates_part_of_neighbor_without_alias_merge(
    tmp_path: Path,
) -> None:
    curator = _curator(
        scene_enabled=False,
        resolver_enabled=True,
        embed_on_ingest=True,
    )
    forest_id = _ingest(
        curator,
        crop=_crop(tmp_path, "forest.png"),
        kind=_LOCATION,
        name="Forest",
        segment_id=1,
        meta={},
        entity_id="forest",
    )
    trail_id = _ingest(
        curator,
        crop=_crop(tmp_path, "forest_trail.png"),
        kind=_LOCATION,
        name="Forest Trail",
        segment_id=2,
        meta={},
        entity_id="forest-trail",
    )

    assert forest_id != trail_id
    forest = curator.bank.get_asset(forest_id)
    trail = curator.bank.get_asset(trail_id)
    assert forest.metadata.get("aliases") is None
    assert trail.metadata.get("aliases") is None
    assert [
        (relation.relation_type, relation.target_asset_id)
        for relation in trail.relations
    ] == [(RelationType.PART_OF, forest_id)]
    record = trail.representations[-1].annotations[
        "location_relation_proposals"
    ][0]
    assert record["applied"] is True
    assert record["read_neighbor"] is True
