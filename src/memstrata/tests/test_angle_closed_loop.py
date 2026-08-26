"""Production closed loop: classify crop angles → store [image + angle] → diversify R_j."""

from __future__ import annotations

import tempfile
from pathlib import Path

from memstrata.bank import AssetBank, AssetType, SpatialAngle, StateAngle
from memstrata.encoders import HashEmbedding
from memstrata.mllm.angle_classifier import HeuristicAngleClassifier, NullAngleClassifier
from memstrata.pipeline import MemStrata
from memstrata.steps.curate import AssetCurator, EntityObservation
from memstrata.steps.decompose import NamedEntity, RoleAwareDecomposer
from memstrata.lib.dedup import select_angle_diverse


def _write(directory: Path, name: str, content: bytes) -> str:
    path = directory / name
    path.write_bytes(content)
    return str(path)


def test_heuristic_classifier_reads_filename() -> None:
    clf = HeuristicAngleClassifier()
    hit = clf.classify("/tmp/hero_front_default.jpg")
    assert hit.spatial_angle == SpatialAngle.FRONT
    assert hit.state_angle == StateAngle.DEFAULT
    assert hit.source == "heuristic"


def test_decompose_classifies_unknown_angles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop = _write(root, "char_side_changed.jpg", b"side-view-bytes")
        decomposer = RoleAwareDecomposer(
            embedder=HashEmbedding(),
            angle_classifier=HeuristicAngleClassifier(),
        )
        obs = decomposer.decompose(
            segment_id=0,
            named_entities=[
                NamedEntity(
                    name="Hero",
                    kind=AssetType.CHARACTER,
                    entity_id="char_hero",
                    crop_path=crop,
                )
            ],
        )
        assert len(obs) == 1
        assert obs[0].spatial_angle == SpatialAngle.SIDE
        assert obs[0].state_angle == StateAngle.CHANGED
        assert obs[0].angle_meta.get("angle_source") == "heuristic"


def test_explicit_angles_not_overwritten() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop = _write(root, "char_side_changed.jpg", b"bytes")
        decomposer = RoleAwareDecomposer(
            embedder=HashEmbedding(),
            angle_classifier=HeuristicAngleClassifier(),
        )
        obs = decomposer.decompose(
            segment_id=0,
            named_entities=[
                NamedEntity(
                    name="Hero",
                    kind=AssetType.CHARACTER,
                    crop_path=crop,
                    spatial_angle=SpatialAngle.FRONT,
                    state_angle=StateAngle.DEFAULT,
                )
            ],
        )
        assert obs[0].spatial_angle == SpatialAngle.FRONT
        assert obs[0].state_angle == StateAngle.DEFAULT
        assert obs[0].angle_meta.get("angle_source") == "explicit"


def test_ingest_keeps_distinct_angle_buckets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        front = _write(root, "hero_front_default.jpg", b"front-view")
        side = _write(root, "hero_side_default.jpg", b"side-view")
        bank = AssetBank()
        curator = AssetCurator(
            bank,
            HashEmbedding(),
            angle_classifier=HeuristicAngleClassifier(),
            redundancy_threshold=0.0,  # force embedding-near-dup path if angles ignored
        )
        curator.ingest_observation(
            EntityObservation("o0", AssetType.CHARACTER, "Hero", front),
            segment_id=0,
        )
        curator.ingest_observation(
            EntityObservation("o1", AssetType.CHARACTER, "Hero", side),
            segment_id=1,
        )
        asset = next(iter(bank.assets.values()))
        angles = {rep.spatial_angle for rep in asset.representations}
        assert SpatialAngle.FRONT in angles
        assert SpatialAngle.SIDE in angles
        assert len(asset.representations) == 2


def test_same_angle_bucket_not_duplicated() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = _write(root, "hero_front_default_a.jpg", b"front-a-unique")
        b = _write(root, "hero_front_default_b.jpg", b"front-b-unique-bytes")
        bank = AssetBank()
        curator = AssetCurator(
            bank,
            HashEmbedding(),
            angle_classifier=HeuristicAngleClassifier(),
            redundancy_threshold=0.99,
        )
        curator.ingest_observation(
            EntityObservation("o0", AssetType.CHARACTER, "Hero", a),
            segment_id=0,
        )
        curator.ingest_observation(
            EntityObservation("o1", AssetType.CHARACTER, "Hero", b),
            segment_id=1,
        )
        asset = next(iter(bank.assets.values()))
        assert len(asset.representations) == 1
        assert asset.representations[0].spatial_angle == SpatialAngle.FRONT


def test_select_angle_diverse_covers_buckets() -> None:
    kept = select_angle_diverse(
        spatial_angles=["front", "front", "side", "back"],
        state_angles=["default", "default", "default", "default"],
        vectors=None,
        quality=[0.5, 0.9, 0.8, 0.7],
        max_keep=3,
    )
    # front keeps higher quality index 1; plus side and back.
    assert set(kept) == {1, 2, 3}


def test_pipeline_wires_classifier() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop = _write(root, "rabbit_front_default.jpg", b"rabbit")
        sut = MemStrata(
            angle_classifier=HeuristicAngleClassifier(),
            embedder=HashEmbedding(),
        )
        result = sut.run_segment(
            "A rabbit.",
            segment_id=0,
            skip_generate=True,
            named_entities=[
                NamedEntity(
                    name="Rabbit",
                    kind=AssetType.CHARACTER,
                    entity_id="char_rabbit",
                    crop_path=crop,
                )
            ],
        )
        assert result.observations[0].spatial_angle == SpatialAngle.FRONT
        asset = sut.bank.get_asset("char_rabbit")
        assert asset is not None
        assert asset.representations[0].spatial_angle == SpatialAngle.FRONT


def test_packet_ingest_skips_classifier() -> None:
    """Track A: packet angles authoritative; Null classifier must not invent angles."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop = _write(root, "apple_front_default.jpg", b"apple")
        bank = AssetBank()
        curator = AssetCurator(
            bank,
            HashEmbedding(),
            angle_classifier=HeuristicAngleClassifier(),
        )
        curator.ingest_packet(
            {
                "segment_id": 0,
                "observations": [
                    {
                        "entity_id": "prop_apple",
                        "kind": "prop",
                        "name": "Apple",
                        "crop_path": crop,
                        "representation_id": "prop_apple@s000",
                        # no angles → stay unknown even though filename has hints
                    }
                ],
                "state_events": [],
            }
        )
        asset = bank.get_asset("prop_apple")
        assert asset is not None
        assert asset.representations[0].spatial_angle == SpatialAngle.UNKNOWN
        assert asset.representations[0].annotations.get("angle_source") == "packet"


def test_null_classifier_default() -> None:
    clf = NullAngleClassifier()
    out = clf.classify("/tmp/anything.jpg")
    assert out.spatial_angle == SpatialAngle.UNKNOWN
    assert out.source == "null"


if __name__ == "__main__":
    test_heuristic_classifier_reads_filename()
    test_decompose_classifies_unknown_angles()
    test_explicit_angles_not_overwritten()
    test_ingest_keeps_distinct_angle_buckets()
    test_same_angle_bucket_not_duplicated()
    test_select_angle_diverse_covers_buckets()
    test_pipeline_wires_classifier()
    test_packet_ingest_skips_classifier()
    test_null_classifier_default()
    print("all angle closed-loop tests passed")
