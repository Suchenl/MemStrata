"""Assert-based production curation tests (name-anchored identity + redundancy)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from memstrata.steps.curate import EntityObservation, InverseIngester
from memstrata.bank import AssetKind, ProductionAssetSpace
from memstrata.encoders import HashEmbedding, cosine_similarity


def _write_crop(directory: Path, name: str, content: bytes) -> str:
    path = directory / name
    path.write_bytes(content)
    return str(path)


def test_same_name_merges_into_one_asset() -> None:
    """Paper Step 4: identity is name-anchored — same name+kind attaches to one asset."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop_a = _write_crop(root, "a.jpg", b"view-a")
        crop_b = _write_crop(root, "b.jpg", b"view-b-different-bytes")

        asset_space = ProductionAssetSpace()
        ingester = InverseIngester(asset_space, HashEmbedding(), redundancy_threshold=0.99)

        id_1 = ingester.ingest_observation(
            EntityObservation("obs001", AssetKind.CHARACTER, "Hero", crop_a),
            segment_id=0,
        )
        id_2 = ingester.ingest_observation(
            EntityObservation("obs002", AssetKind.CHARACTER, "Hero", crop_b),
            segment_id=1,
        )

        assert id_1 == id_2
        assert len(asset_space.assets) == 1


def test_different_names_create_separate_assets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop_a = _write_crop(root, "a.jpg", b"image-alpha-bytes-unique")
        crop_b = _write_crop(root, "b.jpg", b"totally-different-image-beta")

        asset_space = ProductionAssetSpace()
        ingester = InverseIngester(asset_space, HashEmbedding())

        id_a = ingester.ingest_observation(
            EntityObservation("obs_a", AssetKind.PROP, "ItemA", crop_a),
            segment_id=0,
        )
        id_b = ingester.ingest_observation(
            EntityObservation("obs_b", AssetKind.PROP, "ItemB", crop_b),
            segment_id=0,
        )

        assert id_a != id_b
        assert len(asset_space.assets) == 2


def test_near_duplicate_rep_discarded() -> None:
    """Same crop twice → second representation discarded as redundant."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop = _write_crop(root, "same.jpg", b"identical-image-bytes")

        asset_space = ProductionAssetSpace()
        ingester = InverseIngester(asset_space, HashEmbedding(), redundancy_threshold=0.99)

        ingester.ingest_observation(
            EntityObservation("obs001", AssetKind.CHARACTER, "Hero", crop),
            segment_id=0,
        )
        ingester.ingest_observation(
            EntityObservation("obs002", AssetKind.CHARACTER, "Hero", crop),
            segment_id=1,
        )

        asset = next(iter(asset_space.assets.values()))
        assert len(asset.representations) == 1


def test_cosine_helper_self_similarity() -> None:
    embedder = HashEmbedding()
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_crop(Path(tmp), "img.jpg", b"content-0")
        vector = embedder.embed_image(path)
        assert abs(cosine_similarity(vector, vector) - 1.0) < 1e-6


if __name__ == "__main__":
    test_same_name_merges_into_one_asset()
    print("test_same_name_merges_into_one_asset passed")
    test_different_names_create_separate_assets()
    print("test_different_names_create_separate_assets passed")
    test_near_duplicate_rep_discarded()
    print("test_near_duplicate_rep_discarded passed")
    test_cosine_helper_self_similarity()
    print("test_cosine_helper_self_similarity passed")
    print("all production curation tests passed")
