"""Assert-based replay roundtrip tests (spec §7)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from memstrata.adapters.bench import BenchReplayAdapter
from memstrata.steps.compose import ActiveComposer
from memstrata.steps.curate import InverseIngester
from memstrata.bank import ProductionAssetSpace
from memstrata.encoders import HashEmbedding

SCHEMA_VERSION = "2.0.0"
CONTINUITY = "continuity"


def _write_crop(directory: Path, name: str, content: bytes) -> str:
    path = directory / name
    path.write_bytes(content)
    return str(path)


def _build_adapter() -> BenchReplayAdapter:
    asset_space = ProductionAssetSpace()
    embedder = HashEmbedding()
    ingester = InverseIngester(asset_space, embedder)
    composer = ActiveComposer(asset_space)
    return BenchReplayAdapter(asset_space, ingester, composer, planner=None)


def test_segment0_authoritative_ingest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rabbit_crop = _write_crop(root, "rabbit.jpg", b"rabbit-crop-bytes")
        apple_crop = _write_crop(root, "apple.jpg", b"apple-crop-bytes")

        adapter = _build_adapter()
        packet = {
            "schema_version": SCHEMA_VERSION,
            "segment_id": 0,
            "segment_video": "segment_000.mp4",
            "observations": [
                {
                    "entity_id": "char_rabbit_01",
                    "kind": "character",
                    "name": "Rabbit",
                    "representation_id": "rep_rabbit_c0",
                    "crop_path": rabbit_crop,
                    "description": "A fluffy white rabbit with pink ears.",
                },
                {
                    "entity_id": "prop_apple_01",
                    "kind": "prop",
                    "name": "Apple",
                    "representation_id": "rep_apple_c0",
                    "crop_path": apple_crop,
                    "description": "",
                },
            ],
            "state_events": [],
        }
        adapter.handle_observation(packet)

        rabbit = adapter.asset_space.get_asset("char_rabbit_01")
        apple = adapter.asset_space.get_asset("prop_apple_01")
        assert rabbit is not None
        assert apple is not None
        assert rabbit.asset_id == "char_rabbit_01"
        assert apple.asset_id == "prop_apple_01"
        assert rabbit.metadata.get("description") == "A fluffy white rabbit with pink ears."
        assert "description" not in apple.metadata or apple.metadata.get("description") == ""


def test_segment1_prompt_selects_rabbit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rabbit_crop = _write_crop(root, "rabbit.jpg", b"rabbit-crop-bytes")
        apple_crop = _write_crop(root, "apple.jpg", b"apple-crop-bytes")

        adapter = _build_adapter()
        adapter.handle_observation({
            "schema_version": SCHEMA_VERSION,
            "segment_id": 0,
            "segment_video": "segment_000.mp4",
            "observations": [
                {
                    "entity_id": "char_rabbit_01",
                    "kind": "character",
                    "name": "Rabbit",
                    "representation_id": "rep_rabbit_c0",
                    "crop_path": rabbit_crop,
                    "description": "A fluffy white rabbit with pink ears.",
                },
                {
                    "entity_id": "prop_apple_01",
                    "kind": "prop",
                    "name": "Apple",
                    "representation_id": "rep_apple_c0",
                    "crop_path": apple_crop,
                    "description": "",
                },
            ],
            "state_events": [],
        })

        record = adapter.handle_prompt({
            "schema_version": SCHEMA_VERSION,
            "segment_id": 1,
            "prompt": "The Rabbit hops through the meadow.",
        })

        selected_ids = {item["asset_id"] for item in record["selected"]}
        assert "char_rabbit_01" in selected_ids
        assert "prop_apple_01" not in selected_ids
        rabbit_instr = next(i for i in record["instruction"]["per_asset"] if i["asset_ref"] == "char_rabbit_01")
        assert rabbit_instr["requirement"] == CONTINUITY
        assert record["memory_keys"] == ["char_rabbit_01"]
        assert record["schema_version"] == SCHEMA_VERSION
        assert record["timing_ms"] >= 0.0


def test_segment1_state_event_deprecates_apple_rep() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rabbit_crop = _write_crop(root, "rabbit.jpg", b"rabbit-crop-bytes")
        apple_crop = _write_crop(root, "apple.jpg", b"apple-crop-bytes")

        adapter = _build_adapter()
        adapter.handle_observation({
            "schema_version": SCHEMA_VERSION,
            "segment_id": 0,
            "segment_video": "segment_000.mp4",
            "observations": [
                {
                    "entity_id": "char_rabbit_01",
                    "kind": "character",
                    "name": "Rabbit",
                    "representation_id": "rep_rabbit_c0",
                    "crop_path": rabbit_crop,
                    "description": "A fluffy white rabbit with pink ears.",
                },
                {
                    "entity_id": "prop_apple_01",
                    "kind": "prop",
                    "name": "Apple",
                    "representation_id": "rep_apple_c0",
                    "crop_path": apple_crop,
                    "description": "",
                },
            ],
            "state_events": [],
        })

        adapter.handle_observation({
            "schema_version": SCHEMA_VERSION,
            "segment_id": 1,
            "segment_video": "segment_001.mp4",
            "observations": [],
            "state_events": [
                {
                    "event_id": "evt_apple_eaten",
                    "segment_id": 1,
                    "description": "The apple is eaten.",
                    "deprecates": ["rep_apple_c0"],
                }
            ],
        })

        found = adapter.asset_space.find_representation("rep_apple_c0")
        assert found is not None
        _, rep = found
        assert rep.deprecated is True
        assert rep.deprecated_by == "evt_apple_eaten"


def test_segment2_prompt_excludes_deprecated_rep() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rabbit_crop = _write_crop(root, "rabbit.jpg", b"rabbit-crop-bytes")
        apple_crop = _write_crop(root, "apple.jpg", b"apple-crop-bytes")

        adapter = _build_adapter()
        adapter.handle_observation({
            "schema_version": SCHEMA_VERSION,
            "segment_id": 0,
            "segment_video": "segment_000.mp4",
            "observations": [
                {
                    "entity_id": "char_rabbit_01",
                    "kind": "character",
                    "name": "Rabbit",
                    "representation_id": "rep_rabbit_c0",
                    "crop_path": rabbit_crop,
                    "description": "A fluffy white rabbit with pink ears.",
                },
                {
                    "entity_id": "prop_apple_01",
                    "kind": "prop",
                    "name": "Apple",
                    "representation_id": "rep_apple_c0",
                    "crop_path": apple_crop,
                    "description": "",
                },
            ],
            "state_events": [],
        })
        adapter.handle_observation({
            "schema_version": SCHEMA_VERSION,
            "segment_id": 1,
            "segment_video": "segment_001.mp4",
            "observations": [],
            "state_events": [
                {
                    "event_id": "evt_apple_eaten",
                    "segment_id": 1,
                    "description": "The apple is eaten.",
                    "deprecates": ["rep_apple_c0"],
                }
            ],
        })

        record = adapter.handle_prompt({
            "schema_version": SCHEMA_VERSION,
            "segment_id": 2,
            "prompt": "Only the core of the Apple remains on the ground.",
        })

        assert "rep_apple_c0" in record["instruction"]["exclusions"]
        apple_selected = next(item for item in record["selected"] if item["asset_id"] == "prop_apple_01")
        assert "rep_apple_c0" not in apple_selected["representation_ids"]


def test_idempotent_reingest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rabbit_crop = _write_crop(root, "rabbit.jpg", b"rabbit-crop-bytes")

        adapter = _build_adapter()
        packet = {
            "schema_version": SCHEMA_VERSION,
            "segment_id": 0,
            "segment_video": "segment_000.mp4",
            "observations": [
                {
                    "entity_id": "char_rabbit_01",
                    "kind": "character",
                    "name": "Rabbit",
                    "representation_id": "rep_rabbit_c0",
                    "crop_path": rabbit_crop,
                    "description": "A fluffy white rabbit with pink ears.",
                }
            ],
            "state_events": [],
        }
        adapter.handle_observation(packet)
        adapter.handle_observation(packet)

        rabbit = adapter.asset_space.get_asset("char_rabbit_01")
        assert rabbit is not None
        assert len(rabbit.representations) == 1
        assert rabbit.representations[0].representation_id == "rep_rabbit_c0"


if __name__ == "__main__":
    test_segment0_authoritative_ingest()
    print("test_segment0_authoritative_ingest passed")
    test_segment1_prompt_selects_rabbit()
    print("test_segment1_prompt_selects_rabbit passed")
    test_segment1_state_event_deprecates_apple_rep()
    print("test_segment1_state_event_deprecates_apple_rep passed")
    test_segment2_prompt_excludes_deprecated_rep()
    print("test_segment2_prompt_excludes_deprecated_rep passed")
    test_idempotent_reingest()
    print("test_idempotent_reingest passed")
    print("all replay roundtrip tests passed")
