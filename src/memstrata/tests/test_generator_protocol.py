"""Generator protocol + full MemStrata loop with MediaTaskGenerator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from memstrata.bank import AssetBank, AssetType
from memstrata.encoders import HashEmbedding
from memstrata.pipeline import MemStrata
from memstrata.steps.curate import AssetCurator
from memstrata.steps.decompose import Observation
from memstrata.steps.generate import MediaTaskGenerator, composed_reference_images
from memstrata.steps.generate.backends import RecordingBackend


def _seed(bank: AssetBank, crop_a: Path, crop_b: Path) -> None:
    AssetCurator(bank, HashEmbedding()).ingest_packet({
        "segment_id": -1,
        "observations": [
            {
                "entity_id": "char_rabbit_01",
                "kind": "character",
                "name": "Rabbit",
                "description": "white rabbit",
                "crop_path": str(crop_a),
                "representation_id": "char_rabbit_01@seed",
            },
            {
                "entity_id": "prop_apple_01",
                "kind": "prop",
                "name": "Apple",
                "description": "red apple",
                "crop_path": str(crop_b),
                "representation_id": "prop_apple_01@seed",
            },
        ],
        "state_events": [],
    })


def test_media_task_shape_and_continuation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop_a = root / "rabbit.jpg"
        crop_b = root / "apple.jpg"
        crop_a.write_bytes(b"rabbit")
        crop_b.write_bytes(b"apple")
        run_dir = root / "run"

        bank = AssetBank()
        _seed(bank, crop_a, crop_b)
        backend = RecordingBackend(root / "media")
        gen = MediaTaskGenerator(
            backend,
            bank=bank,
            model_name="recording",
            default_controls={"transition": "continue", "duration_sec": 2.0},
            log_dir=run_dir / "gen",
        )
        mem = MemStrata(bank=bank, generator=gen, run_dir=run_dir / "pipe")

        r0 = mem.run_segment(
            "The Rabbit finds an Apple.",
            segment_id=0,
            generation_controls={"transition": "cut", "source_video": str(crop_a)},
            oracle_observations=[
                Observation("char_rabbit_01@s000", AssetType.CHARACTER, "Rabbit", str(crop_a), entity_id="char_rabbit_01"),
                Observation("prop_apple_01@s000", AssetType.PROP, "Apple", str(crop_b), entity_id="prop_apple_01"),
            ],
        )
        assert r0.generation is not None
        assert r0.generation.task is not None
        task0 = r0.generation.task
        assert task0.task_type.value == "video_segment"
        assert "composed_references" in task0.controls
        assert len(task0.controls["composed_references"]) >= 1
        assert all("image" in ref for ref in task0.controls["composed_references"])
        assert "continuation" not in task0.controls
        assert "char_rabbit_01" in r0.context.asset_ids

        r1 = mem.run_segment(
            "The Rabbit holds the Apple.",
            segment_id=1,
            generation_controls={"transition": "continue", "source_video": str(crop_a)},
            oracle_observations=[
                Observation("char_rabbit_01@s001", AssetType.CHARACTER, "Rabbit", str(crop_a), entity_id="char_rabbit_01"),
            ],
        )
        assert r1.generation is not None and r1.generation.task is not None
        assert "continuation" in r1.generation.task.controls
        cont = r1.generation.task.controls["continuation"]
        assert "source_video" in cont and "source_artifact_id" in cont
        assert r1.generation.meta.get("has_continuation") is True
        assert "char_rabbit_01" in mem.bank.assets
        assert len(mem.segment_log) == 2
        assert (run_dir / "pipe" / "run_ledger.json").is_file()
        ledger = json.loads((run_dir / "pipe" / "run_ledger.json").read_text())
        assert ledger["segments"][0]["selected_assets"]


def test_composed_reference_images_materialize() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crop = root / "r.jpg"
        crop.write_bytes(b"x")
        bank = AssetBank()
        _seed(bank, crop, crop)
        from memstrata.steps.compose import compose
        from memstrata.steps.intent import IntentInterpreter

        req, _ = IntentInterpreter(bank).interpret("Rabbit", segment_id=0)
        ctx = compose(bank, req)
        refs = composed_reference_images(ctx, bank)
        assert refs
        assert Path(refs[0]["image"]).is_file()


if __name__ == "__main__":
    test_media_task_shape_and_continuation()
    print("test_media_task_shape_and_continuation passed")
    test_composed_reference_images_materialize()
    print("test_composed_reference_images_materialize passed")
    print("all generator protocol tests passed")
