"""Minimal pipeline smoke: Step1 → Step4 with oracle observations (no generator)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from memstrata.steps.decompose import Observation
from memstrata.bank import AssetType
from memstrata.pipeline import MemStrata


def test_run_segment_oracle_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = Path(tmp) / "rabbit.jpg"
        crop.write_bytes(b"rabbit")

        mem = MemStrata()
        # Seed bank so Step 1 can resolve the name.
        from memstrata.steps.curate import AssetCurator
        from memstrata.encoders import HashEmbedding

        curator = AssetCurator(mem.bank, HashEmbedding())
        curator.ingest_packet({
            "segment_id": 0,
            "observations": [{
                "entity_id": "char_rabbit_01",
                "kind": "character",
                "name": "Rabbit",
                "description": "white rabbit",
                "crop_path": str(crop),
                "representation_id": "char_rabbit_01@s000",
            }],
            "state_events": [],
        })

        result = mem.run_segment(
            "The Rabbit hops.",
            segment_id=1,
            skip_generate=True,
            oracle_observations=[
                Observation(
                    observation_id="char_rabbit_01@s001",
                    kind=AssetType.CHARACTER,
                    name="Rabbit",
                    image_path=str(crop),
                    entity_id="char_rabbit_01",
                )
            ],
        )
        assert "char_rabbit_01" in result.context.asset_ids
        assert result.touched_asset_ids == ["char_rabbit_01"]
        rabbit = mem.bank.get_asset("char_rabbit_01")
        assert rabbit is not None
        # Second obs may be discarded as near-dup of same crop; bank still has the asset.
        assert len(rabbit.representations) >= 1


if __name__ == "__main__":
    test_run_segment_oracle_path()
    print("test_run_segment_oracle_path passed")
