"""The production memory bank is a portable deliverable at the run root."""

import json
from pathlib import Path

from memstrata.bank.schema import (
    Asset,
    AssetBank,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    StateAngle,
)
from memstrata.pipeline import MemStrata
from memstrata.production.run import _anchor_membank_to_film


def _bank_with_reps(tmp_path: Path) -> AssetBank:
    image = tmp_path / "crop.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n dummy")
    bank = AssetBank()
    bank.add_asset(
        Asset(
            asset_id="Elias",
            kind=AssetType.CHARACTER,
            name="Elias",
            status=LifecycleStatus.REUSABLE,
            description="lighthouse keeper",
            representations=[
                AssetRepresentation(
                    representation_id="Elias@s000",
                    asset_id="Elias",
                    object_uri=str(image),
                    origin_segment_id=0,
                    state_angle=StateAngle.DEFAULT,
                ),
                AssetRepresentation(
                    representation_id="Elias@s002",
                    asset_id="Elias",
                    object_uri=str(image),
                    origin_segment_id=2,
                    state_angle=StateAngle.DEFAULT,
                ),
            ],
        )
    )
    return bank


def _mem(tmp_path: Path) -> MemStrata:
    return MemStrata(
        _bank_with_reps(tmp_path),
        run_dir=tmp_path / "run" / "pipeline",
        membank_dir=tmp_path / "run" / "membank",
        movie_id="0001_lighthouse_keeper",
    )


def test_snapshot_goes_to_membank_root_not_pipeline_dir(tmp_path: Path) -> None:
    mem = _mem(tmp_path)
    written = mem.write_memory_snapshot()
    assert written == tmp_path / "run" / "membank" / "memory.json"
    assert (tmp_path / "run" / "membank" / "visual").is_dir()
    assert not (tmp_path / "run" / "pipeline" / "memory.json").exists()


def test_run_dir_stays_the_target_when_no_membank_dir_given(tmp_path: Path) -> None:
    mem = MemStrata(_bank_with_reps(tmp_path), run_dir=tmp_path / "work")
    assert mem.write_memory_snapshot() == tmp_path / "work" / "memory.json"


def test_anchoring_copies_film_and_dates_appearances(tmp_path: Path) -> None:
    mem = _mem(tmp_path)
    film = tmp_path / "review_long.mp4"
    film.write_bytes(b"fake film bytes")

    _anchor_membank_to_film(
        mem,
        tmp_path / "run",
        {
            "long_video": str(film),
            "fps": 24.0,
            "duration_sec": 15.125,
            "segment_durations": [5.0, 5.0, 5.125],
        },
    )

    copied = tmp_path / "run" / "membank" / "long_video.mp4"
    assert copied.is_file()
    assert not copied.is_symlink()
    payload = json.loads((tmp_path / "run" / "membank" / "memory.json").read_text())
    assert payload["video"] == {"path": "long_video.mp4", "duration_sec": 15.125}
    appearances = payload["entities"]["Elias"]["states"]["default"]["appearances"]
    assert [(item["segment"], item["sec"]) for item in appearances] == [(0, 0.0), (2, 10.0)]
