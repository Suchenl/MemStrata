"""The memory bank must be a self-contained deliverable at the run root.

A Track B run used to leave the bank inside ``<run>/pipeline/`` next to the per-segment debug
dumps, with a dead header: ``video: null``, empty ``movie_id``, ``fps: None`` and every ``sec``
None, because the producer never told the pipeline which film the timeline belongs to. These
tests pin the fixed contract: an own ``membank/`` root, a real (non-symlink) film inside it, and
timestamps on that film's timeline. (An unset film omits the ``video`` header entirely, which is
what the run's ``memory.json`` showed.)
"""

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
    img = tmp_path / "crop.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n dummy")
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
                    object_uri=str(img),
                    origin_segment_id=0,
                    state_angle=StateAngle.DEFAULT,
                ),
                AssetRepresentation(
                    representation_id="Elias@s002",
                    asset_id="Elias",
                    object_uri=str(img),
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


def test_snapshot_goes_to_membank_root_not_pipeline_dir(tmp_path):
    mem = _mem(tmp_path)
    written = mem.write_memory_snapshot()
    assert written == tmp_path / "run" / "membank" / "memory.json"
    assert (tmp_path / "run" / "membank" / "visual").is_dir()
    assert not (tmp_path / "run" / "pipeline" / "memory.json").exists()


def test_run_dir_stays_the_target_when_no_membank_dir_given(tmp_path):
    """Existing callers (e.g. the Track A adapter) keep their current layout."""
    mem = MemStrata(_bank_with_reps(tmp_path), run_dir=tmp_path / "work")
    assert mem.write_memory_snapshot() == tmp_path / "work" / "memory.json"


def test_seconds_are_none_until_the_producer_anchors_the_film(tmp_path):
    """An unknown timeline must stay explicitly unknown rather than be guessed."""
    mem = _mem(tmp_path)
    payload = json.loads(mem.write_memory_snapshot().read_text())
    assert "video" not in payload
    appearances = payload["entities"]["Elias"]["states"]["default"]["appearances"]
    assert [a["sec"] for a in appearances] == [None, None]


def test_anchoring_copies_a_real_film_and_dates_every_appearance(tmp_path):
    mem = _mem(tmp_path)
    mem.write_memory_snapshot()
    film = tmp_path / "review_long.mp4"
    film.write_bytes(b"fake film bytes")
    run_dir = tmp_path / "run"

    _anchor_membank_to_film(
        mem,
        run_dir,
        {"long_video": str(film), "fps": 24.0, "duration_sec": 15.125,
         "segment_durations": [5.0, 5.0, 5.125]},
    )

    copied = run_dir / "membank" / "long_video.mp4"
    assert copied.is_file()
    # The bank has to survive being moved on its own, so the film is a copy, not a link.
    assert not copied.is_symlink()
    assert copied.read_bytes() == b"fake film bytes"

    payload = json.loads((run_dir / "membank" / "memory.json").read_text())
    assert payload["movie_id"] == "0001_lighthouse_keeper"
    assert payload["fps"] == 24.0
    assert payload["video"] == {"path": "long_video.mp4", "duration_sec": 15.125}
    appearances = payload["entities"]["Elias"]["states"]["default"]["appearances"]
    # segment 0 starts at 0s, segment 2 starts after two 5s clips.
    assert [(a["segment"], a["sec"]) for a in appearances] == [(0, 0.0), (2, 10.0)]
    assert payload["entities"]["Elias"]["first_seen_sec"] == 0.0


def test_reanchoring_does_not_recopy_an_unchanged_film(tmp_path):
    mem = _mem(tmp_path)
    film = tmp_path / "review_long.mp4"
    film.write_bytes(b"fake film bytes")
    run_dir = tmp_path / "run"
    info = {"long_video": str(film), "fps": 24.0, "duration_sec": 5.0,
            "segment_durations": [5.0]}

    _anchor_membank_to_film(mem, run_dir, info)
    copied = run_dir / "membank" / "long_video.mp4"
    first_mtime = copied.stat().st_mtime_ns

    _anchor_membank_to_film(mem, run_dir, info)
    assert copied.stat().st_mtime_ns == first_mtime
