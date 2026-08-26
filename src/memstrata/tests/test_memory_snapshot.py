"""Unit test for the memory snapshot exporter (dependency-light, assert-based)."""

import json
import tempfile
from pathlib import Path

from memstrata.skills.memory_update.snapshot import export_memory_snapshot
from memstrata.bank.schema import (
    Asset,
    AssetBank,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    StateAngle,
)


def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n dummy")


def _build_bank(src_dir: Path):
    """One CHARACTER asset (2 reps / 2 states) + one LOCATION asset (1 rep)."""
    char_default = src_dir / "eli_default.png"
    char_changed = src_dir / "eli_changed.png"
    loc_img = src_dir / "lighthouse.png"
    for img in (char_default, char_changed, loc_img):
        _write_dummy_image(img)

    char = Asset(
        asset_id="character_eli",
        kind=AssetType.CHARACTER,
        name="Elias",
        status=LifecycleStatus.REUSABLE,
        description="A weathered lighthouse keeper.",
        representations=[
            AssetRepresentation(
                representation_id="character_eli@s000",
                asset_id="character_eli",
                object_uri=str(char_default),
                origin_segment_id=0,
                state_angle=StateAngle.DEFAULT,
                annotations={"observation_description": "middle-aged, dark coat"},
            ),
            AssetRepresentation(
                representation_id="character_eli@s005",
                asset_id="character_eli",
                object_uri=str(char_changed),
                origin_segment_id=5,
                state_angle=StateAngle.CHANGED,
                annotations={"observation_description": "aged, white hair"},
            ),
        ],
    )
    loc = Asset(
        asset_id="location_lighthouse",
        kind=AssetType.LOCATION,
        name="The Lighthouse",
        status=LifecycleStatus.REUSABLE,
        description="A white granite tower on a black cape.",
        representations=[
            AssetRepresentation(
                representation_id="location_lighthouse@s000",
                asset_id="location_lighthouse",
                object_uri=str(loc_img),
                origin_segment_id=0,
                state_angle=StateAngle.DEFAULT,
            ),
        ],
    )
    bank = AssetBank()
    bank.add_asset(char)
    bank.add_asset(loc)
    return bank


def test_export_memory_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        bank = _build_bank(src_dir)

        rep_seconds = {
            "character_eli@s000": 12.0,
            "character_eli@s005": 88.5,
            "location_lighthouse@s000": 12.0,
        }

        memory_path = export_memory_snapshot(
            bank,
            out_dir,
            movie_id="0001_lighthouse_keeper",
            fps=24.0,
            rep_seconds=rep_seconds,
        )

        # memory.json exists + parses.
        assert memory_path.exists()
        assert memory_path == out_dir / "memory.json"
        data = json.loads(memory_path.read_text(encoding="utf-8"))

        # Header.
        assert data["schema"] == "memstrata-memory-1.0"
        assert data["movie_id"] == "0001_lighthouse_keeper"
        assert data["fps"] == 24.0
        # updated_sec defaults to the max known rep second.
        assert data["updated_sec"] == 88.5

        entities = data["entities"]
        assert set(entities) == {"character_eli", "location_lighthouse"}

        # Character entity: right kind / name / description, and BOTH states present.
        char = entities["character_eli"]
        assert char["kind"] == "character"
        assert char["name"] == "Elias"
        assert char["description"] == "A weathered lighthouse keeper."
        assert set(char["states"]) == {"default", "changed"}
        assert char["first_seen_sec"] == 12.0

        # Per-state first_seen_sec matches expected.
        assert char["states"]["default"]["first_seen_sec"] == 12.0
        assert char["states"]["changed"]["first_seen_sec"] == 88.5

        # Appearances secs match and are sorted by (sec, segment).
        default_apps = char["states"]["default"]["appearances"]
        changed_apps = char["states"]["changed"]["appearances"]
        assert [a["sec"] for a in default_apps] == [12.0]
        assert [a["sec"] for a in changed_apps] == [88.5]
        assert default_apps[0]["segment"] == 0
        assert changed_apps[0]["segment"] == 5

        # Location entity present with a single default state.
        loc = entities["location_lighthouse"]
        assert loc["kind"] == "location"
        assert loc["name"] == "The Lighthouse"
        assert loc["description"] == "A white granite tower on a black cape."
        assert set(loc["states"]) == {"default"}

        # Images: relative POSIX paths that exist under the visual tree; copies non-empty.
        char_images = char["states"]["default"]["images"] + char["states"]["changed"]["images"]
        loc_images = loc["states"]["default"]["images"]
        assert len(char_images) == 2
        assert len(loc_images) == 1

        for rel in char_images:
            assert rel.startswith("visual/characters/")
            assert "/" in rel and "\\" not in rel  # POSIX relative
            abs_path = out_dir / rel
            assert abs_path.exists()
            assert abs_path.stat().st_size > 0

        for rel in loc_images:
            assert rel.startswith("visual/locations/")
            abs_path = out_dir / rel
            assert abs_path.exists()
            assert abs_path.stat().st_size > 0

        # No video header unless requested.
        assert "video" not in data


def test_export_memory_snapshot_records_long_video():
    """The sibling grown film is recorded as a relative path + duration."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_dir = tmp_path / "out"
        bank = _build_bank(tmp_path / "src")
        # The film sits inside out_dir (created so the relative path resolves).
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "long_video.mp4").write_bytes(b"\x00\x00")

        memory_path = export_memory_snapshot(
            bank,
            out_dir,
            movie_id="0001_lighthouse_keeper",
            video_path=str(out_dir / "long_video.mp4"),
            video_duration_sec=88.5,
        )
        data = json.loads(memory_path.read_text(encoding="utf-8"))
        assert data["video"] == {"path": "long_video.mp4", "duration_sec": 88.5}
