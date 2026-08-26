"""An interrupted story must continue from its persisted memory, not regenerate or duplicate it.

MemStrata's bank is an external artifact, so a run that died at shot 40 already paid for 40 shots of
identity. Resuming reopens that bank and replays only the shots that are actually missing.

Shots are addressed by id rather than counted because a segment the generator could not produce is
skipped WITHOUT writing a record. A count-based resume therefore restarts before the shot the run
really reached: it re-produces a shot that already exists and leaves the skipped one missing forever,
so the story ends with a duplicate plus a hole and every later shot is misaligned against the prompt
stream it is scored against. This was observed in production — one story finished "86/86" while
holding 85 unique shots, missing shot 1 and carrying a duplicate.
"""

from __future__ import annotations

import json
from pathlib import Path

from memstrata.production.run import _resume_state


def _write_progress(run_dir: Path, ids: list[int], total: int,
                    assets: dict[int, list[str]] | None = None) -> None:
    assets = assets or {}
    segments = [{"segment_id": i, "selected_assets": assets.get(i, ["x"]), "video": f"seg_{i}.mp4"}
                for i in ids]
    (run_dir / "progress.json").write_text(
        json.dumps({"done": len(ids), "total": total, "segments": segments}), encoding="utf-8")


def test_closed_shots_are_recovered_and_only_the_rest_is_queued(tmp_path: Path) -> None:
    _write_progress(tmp_path, list(range(40)), total=131)
    closed, todo = _resume_state(tmp_path, 131)
    assert sorted(closed) == list(range(40))
    assert todo == list(range(40, 131))


def test_a_skipped_shot_is_refilled_instead_of_leaving_a_hole(tmp_path: Path) -> None:
    """The defect this guards: shot 7 was skipped, so a count-based resume redid 39 and lost 7."""
    ids = [i for i in range(40) if i != 7]
    _write_progress(tmp_path, ids, total=131)
    closed, todo = _resume_state(tmp_path, 131)
    assert 7 in todo, "the skipped shot must be produced"
    assert 39 not in todo, "an already produced shot must not be produced again"
    assert todo[:2] == [7, 40]


def test_the_predecessor_of_a_hole_is_recoverable_for_continuity(tmp_path: Path) -> None:
    """Filling shot 13 needs shot 12's on-screen set, not the last-appended record's."""
    ids = [i for i in range(74) if i != 13]
    _write_progress(tmp_path, ids, total=119, assets={12: ["mara", "vitrine"], 73: ["late", "stuff"]})
    closed, _ = _resume_state(tmp_path, 119)
    assert list(closed[12]["selected_assets"]) == ["mara", "vitrine"]


def test_a_run_with_no_progress_file_produces_every_shot(tmp_path: Path) -> None:
    closed, todo = _resume_state(tmp_path, 5)
    assert closed == {}
    assert todo == [0, 1, 2, 3, 4]


def test_a_truncated_progress_file_is_declined_rather_than_half_read(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text('{"done": 12, "segm', encoding="utf-8")
    assert _resume_state(tmp_path, 3) == ({}, [0, 1, 2])


def test_shots_beyond_the_requested_count_are_not_inherited(tmp_path: Path) -> None:
    """A --segments-limited probe reopening a full story must not inherit shots it will not produce."""
    _write_progress(tmp_path, list(range(40)), total=131)
    closed, todo = _resume_state(tmp_path, 10)
    assert sorted(closed) == list(range(10))
    assert todo == []


def test_a_duplicated_record_collapses_to_one_shot(tmp_path: Path) -> None:
    """Runs damaged by the count-based resume already carry duplicates; reopening must normalise."""
    (tmp_path / "progress.json").write_text(
        json.dumps({"segments": [{"segment_id": 0}, {"segment_id": 1}, {"segment_id": 1}]}),
        encoding="utf-8")
    closed, todo = _resume_state(tmp_path, 3)
    assert sorted(closed) == [0, 1]
    assert todo == [2]


def test_unusable_entries_are_dropped(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"segments": [{"segment_id": 0}, "corrupt", {"no_id": True},
                                 {"segment_id": "abc"}, {"segment_id": -1}]}),
        encoding="utf-8")
    closed, todo = _resume_state(tmp_path, 2)
    assert sorted(closed) == [0]
    assert todo == [1]
