from __future__ import annotations

import json

import pytest

from memstrata.lib import observe_profile
from memstrata.lib.observe_profile import capture_profile, merge_profile, profile_span
from memstrata.pipeline import MemStrata


def test_profile_records_nested_and_merged_subphases(tmp_path, monkeypatch) -> None:
    ticks = iter([10.0, 11.0, 13.0, 14.0])
    monkeypatch.setattr(observe_profile.time, "perf_counter", lambda: next(ticks))
    output = tmp_path / "observe_profile.jsonl"

    with capture_profile("observe_segment", segment_id=7, output_path=output):
        with profile_span("local"):
            pass
        merge_profile(
            {
                "subphases": {
                    "wedetect": {"calls": 2, "total_ms": 125.0, "max_ms": 75.0}
                }
            },
            prefix="crop_server.",
        )

    payload = json.loads(output.read_text())
    assert payload["kind"] == "observe_segment"
    assert payload["segment_id"] == 7
    assert payload["wall_ms"] == 4000.0
    assert payload["subphases"]["local"] == {
        "calls": 1,
        "total_ms": 2000.0,
        "max_ms": 2000.0,
    }
    assert payload["subphases"]["crop_server.wedetect"] == {
        "calls": 2,
        "total_ms": 125.0,
        "max_ms": 75.0,
    }


def test_profile_writes_error_status(tmp_path) -> None:
    output = tmp_path / "observe_profile.jsonl"

    with pytest.raises(RuntimeError, match="expected"):
        with capture_profile("observe_segment", segment_id=8, output_path=output):
            raise RuntimeError("expected")

    assert json.loads(output.read_text())["status"] == "error"


def test_observe_realized_segment_emits_profile_without_changing_result(tmp_path) -> None:
    pipeline = MemStrata(
        run_dir=tmp_path,
        persist_path=tmp_path / "bank.json",
    )

    result = pipeline.observe_realized_segment(
        segment_id=3,
        segment_video=None,
        observations=[],
    )

    assert result.observations == []
    assert result.touched_asset_ids == []
    payload = json.loads((tmp_path / "observe_profile.jsonl").read_text())
    assert payload["kind"] == "observe_segment"
    assert payload["segment_id"] == 3
    assert payload["status"] == "ok"
    assert payload["subphases"]["curate"]["calls"] == 1
    assert payload["subphases"]["bank_save"]["calls"] == 1
