from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from memstrata.production.profiles import PAPER_TRACKA_202607
from memstrata.production.realized import build_realized_segment_pipeline
from memstrata.skills.crop_acquisition.orchestrator import _propose_candidates


def test_paper_tracka_profile_is_explicit_and_audited() -> None:
    assert PAPER_TRACKA_202607.write_naming == "mllm"
    assert PAPER_TRACKA_202607.embedder_provider == "dinov3"
    assert PAPER_TRACKA_202607.read_slow_fallback is True
    assert PAPER_TRACKA_202607.read_max_reps_per_asset == 1
    assert PAPER_TRACKA_202607.require_wedetect is True
    assert PAPER_TRACKA_202607.mllm_model == "Qwen3.5-9B-Instruct"
    assert PAPER_TRACKA_202607.require_mllm is True


def test_wedetect_hit_is_authoritative_for_location(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), "white").save(frame)

    class Grounder:
        strict = False

        def ground(self, *_args, **_kwargs):
            return [([100, 100, 800, 900], 0.9)]

    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError(f"fallback backend unexpectedly used: {name}")

    candidates = _propose_candidates(
        frame_path=frame,
        entity_name="meadow",
        entity_kind="location",
        entity_description="sunny green meadow behind the rabbit",
        scratch_dir=tmp_path / "scratch",
        segmenter=MustNotRun(),
        detector=MustNotRun(),
        grounder=Grounder(),
        max_character_bbox_area=1.0,
        min_mask_fill=0.18,
        min_side_px=16,
        iou_threshold=0.7,
    )

    assert [candidate["source"] for candidate in candidates] == ["wedetect_ref"]


def test_strict_profile_fails_before_silent_fallback(monkeypatch, tmp_path: Path) -> None:
    from memstrata.skills.crop_acquisition.wedetect_client import WeDetectRefGrounder

    monkeypatch.setattr(WeDetectRefGrounder, "healthy", lambda self: False)
    with pytest.raises(RuntimeError, match="requires a healthy WeDetect-Ref"):
        build_realized_segment_pipeline(
            run_dir=tmp_path,
            profile="paper_tracka_202607",
            wedetect_url="http://127.0.0.1:1",
        )


def test_paper_profile_rejects_behavior_changing_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="immutable"):
        build_realized_segment_pipeline(
            run_dir=tmp_path,
            profile="paper_tracka_202607",
            write_naming="perception",
        )


def test_strict_profile_fails_before_silent_mllm_degradation(
    monkeypatch, tmp_path: Path
) -> None:
    from memstrata.production import realized
    from memstrata.skills.crop_acquisition.wedetect_client import WeDetectRefGrounder

    monkeypatch.setattr(WeDetectRefGrounder, "healthy", lambda self: True)
    monkeypatch.setattr(realized, "_openai_model_ready", lambda *_args, **_kwargs: False)
    with pytest.raises(RuntimeError, match="requires MLLM model"):
        build_realized_segment_pipeline(
            run_dir=tmp_path,
            profile="paper_tracka_202607",
            mllm_base_url="http://127.0.0.1:1/v1",
        )


def test_finalize_persists_manifest_with_actual_backend_counts(tmp_path: Path) -> None:
    mem = build_realized_segment_pipeline(
        run_dir=tmp_path,
        profile="production",
        write_naming="perception",
        embedder_provider="hash",
    )
    cropper = mem.decomposer.cropper
    cropper._record_attempt(
        "hero",
        hit=True,
        payload={"source": "wedetect_ref", "source_detail": {}},
    )
    cropper._record_attempt(
        "prop",
        hit=True,
        payload={
            "source": "sam3_concept",
            "source_detail": {"fallback_from": "no_hit"},
        },
    )

    result = mem.observe_realized_segment(
        segment_id=0,
        segment_video=None,
        observations=[],
        source_start_sec=2.0,
        source_duration_sec=4.0,
        fps=24.0,
    )
    snapshot = mem.finalize({"system": "memstrata"})

    assert result.observations == []
    assert mem.interpreter.max_reps_per_asset == 1
    assert mem.interpreter.slow_on_miss is True
    assert (tmp_path / "bank.json").is_file()
    assert snapshot == tmp_path / "membank" / "memory.json"

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    provenance = manifest["production"]
    assert provenance["profile"] == "production"
    assert provenance["crop_acquisition"]["backend_counts"] == {
        "sam3_concept": 1,
        "wedetect_ref": 1,
    }
    assert provenance["crop_acquisition"]["fallback_counts"] == {"no_hit": 1}
    assert manifest["finalized"] is True
