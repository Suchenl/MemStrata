"""Production scene-evidence producer, cache, and schema plumbing tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from memstrata.bank import AssetType, SpatialAngle, StateAngle
from memstrata.encoders import HashEmbedding
from memstrata.production.realized import build_realized_segment_pipeline
from memstrata.skills.crop_acquisition.crop_client import ProposeIdentifyCropper
from memstrata.skills.crop_acquisition.orchestrator import acquire_entity_crop
from memstrata.skills.crop_acquisition.scene_evidence import (
    COVERAGE_COMPLETE,
    COVERAGE_LOWER_BOUND,
    COVERAGE_MISSING,
    CachedSegmentSceneEvidenceProducer,
    SCENE_EVIDENCE_SCHEMA_VERSION,
    bbox_coverage,
)
from memstrata.skills.location_scene_validity import (
    LocationSceneEvidence,
    SceneValidityStatus,
    evaluate_location_scene,
)
from memstrata.steps.decompose import NamedEntity, RoleAwareDecomposer


def _frame(path: Path, *, bars: bool = False) -> Path:
    yy, xx = np.indices((80, 120))
    array = np.stack(
        (
            (xx * 7 + yy * 3) % 220 + 20,
            (xx * 5 + yy * 11) % 220 + 20,
            (xx * 13 + yy * 2) % 220 + 20,
        ),
        axis=-1,
    ).astype(np.uint8)
    if bars:
        array[:10] = 0
        array[-10:] = 0
    Image.fromarray(array, mode="RGB").save(path)
    return path


class _BatchDetector:
    def __init__(self, rows=None, *, fail: bool = False) -> None:
        self.rows = rows
        self.fail = fail
        self.calls: list[tuple[list[Path], str]] = []

    def detect_batch(self, paths: list[Path], prompt: str):
        self.calls.append((list(paths), prompt))
        if self.fail:
            raise TimeoutError("detector timed out")
        if self.rows is None:
            return [[] for _ in paths]
        return [self.rows[index] for index in range(len(paths))]


class _BatchEmbedder:
    name = "dinov3:test"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[Path]] = []

    def embed_batch(self, paths: list[Path]) -> list[list[float]]:
        self.calls.append(list(paths))
        return self.vectors[: len(paths)]


def _candidate(
    frame: Path,
    *,
    bbox: list[int] | None = None,
    frame_index: int = 0,
    source: str = "wedetect_ref",
    score: float = 0.8,
) -> dict:
    return {
        "bbox_norm": bbox or [0, 0, 1000, 1000],
        "frame_path": frame,
        "frame_index": frame_index,
        "source": source,
        "score": score,
    }


def test_bbox_max_and_union_coverage_are_exact() -> None:
    maximum, union = bbox_coverage(
        [0, 0, 1000, 1000],
        [
            [0, 0, 500, 500],
            [0, 500, 500, 1000],
            [0, 250, 500, 750],
        ],
    )
    assert maximum == pytest.approx(0.25)
    assert union == pytest.approx(0.50)


def test_detector_success_zero_is_complete_but_failure_is_missing(
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path / "frame.png")
    success = CachedSegmentSceneEvidenceProducer(
        detector=_BatchDetector(),
        embedder=None,
    ).collect_candidates(
        frame_paths=[frame],
        candidates=[_candidate(frame)],
    )[0]
    failure = CachedSegmentSceneEvidenceProducer(
        detector=_BatchDetector(fail=True),
        embedder=None,
    ).collect_candidates(
        frame_paths=[frame],
        candidates=[_candidate(frame)],
    )[0]

    assert success["coverage_semantics"] == COVERAGE_COMPLETE
    assert success["foreground_max_coverage"] == 0.0
    assert success["foreground_union_coverage"] == 0.0
    assert failure["coverage_semantics"] == COVERAGE_MISSING
    assert failure["foreground_max_coverage"] is None
    assert failure["foreground_union_coverage"] is None
    assert (
        evaluate_location_scene(
            LocationSceneEvidence.from_annotations(
                {"scene_validity_evidence": failure}
            )
        ).status
        is SceneValidityStatus.QUARANTINE
    )


def test_exact_and_content_corrected_area_are_both_reported(
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path / "letterboxed.png", bars=True)
    evidence = CachedSegmentSceneEvidenceProducer(
        detector=_BatchDetector(),
        embedder=None,
    ).collect_candidates(
        frame_paths=[frame],
        candidates=[_candidate(frame)],
    )[0]

    assert evidence["crop_area_fraction"] == 1.0
    assert evidence["content_area_fraction"] == 1.0
    assert evidence["content_box_status"] == "detected"
    assert evidence["content_bbox"][0] > 0
    assert evidence["content_bbox"][2] < 1000


def test_lower_bound_can_reject_but_cannot_accept(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    producer = CachedSegmentSceneEvidenceProducer(detector=None, embedder=None)
    high, low = producer.collect_candidates(
        frame_paths=[frame],
        candidates=[
            _candidate(frame),
            _candidate(frame, bbox=[0, 0, 500, 500]),
        ],
        lower_bound_bboxes={0: [[0, 0, 800, 1000]]},
    )

    high_decision = evaluate_location_scene(
        LocationSceneEvidence.from_annotations({"scene_validity_evidence": high})
    )
    low["foreground_max_coverage"] = 0.05
    low["foreground_union_coverage"] = 0.05
    low_decision = evaluate_location_scene(
        LocationSceneEvidence.from_annotations({"scene_validity_evidence": low})
    )
    assert high["coverage_semantics"] == COVERAGE_LOWER_BOUND
    assert high_decision.status is SceneValidityStatus.REJECT
    assert low_decision.status is SceneValidityStatus.QUARANTINE
    assert low_decision.reasons == ("foreground_lower_bound_only",)


def test_recorded_character_bbox_is_only_a_lower_bound(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    producer = CachedSegmentSceneEvidenceProducer(detector=None, embedder=None)
    producer.record_subject_bbox(
        frame_path=frame,
        bbox_norm=[0, 0, 700, 1000],
    )
    evidence = producer.collect_candidates(
        frame_paths=[frame],
        candidates=[_candidate(frame)],
    )[0]

    assert evidence["coverage_semantics"] == COVERAGE_LOWER_BOUND
    assert evidence["foreground_max_coverage"] == pytest.approx(0.70)
    assert producer.cache_stats()["known_subject_frame_entries"] == 1


def test_arbitrary_frame_count_and_multi_location_cache(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / f"frame_{index}.png") for index in range(4)]
    detector = _BatchDetector()
    embedder = _BatchEmbedder([[1.0, 0.0] for _ in frames])
    producer = CachedSegmentSceneEvidenceProducer(
        detector=detector,
        embedder=embedder,
    )
    candidates = [
        _candidate(frame, bbox=[0, 0, 400, 1000], frame_index=index)
        for index, frame in enumerate(frames)
    ]
    first = producer.collect_candidates(
        frame_paths=frames,
        candidates=candidates,
    )
    second = producer.collect_candidates(
        frame_paths=frames,
        candidates=candidates,
    )

    assert len(first) == len(second) == 4
    assert len(detector.calls) == 1
    assert len(embedder.calls) == 1
    assert producer.cache_stats()["detector_frame_calls"] == 4
    assert first[0]["temporal_visual_support_count"] == 4
    assert first[0]["temporal_visual_observation_count"] == 4
    decision = evaluate_location_scene(
        LocationSceneEvidence.from_annotations(
            {"scene_validity_evidence": first[0]}
        )
    )
    assert decision.status is SceneValidityStatus.ACCEPT
    assert "temporal_visual_support" in decision.reasons
    assert first[0]["temporal_semantics"] == "generic_visual_not_vpr"
    assert "place" not in first[0]["temporal_semantics"]


def test_schema_mismatch_is_quarantined() -> None:
    evidence = LocationSceneEvidence.from_annotations(
        {
            "scene_validity_evidence": {
                "schema_version": "future.v99",
                "coverage_semantics": COVERAGE_COMPLETE,
                "foreground_max_coverage": 0.0,
                "crop_area_fraction": 1.0,
            }
        }
    )
    decision = evaluate_location_scene(evidence)
    assert decision.status is SceneValidityStatus.QUARANTINE
    assert decision.reasons == ("scene_evidence_schema_mismatch",)


class _SourceAwareProvider:
    def collect_candidates(self, *, frame_paths, candidates, lower_bound_bboxes=None):
        del frame_paths, lower_bound_bboxes
        rows = []
        for candidate in candidates:
            clean = candidate["source"] == "whole_frame_location_candidate"
            rows.append(
                {
                    "schema_version": SCENE_EVIDENCE_SCHEMA_VERSION,
                    "coverage_semantics": COVERAGE_COMPLETE,
                    "coverage_geometry": "bbox",
                    "foreground_max_coverage": 0.0 if clean else 0.80,
                    "foreground_union_coverage": 0.0 if clean else 0.80,
                    "crop_area_fraction": 1.0,
                    "temporal_visual_status": "missing",
                }
            )
        return rows


class _Grounder:
    strict = False

    def ground(self, frame_path, query, *, kind=""):
        del frame_path, query, kind
        return [([20, 20, 980, 980], 0.99)]


def test_whole_frame_survives_iou_and_text_score_but_needs_scene_decision(
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path / "frame.png")
    result = acquire_entity_crop(
        frame,
        entity_name="Any location",
        entity_kind="location",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[],
        out_dir=tmp_path / "out",
        grounder=_Grounder(),
        location_scene_plate_candidates=True,
        location_scene_evidence_enabled=True,
        scene_evidence_provider=_SourceAwareProvider(),
    )

    assert result is not None
    assert result["source"] == "whole_frame_location_candidate"
    assert result["identity_gate"] == "off_location_scene"
    assert result["selected_score"] == 0.0
    assert result["scene_validity_evidence"]["coverage_semantics"] == COVERAGE_COMPLETE


class _Bank:
    def get_asset(self, entity_id: str):
        del entity_id
        return None


class _LocationCropper(ProposeIdentifyCropper):
    def __init__(self, *args, frame: Path, crop: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frame = frame
        self.crop_path = crop

    def frame_paths_for_segment(self, segment_video: str, *, segment_id: int):
        del segment_video, segment_id
        return [self.frame], [0.5]

    def _ensure_server(self) -> None:
        return None

    def _submit_and_wait(self, request: dict):
        evidence = {
            "schema_version": SCENE_EVIDENCE_SCHEMA_VERSION,
            "coverage_semantics": COVERAGE_COMPLETE,
            "coverage_geometry": "bbox",
            "foreground_max_coverage": 0.0,
            "foreground_union_coverage": 0.0,
            "crop_area_fraction": 0.8,
        }
        # Exercise the real file-queue serialization constraint.
        return json.loads(
            json.dumps(
                {
                    "status": "ok",
                    "result": {
                        "crop_path": str(self.crop_path),
                        "bbox": [0, 0, 800, 1000],
                        "mask_path": str(self.crop_path.with_suffix(".mask.png")),
                        "frame_path": str(self.frame),
                        "frame_position": 0.5,
                        "identity_sim": None,
                        "identity_gate": "off",
                        "identity_verification": {"gate": "off_first_sighting"},
                        "novelty_score": 1.0,
                        "source": "whole_frame_location_candidate",
                        "source_detail": {"frame_index": 0, "frame_position": 0.5},
                        "selected_score": 0.0,
                        "candidate_count": 1,
                        "identity_threshold": request["identity_threshold"],
                        "identity_verification_required": False,
                        "identity_verification_threshold": 0.9,
                        "min_side_px": 16,
                        "max_character_bbox_area": 1.0,
                        "min_mask_fill": 0.18,
                        "qa": {"accepted": True},
                        "scene_validity_evidence": evidence,
                    },
                }
            )
        )


def test_location_fields_cross_client_and_decomposer_boundary(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "source.png")
    crop = _frame(tmp_path / "crop.png")
    cropper = _LocationCropper(
        _Bank(),
        server_dir=tmp_path / "server",
        work_dir=tmp_path / "work",
        auto_start=False,
        frame=frame,
        crop=crop,
        extra_acquire_kwargs={"location_scene_evidence_enabled": True},
    )
    decomposer = RoleAwareDecomposer(
        embedder=HashEmbedding(),
        cropper=cropper,
    )
    observations = decomposer.decompose(
        segment_id=7,
        segment_video="segment.mp4",
        named_entities=[
            NamedEntity(
                name="Location",
                kind=AssetType.LOCATION,
                spatial_angle=SpatialAngle.FRONT,
                state_angle=StateAngle.DEFAULT,
            )
        ],
    )

    assert len(observations) == 1
    observation = observations[0]
    acquisition = observation.angle_meta["crop_acquisition"]
    assert observation.source_frame_path == str(frame)
    assert acquisition["mask_path"].endswith(".mask.png")
    assert acquisition["frame_index"] == 0
    assert acquisition["selected_score"] == 0.0
    assert (
        acquisition["scene_validity_evidence"]["schema_version"]
        == SCENE_EVIDENCE_SCHEMA_VERSION
    )


def test_evidence_wiring_is_adaptive_or_explicit_only(tmp_path: Path) -> None:
    legacy = build_realized_segment_pipeline(
        run_dir=tmp_path / "legacy",
        profile="production",
        write_naming="perception",
        embedder_provider="hash",
    )
    adaptive = build_realized_segment_pipeline(
        run_dir=tmp_path / "adaptive",
        profile="production",
        write_naming="perception",
        embedder_provider="hash",
        location_adaptive_enabled=True,
    )
    explicit = build_realized_segment_pipeline(
        run_dir=tmp_path / "explicit",
        profile="production",
        write_naming="perception",
        embedder_provider="hash",
        location_scene_evidence_enabled=True,
    )

    assert "location_scene_evidence_enabled" not in legacy.decomposer.cropper.extra_acquire_kwargs
    assert adaptive.decomposer.cropper.extra_acquire_kwargs[
        "location_scene_evidence_enabled"
    ] is True
    assert explicit.decomposer.cropper.extra_acquire_kwargs[
        "location_scene_evidence_enabled"
    ] is True
    with pytest.raises(ValueError, match="immutable"):
        build_realized_segment_pipeline(
            run_dir=tmp_path / "paper",
            profile="paper_tracka_202607",
            location_scene_evidence_enabled=True,
        )
