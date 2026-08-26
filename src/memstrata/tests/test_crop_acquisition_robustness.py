from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from memstrata.bank import AssetType
from memstrata.skills.crop_acquisition.crop_client import ProposeIdentifyCropper
from memstrata.skills.crop_acquisition.orchestrator import (
    DEFAULT_IDENTITY_THRESHOLD,
    _concepts_for,
    _grounding_phrases_for,
    acquire_entity_crop,
)
from memstrata.steps.decompose import NamedEntity


def _frame(path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr = rng.integers(40, 230, size=(100, 100, 3), dtype="uint8")
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _mask(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class _Segmenter:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.masks = masks
        self.concepts: list[str] = []

    def segment_multi(self, frame_path: Path, concepts: list[str]):
        del frame_path
        self.concepts.extend(concepts)
        return {
            concepts[0]: [
                ((int(m.argmax() % 100), int(m.argmax() // 100), 90, 90), 1.0 - i * 0.1, m)
                for i, m in enumerate(self.masks)
            ]
        }


class _Embedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def embed_batch(self, paths: list[Path]) -> list[list[float]]:
        assert len(paths) <= len(self.vectors)
        return self.vectors[: len(paths)]


def test_close_up_character_crop_is_not_area_filtered(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    segmenter = _Segmenter([_mask(10, 10, 90, 90)])  # 64% area: close-up, not full-frame

    result = acquire_entity_crop(
        frame,
        entity_name="Hero",
        entity_kind="character",
        entity_description="person in a red coat",
        exemplar_vectors=[],
        existing_rep_vectors=[],
        out_dir=tmp_path / "out",
        segmenter=segmenter,
        detector=None,
        embedder=None,
    )

    assert result is not None
    assert result["qa"]["accepted"] is True
    assert result["max_character_bbox_area"] == 1.0


def test_identity_gate_miss_does_not_bypass_to_stranger(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    segmenter = _Segmenter([_mask(10, 10, 50, 50)])
    embedder = _Embedder([[0.0, 1.0]])

    result = acquire_entity_crop(
        frame,
        entity_name="Hero",
        entity_kind="character",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[],
        out_dir=tmp_path / "out",
        segmenter=segmenter,
        detector=None,
        embedder=embedder,
        identity_threshold=DEFAULT_IDENTITY_THRESHOLD,
    )

    assert result is None


def test_identity_similarity_ranks_before_novelty(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    segmenter = _Segmenter([_mask(10, 10, 50, 50), _mask(55, 55, 90, 90)])
    embedder = _Embedder([[1.0, 0.0], [0.3, 0.954]])

    result = acquire_entity_crop(
        frame,
        entity_name="Hero",
        entity_kind="character",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[[1.0, 0.0]],
        out_dir=tmp_path / "out",
        segmenter=segmenter,
        detector=None,
        embedder=embedder,
        identity_threshold=DEFAULT_IDENTITY_THRESHOLD,
    )

    assert result is not None
    assert result["identity_sim"] > 0.99
    assert result["novelty_score"] < 0.01


def test_location_and_gdino_prompts_use_categories_not_names() -> None:
    assert _concepts_for("location") == ("room", "building", "landscape")
    phrases = _grounding_phrases_for("character", "rabbit in a blue jacket")
    assert phrases[0] == "animal rabbit in a blue jacket"
    assert "person" in phrases and "animal" in phrases


class _Bank:
    def get_asset(self, entity_id: str):
        del entity_id
        return None


class _CapturingCropper(ProposeIdentifyCropper):
    def __init__(self, *args, frame_paths: list[Path], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fake_frame_paths = frame_paths
        self.last_request = None

    def frame_paths_for_segment(self, segment_video: str, *, segment_id: int):
        del segment_video, segment_id
        return self._fake_frame_paths, [0.2, 0.8]

    def _ensure_server(self) -> None:
        return None

    def _submit_and_wait(self, request: dict):
        self.last_request = request
        return {
            "status": "ok",
            "result": {
                "crop_path": str(Path(request["out_dir"]) / "crop_hero.png"),
                "bbox": [100, 100, 500, 500],
                "identity_sim": None,
                "identity_gate": "off",
                "novelty_score": 1.0,
                "source": "sam3_concept",
                "source_detail": {"frame_position": 0.8},
                "frame_position": 0.8,
                "candidate_count": 2,
                "identity_threshold": request["identity_threshold"],
                "min_side_px": 16,
                "max_character_bbox_area": 1.0,
                "min_mask_fill": 0.18,
            },
        }


def test_client_submits_multi_frame_pool_and_writes_summary(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "a.jpg"), _frame(tmp_path / "b.jpg")]
    cropper = _CapturingCropper(
        _Bank(),
        server_dir=tmp_path / "server",
        work_dir=tmp_path / "work",
        frame_paths=frames,
    )

    payload = cropper.crop(
        "segment.mp4",
        NamedEntity(
            name="Hero",
            kind=AssetType.CHARACTER,
            entity_id="char_hero",
            description="person in a blue jacket",
        ),
        segment_id=3,
    )

    assert payload is not None
    assert cropper.last_request["frame_paths"] == [str(p.resolve()) for p in frames]
    assert cropper.last_request["frame_positions"] == [0.2, 0.8]
    assert cropper.last_request["identity_threshold"] == DEFAULT_IDENTITY_THRESHOLD
    assert cropper.last_request["entity_description"] == "person in a blue jacket"
    summary = json.loads((tmp_path / "work" / "crop_acquisition_summary.json").read_text())
    assert summary["config"]["identity_threshold"] == DEFAULT_IDENTITY_THRESHOLD
    assert summary["config"]["min_side_px"] == 16
    assert summary["entities"]["char_hero"]["miss_rate"] == 0.0
