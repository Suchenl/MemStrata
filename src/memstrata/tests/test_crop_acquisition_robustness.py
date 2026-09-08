from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import pytest

from memstrata.bank import AssetType
from memstrata.mllm.identity_judge import IdentityVerdict, JUDGE_PROMPT
from memstrata.skills.crop_acquisition.crop_client import ProposeIdentifyCropper, _pid_alive
from memstrata.skills.crop_acquisition.orchestrator import (
    DEFAULT_IDENTITY_THRESHOLD,
    DEFAULT_IDENTITY_VERIFICATION_THRESHOLD,
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


class _IdentityVerifier:
    def __init__(self, *, same: bool | None, confidence: float = 0.99) -> None:
        self.verdict = IdentityVerdict(
            same=same,
            confidence=confidence,
            source="test",
            reasoning="visual comparison",
        )
        self.calls: list[tuple[str, list[str], str]] = []

    def judge(self, crop, references, *, kind="", name_a="", name_b=""):  # noqa: ANN001
        assert name_a == "" and name_b == ""
        self.calls.append((str(crop), list(references), kind))
        return self.verdict


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


def test_image_only_verifier_rejects_wrong_entity_despite_passing_dino(
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    reference = _frame(tmp_path / "big_rabbit_reference.jpg")
    segmenter = _Segmenter([_mask(10, 10, 50, 50)])
    # Reproduces the failure shape: 0.548 passes the deliberately lenient 0.25 recall gate.
    embedder = _Embedder([[0.548, 0.836]])
    verifier = _IdentityVerifier(same=False)

    result = acquire_entity_crop(
        frame,
        entity_name="大兔子",
        entity_kind="character",
        entity_description="a brown big rabbit",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[],
        exemplar_image_paths=[reference],
        out_dir=tmp_path / "out",
        segmenter=segmenter,
        detector=None,
        embedder=embedder,
        identity_verifier=verifier,
        identity_verification_required=True,
    )

    assert result is None
    assert len(verifier.calls) == 1
    assert verifier.calls[0][2] == "character"


def test_image_only_verifier_preserves_cross_view_recall(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    reference = _frame(tmp_path / "hero_front_reference.jpg")
    segmenter = _Segmenter([_mask(10, 10, 50, 50)])
    # A novel view barely clears DINO recall, then visual reference adjudication confirms it.
    embedder = _Embedder([[0.30, 0.954]])
    verifier = _IdentityVerifier(same=True, confidence=0.95)

    result = acquire_entity_crop(
        frame,
        entity_name="Hero",
        entity_kind="character",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[[1.0, 0.0]],
        exemplar_image_paths=[reference],
        out_dir=tmp_path / "out",
        segmenter=segmenter,
        detector=None,
        embedder=embedder,
        identity_verifier=verifier,
        identity_verification_required=True,
    )

    assert result is not None
    assert math.isclose(result["identity_sim"], 0.30, abs_tol=1e-3)
    assert result["identity_verification"] == {
        "gate": "applied",
        "same": True,
        "confidence": 0.95,
        "source": "test",
        "reasoning": "visual comparison",
        "reference_count": 1,
        "threshold": DEFAULT_IDENTITY_VERIFICATION_THRESHOLD,
    }


def test_required_identity_verification_fails_closed_on_abstention(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    reference = _frame(tmp_path / "reference.jpg")
    result = acquire_entity_crop(
        frame,
        entity_name="Hero",
        entity_kind="character",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[],
        exemplar_image_paths=[reference],
        out_dir=tmp_path / "out",
        segmenter=_Segmenter([_mask(10, 10, 50, 50)]),
        detector=None,
        embedder=_Embedder([[1.0, 0.0]]),
        identity_verifier=_IdentityVerifier(same=None),
        identity_verification_required=True,
    )

    assert result is None


def test_identity_judge_prompt_is_name_free_and_handles_animal_characters() -> None:
    prompt = JUDGE_PROMPT.format(kind="character", n=2)
    assert "entity name" in prompt.lower()
    assert "animal" in prompt
    assert "{name" not in JUDGE_PROMPT


def test_location_and_gdino_prompts_use_categories_not_names() -> None:
    assert _concepts_for("location") == ("room", "building", "landscape")
    phrases = _grounding_phrases_for("character", "rabbit in a blue jacket")
    assert phrases[0] == "animal rabbit in a blue jacket"
    assert "person" in phrases and "animal" in phrases


class _Bank:
    def get_asset(self, entity_id: str):
        del entity_id
        return None


def test_pid_alive_rejects_zombie(monkeypatch) -> None:
    monkeypatch.setattr("os.kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: "123 (python worker) Z 1 2 3"
        if str(self) == "/proc/123/stat"
        else "",
    )

    assert _pid_alive(123) is False


def test_submit_fails_fast_when_server_disappears(tmp_path: Path) -> None:
    cropper = ProposeIdentifyCropper(
        _Bank(),
        server_dir=tmp_path / "server",
        auto_start=False,
        job_timeout=1800,
    )

    with pytest.raises(RuntimeError, match="exited while job .* was pending"):
        cropper._submit_and_wait({"entity_name": "mouse"})


def test_server_ready_reaps_exited_autostart_child(tmp_path: Path) -> None:
    class _ExitedProcess:
        pid = 123

        @staticmethod
        def poll():
            return 0

    cropper = ProposeIdentifyCropper(
        _Bank(),
        server_dir=tmp_path / "server",
        auto_start=False,
    )
    cropper._proc = _ExitedProcess()

    assert cropper._server_ready() is False
    assert cropper._proc is None


def test_server_env_preserves_public_models_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PUBLIC_MODELS_ROOT", "/tmp/public-models")
    cropper = ProposeIdentifyCropper(
        _Bank(),
        server_dir=tmp_path / "server",
        auto_start=False,
    )

    assert cropper._server_env()["PUBLIC_MODELS_ROOT"] == "/tmp/public-models"


def test_timed_out_job_is_quarantined_and_server_retired(
    monkeypatch, tmp_path: Path
) -> None:
    cropper = ProposeIdentifyCropper(
        _Bank(),
        server_dir=tmp_path / "server",
        auto_start=False,
        job_timeout=0,
    )
    retired: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cropper,
        "_retire_server",
        lambda *, reason, job_id: retired.append((reason, job_id)),
    )

    with pytest.raises(TimeoutError, match="server was quarantined"):
        cropper._submit_and_wait({"job_id": "stuck", "entity_name": "Hero"})

    failure = json.loads(
        (tmp_path / "server" / "failed" / "stuck.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "timed_out"
    assert failure["request"]["entity_name"] == "Hero"
    assert not (tmp_path / "server" / "pending" / "stuck.json").exists()
    assert retired == [("job_timeout", "stuck")]


def test_worker_exit_fails_before_full_job_timeout(monkeypatch, tmp_path: Path) -> None:
    cropper = ProposeIdentifyCropper(
        _Bank(),
        server_dir=tmp_path / "server",
        auto_start=False,
        job_timeout=1800,
    )
    monkeypatch.setattr(cropper, "_server_ready", lambda: False)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="worker exited during job gone"):
        cropper._submit_and_wait({"job_id": "gone"})

    assert time.monotonic() - started < 1
    failure = json.loads(
        (tmp_path / "server" / "failed" / "gone.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "worker_exited"


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
    assert cropper.last_request["identity_verification_required"] is False
    assert (
        cropper.last_request["identity_verification_threshold"]
        == DEFAULT_IDENTITY_VERIFICATION_THRESHOLD
    )
    assert cropper.last_request["entity_description"] == "person in a blue jacket"
    summary = json.loads((tmp_path / "work" / "crop_acquisition_summary.json").read_text())
    assert summary["config"]["identity_threshold"] == DEFAULT_IDENTITY_THRESHOLD
    assert summary["config"]["min_side_px"] == 16
    assert summary["entities"]["char_hero"]["miss_rate"] == 0.0
