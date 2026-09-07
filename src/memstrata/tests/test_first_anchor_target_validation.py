from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from memstrata.bank import AssetBank, AssetType
from memstrata.encoders import HashEmbedding
from memstrata.mllm.crop_attributes import HeuristicCropAttributeClassifier
from memstrata.mllm.identity_judge import IdentityVerdict
from memstrata.skills.crop_acquisition.orchestrator import acquire_entity_crop
from memstrata.skills.decomposition import SOURCE_DISCOVERED, Observation
from memstrata.skills.memory_update.curator import AssetCurator


def _image(path: Path, *, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(30, 225, size=(96, 96, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)
    return str(path)


class _CountingClassifier:
    def __init__(self) -> None:
        self.inner = HeuristicCropAttributeClassifier()
        self.batch_calls = 0
        self.single_calls = 0
        self.targets: list[list[str | None]] = []

    def classify(self, image_path, **kwargs):
        self.single_calls += 1
        return self.inner.classify(image_path, **kwargs)

    def classify_batch(self, items, *, target_descriptions=None):
        self.batch_calls += 1
        self.targets.append(list(target_descriptions or []))
        return self.inner.classify_batch(
            items,
            target_descriptions=target_descriptions,
        )


class _NoVerdictClassifier(_CountingClassifier):
    def classify_batch(self, items, *, target_descriptions=None):
        packs = super().classify_batch(
            items,
            target_descriptions=target_descriptions,
        )
        for pack in packs:
            pack.extra.pop("matches_target", None)
        return packs


def _curator(bank: AssetBank, classifier: _CountingClassifier) -> AssetCurator:
    return AssetCurator(
        bank,
        HashEmbedding(),
        crop_attribute_classifier=classifier,
        dark_gate=False,
    )


def _observation(
    path: str,
    *,
    entity_id: str | None = "char_hero",
    name: str = "Hero",
    kind: AssetType = AssetType.CHARACTER,
    description: str = "red hat",
    source: str = "requested",
) -> Observation:
    return Observation(
        observation_id=f"obs_{name}",
        entity_id=entity_id,
        name=name,
        kind=kind,
        image_path=path,
        description=description,
        target_description=description,
        source=source,
    )


def test_wrong_named_first_anchor_is_vetoed_without_empty_asset(tmp_path: Path) -> None:
    bank = AssetBank()
    classifier = _CountingClassifier()
    obs = _observation(_image(tmp_path / "blue_scarf.png"))

    touched = _curator(bank, classifier).curate_observations([obs], segment_id=0)

    assert touched == []
    assert bank.get_asset("char_hero") is None
    assert bank.assets == {}
    assert obs.angle_meta["target_validation"]["matches_target"] is False
    assert obs.angle_meta["target_validation"]["decision"] == "rejected_explicit_mismatch"


def test_correct_named_first_anchor_is_admitted_with_audit_metadata(tmp_path: Path) -> None:
    bank = AssetBank()
    classifier = _CountingClassifier()
    obs = _observation(_image(tmp_path / "hero_red_hat.png"))

    touched = _curator(bank, classifier).curate_observations([obs], segment_id=0)

    assert touched == ["char_hero"]
    asset = bank.get_asset("char_hero")
    assert asset is not None and len(asset.representations) == 1
    validation = asset.representations[0].annotations["target_validation"]
    assert validation["matches_target"] is True
    assert validation["decision"] == "admitted_explicit_match"


def test_missing_target_verdict_preserves_previous_admission_behavior(tmp_path: Path) -> None:
    bank = AssetBank()
    classifier = _NoVerdictClassifier()
    obs = _observation(_image(tmp_path / "hero_red_hat.png"))

    touched = _curator(bank, classifier).curate_observations([obs], segment_id=0)

    assert touched == ["char_hero"]
    asset = bank.get_asset("char_hero")
    assert asset is not None and len(asset.representations) == 1
    validation = asset.representations[0].annotations["target_validation"]
    assert validation["matches_target"] is None
    assert validation["decision"] == "preserved_no_verdict"
    assert classifier.batch_calls == 1
    assert classifier.single_calls == 0


def test_established_mismatch_is_vetoed_before_representation_mutation(
    tmp_path: Path,
) -> None:
    bank = AssetBank()
    classifier = _CountingClassifier()
    curator = _curator(bank, classifier)
    first = _observation(_image(tmp_path / "hero_red_hat_front.png", seed=1))
    assert curator.curate_observations([first], segment_id=0) == ["char_hero"]
    original_reps = list(bank.get_asset("char_hero").representations)  # type: ignore[union-attr]

    mismatch = _observation(_image(tmp_path / "blue_scarf_side.png", seed=2))
    touched = curator.curate_observations([mismatch], segment_id=1)

    assert touched == []
    assert bank.get_asset("char_hero").representations == original_reps  # type: ignore[union-attr]
    validation = mismatch.angle_meta["target_validation"]
    assert validation["target_description"] == "red hat"
    assert validation["matches_target"] is False
    assert validation["decision"] == "rejected_explicit_mismatch"
    assert classifier.batch_calls == 2
    assert classifier.single_calls == 0


def test_established_match_is_accepted_by_same_segment_batch(tmp_path: Path) -> None:
    bank = AssetBank()
    classifier = _CountingClassifier()
    curator = _curator(bank, classifier)
    first = _observation(_image(tmp_path / "hero_red_hat_front.png", seed=3))
    assert curator.curate_observations([first], segment_id=0) == ["char_hero"]

    matching = _observation(_image(tmp_path / "hero_red_hat_side.png", seed=4))
    touched = curator.curate_observations([matching], segment_id=1)

    assert touched == ["char_hero"]
    assert matching.angle_meta["target_validation"]["matches_target"] is True
    assert matching.angle_meta["target_validation"]["decision"] == "admitted_explicit_match"
    assert classifier.batch_calls == 2
    assert classifier.single_calls == 0


def test_wrong_discovered_first_anchor_is_vetoed(tmp_path: Path) -> None:
    bank = AssetBank()
    classifier = _CountingClassifier()
    obs = _observation(
        _image(tmp_path / "blue_scarf.png"),
        entity_id=None,
        name="New Hero",
        source=SOURCE_DISCOVERED,
    )

    touched = _curator(bank, classifier).curate_observations([obs], segment_id=3)

    assert touched == []
    assert bank.assets == {}
    assert obs.angle_meta["target_validation"]["matches_target"] is False


def test_new_location_uses_target_validation_in_same_batch(tmp_path: Path) -> None:
    bank = AssetBank()
    classifier = _CountingClassifier()
    observations = [
        _observation(
            _image(tmp_path / "green_meadow.png"),
            entity_id="loc_meadow",
            name="Meadow",
            kind=AssetType.LOCATION,
            description="green meadow",
        ),
        _observation(
            _image(tmp_path / "indoor_hall.png"),
            entity_id="loc_forest",
            name="Forest",
            kind=AssetType.LOCATION,
            description="dense forest",
        ),
    ]

    touched = _curator(bank, classifier).curate_observations(observations, segment_id=0)

    assert touched == ["loc_meadow"]
    assert bank.get_asset("loc_meadow") is not None
    assert bank.get_asset("loc_forest") is None
    assert classifier.batch_calls == 1
    assert classifier.single_calls == 0
    assert classifier.targets == [["green meadow", "dense forest"]]


class _Segmenter:
    def segment_multi(self, frame_path: Path, concepts: list[str]):
        del frame_path
        mask = np.zeros((96, 96), dtype=bool)
        mask[8:88, 8:88] = True
        return {concepts[0]: [((8, 8, 88, 88), 0.9, mask)]}


class _Embedder:
    def embed_batch(self, paths: list[Path]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in paths]


class _RejectingVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, crop, references, **kwargs):
        del crop, references, kwargs
        self.calls += 1
        return IdentityVerdict(same=False, confidence=1.0, source="test")


def test_manual_compact_identity_opt_in_is_fail_closed_but_skips_locations(
    tmp_path: Path,
) -> None:
    frame = _image(tmp_path / "frame.png")
    reference = _image(tmp_path / "reference.png")

    character_verifier = _RejectingVerifier()
    character = acquire_entity_crop(
        frame,
        entity_name="Hero",
        entity_kind="character",
        entity_description="red hat",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[],
        exemplar_image_paths=[reference],
        out_dir=tmp_path / "character",
        segmenter=_Segmenter(),
        embedder=_Embedder(),
        identity_verifier=character_verifier,
        identity_verification_required=True,
    )
    assert character is None
    assert character_verifier.calls == 1

    location_verifier = _RejectingVerifier()
    location = acquire_entity_crop(
        frame,
        entity_name="Meadow",
        entity_kind="location",
        entity_description="green meadow",
        exemplar_vectors=[[1.0, 0.0]],
        existing_rep_vectors=[],
        exemplar_image_paths=[reference],
        out_dir=tmp_path / "location",
        segmenter=_Segmenter(),
        embedder=None,
        identity_verifier=location_verifier,
        identity_verification_required=True,
    )
    assert location is not None
    assert location["identity_gate"] == "not_applicable_location"
    assert location["identity_verification"] == {"gate": "not_applicable_location"}
    assert location_verifier.calls == 0
