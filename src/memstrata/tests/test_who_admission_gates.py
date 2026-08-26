"""WHO-before-WHERE admission gates for the memory bank (design_philosophy.md §2).

Covers ① deterministic dark/low-info gate, ③ identity_visible anchor gating, and
② embedding-cohesion admission. Assert-based; runnable via ``python3`` or pytest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from memstrata.bank import AssetBank, AssetType, SpatialAngle, StateAngle
from memstrata.encoders import HashEmbedding
from memstrata.lib.dedup import medoid_cohesion, similarity_to_set
from memstrata.mllm.crop_attributes import HeuristicCropAttributeClassifier
from memstrata.steps.curate import AssetCurator, EntityObservation
from memstrata.steps.decompose import Observation

_CHAR = AssetType.CHARACTER
_ANCHOR = "identity_anchor"


def _dark_png(directory: Path, name: str = "dark.png") -> str:
    path = directory / name
    Image.new("RGB", (48, 48), (5, 5, 5)).save(path)  # near-black, flat
    return str(path)


def _bright_png(directory: Path, name: str = "bright.png") -> str:
    path = directory / name
    img = Image.new("RGB", (48, 48))
    img.putdata([((i * 7) % 256, (i * 13) % 256, (i * 29) % 256) for i in range(48 * 48)])
    img.save(path)
    return str(path)


# --- ① deterministic dark / low-information gate --------------------------------

def test_dark_crop_is_not_banked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = _dark_png(Path(tmp))
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), attributes_when_angles_known=False)
        cur.ingest_observation(
            EntityObservation(
                "o1", _CHAR, "Hero", crop,
                spatial_angle=SpatialAngle.FRONT, state_angle=StateAngle.DEFAULT,
            ),
            segment_id=0,
        )
        asset = bank.find_by_name("Hero", kind=_CHAR)
        assert asset is not None
        assert len(asset.representations) == 0  # rejected before banking


def test_bright_crop_is_banked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = _bright_png(Path(tmp))
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), attributes_when_angles_known=False)
        cur.ingest_observation(
            EntityObservation(
                "o1", _CHAR, "Hero", crop,
                spatial_angle=SpatialAngle.FRONT, state_angle=StateAngle.DEFAULT,
            ),
            segment_id=0,
        )
        asset = bank.find_by_name("Hero", kind=_CHAR)
        assert asset is not None and len(asset.representations) == 1


def test_dark_gate_can_be_disabled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = _dark_png(Path(tmp))
        bank = AssetBank()
        cur = AssetCurator(
            bank, HashEmbedding(), attributes_when_angles_known=False, dark_gate=False
        )
        cur.ingest_observation(
            EntityObservation(
                "o1", _CHAR, "Hero", crop,
                spatial_angle=SpatialAngle.FRONT, state_angle=StateAngle.DEFAULT,
            ),
            segment_id=0,
        )
        asset = bank.find_by_name("Hero", kind=_CHAR)
        assert asset is not None and len(asset.representations) == 1


# --- ③ identity_visible → anchor eligibility (keep as diversity) -----------------

def test_not_visible_crop_kept_but_not_anchor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Filename drives the heuristic classifier: bright + front + not-visible.
        crop = _bright_png(Path(tmp), "hero_front_default_notvisible.png")
        bank = AssetBank()
        cur = AssetCurator(
            bank, HashEmbedding(),
            crop_attribute_classifier=HeuristicCropAttributeClassifier(),
        )
        cur.ingest_observation(EntityObservation("o1", _CHAR, "Hero", crop), segment_id=0)
        asset = bank.find_by_name("Hero", kind=_CHAR)
        assert asset is not None and len(asset.representations) == 1  # kept for diversity
        rep = asset.representations[0]
        assert rep.annotations["crop_attributes"]["identity_visible"] is False
        assert _ANCHOR not in rep.reference_aspects
        assert _ANCHOR in rep.excluded_aspects
        assert rep.annotations["identity_anchor_eligible"] is False


def test_visible_crop_is_an_anchor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = _bright_png(Path(tmp), "hero_front_default.png")
        bank = AssetBank()
        cur = AssetCurator(
            bank, HashEmbedding(),
            crop_attribute_classifier=HeuristicCropAttributeClassifier(),
        )
        cur.ingest_observation(EntityObservation("o1", _CHAR, "Hero", crop), segment_id=0)
        rep = bank.find_by_name("Hero", kind=_CHAR).representations[0]
        assert rep.annotations["crop_attributes"]["identity_visible"] is True
        assert _ANCHOR in rep.reference_aspects
        assert rep.annotations.get("identity_anchor_eligible", True) is True


# --- ② embedding cohesion admission (low mixing) --------------------------------

def _obs(oid: str, vec: list[float], angle: SpatialAngle) -> Observation:
    return Observation(
        observation_id=oid,
        kind=_CHAR,
        name="Hero",
        image_path=f"/nonexistent/{oid}.png",  # dark gate off in this test
        embedding=vec,
        spatial_angle=angle,
        state_angle=StateAngle.DEFAULT,
    )


def test_cohesion_rejects_outlier_when_floor_set() -> None:
    bank = AssetBank()
    cur = AssetCurator(
        bank, HashEmbedding(),
        embed_on_ingest=False, dark_gate=False,
        attributes_when_angles_known=False,
        cohesion_floor=0.5, cohesion_min_refs=2,
    )
    cur.curate_observations([_obs("o0", [1.0, 0.0], SpatialAngle.FRONT)], segment_id=0)
    cur.curate_observations([_obs("o1", [0.8, 0.6], SpatialAngle.SIDE)], segment_id=1)
    # Outlier: max cosine to {v0, v1} = max(-0.6, 0.0) = 0.0 < 0.5 → rejected.
    cur.curate_observations([_obs("o2", [-0.6, 0.8], SpatialAngle.BACK)], segment_id=2)
    asset = bank.find_by_name("Hero", kind=_CHAR)
    active = [r for r in asset.representations if not r.deprecated]
    assert len(active) == 2


def test_cohesion_disabled_by_default_keeps_outlier() -> None:
    bank = AssetBank()
    cur = AssetCurator(
        bank, HashEmbedding(),
        embed_on_ingest=False, dark_gate=False,
        attributes_when_angles_known=False,
    )  # cohesion_floor defaults to 0.0
    cur.curate_observations([_obs("o0", [1.0, 0.0], SpatialAngle.FRONT)], segment_id=0)
    cur.curate_observations([_obs("o1", [0.8, 0.6], SpatialAngle.SIDE)], segment_id=1)
    cur.curate_observations([_obs("o2", [-0.6, 0.8], SpatialAngle.BACK)], segment_id=2)
    asset = bank.find_by_name("Hero", kind=_CHAR)
    active = [r for r in asset.representations if not r.deprecated]
    assert len(active) == 3


def test_dedup_cohesion_helpers() -> None:
    assert similarity_to_set([1.0, 0.0], []) == -1.0
    assert abs(similarity_to_set([1.0, 0.0], [[0.8, 0.6], [0.0, 1.0]]) - 0.8) < 1e-9
    medoid, sims, min_pair = medoid_cohesion([[1.0, 0.0], [0.9, 0.436], [-1.0, 0.0]])
    assert medoid in (0, 1)  # the two close vectors, not the opposite one
    assert min_pair < 0.0  # the [-1,0] outlier drags min pairwise negative
    _, _, single = medoid_cohesion([[1.0, 0.0]])
    assert single == 1.0


if __name__ == "__main__":
    for fn in [
        test_dark_crop_is_not_banked,
        test_bright_crop_is_banked,
        test_dark_gate_can_be_disabled,
        test_not_visible_crop_kept_but_not_anchor,
        test_visible_crop_is_an_anchor,
        test_cohesion_rejects_outlier_when_floor_set,
        test_cohesion_disabled_by_default_keeps_outlier,
        test_dedup_cohesion_helpers,
    ]:
        fn()
        print(f"{fn.__name__} passed")
    print("all WHO admission gate tests passed")
