"""Crop attribute pack + attribute-diverse dedup."""

from __future__ import annotations

from memstrata.bank import SpatialAngle, StateAngle
from memstrata.mllm.angle_classifier import HeuristicAngleClassifier
from memstrata.mllm.crop_attributes import (
    CropAttributePack,
    HeuristicCropAttributeClassifier,
    Lighting,
    NullCropAttributeClassifier,
    ShotSize,
)
from memstrata.lib.dedup import select_angle_diverse, select_attribute_diverse


def test_heuristic_pack_from_filename() -> None:
    clf = HeuristicCropAttributeClassifier()
    pack = clf.classify("/tmp/hero_front_default_close_up_day.png")
    assert pack.spatial_angle == SpatialAngle.FRONT
    assert pack.state_angle == StateAngle.DEFAULT
    assert pack.shot_size == ShotSize.CLOSE_UP
    assert pack.lighting == Lighting.DAY
    assert pack.source == "heuristic"
    assert pack.diversity_bucket() == ("front", "default", "close_up", "day")


def test_null_pack_unknown() -> None:
    pack = NullCropAttributeClassifier().classify("/tmp/x.png", segment_id=3, frame_index=10)
    assert pack.spatial_angle == SpatialAngle.UNKNOWN
    assert pack.segment_id == 3
    assert pack.frame_index == 10
    assert "crop_attributes" in pack.to_annotations()


def test_pack_roundtrip() -> None:
    original = CropAttributePack(
        spatial_angle=SpatialAngle.SIDE,
        state_angle=StateAngle.CHANGED,
        shot_size=ShotSize.MEDIUM,
        lighting=Lighting.NIGHT,
        segment_id=1,
        seconds=12.5,
        source="test",
    )
    restored = CropAttributePack.from_dict(original.to_dict())
    assert restored.spatial_angle == SpatialAngle.SIDE
    assert restored.shot_size == ShotSize.MEDIUM
    assert restored.seconds == 12.5


def test_select_attribute_diverse_covers_buckets() -> None:
    kept = select_attribute_diverse(
        bucket_keys=[
            ("front", "default", "close_up", "day"),
            ("front", "default", "close_up", "day"),
            ("front", "default", "medium", "day"),
            ("side", "default", "close_up", "night"),
        ],
        vectors=None,
        quality=[0.5, 0.9, 0.8, 0.7],
        max_keep=3,
    )
    assert set(kept) == {1, 2, 3}


def test_select_angle_diverse_wrapper() -> None:
    kept = select_angle_diverse(
        spatial_angles=["front", "front", "side", "back"],
        state_angles=["default", "default", "default", "default"],
        vectors=None,
        quality=[0.5, 0.9, 0.8, 0.7],
        max_keep=3,
    )
    assert set(kept) == {1, 2, 3}


def test_angle_classifier_projects_pack() -> None:
    hit = HeuristicAngleClassifier().classify("/tmp/prop_back_damaged_wide_indoor.png")
    assert hit.spatial_angle == SpatialAngle.BACK
    assert hit.state_angle == StateAngle.DAMAGED
    assert hit.pack is not None
    assert hit.pack.shot_size == ShotSize.WIDE
    assert hit.pack.lighting == Lighting.INDOOR
    assert "crop_attributes" in hit.to_annotations()
