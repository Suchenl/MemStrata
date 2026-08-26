"""Angle-aware representation selection (visual stratum)."""

from __future__ import annotations

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    SpatialAngle,
    StateAngle,
)
from memstrata.steps.compose import compose, select_reps
from memstrata.steps.intent import AssetReference, CompositionRequest


def _asset_with_views() -> Asset:
    asset = Asset(
        asset_id="char_alice",
        kind=AssetType.CHARACTER,
        name="Alice",
        status=LifecycleStatus.REUSABLE,
    )
    asset.representations = [
        AssetRepresentation(
            representation_id="alice@s001",
            asset_id="char_alice",
            object_uri="/tmp/front.png",
            origin_segment_id=1,
            spatial_angle=SpatialAngle.FRONT,
            state_angle=StateAngle.DEFAULT,
            temporal_tag="segment_1",
        ),
        AssetRepresentation(
            representation_id="alice@s002",
            asset_id="char_alice",
            object_uri="/tmp/side.png",
            origin_segment_id=2,
            spatial_angle=SpatialAngle.SIDE,
            state_angle=StateAngle.DEFAULT,
            temporal_tag="segment_2",
        ),
        AssetRepresentation(
            representation_id="alice@s003",
            asset_id="char_alice",
            object_uri="/tmp/changed.png",
            origin_segment_id=3,
            spatial_angle=SpatialAngle.FRONT,
            state_angle=StateAngle.CHANGED,
            temporal_tag="segment_3",
        ),
    ]
    return asset


def test_recency_without_preference() -> None:
    asset = _asset_with_views()
    chosen = select_reps(asset, function="identity_anchor")
    assert chosen == ["alice@s003"]


def test_preferred_spatial_over_recency() -> None:
    asset = _asset_with_views()
    chosen = select_reps(
        asset,
        function="identity_anchor",
        preferred_spatial=SpatialAngle.SIDE,
    )
    assert chosen == ["alice@s002"]


def test_preferred_state_over_recency() -> None:
    asset = _asset_with_views()
    chosen = select_reps(
        asset,
        function="identity_anchor",
        preferred_state=StateAngle.CHANGED,
    )
    assert chosen == ["alice@s003"]


def test_compose_wires_preferred_angles() -> None:
    bank = AssetBank()
    bank.add_asset(_asset_with_views())
    ctx = compose(
        bank,
        CompositionRequest(
            references=[
                AssetReference(
                    asset_id="char_alice",
                    preferred_spatial=SpatialAngle.SIDE,
                )
            ]
        ),
    )
    assert ctx.representation_ids["char_alice"] == ["alice@s002"]


if __name__ == "__main__":
    test_recency_without_preference()
    test_preferred_spatial_over_recency()
    test_preferred_state_over_recency()
    test_compose_wires_preferred_angles()
    print("ok")
