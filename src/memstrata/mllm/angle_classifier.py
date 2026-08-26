"""Production VLM classifier: crop → spatial_angle + state_angle.

Thin compatibility layer over :mod:`memstrata.mllm.crop_attributes`. Full packs
(shot size / lighting / occlusion) live there; this module projects the subset
used by decompose / curate angle fields.

Used on the production closed loop (decompose / ingest_observation). Track A
ObservationPacket angles remain authoritative and bypass this classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from memstrata.bank import SpatialAngle, StateAngle
from memstrata.mllm.crop_attributes import (
    CropAttributeClassifier,
    CropAttributePack,
    HeuristicCropAttributeClassifier,
    NullCropAttributeClassifier,
    VlmCropAttributeClassifier,
    build_crop_attribute_classifier,
)

# Re-export schema/prompt names used by older callers / docs.
ANGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "spatial_angle": {
            "type": "string",
            "enum": ["front", "side", "back", "top", "unknown"],
        },
        "state_angle": {
            "type": "string",
            "enum": ["default", "changed", "damaged", "unknown"],
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["spatial_angle", "state_angle", "confidence", "reasoning"],
    "additionalProperties": False,
}

CLASSIFY_PROMPT = (
    "You classify one entity crop for a stratified visual memory bank.\n"
    "Decide the entity's spatial viewpoint relative to the camera, and its "
    "appearance/condition state.\n"
    "Entity kind: {kind}\n"
    "Entity name: {name}\n"
    "Return JSON only."
)


@dataclass(slots=True)
class AngleClassification:
    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    confidence: float = 0.0
    reasoning: str = ""
    source: str = "unknown"
    pack: CropAttributePack | None = None

    def to_annotations(self) -> dict[str, Any]:
        if self.pack is not None:
            return self.pack.to_annotations()
        return {
            "angle_source": self.source,
            "angle_confidence": float(self.confidence),
            "angle_reasoning": self.reasoning,
        }


class AngleClassifier(Protocol):
    """Classify a crop into spatial / state visual strata."""

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
    ) -> AngleClassification: ...


def _pack_to_angle(pack: CropAttributePack) -> AngleClassification:
    return AngleClassification(
        spatial_angle=pack.spatial_angle,
        state_angle=pack.state_angle,
        confidence=pack.confidence,
        reasoning=pack.reasoning,
        source=pack.source,
        pack=pack,
    )


class NullAngleClassifier:
    """No-op classifier: leave angles unknown (default offline / Track A safe)."""

    def __init__(self) -> None:
        self._inner: CropAttributeClassifier = NullCropAttributeClassifier()

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
    ) -> AngleClassification:
        return _pack_to_angle(self._inner.classify(image_path, kind=kind, name=name))


class HeuristicAngleClassifier:
    """Deterministic stub for tests: read hints from filename stem.

    Examples: ``hero_front_default.jpg`` → front/default;
    ``prop_side_damaged.png`` → side/damaged.
    """

    def __init__(self) -> None:
        self._inner: CropAttributeClassifier = HeuristicCropAttributeClassifier()

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
    ) -> AngleClassification:
        return _pack_to_angle(self._inner.classify(image_path, kind=kind, name=name))


class VlmAngleClassifier:
    """OpenAI-compatible multimodal classifier (full crop attribute pack under the hood)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout_sec: float = 60.0,
    ) -> None:
        self._inner = VlmCropAttributeClassifier(
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
        )

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
    ) -> AngleClassification:
        return _pack_to_angle(self._inner.classify(image_path, kind=kind, name=name))


def build_angle_classifier(
    *,
    mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> AngleClassifier:
    """Factory: ``vlm`` | ``heuristic`` | ``null`` (default null)."""
    inner = build_crop_attribute_classifier(mode=mode, base_url=base_url, model=model)
    if isinstance(inner, VlmCropAttributeClassifier):
        return VlmAngleClassifier(base_url=base_url, model=model)
    if isinstance(inner, HeuristicCropAttributeClassifier):
        return HeuristicAngleClassifier()
    return NullAngleClassifier()


__all__ = [
    "ANGLE_SCHEMA",
    "CLASSIFY_PROMPT",
    "AngleClassification",
    "AngleClassifier",
    "HeuristicAngleClassifier",
    "NullAngleClassifier",
    "VlmAngleClassifier",
    "build_angle_classifier",
]
