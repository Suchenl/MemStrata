"""Crop attribute pack: shared visual-stratum labels for MemStrata memory crops.

Contract (keep in sync with ``vmem_bench.common.crop_attributes``):

Time is filled by the caller (segment/frame/seconds). Viewpoint, appearance state,
shot size, lighting, and occlusion are VLM multiple-choice fields with closed
enums. Failures degrade to ``unknown`` and never block the pipeline.

See also: ``docs/benchmark/crop_contract.md``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from memstrata.bank import SpatialAngle, StateAngle


class ShotSize(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    INSERT = "insert"
    UNKNOWN = "unknown"


class Lighting(str, Enum):
    DAY = "day"
    NIGHT = "night"
    INDOOR = "indoor"
    OUTDOOR_OVERCAST = "outdoor_overcast"
    ARTIFICIAL = "artificial"
    BACKLIGHT = "backlight"
    UNKNOWN = "unknown"


class Occlusion(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class Pose(str, Enum):
    """Body/handling pose — orthogonal to ``state_angle`` (appearance condition).

    A diversity axis (design_philosophy.md axiom 5): the same identity in
    different poses is legitimate cross-pose evidence to retain, not redundancy.
    """

    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    ACTION = "action"
    UNKNOWN = "unknown"


CROP_ATTRIBUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spatial_angle": {
            "type": "string",
            "enum": [e.value for e in SpatialAngle],
        },
        "state_angle": {
            "type": "string",
            "enum": [e.value for e in StateAngle],
        },
        "shot_size": {
            "type": "string",
            "enum": [e.value for e in ShotSize],
        },
        "lighting": {
            "type": "string",
            "enum": [e.value for e in Lighting],
        },
        "occlusion": {
            "type": "string",
            "enum": [e.value for e in Occlusion],
        },
        "pose": {
            "type": "string",
            "enum": [e.value for e in Pose],
        },
        "identity_visible": {"type": "boolean"},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "spatial_angle",
        "state_angle",
        "shot_size",
        "lighting",
        "occlusion",
        "pose",
        "identity_visible",
        "description",
        "confidence",
        "reasoning",
    ],
    "additionalProperties": False,
}

def _batch_crop_attribute_schema(*, with_target: bool) -> dict[str, Any]:
    """Array form of ``CROP_ATTRIBUTE_SCHEMA``: one object per crop, in image order.

    When ``with_target`` is set, each item additionally reports ``matches_target`` — a
    boolean saying whether the crop matches its requested new-entity description (the
    first-sighting / new-entity verification path C).
    """
    item_props = dict(CROP_ATTRIBUTE_SCHEMA["properties"])
    required = list(CROP_ATTRIBUTE_SCHEMA["required"])
    if with_target:
        item_props = {**item_props, "matches_target": {"type": "boolean"}}
        required = [*required, "matches_target"]
    item_schema = {
        "type": "object",
        "properties": item_props,
        "required": required,
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item_schema}},
        "required": ["items"],
        "additionalProperties": False,
    }


CLASSIFY_PROMPT = (
    "You classify one entity crop for a stratified visual memory bank.\n"
    "Pick exactly one value from each closed enum. Do not invent labels.\n\n"
    "spatial_angle: front | side | back | top | unknown\n"
    "state_angle: default | changed | damaged | unknown\n"
    "shot_size: wide | medium | close_up | extreme_close_up | insert | unknown\n"
    "lighting: day | night | indoor | outdoor_overcast | artificial | backlight | unknown\n"
    "occlusion: none | partial | heavy | unknown\n"
    "pose: standing | sitting | lying | action | unknown — body/handling pose,\n"
    "  independent of appearance state. Use 'action' for running/fighting/gesturing.\n"
    "identity_visible: true | false — whether this crop actually shows a view that\n"
    "  lets you verify WHO/WHAT this entity is. A character needs a visible frontal or\n"
    "  clear side face / distinctive identifying features; a prop needs its recognizable\n"
    "  key features. A back-of-head/back view, severe blur, near-black darkness, or\n"
    "  occlusion so heavy you cannot tell who it is → identity_visible=false. Do NOT set\n"
    "  it true just from hair color, clothing, body shape, or silhouette. Note: a clean\n"
    "  back view is still spatial_angle=back with identity_visible=false.\n"
    "description: one short English clause naming the entity's stable, recognizable\n"
    "  APPEARANCE (build, hair, clothing, colors, distinctive markings or shape). Describe\n"
    "  only what persists across shots — never the momentary action, camera, or background.\n"
    "  Empty string if the crop shows too little to describe.\n\n"
    "Entity kind: {kind}\n"
    "Entity name: {name}\n"
    "Return JSON only."
)

BATCH_CLASSIFY_PROMPT = (
    "You classify a BATCH of entity crops for a stratified visual memory bank.\n"
    "The user message contains N images IN ORDER. Return a JSON object of the form\n"
    '  {"items": [ {..attributes for image 1..}, {..for image 2..}, ... ]}\n'
    "with EXACTLY one attribute object per image, in the SAME order as the images.\n"
    "For every image, pick exactly one value from each closed enum (do not invent):\n\n"
    "spatial_angle: front | side | back | top | unknown\n"
    "state_angle: default | changed | damaged | unknown\n"
    "shot_size: wide | medium | close_up | extreme_close_up | insert | unknown\n"
    "lighting: day | night | indoor | outdoor_overcast | artificial | backlight | unknown\n"
    "occlusion: none | partial | heavy | unknown\n"
    "pose: standing | sitting | lying | action | unknown\n"
    "identity_visible: true | false — whether the crop lets you verify WHO/WHAT it is.\n"
    "description: one short English clause naming the stable, recognizable appearance;\n"
    "  empty string if the crop shows too little.\n"
    "{target_clause}"
    "Per-image context (image_index: kind / name / target):\n"
    "{items_block}\n"
    "Return JSON only."
)
_MATCHES_TARGET_CLAUSE = (
    "matches_target: true | false — whether this crop matches the requested new-entity\n"
    "  description given for its image below (omit/ignore when no target is given).\n"
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen3.5-9B-Instruct"


def _normalize_batch_item(item: Any) -> dict[str, Any]:
    """Coerce a batch item (dict or (path, ...) tuple/str) into classify() fields."""
    if isinstance(item, Mapping):
        return {
            "image_path": str(item.get("image_path") or item.get("path") or ""),
            "kind": str(item.get("kind", "") or ""),
            "name": str(item.get("name", "") or ""),
            "segment_id": item.get("segment_id"),
            "frame_index": item.get("frame_index"),
            "seconds": item.get("seconds"),
        }
    if isinstance(item, (tuple, list)):
        image_path = str(item[0]) if item else ""
        return {
            "image_path": image_path,
            "kind": "",
            "name": "",
            "segment_id": None,
            "frame_index": None,
            "seconds": None,
        }
    return {
        "image_path": str(item),
        "kind": "",
        "name": "",
        "segment_id": None,
        "frame_index": None,
        "seconds": None,
    }


def _target_matches_stem(image_path: str, target: Any) -> bool:
    """Deterministic offline stand-in for path-C verification: do the target words
    appear in the crop's filename stem? True only when every target word is present."""
    text = str(target or "").strip().lower()
    if not text:
        return False
    stem = Path(image_path).stem.lower()
    words = [w for w in text.replace("-", " ").replace("_", " ").split() if w]
    return bool(words) and all(w in stem for w in words)


def _parse_enum(enum_cls: type[Enum], raw: Any) -> Any:
    try:
        return enum_cls(str(raw))
    except ValueError:
        return enum_cls["UNKNOWN"]


@dataclass(slots=True)
class CropAttributePack:
    """Full attribute pack attached to one crop / representation."""

    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    shot_size: ShotSize = ShotSize.UNKNOWN
    lighting: Lighting = Lighting.UNKNOWN
    occlusion: Occlusion = Occlusion.UNKNOWN
    # Diversity axis (axiom 5): body/handling pose, orthogonal to state_angle.
    pose: Pose = Pose.UNKNOWN
    # WHO signal (design_philosophy.md §2 gate ③): can this crop verify identity?
    # Default True so null/heuristic/packet paths never strip identity-anchor status.
    identity_visible: bool = True
    # Observation-level appearance description d̂_i (paper Evidence Acquisition).
    # Rides the SAME structured VLM request as the labels above, so filling it costs
    # zero extra model calls; empty whenever no pack was fetched.
    description: str = ""
    # Caller-owned time fields (not from VLM).
    segment_id: int | None = None
    frame_index: int | None = None
    seconds: float | None = None
    confidence: float = 0.0
    reasoning: str = ""
    source: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def diversity_bucket(self) -> tuple[str, str, str, str]:
        """Bucket used by attribute-diverse dedup (occlusion excluded)."""
        return (
            self.spatial_angle.value,
            self.state_angle.value,
            self.shot_size.value,
            self.lighting.value,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "spatial_angle": self.spatial_angle.value,
            "state_angle": self.state_angle.value,
            "shot_size": self.shot_size.value,
            "lighting": self.lighting.value,
            "occlusion": self.occlusion.value,
            "pose": self.pose.value,
            "identity_visible": bool(self.identity_visible),
            "description": self.description,
            "segment_id": self.segment_id,
            "frame_index": self.frame_index,
            "seconds": self.seconds,
            "confidence": float(self.confidence),
            "reasoning": self.reasoning,
            "source": self.source,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    def to_annotations(self) -> dict[str, Any]:
        """Flatten into AssetRepresentation.annotations / proposal metadata."""
        return {
            "crop_attributes": self.to_dict(),
            "angle_source": self.source,
            "angle_confidence": float(self.confidence),
            "angle_reasoning": self.reasoning,
            "shot_size": self.shot_size.value,
            "lighting": self.lighting.value,
            "occlusion": self.occlusion.value,
            "identity_visible": bool(self.identity_visible),
            "observation_description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropAttributePack:
        return cls(
            spatial_angle=_parse_enum(SpatialAngle, data.get("spatial_angle")),  # type: ignore[arg-type]
            state_angle=_parse_enum(StateAngle, data.get("state_angle")),  # type: ignore[arg-type]
            shot_size=_parse_enum(ShotSize, data.get("shot_size")),  # type: ignore[arg-type]
            lighting=_parse_enum(Lighting, data.get("lighting")),  # type: ignore[arg-type]
            occlusion=_parse_enum(Occlusion, data.get("occlusion")),  # type: ignore[arg-type]
            pose=_parse_enum(Pose, data.get("pose")),  # type: ignore[arg-type]
            identity_visible=bool(data.get("identity_visible", True)),
            description=str(data.get("description", "")),
            segment_id=(int(data["segment_id"]) if data.get("segment_id") is not None else None),
            frame_index=(
                int(data["frame_index"]) if data.get("frame_index") is not None else None
            ),
            seconds=(float(data["seconds"]) if data.get("seconds") is not None else None),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reasoning=str(data.get("reasoning", "")),
            source=str(data.get("source", "unknown")),
            extra=dict(data.get("extra") or {}),
        )


class CropAttributeClassifier(Protocol):
    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        segment_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack: ...

    def classify_batch(
        self,
        items: list[Any],
        *,
        target_descriptions: list[str | None] | None = None,
    ) -> list[CropAttributePack]:
        """Attribute several crops in ONE call. ``items`` carry ``image_path`` (+ optional
        ``kind``/``name``/``segment_id``/``frame_index``/``seconds``). When
        ``target_descriptions`` is given (aligned per item), each pack records
        ``extra["matches_target"]`` for the new-entity verification path C."""
        ...


def _image_data_url(image_path: str) -> str:
    import io

    from memstrata.lib.media import load_crop_rgb_for_model

    rgb = load_crop_rgb_for_model(image_path)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class NullCropAttributeClassifier:
    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        segment_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack:
        _ = image_path, kind, name
        return CropAttributePack(
            segment_id=segment_id,
            frame_index=frame_index,
            seconds=seconds,
            source="null",
        )

    def classify_batch(
        self,
        items: list[Any],
        *,
        target_descriptions: list[str | None] | None = None,
    ) -> list[CropAttributePack]:
        _ = target_descriptions
        return [
            self.classify(
                fields["image_path"],
                kind=fields["kind"],
                name=fields["name"],
                segment_id=fields["segment_id"],
                frame_index=fields["frame_index"],
                seconds=fields["seconds"],
            )
            for fields in (_normalize_batch_item(it) for it in items)
        ]


class HeuristicCropAttributeClassifier:
    """Filename-stem hints for tests (e.g. ``hero_front_default_close_up_day``)."""

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        segment_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack:
        stem = Path(image_path).stem.lower().replace("-", "_")
        tokens = set(stem.split("_"))
        stem_compact = stem.replace("_", "")

        def _pick(enum_cls: type[Enum]) -> Enum:
            for item in enum_cls:
                if item.value == "unknown":
                    continue
                value = item.value
                if (
                    value in tokens
                    or value in stem
                    or value.replace("_", "") in stem_compact
                ):
                    return item
            return enum_cls["UNKNOWN"]  # type: ignore[return-value]

        # Identity is assumed visible unless the filename explicitly says otherwise
        # (test hook: ``*_notvisible*`` / ``*_hidden*``). A bare ``back`` token is a
        # valid diversity angle and does NOT by itself flip identity_visible here.
        identity_visible = not (
            "notvisible" in tokens or "hidden" in tokens or "notvisible" in stem_compact
        )
        spatial = _pick(SpatialAngle)
        state = _pick(StateAngle)
        shot = _pick(ShotSize)
        pack = CropAttributePack(
            spatial_angle=spatial,  # type: ignore[arg-type]
            state_angle=state,  # type: ignore[arg-type]
            shot_size=shot,  # type: ignore[arg-type]
            lighting=_pick(Lighting),  # type: ignore[arg-type]
            occlusion=_pick(Occlusion),  # type: ignore[arg-type]
            pose=_pick(Pose),  # type: ignore[arg-type]
            identity_visible=identity_visible,
            # Deterministic stand-in for the VLM appearance clause, so the offline
            # default still exercises the description write-through end to end.
            description=" ".join(
                part
                for part in (str(kind), str(name), spatial.value, state.value, shot.value)
                if part and part != "unknown"
            ).strip(),
            segment_id=segment_id,
            frame_index=frame_index,
            seconds=seconds,
            confidence=1.0,
            reasoning="heuristic_filename",
            source="heuristic",
        )
        return pack

    def classify_batch(
        self,
        items: list[Any],
        *,
        target_descriptions: list[str | None] | None = None,
    ) -> list[CropAttributePack]:
        packs: list[CropAttributePack] = []
        for index, item in enumerate(items):
            fields = _normalize_batch_item(item)
            pack = self.classify(
                fields["image_path"],
                kind=fields["kind"],
                name=fields["name"],
                segment_id=fields["segment_id"],
                frame_index=fields["frame_index"],
                seconds=fields["seconds"],
            )
            if target_descriptions is not None:
                target = (
                    target_descriptions[index]
                    if index < len(target_descriptions)
                    else None
                )
                if target:
                    pack.extra["matches_target"] = _target_matches_stem(
                        fields["image_path"], target
                    )
            packs.append(pack)
        return packs


class VlmCropAttributeClassifier:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout_sec: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MEMSTRATA_CROP_ATTR_BASE_URL")
            or os.environ.get("MEMSTRATA_ANGLE_CLASSIFIER_BASE_URL")
            or os.environ.get("MEMSTRATA_CONTEXT_JUDGER_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.model = (
            model
            or os.environ.get("MEMSTRATA_CROP_ATTR_MODEL")
            or os.environ.get("MEMSTRATA_ANGLE_CLASSIFIER_MODEL")
            or DEFAULT_MODEL
        )
        self.timeout_sec = timeout_sec

    def classify(
        self,
        image_path: str,
        *,
        kind: str = "",
        name: str = "",
        segment_id: int | None = None,
        frame_index: int | None = None,
        seconds: float | None = None,
    ) -> CropAttributePack:
        try:
            data_url = _image_data_url(image_path)
        except OSError:
            return CropAttributePack(
                segment_id=segment_id,
                frame_index=frame_index,
                seconds=seconds,
                source="vlm_error",
                reasoning="unreadable_image",
            )

        prompt = CLASSIFY_PROMPT.format(kind=kind or "unknown", name=name or "unknown")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        try:
            result = self._call_api(messages)
        except Exception as exc:  # noqa: BLE001
            return CropAttributePack(
                segment_id=segment_id,
                frame_index=frame_index,
                seconds=seconds,
                source="vlm_error",
                reasoning=str(exc)[:200],
            )

        return CropAttributePack(
            spatial_angle=_parse_enum(SpatialAngle, result.get("spatial_angle")),  # type: ignore[arg-type]
            state_angle=_parse_enum(StateAngle, result.get("state_angle")),  # type: ignore[arg-type]
            shot_size=_parse_enum(ShotSize, result.get("shot_size")),  # type: ignore[arg-type]
            lighting=_parse_enum(Lighting, result.get("lighting")),  # type: ignore[arg-type]
            occlusion=_parse_enum(Occlusion, result.get("occlusion")),  # type: ignore[arg-type]
            pose=_parse_enum(Pose, result.get("pose")),  # type: ignore[arg-type]
            identity_visible=bool(result.get("identity_visible", True)),
            description=str(result.get("description", "") or "").strip(),
            segment_id=segment_id,
            frame_index=frame_index,
            seconds=seconds,
            confidence=float(result.get("confidence", 0.0) or 0.0),
            reasoning=str(result.get("reasoning", "")),
            source="vlm",
        )

    def classify_batch(
        self,
        items: list[Any],
        *,
        target_descriptions: list[str | None] | None = None,
    ) -> list[CropAttributePack]:
        """Attribute all crops in ONE chat request (multiple image blocks).

        Sends the whole batch as a single structured call returning a JSON array of
        attribute objects (one per image, in order). When ``target_descriptions`` is
        provided each item also reports ``matches_target`` for path-C verification. On
        ANY error (unreadable image, transport failure, malformed / length-mismatched
        response) it degrades safely to per-item :meth:`classify`.
        """
        normalized = [_normalize_batch_item(it) for it in items]
        if not normalized:
            return []

        def _fallback() -> list[CropAttributePack]:
            return [
                self.classify(
                    fields["image_path"],
                    kind=fields["kind"],
                    name=fields["name"],
                    segment_id=fields["segment_id"],
                    frame_index=fields["frame_index"],
                    seconds=fields["seconds"],
                )
                for fields in normalized
            ]

        targets: list[str | None] = list(target_descriptions or [])
        with_target = any(bool(t) for t in targets)
        try:
            item_lines = []
            content: list[dict[str, Any]] = [{"type": "text", "text": ""}]
            for index, fields in enumerate(normalized):
                target = targets[index] if index < len(targets) else None
                item_lines.append(
                    f"  image {index}: {fields['kind'] or 'unknown'} / "
                    f"{fields['name'] or 'unknown'} / "
                    f"target={target or 'none'}"
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(fields["image_path"])},
                    }
                )
            content[0]["text"] = BATCH_CLASSIFY_PROMPT.format(
                target_clause=_MATCHES_TARGET_CLAUSE if with_target else "",
                items_block="\n".join(item_lines),
            )
            result = self._call_batch_api([{"role": "user", "content": content}], with_target=with_target)
        except Exception:  # noqa: BLE001 - any failure degrades to per-item classify
            return _fallback()

        rows = result.get("items") if isinstance(result, dict) else None
        if not isinstance(rows, list) or len(rows) != len(normalized):
            return _fallback()

        packs: list[CropAttributePack] = []
        for index, (fields, row) in enumerate(zip(normalized, rows)):
            if not isinstance(row, dict):
                return _fallback()
            pack = CropAttributePack.from_dict(row)
            pack.segment_id = fields["segment_id"]
            pack.frame_index = fields["frame_index"]
            pack.seconds = fields["seconds"]
            pack.source = "vlm_batch"
            target = targets[index] if index < len(targets) else None
            if target and "matches_target" in row:
                pack.extra["matches_target"] = bool(row.get("matches_target"))
            packs.append(pack)
        return packs

    def _call_api(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post_chat(
            messages,
            schema_name="crop_attribute_pack",
            schema=CROP_ATTRIBUTE_SCHEMA,
        )

    def _call_batch_api(
        self, messages: list[dict[str, Any]], *, with_target: bool
    ) -> dict[str, Any]:
        return self._post_chat(
            messages,
            schema_name="crop_attribute_pack_batch",
            schema=_batch_crop_attribute_schema(with_target=with_target),
        )

    def _post_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": int(
                os.environ.get("MEMSTRATA_CROP_ATTR_MAX_TOKENS")
                or os.environ.get("MEMSTRATA_BENCH_CROP_ATTR_MAX_TOKENS")
                or "2048"
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
            return json.loads(content)


def build_crop_attribute_classifier(
    *,
    mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> CropAttributeClassifier:
    chosen = (
        mode
        or os.environ.get("MEMSTRATA_CROP_ATTR_CLASSIFIER")
        or os.environ.get("MEMSTRATA_ANGLE_CLASSIFIER")
        or "null"
    ).strip().lower()
    if chosen in {"vlm", "mllm", "api"}:
        return VlmCropAttributeClassifier(base_url=base_url, model=model)
    if chosen in {"heuristic", "stub", "test"}:
        return HeuristicCropAttributeClassifier()
    return NullCropAttributeClassifier()


__all__ = [
    "CLASSIFY_PROMPT",
    "CROP_ATTRIBUTE_SCHEMA",
    "CropAttributeClassifier",
    "CropAttributePack",
    "HeuristicCropAttributeClassifier",
    "Lighting",
    "NullCropAttributeClassifier",
    "Occlusion",
    "Pose",
    "ShotSize",
    "VlmCropAttributeClassifier",
    "build_crop_attribute_classifier",
]
