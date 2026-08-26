"""Decomposition skill — Role-aware Asset Decomposition (paper Evidence Acquisition).

Produces ``O_n = O_n^req ∪ O_n^disc`` out of the *realized* segment:

* ``O_n^req`` — **requested-asset grounding**: for every entity the intent named, cut a
  fresh crop, encode it by type, and emit one ``Observation`` tagged
  ``source="requested"``. Identity is anchored by the caller-supplied id.
* ``O_n^disc`` — **type-constrained discovery**: an optional ``Discoverer`` proposes
  candidates that belong to a supported type but were *not* requested. Candidates
  already explained by the requested crops (``covered``, by bbox IoU) are dropped; the
  rest are emitted as ``source="discovered"`` with provisional identifiers, and the
  curate step decides via identity reconciliation whether each extends an existing
  asset or creates a new one.

Discovery is opt-in (``discoverer=None`` → requested-only, the historical behaviour) and
is type-constrained by construction: a ``Discoverer`` is only ever asked for the asset
types in ``AssetType``, so this stays controlled expansion rather than unrestricted
open-vocabulary accumulation.

This is a reusable capability, so it lives under ``memstrata.skills`` (``steps/decompose.py``
is a thin re-export shim for backward compatibility). The crop-isolation half is pluggable
via the ``Cropper`` protocol and is provided by sibling skills:
  - ``memstrata.skills.crop_acquisition`` — S5-derived SAM3+GDINO+DINOv3 propose/identify/
    novelty perception (production default; clean masked entity crops).
  - ``memstrata.skills.entity_grounding`` — lightweight single-VLM (Qwen) tight-box grounding.
Angle/attribute resolution routes through ``memstrata.mllm`` classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from memstrata.bank import AssetType, SpatialAngle, StateAngle
from memstrata.encoders import EmbeddingModel, HashEmbedding, RoleRoutedEmbedding, Vector
from memstrata.lib.crop_qa import audit_crop
from memstrata.mllm.angle_classifier import AngleClassification, AngleClassifier, NullAngleClassifier

# ω_i — how an observation was acquired (paper Evidence Acquisition).
SOURCE_REQUESTED = "requested"
SOURCE_DISCOVERED = "discovered"

# Default IoU above which a discovered proposal counts as already explained by a
# requested crop (``covered(m, q_n)`` in the paper).
DEFAULT_COVERED_IOU = 0.5


@dataclass(slots=True)
class NamedEntity:
    """Entity that g_n already named as meaningful (targeted grounding, not open-vocab)."""

    name: str
    kind: AssetType
    entity_id: str | None = None  # stable id when known (production naming / seed)
    crop_path: str | None = None  # when a crop is already isolated
    category: str = ""  # short English common noun for the open-vocab segmenter concept
    description: str = ""  # appearance hint for perception prompts (not a name oracle)
    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    temporal_tag: str = ""


@dataclass(slots=True)
class Observation:
    """Candidate observation o ∈ O_n after type-routed encoding.

    Visual evidence is stored as [image + angle]: spatial / temporal / state.
    """

    observation_id: str
    kind: AssetType
    name: str
    image_path: str
    entity_id: str | None = None
    embedding: Vector | None = None
    encoder_route: str = "general"
    quality: float = 1.0
    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    temporal_tag: str = ""
    angle_meta: dict[str, Any] = field(default_factory=dict)
    # d̂_i — observation-level appearance description (rides the attribute VLM call).
    description: str = ""
    # ω_i — requested | discovered. Only discovered observations go through identity
    # reconciliation; requested ones are anchored by their symbolic id.
    source: str = SOURCE_REQUESTED
    # Normalized [ymin,xmin,ymax,xmax] on a 0-1000 grid when the cropper reports it;
    # used to drop discoveries already explained by a requested crop.
    bbox_norm: list[int] | None = None
    # Absolute path to the FULL source frame this crop was cut from (832×480), kept
    # alongside the crop so the bank can self-audit crop↔frame provenance and support
    # frame-level retrieval. Empty when the observation has no distinct source frame
    # (e.g. a whole-frame location, where image_path already IS the frame).
    source_frame_path: str = ""


@dataclass(slots=True)
class DiscoveredEntity:
    """A type-constrained candidate the perception found without being asked.

    ``name`` is provisional (the discoverer has no naming oracle); the curate step
    resolves it against existing assets via identity reconciliation.
    """

    kind: AssetType
    crop_path: str
    name: str = ""
    bbox_norm: list[int] | None = None
    quality: float = 1.0
    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    meta: dict[str, Any] = field(default_factory=dict)


def _normalize_bbox(raw: Any) -> list[int] | None:
    try:
        box = [int(v) for v in raw]
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 else None


def _crop_path_bbox_meta(acquired: Any) -> tuple[str | None, list[int] | None, dict[str, Any]]:
    """Accept either a bare crop path or a ``{crop_path, bbox}`` mapping.

    Croppers that report a bbox let discovery tell "already explained by a requested
    crop" from "genuinely new region"; ones that only return a path still work.
    """
    if acquired is None:
        return None, None, {}
    if isinstance(acquired, dict):
        path = acquired.get("crop_path") or acquired.get("path")
        meta = acquired.get("meta") if isinstance(acquired.get("meta"), dict) else {}
        return (
            str(path) if path else None,
            _normalize_bbox(acquired.get("bbox") or acquired.get("bbox_norm")),
            dict(meta),
        )
    return str(acquired), None, {}


def _is_covered(bbox: list[int] | None, covered: list[list[int]], iou_threshold: float) -> bool:
    """``covered(m, q_n)`` — does this proposal overlap an already-acquired crop?

    Without a bbox we cannot prove overlap, so the candidate is *not* dropped; the
    curate step's identity reconciliation is the second line of defence there.
    """
    if not bbox or not covered:
        return False
    from memstrata.skills.crop_acquisition.geometry import bbox_iou

    return any(bbox_iou(bbox, other) >= iou_threshold for other in covered)


class Cropper(Protocol):
    """Isolate a crop for a named entity from a realized segment. Deploy-time concern.

    Returns the crop path, or a ``{crop_path, bbox}`` mapping when the implementation can
    also report where in the frame it cut (which lets discovery skip that region).
    """

    def crop(
        self, segment_video: str, entity: NamedEntity, *, segment_id: int
    ) -> str | dict[str, Any] | None: ...


class Discoverer(Protocol):
    """Propose type-constrained candidates from a realized segment (D_T in the paper).

    Implementations must only ever return candidates whose ``kind`` is in ``kinds``;
    that type restriction is what keeps memory expansion controlled.
    """

    def discover(
        self,
        segment_video: str,
        *,
        segment_id: int,
        kinds: tuple[AssetType, ...],
    ) -> list[DiscoveredEntity]: ...


class RoleAwareDecomposer:
    """Paper Step 3: only entities named by the intent; encoder routed by τ_j."""

    def __init__(
        self,
        embedder: EmbeddingModel | RoleRoutedEmbedding | None = None,
        *,
        cropper: Cropper | None = None,
        angle_classifier: AngleClassifier | None = None,
        crop_quality_gate: bool = False,
        discoverer: Discoverer | None = None,
        discovery_kinds: tuple[AssetType, ...] = (
            AssetType.CHARACTER,
            AssetType.PROP,
            AssetType.LOCATION,
        ),
        covered_iou: float = DEFAULT_COVERED_IOU,
    ) -> None:
        if embedder is None:
            general = HashEmbedding()
            self.embedder: EmbeddingModel | RoleRoutedEmbedding = RoleRoutedEmbedding(
                general=general,
                face=None,
                location=None,
            )
        else:
            self.embedder = embedder
        self.cropper = cropper
        self.angle_classifier: AngleClassifier = angle_classifier or NullAngleClassifier()
        # Track A packets are authoritative; production callers opt in at acquisition time.
        self.crop_quality_gate = crop_quality_gate
        # Type-constrained discovery (paper D_T). ``None`` → requested-only decomposition.
        self.discoverer = discoverer
        self.discovery_kinds = tuple(discovery_kinds)
        self.covered_iou = float(covered_iou)

    def _embed(self, image_path: str, kind: AssetType) -> tuple[Vector, str]:
        if isinstance(self.embedder, RoleRoutedEmbedding):
            return self.embedder.embed_with_route(image_path, kind.value)
        return self.embedder.embed_image(image_path), getattr(self.embedder, "name", "general")

    def _resolve_angles(
        self,
        *,
        crop: str,
        kind: AssetType,
        name: str,
        spatial: SpatialAngle,
        state: StateAngle,
    ) -> tuple[SpatialAngle, StateAngle, dict[str, Any]]:
        """Use explicit angles when known; otherwise classify the crop.

        The classifier call also carries the observation-level description d̂_i, so
        ``angle_meta['observation_description']`` is populated for free whenever a pack
        is fetched (never an extra request).
        """
        meta: dict[str, Any] = {}
        needs_classify = spatial == SpatialAngle.UNKNOWN or state == StateAngle.UNKNOWN
        if not needs_classify:
            meta["angle_source"] = "explicit"
            return spatial, state, meta

        classified: AngleClassification = self.angle_classifier.classify(
            crop,
            kind=kind.value,
            name=name,
        )
        meta.update(classified.to_annotations())
        if spatial == SpatialAngle.UNKNOWN:
            spatial = classified.spatial_angle
        if state == StateAngle.UNKNOWN:
            state = classified.state_angle
        return spatial, state, meta

    def decompose(
        self,
        *,
        segment_id: int,
        named_entities: list[NamedEntity],
        segment_video: str | None = None,
    ) -> list[Observation]:
        """Produce ``O_n = O_n^req ∪ O_n^disc`` for one realized segment.

        Requested observations come first and keep their original order, so callers
        that index into the result (and the curate step's identity anchoring) are
        unaffected by discovery.
        """
        observations = self._decompose_requested(
            segment_id=segment_id, named_entities=named_entities, segment_video=segment_video
        )
        observations.extend(
            self._decompose_discovered(
                segment_id=segment_id, segment_video=segment_video, requested=observations
            )
        )
        return observations

    def _decompose_requested(
        self,
        *,
        segment_id: int,
        named_entities: list[NamedEntity],
        segment_video: str | None,
    ) -> list[Observation]:
        """O_n^req — one observation per named entity that has (or can get) a crop."""
        observations: list[Observation] = []
        for index, entity in enumerate(named_entities):
            crop = entity.crop_path
            bbox: list[int] | None = None
            acquisition_meta: dict[str, Any] = {}
            if crop is None and segment_video and self.cropper is not None:
                acquired = self.cropper.crop(segment_video, entity, segment_id=segment_id)
                crop, bbox, acquisition_meta = _crop_path_bbox_meta(acquired)
            if not crop:
                continue
            quality_meta: dict[str, Any] = {}
            if self.crop_quality_gate:
                report = audit_crop(crop)
                quality_meta["crop_quality"] = report.to_dict()
                if not report.accepted:
                    continue
            vector, route = self._embed(crop, entity.kind)
            spatial, state, angle_meta = self._resolve_angles(
                crop=crop,
                kind=entity.kind,
                name=entity.name,
                spatial=entity.spatial_angle,
                state=entity.state_angle,
            )
            angle_meta.update(quality_meta)
            angle_meta.update(acquisition_meta)
            obs_id = entity.entity_id or f"{entity.kind.value}_{entity.name}_{segment_id}_{index}"
            observations.append(
                Observation(
                    observation_id=str(obs_id),
                    kind=entity.kind,
                    name=entity.name,
                    image_path=crop,
                    entity_id=entity.entity_id,
                    embedding=vector,
                    encoder_route=route,
                    spatial_angle=spatial,
                    state_angle=state,
                    temporal_tag=entity.temporal_tag or f"segment_{segment_id}",
                    angle_meta=angle_meta,
                    description=str(angle_meta.get("observation_description", "")),
                    source=SOURCE_REQUESTED,
                    bbox_norm=bbox,
                )
            )
        return observations

    def _decompose_discovered(
        self,
        *,
        segment_id: int,
        segment_video: str | None,
        requested: list[Observation],
    ) -> list[Observation]:
        """O_n^disc — type-constrained candidates not explained by ``requested``."""
        if self.discoverer is None or not segment_video or not self.discovery_kinds:
            return []
        try:
            candidates = self.discoverer.discover(
                segment_video, segment_id=segment_id, kinds=self.discovery_kinds
            )
        except Exception:  # noqa: BLE001 - discovery is best-effort, never fails a segment
            return []

        allowed = set(self.discovery_kinds)
        covered_boxes = [o.bbox_norm for o in requested if o.bbox_norm]
        observations: list[Observation] = []
        for index, cand in enumerate(candidates):
            if cand.kind not in allowed or not cand.crop_path:
                continue  # type constraint: never admit an unsupported type
            if _is_covered(cand.bbox_norm, covered_boxes, self.covered_iou):
                continue
            if self.crop_quality_gate:
                report = audit_crop(cand.crop_path)
                if not report.accepted:
                    continue
            vector, route = self._embed(cand.crop_path, cand.kind)
            provisional = cand.name or f"{cand.kind.value}_disc_c{segment_id:03d}_{index}"
            spatial, state, angle_meta = self._resolve_angles(
                crop=cand.crop_path,
                kind=cand.kind,
                name=provisional,
                spatial=cand.spatial_angle,
                state=cand.state_angle,
            )
            angle_meta.update(cand.meta)
            observations.append(
                Observation(
                    observation_id=f"disc_{cand.kind.value}_c{segment_id:03d}_{index}",
                    kind=cand.kind,
                    name=provisional,
                    image_path=cand.crop_path,
                    entity_id=None,  # provisional: curate reconciles or creates
                    embedding=vector,
                    encoder_route=route,
                    quality=float(cand.quality),
                    spatial_angle=spatial,
                    state_angle=state,
                    temporal_tag=f"segment_{segment_id}",
                    angle_meta=angle_meta,
                    description=str(angle_meta.get("observation_description", "")),
                    source=SOURCE_DISCOVERED,
                    bbox_norm=cand.bbox_norm,
                )
            )
            if cand.bbox_norm:
                # Later candidates must not re-report the same region.
                covered_boxes.append(cand.bbox_norm)
        return observations

    def observations_from_packet_dicts(self, packet: dict[str, Any]) -> list[Observation]:
        """Helper: map bench ObservationPacket rows into Observation (no open-vocab discovery)."""
        segment_id = int(packet["segment_id"])
        out: list[Observation] = []
        for index, raw in enumerate(packet.get("observations", [])):
            kind = AssetType(str(raw["kind"]))
            crop = str(raw["crop_path"])
            vector, route = self._embed(crop, kind)
            spatial = str(raw.get("spatial_angle", SpatialAngle.UNKNOWN.value))
            state = str(raw.get("state_angle", StateAngle.UNKNOWN.value))
            try:
                spatial_angle = SpatialAngle(spatial)
            except ValueError:
                spatial_angle = SpatialAngle.UNKNOWN
            try:
                state_angle = StateAngle(state)
            except ValueError:
                state_angle = StateAngle.UNKNOWN
            out.append(
                Observation(
                    observation_id=str(raw.get("representation_id", f"obs_{segment_id}_{index}")),
                    kind=kind,
                    name=str(raw["name"]),
                    image_path=crop,
                    entity_id=str(raw["entity_id"]),
                    embedding=vector,
                    encoder_route=route,
                    spatial_angle=spatial_angle,
                    state_angle=state_angle,
                    temporal_tag=str(raw.get("temporal_tag", f"segment_{segment_id}")),
                    description=str(raw.get("description", "")),
                )
            )
        return out
