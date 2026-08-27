"""Curation skill — Asset Memory & Management / 落库 (paper Stratified Update).

Admits decompose Observations into the stratified `AssetBank`:

* **identity anchoring** — requested observations are anchored by symbolic identifier;
  *discovered* ones go through type-restricted identity reconciliation (χ) that blends
  description and visual similarity before merging or creating a record;
* **WHO-before-WHERE admission gates** — dark / identity-visible / embedding-cohesion;
* **stratum-restricted novelty** — near-duplicate suppression compares only against
  evidence in a *compatible* visual stratum, so a new view/state is never dropped for
  looking unlike a stored one;
* **angle-&-attribute-diverse selection** under a per-type budget B_τ, with a reserved
  slot for the newest observation so a full asset can still learn;
* **lifecycle deprecation** with traceable `deprecated_by` markers (留痕) — including the
  bank-wide budget, which isolates rather than deletes;
* a **cohesion self-audit** sweep that retroactively isolates other-identity intruders.

All thresholds are per-type (B_τ / γ_τ / β_τ / λ_τ) and carried by `MemoryPolicy`, which
the pipeline and the production runner share so a preset can never be silently dropped.

Reusable capability, so it lives under `memstrata.skills` (`steps/curate.py` is a thin
re-export shim). Pairs with the `decomposition` skill (which produces the Observations) and
mirrors the bench annotation pipeline's library-building logic without importing vmem_bench.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from memstrata.skills.decomposition import SOURCE_DISCOVERED, Observation
from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRelation,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    RelationType,
    SpatialAngle,
    StateAngle,
)
from memstrata.encoders import EmbeddingModel, HashEmbedding, Vector
from memstrata.lib.crop_quality import (
    is_dark_low_information,
    is_overexposed_low_information,
)
from memstrata.lib.dedup import (
    compatible_stratum,
    cosine_or_none,
    largest_cohesive_subcluster,
    medoid_cohesion,
    select_attribute_diverse,
    similarity_to_set,
    text_similarity,
)
from memstrata.mllm.angle_classifier import AngleClassifier, NullAngleClassifier
from memstrata.mllm.crop_attributes import (
    CropAttributeClassifier,
    CropAttributePack,
    NullCropAttributeClassifier,
    Occlusion,
)
from memstrata.mllm.identity_judge import (
    IdentityJudge,
    NullIdentityJudge,
    build_identity_judge,
)

# Embedding backends that carry no semantic signal. Similarity thresholds calibrated for
# a real encoder are meaningless against these, so gates that compare embeddings across
# *different* crops must stay off rather than reject everything at random.
_NON_SEMANTIC_EMBEDDERS = frozenset({"hash-fallback"})


def _is_semantic_embedder(embedder: Any) -> bool:
    """Whether ``embedder`` produces comparable, semantically meaningful vectors."""
    name = str(getattr(embedder, "name", "") or "")
    if name == "role-routed":  # inspect the route actually used for general crops
        return _is_semantic_embedder(getattr(embedder, "general", None))
    return bool(name) and name not in _NON_SEMANTIC_EMBEDDERS


def _resolve_by_type(
    default: Any,
    by_type: Mapping[str, Any] | None,
    kind: AssetType | str | None,
) -> Any:
    """Look up a per-type override, falling back to the shared default."""
    if not by_type:
        return default
    key = str(getattr(kind, "value", kind) or "")
    return by_type.get(key, default)


@dataclass(frozen=True)
class MemoryPolicy:
    """Write-path policy shared by ``MemStrata`` and the production runner.

    This type exists because the production preset used to be applied *only* when
    ``MemStrata`` constructed the curator/decomposer itself. The production runner built
    them first, so every preset knob (crop-quality gate, bank-wide budget, angle
    classifier) was silently discarded and the "stratified" bank ran with all-unknown
    angles. Passing one policy object to whoever builds the components removes that
    whole failure mode.

    Thresholds are per asset type (paper B_τ / γ_τ / β_τ / λ_τ). The values below are
    *starting points*, not calibrated constants — run
    ``experiments/*_memstrata_write_path_calibration`` against a labelled control set
    whenever the encoder changes, because every one of them is encoder-relative.
    """

    name: str = "default"

    # --- B_τ: per-asset representation budget -----------------------------------
    max_reps_per_asset: int = 5
    max_reps_by_type: Mapping[str, int] = field(default_factory=dict)
    # Farthest-point spacing for the diversity fill.
    min_distance: float = 0.15

    # --- γ_τ: near-duplicate ceiling inside a compatible stratum -----------------
    redundancy_threshold: float = 0.92
    redundancy_by_type: Mapping[str, float] = field(default_factory=dict)

    # --- gate ②: embedding-cohesion admission floor ------------------------------
    cohesion_floor: float = 0.0
    cohesion_by_type: Mapping[str, float] = field(default_factory=dict)
    cohesion_min_refs: int = 2
    selfaudit_cohesion_floor: float | None = None
    selfaudit_reference: str = "medoid"
    # Run the library-level self-audit after every segment (the fourth, retroactive gate).
    selfaudit_each_segment: bool = False

    # --- β_τ / λ_τ: identity reconciliation for discovered observations ----------
    reconcile_threshold: float = 0.55
    reconcile_by_type: Mapping[str, float] = field(default_factory=dict)
    reconcile_text_weight: float = 0.5
    reconcile_text_weight_by_type: Mapping[str, float] = field(default_factory=dict)

    # --- VLM-first identity adjudication (slow write path, paper §4.5) -----------
    # The encoder score χ still drives the shortlist and the high-confidence short-circuit;
    # a configured VLM only adjudicates the χ *gray zone* around β_τ. Off by default and
    # inert without a real judge (NullIdentityJudge abstains), so behavior is unchanged
    # until BOTH the flag is on AND a VLM judge is injected. See
    # the original calibration workspace/20260725_vlm_vs_embedding_robustness.
    identity_vlm_enabled: bool = False
    # χ ≥ β_τ + shortcircuit_margin → accept SAME on the encoder alone (skip the VLM);
    # this is the high-precision ArcFace/DINOv3 fast path from the robustness study.
    identity_shortcircuit_margin: float = 0.15
    # β_τ - gray_margin < χ < β_τ + shortcircuit_margin → the gray zone routed to the VLM.
    identity_gray_margin: float = 0.10
    # Minimum VLM confidence required to ACT on a "same" verdict (per-model calibrated:
    # ~0.90 for a 32B judge; a smaller judge must be set far more conservative).
    identity_vlm_theta: float = 0.90
    # Up to this many diverse reference crops of the candidate are sent in the SINGLE judge
    # call (the endpoint accepts several images per prompt), so identity is judged against a
    # cross-view/state reference set rather than one possibly-unlucky frame — at no extra
    # call count. Kept small to bound prompt size.
    identity_max_references: int = 4
    # Strong-blur crops are where every judge false-merges; in the gray zone a crop below
    # this sharpness is deferred to a new provisional record rather than hard-merged.
    identity_blur_defer: bool = True
    identity_blur_min_sharpness: float = 40.0

    # --- bank-wide and component knobs ------------------------------------------
    max_total_representations: int | None = None
    dark_gate: bool = True
    attributes_when_angles_known: bool = True
    crop_quality_gate: bool = False
    relation_hops: int = 0
    discovery: bool = False

    @classmethod
    def production(cls, **overrides: Any) -> MemoryPolicy:
        """Long-video production preset: the lossy/scene-dependent knobs turned on.

        Per-type values follow the asset semantics: characters carry the most legitimate
        view/pose variation so they get the largest budget and the most permissive
        redundancy ceiling; locations repeat near-identically across shots so they are
        deduplicated hardest. Reconciliation is strictest for characters because merging
        two people is the memory bank's first red line.
        """
        # Per-type similarity thresholds below are CALIBRATED for the DINOv3 encoder on the
        # labelled LSMDC control set (the original calibration workspace/
        # 20260725_memstrata_write_path_calibration/calibration_result.json). They replaced the
        # earlier hand-tuned starting points:
        #   redundancy_by_type old: {"character": 0.94, "prop": 0.92, "location": 0.88}
        #   cohesion_by_type   old: {"character": 0.35, "prop": 0.30, "location": 0.25}
        #   reconcile_by_type  old: {"character": 0.60, "prop": 0.55, "location": 0.50}
        #   cohesion_floor     old: 0.30
        # Provenance / caveats: redundancy = same-entity p99; reconcile = P>=0.95 operating
        # point (a wrong merge is unrecoverable); cohesion = self-audit medoid max-F1 point
        # (DINOv3 cannot reach P>=0.9 on LSMDC faces — AUC≈0.79 char / 0.72 prop — so this
        # floor trades precision for recall and WILL isolate some legit off-angle views;
        # consider ArcFace routing for `character`). `location` stays at the old hand-tuned
        # values: the control set has NO labelled location entities, so it is uncalibrated.
        # `prop` β=0.21 comes from only 11 labelled entities (10 positive pairs) — treat as
        # provisional until more prop labels exist.
        defaults: dict[str, Any] = {
            "name": "production",
            "max_reps_per_asset": 5,
            "max_reps_by_type": {"character": 6, "prop": 4, "location": 3},
            "redundancy_threshold": 0.92,
            "redundancy_by_type": {"character": 0.89, "prop": 0.82, "location": 0.88},
            "cohesion_floor": 0.35,
            "cohesion_by_type": {"character": 0.51, "prop": 0.35, "location": 0.25},
            "selfaudit_each_segment": True,
            "reconcile_threshold": 0.55,
            "reconcile_by_type": {"character": 0.75, "prop": 0.21, "location": 0.50},
            "reconcile_text_weight": 0.5,
            "reconcile_text_weight_by_type": {
                "character": 0.35,
                "prop": 0.50,
                "location": 0.60,
            },
            "max_total_representations": 512,
            "crop_quality_gate": True,
            "attributes_when_angles_known": True,
            "relation_hops": 1,
            # VLM-first identity gate: enabled for production, but still inert unless a real
            # judge is injected (build_identity_judge honours MEMSTRATA_IDENTITY_JUDGE). θ=0.90
            # is the 32B operating point; drop the judge to 8B only with a more conservative θ.
            "identity_vlm_enabled": True,
            "identity_vlm_theta": 0.90,
        }
        defaults.update(overrides)
        return cls(**defaults)


def _parse_spatial(raw: Any) -> SpatialAngle:
    try:
        return SpatialAngle(str(raw))
    except ValueError:
        return SpatialAngle.UNKNOWN


def _parse_state(raw: Any) -> StateAngle:
    try:
        return StateAngle(str(raw))
    except ValueError:
        return StateAngle.UNKNOWN


def _attr_bucket(rep: AssetRepresentation) -> tuple[str, str, str, str, str]:
    """Diversity stratum key (axiom 5): (spatial, state, shot, lighting, pose).

    ``pose`` is an optional axis — absent inputs default to ``unknown`` (a no-op
    that never *reduces* diversity granularity), present inputs keep same-identity
    different-pose evidence from being collapsed as redundant.
    """
    attrs = rep.annotations.get("crop_attributes")
    if isinstance(attrs, dict):
        return (
            str(attrs.get("spatial_angle", SpatialAngle.UNKNOWN.value)),
            str(attrs.get("state_angle", StateAngle.UNKNOWN.value)),
            str(attrs.get("shot_size", "unknown")),
            str(attrs.get("lighting", "unknown")),
            str(attrs.get("pose", "unknown")),
        )
    return (
        rep.spatial_angle.value,
        rep.state_angle.value,
        str(rep.annotations.get("shot_size", "unknown")),
        str(rep.annotations.get("lighting", "unknown")),
        str(rep.annotations.get("pose", "unknown")),
    )


def _bucket_is_known(bucket: tuple[str, ...]) -> bool:
    return any(part and part != "unknown" for part in bucket)


# Crop-acquisition quality profile of a GroundingDINO bbox-only fallback (no SAM3 mask,
# so the crop still carries background the mask would have removed).
_BBOX_ONLY_QUALITY_PROFILE = "bbox_high_recall_no_mask"


def _annotate_crop_acquisition_review(annotations: dict[str, Any]) -> None:
    """Flag a GDINO bbox-only crop for later review, without changing write-path scoring.

    The high-recall grounding fallback keeps a specialist crop when SAM3 masking fails;
    its box includes background, so it is a weaker identity reference than a mask-gated
    crop. We only *annotate* it (留痕) — the per-type calibration owns the actual quality
    thresholds/eviction, so admission and diversity behaviour must stay unchanged here.
    Downstream audit / compose can down-weight or re-check reps carrying ``needs_review``.
    """
    acq = annotations.get("crop_acquisition")
    if not isinstance(acq, dict):
        return
    detail = acq.get("source_detail")
    profile = detail.get("quality_profile") if isinstance(detail, dict) else None
    if profile == _BBOX_ONLY_QUALITY_PROFILE:
        annotations["needs_review"] = True
        annotations["review_reason"] = "gdino_bbox_only_no_mask"


def _crop_sharpness(image_path: str | None) -> float | None:
    """Variance of the Laplacian (focus measure); ``None`` if the crop is unreadable.

    Higher = sharper. Used only by the VLM-first gray-zone defer: the robustness study
    showed every judge (encoder and VLM) false-merges on strong blur, so a low-sharpness
    crop must not be hard-merged. Deterministic and cheap (single grayscale pass).
    """
    if not image_path:
        return None
    try:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float64)
    except (OSError, ValueError):
        return None
    if gray.size == 0:
        return None
    # 4-neighbour Laplacian without SciPy: sum of shifted differences.
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    if lap.size == 0:
        return None
    return float(lap.var())


_REFERENCE_ASPECT_BY_KIND = {
    AssetType.CHARACTER: "identity_anchor",
    AssetType.PROP: "object_continuity",
    AssetType.LOCATION: "scene_reference",
}


def _reference_aspect(kind: AssetType) -> str:
    """Conditioning role for a type; tolerant of future/unknown enum values."""
    return _REFERENCE_ASPECT_BY_KIND.get(kind, "identity_anchor")


def _rep_quality(rep: AssetRepresentation) -> float:
    """Best per-purpose quality, falling back to the raw annotation score."""
    if rep.quality_by_purpose:
        return max(rep.quality_by_purpose.values())
    return float(rep.annotations.get("quality", 0.0))


def _rep_identity_visible(rep: AssetRepresentation) -> bool:
    """Whether a rep can verify WHO it is (gate ③). Defaults to visible."""
    attrs = rep.annotations.get("crop_attributes")
    if isinstance(attrs, dict) and "identity_visible" in attrs:
        return bool(attrs["identity_visible"])
    return bool(rep.annotations.get("identity_visible", True))


def _rep_occlusion_heavy(rep: AssetRepresentation) -> bool:
    """Whether the crop attributes report heavy occlusion (Occlusion.HEAVY).

    A heavily-occluded crop cannot reliably verify WHO it is even if the classifier
    left ``identity_visible`` at its permissive default, so admission treats it like a
    non-visible view. Reads the same annotation shapes the attribute pack flattens to
    (``crop_attributes.occlusion`` or the top-level ``occlusion`` mirror).
    """
    heavy = Occlusion.HEAVY.value
    attrs = rep.annotations.get("crop_attributes")
    if isinstance(attrs, dict) and "occlusion" in attrs:
        return str(attrs.get("occlusion")) == heavy
    return str(rep.annotations.get("occlusion", "")) == heavy


def _same_dimension_group(reps: list[AssetRepresentation]) -> list[AssetRepresentation]:
    """Keep only the reps whose embeddings share the most common dimensionality."""
    if len(reps) < 2:
        return reps
    counts: dict[int, int] = {}
    for rep in reps:
        dim = len(rep.annotations.get("embedding") or ())
        counts[dim] = counts.get(dim, 0) + 1
    if len(counts) == 1:
        return reps
    modal = max(counts, key=lambda d: (counts[d], d))
    return [rep for rep in reps if len(rep.annotations.get("embedding") or ()) == modal]


def _is_placeholder_rep(rep: AssetRepresentation) -> bool:
    """A pixel-less record (seeded appearance text, no crop yet).

    Screenplay seeding registers every main entity with an empty ``crop_path`` so the
    identity is addressable before it is ever seen. Such a rep carries no visual
    evidence, so it must never consume a slot in the per-asset budget — otherwise the
    very first bank write permanently costs one of the (few) real evidence slots.
    """
    return not str(rep.object_uri or "").strip()


@dataclass
class EntityObservation:
    """Legacy production observation shape (name-anchored; optional entity_id)."""

    observation_id: str
    kind: AssetType
    name: str
    image_path: str
    quality: float = 1.0
    entity_id: str | None = None
    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    temporal_tag: str = ""
    angle_meta: dict[str, Any] = field(default_factory=dict)


class MemoryUpdater:
    """Paper Step 4: identity from naming; embeddings + angle strata gate R_j."""

    def __init__(
        self,
        bank: AssetBank,
        embedder: EmbeddingModel | None = None,
        *,
        policy: MemoryPolicy | None = None,
        max_reps_per_asset: int | None = None,
        max_frames_per_asset: int | None = None,  # alias
        min_distance: float | None = None,
        redundancy_threshold: float | None = None,
        embed_on_ingest: bool = True,
        attributes_when_angles_known: bool | None = None,
        max_total_representations: int | None = None,
        disable_deprecation: bool = False,
        dark_gate: bool | None = None,
        cohesion_floor: float | None = None,
        cohesion_min_refs: int | None = None,
        selfaudit_cohesion_floor: float | None = None,
        selfaudit_reference: str | None = None,
        angle_classifier: AngleClassifier | None = None,
        crop_attribute_classifier: CropAttributeClassifier | None = None,
        identity_judge: IdentityJudge | None = None,
        # Ignored legacy InverseIngester kwargs (appearance-threshold path removed).
        high_threshold: float | None = None,
        low_threshold: float | None = None,
        vlm_judger: Any = None,
        mode: str | None = None,
        workspace_path: Any = None,
    ) -> None:
        _ = high_threshold, low_threshold, vlm_judger, mode, workspace_path
        # Explicit kwargs win over the policy; the policy wins over library defaults.
        self.policy = policy or MemoryPolicy()
        pol = self.policy

        def _pick(explicit: Any, from_policy: Any) -> Any:
            return from_policy if explicit is None else explicit

        self.bank = bank
        self.embedder = embedder or HashEmbedding()
        self.max_reps_per_asset = int(
            max_frames_per_asset or _pick(max_reps_per_asset, pol.max_reps_per_asset)
        )
        self.max_reps_by_type = dict(pol.max_reps_by_type)
        self.min_distance = float(_pick(min_distance, pol.min_distance))
        self.redundancy_threshold = float(
            _pick(redundancy_threshold, pol.redundancy_threshold)
        )
        self.redundancy_by_type = dict(pol.redundancy_by_type)
        self.embed_on_ingest = embed_on_ingest
        # When angles are already known, skip the extra attribute-classifier call
        # unless the caller still wants shot_size/lighting for diversity buckets.
        self.attributes_when_angles_known = bool(
            _pick(attributes_when_angles_known, pol.attributes_when_angles_known)
        )
        # Optional global cap on live representations across the whole bank
        # (None disables it; per-asset diversity still applies first).
        self.max_total_representations = _pick(
            max_total_representations, pol.max_total_representations
        )
        # Ablation switch: when True the lifecycle/avoidance path is disabled — state
        # events never mark representations deprecated, so `compose` keeps stale evidence
        # and emits no exclusions (paper ablation "- lifecycle avoidance").
        self.disable_deprecation = disable_deprecation
        # WHO-before-WHERE admission gates (design_philosophy.md §2).
        # ① dark gate: deterministic, on by default (unreadable crops pass through).
        self.dark_gate = bool(_pick(dark_gate, pol.dark_gate))
        # ② embedding cohesion floor. Similarity floors are encoder-relative: one
        # calibrated for DINOv3 would reject almost everything under the non-semantic
        # offline fallback, where two crops of the same entity hash to near-orthogonal
        # vectors. So a floor inherited from a *preset* is refused unless the encoder can
        # support it — otherwise enabling the production preset offline would quietly
        # gut the bank. An explicitly passed floor is always honoured: that caller is
        # testing the gate itself, or supplying its own embeddings.
        self.embedder_is_semantic = _is_semantic_embedder(self.embedder)
        self.cohesion_by_type = dict(pol.cohesion_by_type)
        if cohesion_floor is not None:
            requested_cohesion = float(cohesion_floor)
        else:
            requested_cohesion = float(pol.cohesion_floor)
            if requested_cohesion > 0.0 and not self.embedder_is_semantic:
                warnings.warn(
                    f"{pol.name!r} policy cohesion floor ignored: embedder "
                    f"{getattr(self.embedder, 'name', '?')!r} is not semantic, so a "
                    "similarity floor cannot be interpreted. Set a real image encoder "
                    "(e.g. MEMSTRATA_GENERAL_EMBEDDER_PROVIDER=dinov3) to enable "
                    "gate ② and the cohesion self-audit.",
                    stacklevel=2,
                )
                requested_cohesion = 0.0
                self.cohesion_by_type = {}
        self.cohesion_floor = requested_cohesion
        self.cohesion_min_refs = max(1, int(_pick(cohesion_min_refs, pol.cohesion_min_refs)))
        # Library-level cohesion self-audit floor (design_philosophy.md §2/§4).
        # Admission gates only guard incoming evidence; this sweep re-checks whole
        # assets to catch pollution that slipped in before ② had a stable cluster
        # (e.g. the FIRST rep was the intruder). None → reuse cohesion_floor.
        requested_selfaudit = _pick(selfaudit_cohesion_floor, pol.selfaudit_cohesion_floor)
        self.selfaudit_cohesion_floor = (
            self.cohesion_floor
            if requested_selfaudit is None
            else float(requested_selfaudit)
        )
        self.selfaudit_each_segment = bool(pol.selfaudit_each_segment)
        # β_τ / λ_τ: identity reconciliation for discovered observations.
        self.reconcile_threshold = float(pol.reconcile_threshold)
        self.reconcile_by_type = dict(pol.reconcile_by_type)
        self.reconcile_text_weight = float(pol.reconcile_text_weight)
        self.reconcile_text_weight_by_type = dict(pol.reconcile_text_weight_by_type)
        selfaudit_reference = _pick(selfaudit_reference, pol.selfaudit_reference)
        # Self-audit reference: "medoid" (default, single centre) or "subcluster"
        # (largest cohesive subcluster). Tied on LSMDC when intruders are a minority;
        # "subcluster" is the opt-in fallback for majority-polluted assets where the
        # medoid can itself be an intruder (see experiments/.../RESULTS.md §Findings).
        if selfaudit_reference not in ("medoid", "subcluster"):
            raise ValueError(
                f"selfaudit_reference must be 'medoid' or 'subcluster', got {selfaudit_reference!r}"
            )
        self.selfaudit_reference = selfaudit_reference
        self.max_frames_per_asset = self.max_reps_per_asset
        self.angle_classifier: AngleClassifier = angle_classifier or NullAngleClassifier()
        self.crop_attribute_classifier: CropAttributeClassifier = (
            crop_attribute_classifier or NullCropAttributeClassifier()
        )
        # VLM-first identity gate (paper §4.5). Default is env-driven and abstains offline
        # (build_identity_judge → NullIdentityJudge unless MEMSTRATA_IDENTITY_JUDGE is set).
        self.identity_judge: IdentityJudge = identity_judge or build_identity_judge()
        self.identity_vlm_enabled = bool(pol.identity_vlm_enabled)
        self.identity_shortcircuit_margin = float(pol.identity_shortcircuit_margin)
        self.identity_gray_margin = float(pol.identity_gray_margin)
        self.identity_vlm_theta = float(pol.identity_vlm_theta)
        self.identity_max_references = max(1, int(pol.identity_max_references))
        self.identity_blur_defer = bool(pol.identity_blur_defer)
        self.identity_blur_min_sharpness = float(pol.identity_blur_min_sharpness)
        # The VLM gray-zone path is active only when the policy enables it AND a judge that
        # actually answers is present; a Null (abstaining) judge keeps the deterministic
        # encoder decision bit-for-bit unchanged.
        self._identity_judge_active = self.identity_vlm_enabled and not isinstance(
            self.identity_judge, NullIdentityJudge
        )

    # --- public aliases matching older InverseIngester API ---
    @property
    def asset_space(self) -> AssetBank:
        return self.bank

    # --- per-type thresholds (paper B_τ / γ_τ / β_τ / λ_τ) ------------------------

    def budget_for(self, kind: AssetType | str | None) -> int:
        """B_τ — how many live representations one asset of this type may hold."""
        return int(_resolve_by_type(self.max_reps_per_asset, self.max_reps_by_type, kind))

    def redundancy_for(self, kind: AssetType | str | None) -> float:
        """γ_τ — similarity at or above which new evidence counts as redundant."""
        return float(
            _resolve_by_type(self.redundancy_threshold, self.redundancy_by_type, kind)
        )

    def cohesion_for(self, kind: AssetType | str | None) -> float:
        """Gate ② admission floor for this type (0 disables)."""
        if self.cohesion_floor <= 0.0:
            return 0.0
        return float(_resolve_by_type(self.cohesion_floor, self.cohesion_by_type, kind))

    def reconcile_for(self, kind: AssetType | str | None) -> float:
        """β_τ — identity-reconciliation floor for a discovered observation."""
        return float(
            _resolve_by_type(self.reconcile_threshold, self.reconcile_by_type, kind)
        )

    def text_weight_for(self, kind: AssetType | str | None) -> float:
        """λ_τ — weight of the description term in the reconciliation score."""
        weight = float(
            _resolve_by_type(
                self.reconcile_text_weight, self.reconcile_text_weight_by_type, kind
            )
        )
        return min(1.0, max(0.0, weight))

    def _embed(self, path: str, kind: AssetType | str | None = None) -> Vector:
        """Encode a crop through the SAME route the decomposer would use.

        Routing matters for comparability: ``RoleRoutedEmbedding.embed_image`` always
        takes the general route, so a seed rep embedded here and a decompose-time rep
        embedded via the face route ended up in different spaces (and different
        dimensionalities) inside one asset, which made every later similarity gate
        meaningless.
        """
        route = getattr(self.embedder, "embed_with_route", None)
        if kind is not None and callable(route):
            return route(path, getattr(kind, "value", kind))[0]
        return self.embedder.embed_image(path)

    def _batch_classify_cache(
        self, observations: list[Observation], segment_id: int
    ) -> dict[str, CropAttributePack]:
        """ONE batched crop-attribute call for a segment (behavior-preserving).

        Mirrors the per-crop decision in :meth:`_classify_if_needed`: an observation is
        classified unless both its angles are already known *and* the policy skips the
        attribute pack. The distinct crop ``image_path``s that would be classified are
        sent to ``classify_batch`` in a single call and cached by path so
        :meth:`_classify_if_needed` reuses each pack instead of calling ``classify``
        per crop. For the Null/Heuristic classifiers ``classify_batch`` just loops
        ``classify`` so the packs are identical — only the call count drops.

        An ``image_path`` that appears with conflicting ``(kind, name)`` within the same
        segment is left uncached (it falls back to per-crop ``classify``), and a batch that
        returns the wrong number of packs is discarded, so the batched path can never
        change a result.
        """
        order: list[str] = []
        by_path: dict[str, tuple[AssetType, str]] = {}
        conflicts: set[str] = set()
        for obs in observations:
            path = obs.image_path
            if not path:
                continue
            needs = (
                obs.spatial_angle == SpatialAngle.UNKNOWN
                or obs.state_angle == StateAngle.UNKNOWN
            )
            if not needs and not self.attributes_when_angles_known:
                continue
            key = (obs.kind, obs.name)
            if path in by_path:
                if by_path[path] != key:
                    conflicts.add(path)
            else:
                by_path[path] = key
                order.append(path)
        paths = [p for p in order if p not in conflicts]
        if not paths:
            return {}
        items = [
            {
                "image_path": p,
                "kind": by_path[p][0].value,
                "name": by_path[p][1],
                "segment_id": segment_id,
            }
            for p in paths
        ]
        packs = self.crop_attribute_classifier.classify_batch(items)
        if len(packs) != len(paths):
            return {}  # unexpected count → fall back to per-crop classify for all
        return dict(zip(paths, packs))

    def _classify_if_needed(
        self,
        *,
        image_path: str,
        kind: AssetType,
        name: str,
        spatial: SpatialAngle,
        state: StateAngle,
        angle_meta: dict[str, Any] | None = None,
        segment_id: int | None = None,
        pack_cache: dict[str, CropAttributePack] | None = None,
    ) -> tuple[SpatialAngle, StateAngle, dict[str, Any]]:
        meta = dict(angle_meta or {})
        needs = spatial == SpatialAngle.UNKNOWN or state == StateAngle.UNKNOWN

        # Cheapest-tool guard: both angles known and no attribute pack requested →
        # do not spend a classifier call (matters when a real VLM is wired in).
        if not needs and not self.attributes_when_angles_known:
            meta["angle_source"] = meta.get("angle_source", "explicit") or "explicit"
            return spatial, state, meta

        # Prefer a pack from the segment-level batched call; copy it so per-observation
        # angle/source mutations below never alias a shared cached pack. On a cache miss
        # (empty batch, conflicting path, batch failure) fall back to a per-crop call.
        pack = None
        if pack_cache is not None:
            cached = pack_cache.get(image_path)
            if cached is not None:
                pack = replace(cached)
        if pack is None:
            pack = self.crop_attribute_classifier.classify(
                image_path,
                kind=kind.value,
                name=name,
                segment_id=segment_id,
            )
        # Angle-only classifier fills gaps when the attribute classifier is null.
        if needs and pack.spatial_angle == SpatialAngle.UNKNOWN and pack.state_angle == StateAngle.UNKNOWN:
            classified = self.angle_classifier.classify(
                image_path,
                kind=kind.value,
                name=name,
            )
            if classified.pack is not None:
                pack = classified.pack
            else:
                pack = CropAttributePack(
                    spatial_angle=classified.spatial_angle,
                    state_angle=classified.state_angle,
                    confidence=classified.confidence,
                    reasoning=classified.reasoning,
                    source=classified.source,
                    segment_id=segment_id,
                )
            meta.update(classified.to_annotations())
        elif not needs:
            pack.spatial_angle = spatial
            pack.state_angle = state
            pack.source = meta.get("angle_source", "explicit") or "explicit"
            meta.update(pack.to_annotations())
            meta["angle_source"] = "explicit"
        else:
            meta.update(pack.to_annotations())

        if spatial == SpatialAngle.UNKNOWN:
            spatial = pack.spatial_angle
        if state == StateAngle.UNKNOWN:
            state = pack.state_angle
        return spatial, state, meta

    def _apply_rep_selection(self, asset: Asset, new_rep: AssetRepresentation) -> bool:
        """Attach ``new_rep`` under the diversity budget. Returns True iff R_j changed."""
        # WHO-before-WHERE admission (design_philosophy.md §2). ① deterministic
        # luminance gate. A near-black/near-flat crop (dark) and a near-white/near-flat
        # crop (overexposed) both carry no identity signal → never bank. Both keep the
        # "unreadable → pass" semantics so a crop that cannot be assessed is not rejected.
        if self.dark_gate and is_dark_low_information(new_rep.object_uri):
            new_rep.annotations["admission"] = "rejected_dark_low_information"
            return False
        if self.dark_gate and is_overexposed_low_information(new_rep.object_uri):
            new_rep.annotations["admission"] = "rejected_overexposed_low_information"
            return False

        deprecated_reps = [rep for rep in asset.representations if rep.deprecated]
        active = [rep for rep in asset.representations if not rep.deprecated]

        # ③ identity visibility → anchor eligibility. A crop that cannot verify WHO
        # (back-of-head / heavy occlusion / unresolvable blur) stays as a cross-view
        # diversity rep but must NOT seed the identity anchor (axiom 5 preserved).
        # Heavy occlusion is downgraded to non-anchor here even when the classifier left
        # identity_visible at its permissive default: you cannot verify WHO through it,
        # but it is NOT hard-rejected (it may still add cross-view diversity).
        identity_visible = _rep_identity_visible(new_rep) and not _rep_occlusion_heavy(new_rep)
        aspect = _reference_aspect(asset.kind)
        if not identity_visible:
            new_rep.reference_aspects = [a for a in new_rep.reference_aspects if a != aspect]
            if aspect not in new_rep.excluded_aspects:
                new_rep.excluded_aspects.append(aspect)
            new_rep.annotations["identity_anchor_eligible"] = False
        else:
            new_rep.annotations.setdefault("identity_anchor_eligible", True)

        # ② embedding cohesion admission: compare only among identity-visible evidence
        # and only once a stable visible cluster exists. Off by default (floor=0), as
        # the offline fallback embedder is non-semantic; production sets a calibrated
        # per-type floor with a real encoder.
        cohesion_floor = self.cohesion_for(asset.kind)
        if cohesion_floor > 0.0 and identity_visible:
            new_emb = new_rep.annotations.get("embedding")
            ref_embs = [
                rep.annotations.get("embedding")
                for rep in active
                if rep.annotations.get("identity_anchor_eligible", True)
            ]
            ref_embs = [emb for emb in ref_embs if emb]
            if new_emb is not None and len(ref_embs) >= self.cohesion_min_refs:
                sim = similarity_to_set(new_emb, ref_embs)
                new_rep.annotations["cohesion_to_bank"] = round(sim, 4)
                if sim < cohesion_floor:
                    new_rep.annotations["admission"] = "rejected_low_cohesion"
                    return False

        new_bucket = _attr_bucket(new_rep)
        covered = {_attr_bucket(rep) for rep in active}
        novel_known_angle = _bucket_is_known(new_bucket) and new_bucket not in covered

        # Near-duplicate discard, restricted to the COMPATIBLE stratum (paper
        # ``compat(r, ŝ, b̂)``): evidence in a different known view/state stratum can
        # never make this observation redundant, so a new angle is not discarded just
        # for resembling a stored one. Unknown labels stay compatible, which keeps the
        # offline default (all reps compatible) unchanged.
        new_emb = new_rep.annotations.get("embedding")
        if new_emb is not None and not novel_known_angle:
            redundancy = self.redundancy_for(asset.kind)
            for rep in active:
                old = rep.annotations.get("embedding")
                if old is None:
                    continue
                if not compatible_stratum(
                    rep_bucket=_attr_bucket(rep), new_bucket=new_bucket
                ):
                    continue
                score = cosine_or_none(new_emb, old)
                if score is None:
                    continue  # different encoder route → cannot judge redundancy
                if score >= redundancy:
                    new_rep.annotations["admission"] = "rejected_near_duplicate"
                    return False

        # Same known attribute bucket already present. Keep the HIGHER-quality
        # representative instead of an unconditional discard: the old behaviour froze
        # the first (possibly blurry) crop of each bucket and rejected every sharper
        # same-bucket crop after it — which becomes the dominant eviction rule the
        # moment a real crop-attribute classifier makes buckets ``known``. If the
        # newcomer beats the weakest same-bucket rep, retire that rep (留痕) and let the
        # newcomer through the budget/diversity fill below; otherwise discard it.
        if _bucket_is_known(new_bucket) and new_bucket in covered:
            same_bucket = [rep for rep in active if _attr_bucket(rep) == new_bucket]
            new_q = float(new_rep.annotations.get("quality", 1.0))
            weakest = min(
                same_bucket,
                key=lambda r: float(r.annotations.get("quality", 1.0)),
                default=None,
            )
            if weakest is None or new_q <= float(weakest.annotations.get("quality", 1.0)):
                new_rep.annotations["admission"] = "rejected_bucket_covered"
                return False
            weakest.deprecated = True
            weakest.deprecated_by = "superseded_by_higher_quality_same_bucket"
            weakest.annotations["admission"] = "retired_lower_quality"
            deprecated_reps = deprecated_reps + [weakest]
            active = [rep for rep in active if rep is not weakest]

        # A pixel-less placeholder is only useful while the identity has no visual
        # evidence at all; once real evidence exists it is retired (with 留痕) so it
        # stops consuming one of the few per-asset slots.
        budget = self.budget_for(asset.kind)
        real = [rep for rep in active if not _is_placeholder_rep(rep)]
        placeholders = [rep for rep in active if _is_placeholder_rep(rep)]
        if _is_placeholder_rep(new_rep) and real:
            new_rep.annotations["admission"] = "rejected_placeholder_superseded"
            return False
        if placeholders and not _is_placeholder_rep(new_rep):
            for rep in placeholders:
                rep.deprecated = True
                rep.deprecated_by = "superseded_by_visual_evidence"
                rep.annotations["admission"] = "retired_placeholder"
            deprecated_reps = deprecated_reps + placeholders
            active = real

        candidates = active + [new_rep]
        if len(candidates) <= budget:
            asset.representations = deprecated_reps + candidates
            return True

        embeddings = [rep.annotations.get("embedding") for rep in candidates]
        qualities = [float(rep.annotations.get("quality", 1.0)) for rep in candidates]
        bucket_keys = [_attr_bucket(rep) for rep in candidates]
        new_index = len(candidates) - 1
        # Reserve a slot for the newest evidence. Without it the newest rep loses every
        # tie-break (bucket winners prefer the lowest index; the diversity fill walks in
        # index order), so once an asset filled its budget it could never record anything
        # again — memory froze after the first few segments of a long video.
        kept = select_attribute_diverse(
            bucket_keys=bucket_keys,
            vectors=embeddings if all(embeddings) else None,
            quality=qualities,
            max_keep=budget,
            min_distance=self.min_distance,
            pin=[new_index],
        )
        asset.representations = deprecated_reps + [candidates[i] for i in kept]
        # R_j changed iff the new rep actually made it in (the pin guarantees it does,
        # but the contract must not depend on that).
        return new_index in set(kept)
    def audit_cohesion(
        self,
        *,
        floor: float | None = None,
        isolate: bool = True,
    ) -> list[dict[str, Any]]:
        """Per-asset identity-cohesion self-audit sweep (design_philosophy.md §2/§4).

        The three admission gates (①②③) only guard *incoming* evidence; they cannot
        repair an asset whose **first** rep was the intruder, or that was polluted
        before gate ② had a stable visible cluster to compare against. This sweep
        recomputes each asset's embedding cohesion against a reference for the majority
        identity mass and treats identity-visible reps whose similarity to that
        reference falls below ``floor`` as suspected other-identity intruders — the
        axiom-4 red line applied *retroactively*.

        The reference is chosen by ``selfaudit_reference``:
          - ``"medoid"`` (default): a single centre (`medoid_cohesion`). Simple; on
            labelled S5 DINOv3 data AUC ≈ 0.79.
          - ``"subcluster"``: the largest cohesive subcluster
            (`largest_cohesive_subcluster`), scored by max-sim to that set. Tied with
            medoid when intruders are a minority, but stronger (AUC 0.79 vs 0.74) when
            an asset is *majority-polluted* — there the single medoid can itself be the
            intruder. Opt-in fallback for high-mixing / distribution-shift regimes.
        See the original calibration workspace/20260722_memstrata_cohesion_calibration/RESULTS.md.

        Conservative isolation (philosophy §4): a flagged rep is **deprecated with a
        traceable ``deprecated_by`` marker (留痕), never silently deleted**, so it is
        excluded from ``compose`` (stops poisoning generation) yet stays in the
        record and is reversible. ``isolate=False`` runs a dry report only.

        Off when the effective floor ≤ 0 (offline HashEmbedding is non-semantic);
        production wires a real encoder and a control-set-calibrated floor. Returns
        one report row per flagged rep.
        """
        eff_floor = self.selfaudit_cohesion_floor if floor is None else float(floor)
        report: list[dict[str, Any]] = []
        if eff_floor <= 0.0:
            return report

        changed = False
        for asset in self.bank.assets.values():
            # Only identity-visible, non-deprecated, embedded reps define the cluster.
            live = [
                rep
                for rep in asset.representations
                if not rep.deprecated
                and _rep_identity_visible(rep)
                and rep.annotations.get("embedding")
            ]
            # Only compare embeddings that live in the same space. A single rep encoded
            # by another route would otherwise make every pairwise cosine raise, taking
            # down the whole sweep; restricting to the dominant dimensionality keeps the
            # audit meaningful and lets the odd one out be re-checked once it is
            # re-encoded.
            live = _same_dimension_group(live)
            # Need a majority to anchor the reference; too few reps → nothing to compare.
            if len(live) <= self.cohesion_min_refs:
                continue

            vectors = [rep.annotations["embedding"] for rep in live]
            # Build (score, extra-fields) per rep, plus the set of reps that are part
            # of the reference itself and so trivially pass (never flagged).
            scored: list[tuple[float, dict[str, Any]]]
            skip: set[int]
            if self.selfaudit_reference == "subcluster":
                core = largest_cohesive_subcluster(vectors, link_threshold=eff_floor)
                # No stable majority identity → cannot adjudicate; skip conservatively.
                if len(core) < self.cohesion_min_refs:
                    continue
                asset.metadata["cohesion_core_size"] = len(core)
                skip = set(core)
                core_vecs = [vectors[j] for j in core]
                scored = [
                    (
                        similarity_to_set(vectors[i], core_vecs),
                        {"cohesion_to_core": 0.0, "core_size": len(core)},
                    )
                    for i in range(len(live))
                ]
            else:  # "medoid"
                medoid_idx, sims_to_medoid, min_pairwise = medoid_cohesion(vectors)
                asset.metadata["cohesion_min_pairwise"] = round(min_pairwise, 4)
                skip = {medoid_idx}
                scored = [
                    (
                        sims_to_medoid[i],
                        {
                            "cohesion_to_medoid": 0.0,
                            "medoid_representation_id": live[medoid_idx].representation_id,
                        },
                    )
                    for i in range(len(live))
                ]

            for i, rep in enumerate(live):
                if i in skip:
                    continue
                sim, extra = scored[i]
                if sim >= eff_floor:
                    continue
                key = "cohesion_to_core" if "cohesion_to_core" in extra else "cohesion_to_medoid"
                extra[key] = round(sim, 4)
                row = {
                    "asset_id": asset.asset_id,
                    "representation_id": rep.representation_id,
                    **extra,
                    "floor": eff_floor,
                    "action": "isolated" if isolate else "flagged",
                }
                report.append(row)
                rep.annotations["cohesion_selfaudit"] = round(sim, 4)
                if isolate:
                    rep.deprecated = True
                    rep.deprecated_by = f"cohesion_selfaudit:{round(sim, 4)}"
                    rep.annotations["admission"] = "isolated_low_cohesion_selfaudit"
                    changed = True
        if changed:
            self.bank.touch()
        return report

    def _resolve_asset(self, *, entity_id: str | None, name: str, kind: AssetType) -> Asset:
        """Name-anchored identity: prefer explicit id, else same name+kind in bank."""
        if entity_id:
            asset = self.bank.get_asset(entity_id)
            if asset is not None:
                self._warn_identity_conflict(asset, name=name, kind=kind, via="entity_id")
                # Authoritative id ties surface-name variants to one identity
                # (axiom 3): record the incoming name as an alias so the name-only
                # read/write path can also aggregate it later.
                if name and name.strip().lower() != asset.name.strip().lower():
                    self.bank.register_alias(asset.asset_id, name)
                return asset
            asset = Asset(
                asset_id=entity_id,
                kind=kind,
                name=name,
                status=LifecycleStatus.REUSABLE,
                description="",
            )
            self.bank.add_asset(asset)
            return asset

        existing = self.bank.find_by_name(name, kind=kind)
        if existing is not None:
            self._warn_identity_conflict(existing, name=name, kind=kind, via="name")
            return existing

        new_id = f"{kind.value}_{name.strip().lower().replace(' ', '_')}"
        collision = self.bank.get_asset(new_id)
        if collision is not None:
            # Same generated id but a different type: two distinct identities that only
            # share a surface name (two characters called "Guard"). Merging them would
            # break the low-mixing red line, so give the newcomer its own record and say
            # so loudly — silent merging is exactly what poisons downstream generation.
            self._warn_identity_conflict(collision, name=name, kind=kind, via="generated_id")
            suffix = 2
            while self.bank.get_asset(f"{new_id}__{suffix}") is not None:
                suffix += 1
            new_id = f"{new_id}__{suffix}"
        asset = Asset(
            asset_id=new_id,
            kind=kind,
            name=name,
            status=LifecycleStatus.REUSABLE,
        )
        self.bank.add_asset(asset)
        return asset

    def _warn_identity_conflict(
        self, asset: Asset, *, name: str, kind: AssetType, via: str
    ) -> None:
        """Flag a suspicious reuse of an existing identity record.

        A type mismatch means the incoming observation is almost certainly a different
        entity that happens to collide; recording it under the same asset would violate
        the low-mixing axiom. We surface it (warning + traceable metadata) rather than
        guessing, because identity is the one decision the write path must never make
        silently.
        """
        if asset.kind == kind:
            return
        note = {
            "incoming_name": name,
            "incoming_kind": kind.value,
            "existing_kind": asset.kind.value,
            "matched_via": via,
        }
        conflicts = asset.metadata.setdefault("identity_conflicts", [])
        if note not in conflicts:
            conflicts.append(note)
            self.bank.touch()
        warnings.warn(
            f"identity conflict on {asset.asset_id!r}: incoming {kind.value}/{name!r} "
            f"matched an existing {asset.kind.value} via {via}",
            stacklevel=3,
        )

    def identity_score(self, obs: Observation, asset: Asset) -> tuple[float, float, float]:
        """χ_{i,j} — how strongly a discovered observation belongs to ``asset``.

        ``λ_τ · sim_text(d̂_i, d_j) + (1-λ_τ) · max_r sim_τ(z_i, E(v_r))`` over the asset's
        *active* representations, restricted by the caller to same-type assets. Returns
        ``(chi, text_sim, visual_sim)``; a missing term contributes 0 and the other term
        carries the full weight, so an asset with no description is judged on pixels and
        vice versa. Deterministic by construction — identity decisions must reproduce.
        """
        lam = self.text_weight_for(obs.kind)
        text_sim = text_similarity(obs.description, asset.d) if obs.description and asset.d else -1.0
        visual_sim = -1.0
        if obs.embedding:
            active = [
                rep.annotations.get("embedding")
                for rep in asset.representations
                if not rep.deprecated and rep.annotations.get("embedding")
            ]
            if active:
                visual_sim = similarity_to_set(obs.embedding, active)
        have_text, have_visual = text_sim >= 0.0, visual_sim >= 0.0
        if have_text and have_visual:
            chi = lam * text_sim + (1.0 - lam) * visual_sim
        elif have_text:
            chi = text_sim
        elif have_visual:
            chi = visual_sim
        else:
            chi = -1.0
        return chi, text_sim, visual_sim

    def _candidate_reference_crops(self, asset: Asset, k: int) -> list[str]:
        """Up to ``k`` diverse *active* representation crops for the VLM identity judge.

        Sent together in ONE judge call so identity is decided against a cross-view/state
        reference set rather than a single possibly-unlucky frame. Selection prefers higher
        quality then recency, and spreads across distinct attribute strata (view/state/shot/
        lighting/pose) so the references are complementary; if diversity yields fewer than
        ``k`` it tops up by the quality order.
        """
        reps = [
            rep
            for rep in asset.representations
            if not rep.deprecated and str(rep.object_uri or "").strip()
        ]
        if not reps:
            return []
        reps.sort(key=lambda r: (_rep_quality(r), int(r.origin_segment_id or 0)), reverse=True)
        picked: list[str] = []
        seen_buckets: set[tuple[str, ...]] = set()
        for rep in reps:
            bucket = _attr_bucket(rep)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            picked.append(str(rep.object_uri))
            if len(picked) >= k:
                return picked
        for rep in reps:  # top up if distinct strata were fewer than k
            uri = str(rep.object_uri)
            if uri not in picked:
                picked.append(uri)
                if len(picked) >= k:
                    break
        return picked

    def _reconcile_identity(self, obs: Observation) -> tuple[Asset | None, dict[str, Any]]:
        """Pick the best same-type asset for a discovered observation, or ``None``.

        Type-restricted on purpose: reconciliation may refine *which* record of the right
        type an observation extends, but it must never move an observation across types.
        Below β_τ the observation becomes a new record rather than being forced into the
        closest existing one — a wrong merge poisons every later generation, whereas a
        spurious new asset only costs budget.

        VLM-first (paper §4.5): the encoder score χ still selects the candidate and short-
        circuits high-confidence matches, but the χ *gray zone* around β_τ is adjudicated
        by a VLM (temperature 0, per-model θ). Strong-blur crops in the gray zone are
        deferred to a new provisional record rather than hard-merged, because every judge
        false-merges under strong blur. With no active judge the decision is exactly χ≥β_τ.
        """
        best: Asset | None = None
        best_row: dict[str, Any] = {}
        for asset in self.bank.assets.values():
            if asset.kind != obs.kind or asset.status in (
                LifecycleStatus.REJECTED,
                LifecycleStatus.FAILED,
            ):
                continue
            chi, text_sim, visual_sim = self.identity_score(obs, asset)
            if not best_row or chi > best_row["chi"]:
                best, best_row = asset, {
                    "chi": chi,
                    "text_sim": text_sim,
                    "visual_sim": visual_sim,
                    "asset_id": asset.asset_id,
                }
        threshold = self.reconcile_for(obs.kind)
        meta = {**best_row, "threshold": threshold}
        chi = float(best_row.get("chi", -1.0))

        # Deterministic encoder decision (also the fallback when the VLM abstains). This is
        # the historical rule and stays bit-for-bit identical whenever the judge is inactive.
        def _encoder_decision(gate: str) -> tuple[Asset | None, dict[str, Any]]:
            if best is not None and chi >= threshold:
                return best, {**meta, "decision": "merged", "gate": gate}
            return None, {**meta, "decision": "new_asset", "gate": gate}

        if best is None or not self._identity_judge_active:
            return _encoder_decision("encoder")

        # χ bands relative to β_τ: confident SAME / confident NEW skip the VLM entirely.
        if chi >= threshold + self.identity_shortcircuit_margin:
            return best, {**meta, "decision": "merged", "gate": "encoder_shortcircuit"}
        if chi <= threshold - self.identity_gray_margin:
            return None, {**meta, "decision": "new_asset", "gate": "encoder_reject"}

        # Gray zone. Defer strong-blur crops rather than risk a false merge.
        if self.identity_blur_defer:
            sharpness = _crop_sharpness(obs.image_path)
            if sharpness is not None and sharpness < self.identity_blur_min_sharpness:
                return None, {
                    **meta,
                    "decision": "new_asset",
                    "gate": "deferred_blur",
                    "sharpness": sharpness,
                }

        reference_crops = self._candidate_reference_crops(best, self.identity_max_references)
        if not obs.image_path or not reference_crops:
            return _encoder_decision("encoder_fallback")

        verdict = self.identity_judge.judge(
            obs.image_path,
            reference_crops,
            kind=obs.kind.value,
            name_a=obs.name or "",
            name_b=best.name or "",
        )
        vmeta = {
            **meta,
            "vlm_same": verdict.same,
            "vlm_confidence": verdict.confidence,
            "vlm_source": verdict.source,
            "vlm_n_refs": len(reference_crops),
        }
        if verdict.same is True and verdict.confidence >= self.identity_vlm_theta:
            return best, {**vmeta, "decision": "merged", "gate": "vlm"}
        if verdict.same is False:
            return None, {**vmeta, "decision": "new_asset", "gate": "vlm"}
        # Abstain or low-confidence "same" → trust the deterministic encoder threshold.
        decision = _encoder_decision("vlm_abstain_fallback")
        return decision[0], {**vmeta, **decision[1]}

    def curate_observations(
        self,
        observations: list[Observation],
        *,
        segment_id: int,
        state_events: list[dict[str, Any]] | None = None,
        relations: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        touched: list[str] = []
        # One batched crop-attribute call for this segment's crops (behavior-preserving);
        # _classify_if_needed reuses these packs instead of one classify() per crop.
        pack_cache = self._batch_classify_cache(observations, segment_id)
        for obs in observations:
            reconcile_meta: dict[str, Any] = {}
            # χ reconciliation compares the observation's embedding against existing reps, so a
            # discovered observation must be embedded BEFORE reconcile (otherwise visual_sim is
            # unavailable and identity collapses to text-only, never merging two crops of the
            # same entity). Requested evidence is name-anchored and stays embedded lazily at
            # rep-storage time, so its behaviour is unchanged.
            if (
                obs.embedding is None
                and self.embed_on_ingest
                and obs.source == SOURCE_DISCOVERED
                and not obs.entity_id
            ):
                try:
                    obs.embedding = self._embed(obs.image_path, obs.kind)
                except Exception:  # noqa: BLE001 - embedding is best-effort, never fail a segment
                    obs.embedding = None
            if obs.source == SOURCE_DISCOVERED and not obs.entity_id:
                # Discovered evidence has no symbolic anchor, so identity is decided by
                # type-restricted reconciliation (χ) instead of a name lookup.
                matched, reconcile_meta = self._reconcile_identity(obs)
                if matched is not None:
                    asset = matched
                    # χ merged this observation into an existing identity under a DIFFERENT
                    # surface name. Record that name as an alias (axiom 3) so the name-anchored
                    # read path can also retrieve the identity by the incoming name — otherwise
                    # a cross-segment prompt using the other name silently misses the merged
                    # asset and the bank looks fragmented on the read side.
                    if obs.name and obs.name.strip().lower() != asset.name.strip().lower():
                        self.bank.register_alias(asset.asset_id, obs.name)
                else:
                    asset = self._new_discovered_asset(obs)
            else:
                asset = self._resolve_asset(
                    entity_id=obs.entity_id,
                    name=obs.name,
                    kind=obs.kind,
                )
            if asset.asset_id not in touched:
                touched.append(asset.asset_id)

            # d_j: adopt the observation description when the record has none yet, so the
            # bank carries appearance text that text-keyed retrieval can match against.
            if obs.description and not asset.description:
                asset.description = obs.description
                asset.metadata["description"] = obs.description
                self.bank.touch()

            rep_id = f"{asset.asset_id}@s{segment_id:03d}"
            if any(r.representation_id == rep_id for r in asset.representations):
                rep_id = f"{rep_id}_{obs.observation_id}"
            if any(r.representation_id == rep_id for r in asset.representations):
                continue

            spatial, state, angle_meta = self._classify_if_needed(
                image_path=obs.image_path,
                kind=obs.kind,
                name=obs.name,
                spatial=obs.spatial_angle,
                state=obs.state_angle,
                angle_meta=obs.angle_meta,
                segment_id=segment_id,
                pack_cache=pack_cache,
            )

            # Description upgrade: a clearer observation may populate a still-empty stable
            # description even when ``obs.description`` was empty but the attribute
            # classifier produced one (crop_attributes.description → observation_description).
            # Conservative and deterministic: only fills an empty description, never
            # overwrites a non-empty one (per-state text still flows via rep annotations).
            if not asset.description:
                classified_desc = str(angle_meta.get("observation_description", "")).strip()
                if classified_desc:
                    asset.description = classified_desc
                    asset.metadata["description"] = classified_desc
                    self.bank.touch()

            annotations: dict[str, Any] = {
                "quality": obs.quality,
                "encoder_route": obs.encoder_route,
                "acquisition_source": obs.source,
                **angle_meta,
            }
            if obs.description:
                annotations["observation_description"] = obs.description
            if obs.bbox_norm:
                # Kept so a later segment's discovery can skip regions already banked for
                # a named entity.
                annotations["bbox"] = list(obs.bbox_norm)
            if getattr(obs, "source_frame_path", ""):
                # Full source frame this crop was cut from, for crop↔frame self-audit
                # and frame-level retrieval (snapshot exports it beside the crop).
                annotations["source_frame"] = obs.source_frame_path
            if reconcile_meta:
                annotations["identity_reconciliation"] = reconcile_meta
            if obs.embedding is not None:
                annotations["embedding"] = obs.embedding
            elif self.embed_on_ingest:
                annotations["embedding"] = self._embed(obs.image_path, obs.kind)
            # C item 1/2: the crop-acquisition provenance rides in via ``**angle_meta``;
            # flag bbox-only GDINO fallbacks for review (annotation-only, no scoring change).
            _annotate_crop_acquisition_review(annotations)

            new_rep = AssetRepresentation(
                representation_id=rep_id,
                asset_id=asset.asset_id,
                object_uri=obs.image_path,
                origin_segment_id=segment_id,
                spatial_angle=spatial,
                state_angle=state,
                temporal_tag=obs.temporal_tag or f"segment_{segment_id}",
                reference_aspects=[_reference_aspect(obs.kind)],
                quality_by_purpose={
                    _reference_aspect(obs.kind): float(obs.quality),
                },
                annotations=annotations,
            )
            mutated = self._apply_rep_selection(asset, new_rep)
            if asset.status == LifecycleStatus.CANDIDATE:
                asset.status = LifecycleStatus.REUSABLE
                mutated = True
            if mutated:
                self.bank.touch()

        self._apply_state_events(state_events or [])
        self._apply_relations(relations or [])
        self._enforce_global_budget()
        return touched

    def _new_discovered_asset(self, obs: Observation) -> Asset:
        """Create a record for a discovery that matched nothing above β_τ."""
        handle = str(obs.observation_id).strip().lower().replace(" ", "_")
        prefix = f"{obs.kind.value}_"
        base = handle if handle.startswith(("disc_", prefix)) else f"{prefix}disc_{handle}"
        asset_id = base
        suffix = 2
        while self.bank.get_asset(asset_id) is not None:
            asset_id = f"{base}__{suffix}"
            suffix += 1
        asset = Asset(
            asset_id=asset_id,
            kind=obs.kind,
            name=obs.name or asset_id,
            status=LifecycleStatus.REUSABLE,
            description=obs.description,
            metadata={"provisional": True, "acquisition_source": obs.source},
        )
        if obs.description:
            asset.metadata["description"] = obs.description
        self.bank.add_asset(asset)
        return asset

    def _apply_state_events(self, events: list[dict[str, Any]]) -> None:
        if self.disable_deprecation:
            return  # ablation "- lifecycle avoidance": never deprecate representations
        for event in events:
            event_id = str(event["event_id"])
            affected: set[str] = set()
            for rep_id in event.get("deprecates", []):
                found = self.bank.find_representation(str(rep_id))
                if found is None:
                    continue
                host, rep = found
                rep.deprecated = True
                rep.deprecated_by = event_id
                # State-change events also mark the visual evidence as a changed state angle.
                if rep.state_angle == StateAngle.DEFAULT:
                    rep.state_angle = StateAngle.CHANGED
                affected.add(host.asset_id)
            for asset_id in affected:
                host = self.bank.get_asset(asset_id)
                if host is not None:
                    host.metadata.setdefault("state_events", []).append(dict(event))

            # Auto-build a traceable deprecation chain when the event names a
            # successor asset. Only linked when both endpoints already exist.
            replaced_by = str(event.get("replaced_by", "") or "")
            successor = self.bank.get_asset(replaced_by) if replaced_by else None
            if successor is not None:
                relation_changed = False
                for old_asset_id in affected:
                    if old_asset_id == replaced_by:
                        continue
                    old_asset = self.bank.get_asset(old_asset_id)
                    if old_asset is None:
                        continue
                    attrs = {"event_id": event_id}
                    if self._add_relation(
                        old_asset, RelationType.DEPRECATED_BY, replaced_by, attrs
                    ):
                        relation_changed = True
                    if self._add_relation(
                        successor, RelationType.REPLACES, old_asset_id, attrs
                    ):
                        relation_changed = True
                if relation_changed:
                    self.bank.touch()

            # Asset-level lifecycle is only changed through the explicit `marks_status`
            # channel; deprecating every representation must still leave the asset
            # referenceable so `compose` can surface the reps as exclusions.
            marked_status = event.get("marks_status")
            target_asset_id = event.get("asset_id")
            if marked_status and target_asset_id:
                try:
                    self.bank.update_status(str(target_asset_id), LifecycleStatus(str(marked_status)))
                except ValueError:
                    continue

    def _enforce_global_budget(self) -> None:
        """Retire the weakest live representations when over the bank-wide cap.

        Deterministic: lowest quality then oldest segment goes first, but every asset keeps
        at least one live representation so it stays referenceable.

        Retirement is a *lifecycle transition*, not a delete: the rep is marked
        ``deprecated`` with a traceable ``deprecated_by`` marker (留痕), so it leaves
        ``compose`` immediately yet stays auditable and reversible — the conservative
        rule from philosophy.md §4 ("never silently drop evidence").
        """
        limit = self.max_total_representations
        if limit is None or limit <= 0:
            return
        live: list[tuple[str, AssetRepresentation]] = [
            (asset.asset_id, rep)
            for asset in self.bank.assets.values()
            for rep in asset.representations
            if not rep.deprecated
        ]
        overflow = len(live) - limit
        if overflow <= 0:
            return

        live_per_asset: dict[str, int] = {}
        for asset_id, _rep in live:
            live_per_asset[asset_id] = live_per_asset.get(asset_id, 0) + 1

        evict_order = sorted(live, key=lambda ar: (_rep_quality(ar[1]), ar[1].origin_segment_id))
        retired = 0
        for asset_id, rep in evict_order:
            if retired >= overflow:
                break
            if live_per_asset[asset_id] <= 1:
                continue  # protect each asset's last live representation
            rep.deprecated = True
            rep.deprecated_by = "global_budget"
            rep.annotations["admission"] = "retired_global_budget"
            live_per_asset[asset_id] -= 1
            retired += 1
        if retired:
            self.bank.touch()

    def _add_relation(
        self,
        source: Asset,
        rel_type: RelationType,
        target_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Append a deduplicated relation. Returns True iff it was newly added."""
        if not target_id:
            return False
        if any(
            rel.relation_type == rel_type and rel.target_asset_id == target_id
            for rel in source.relations
        ):
            return False
        source.relations.append(AssetRelation(rel_type, target_id, dict(attributes or {})))
        return True

    def _apply_relations(self, declarations: list[dict[str, Any]]) -> None:
        """Attach explicit, deduplicated structural relations onto source assets.

        Declarations are trusted inputs (never inferred here). Each row is
        ``{asset_id, relation_type, target_asset_id, attributes?}``; unknown
        types, missing endpoints, and duplicates are skipped.
        """
        changed = False
        for decl in declarations:
            src_id = str(decl.get("asset_id", ""))
            target_id = str(decl.get("target_asset_id", ""))
            if not src_id or not target_id:
                continue
            try:
                rel_type = RelationType(str(decl.get("relation_type", "")))
            except ValueError:
                continue
            source = self.bank.get_asset(src_id)
            if source is None:
                continue
            if self._add_relation(source, rel_type, target_id, decl.get("attributes", {})):
                changed = True
        if changed:
            self.bank.touch()

    def ingest_packet(self, packet: dict) -> list[str]:
        """Bench Track A: ObservationPacket with authoritative entity ids (naming oracle).

        Does **not** invoke the VLM classifier — packet angles are authoritative.
        """
        segment_id = int(packet["segment_id"])
        touched: list[str] = []

        for obs in packet.get("observations", []):
            # A malformed row (missing id/kind/crop or a non-paper type) must not
            # drop the rest of the segment's observations.
            try:
                asset_id = str(obs["entity_id"])
                kind = AssetType(str(obs["kind"]))
                name = str(obs["name"])
                rep_id = str(obs["representation_id"])
                crop_path = str(obs["crop_path"])
            except (KeyError, ValueError) as exc:
                warnings.warn(
                    f"skip malformed observation row in segment {segment_id}: {exc!r}",
                    stacklevel=2,
                )
                continue
            description = str(obs.get("description", ""))

            asset = self.bank.get_asset(asset_id)
            if asset is None:
                asset = Asset(
                    asset_id=asset_id,
                    kind=kind,
                    name=name,
                    status=LifecycleStatus.REUSABLE,
                    description=description,
                    metadata={"description": description} if description else {},
                )
                self.bank.add_asset(asset)
            elif description and not asset.description:
                asset.description = description
                asset.metadata["description"] = description

            if asset_id not in touched:
                touched.append(asset_id)

            if any(r.representation_id == rep_id for r in asset.representations):
                continue

            annotations: dict[str, Any] = {
                "angle_source": "packet",
                "acquisition_source": "packet",
            }
            if description:
                annotations["observation_description"] = description
            if self.embed_on_ingest:
                # Video-free "text-gold" carries entity references without pixels
                # (empty crop_path). Seed the deterministic fallback with the unique
                # representation id so distinct references stay distinguishable instead
                # of collapsing to one identical hash (which would over-merge dedup).
                annotations["embedding"] = self._embed(crop_path or rep_id, kind)

            new_rep = AssetRepresentation(
                representation_id=rep_id,
                asset_id=asset_id,
                object_uri=crop_path,
                origin_segment_id=segment_id,
                spatial_angle=_parse_spatial(obs.get("spatial_angle", SpatialAngle.UNKNOWN.value)),
                state_angle=_parse_state(obs.get("state_angle", StateAngle.UNKNOWN.value)),
                temporal_tag=str(obs.get("temporal_tag", f"segment_{segment_id}")),
                quality_by_purpose={
                    _reference_aspect(kind): float(obs.get("quality", 1.0)),
                },
                annotations=annotations,
            )
            if self._apply_rep_selection(asset, new_rep):
                self.bank.touch()

        self._apply_state_events(list(packet.get("state_events", [])))
        self._apply_relations(list(packet.get("relations", [])))
        self._enforce_global_budget()
        return touched

    def ingest_observation(self, obs: EntityObservation, segment_id: int) -> str:
        """Production path: classify angles if unknown, then name-anchored attach."""
        spatial, state, angle_meta = self._classify_if_needed(
            image_path=obs.image_path,
            kind=obs.kind,
            name=obs.name,
            spatial=obs.spatial_angle,
            state=obs.state_angle,
            angle_meta=obs.angle_meta,
            segment_id=segment_id,
        )
        vector = self._embed(obs.image_path, obs.kind) if self.embed_on_ingest else None
        observation = Observation(
            observation_id=obs.observation_id,
            kind=obs.kind,
            name=obs.name,
            image_path=obs.image_path,
            entity_id=obs.entity_id,
            embedding=vector,
            quality=obs.quality,
            spatial_angle=spatial,
            state_angle=state,
            temporal_tag=obs.temporal_tag or f"segment_{segment_id}",
            angle_meta=angle_meta,
            description=str(angle_meta.get("observation_description", "")),
        )
        touched = self.curate_observations([observation], segment_id=segment_id)
        return touched[0] if touched else ""


def stratification_report(bank: AssetBank) -> dict[str, Any]:
    """Is the "stratified" memory actually stratified? (paper §Stratified Update).

    The claim that每 crop carries a spatial / state / temporal stratum is only true if
    those fields are *populated*, and for a long time they silently were not: the
    production runner built its own curator without an attribute classifier, so every
    rep stored ``unknown`` and the whole read-side angle preference was a no-op. This
    report is the direct evidence for (or against) the claim — read
    ``spatial_known_ratio`` / ``state_known_ratio`` first: near 0 means the classifier is
    not wired and no stratification result may be reported.
    """
    total = 0
    spatial_known = 0
    state_known = 0
    temporal_known = 0
    described = 0
    deprecated = 0
    buckets: set[tuple[str, ...]] = set()
    angle_source: dict[str, int] = {}
    by_kind: dict[str, dict[str, Any]] = {}
    acquisition: dict[str, int] = {}

    for asset in bank.assets.values():
        kind = asset.kind.value
        slot = by_kind.setdefault(
            kind,
            {"assets": 0, "reps": 0, "spatial_known": 0, "state_known": 0, "buckets": set()},
        )
        slot["assets"] += 1
        for rep in asset.representations:
            total += 1
            slot["reps"] += 1
            if rep.deprecated:
                deprecated += 1
            source = str(rep.annotations.get("angle_source", "missing"))
            angle_source[source] = angle_source.get(source, 0) + 1
            acq = str(rep.annotations.get("acquisition_source", "unknown"))
            acquisition[acq] = acquisition.get(acq, 0) + 1
            if rep.spatial_angle != SpatialAngle.UNKNOWN:
                spatial_known += 1
                slot["spatial_known"] += 1
            if rep.state_angle != StateAngle.UNKNOWN:
                state_known += 1
                slot["state_known"] += 1
            if str(rep.temporal_tag or "").strip():
                temporal_known += 1
            if str(rep.annotations.get("observation_description", "")).strip():
                described += 1
            bucket = _attr_bucket(rep)
            buckets.add(bucket)
            slot["buckets"].add(bucket)

    def _ratio(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    for slot in by_kind.values():
        reps = slot["reps"]
        slot["bucket_coverage"] = len(slot.pop("buckets"))
        slot["spatial_known_ratio"] = round(slot["spatial_known"] / reps, 4) if reps else 0.0
        slot["state_known_ratio"] = round(slot["state_known"] / reps, 4) if reps else 0.0

    return {
        "assets": len(bank.assets),
        "representations": total,
        "deprecated_representations": deprecated,
        "spatial_known_ratio": _ratio(spatial_known),
        "state_known_ratio": _ratio(state_known),
        "temporal_known_ratio": _ratio(temporal_known),
        "described_ratio": _ratio(described),
        "bucket_coverage": len(buckets),
        "angle_source_counts": dict(sorted(angle_source.items())),
        "acquisition_source_counts": dict(sorted(acquisition.items())),
        "by_kind": by_kind,
        "assets_with_identity_conflicts": sorted(
            aid for aid, a in bank.assets.items() if a.metadata.get("identity_conflicts")
        ),
    }


# Backward-compatible names (pre-rename: AssetCurator / InverseIngester).
AssetCurator = MemoryUpdater
InverseIngester = MemoryUpdater
