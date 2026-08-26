"""MemStrata four-step loop (paper §3.2)."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from memstrata.bank import AssetBank, SpatialAngle, StateAngle
from memstrata.encoders import build_role_routed_embedding_from_env
from memstrata.mllm.angle_classifier import AngleClassifier, build_angle_classifier
from memstrata.mllm.crop_attributes import (
    CropAttributeClassifier,
    build_crop_attribute_classifier,
)
from memstrata.skills.memory_update import export_memory_snapshot
from memstrata.steps.compose import ComposedContext, compose
from memstrata.steps.curate import MemoryPolicy, MemoryUpdater, stratification_report
from memstrata.steps.decompose import (
    Discoverer,
    NamedEntity,
    Observation,
    RoleAwareDecomposer,
)
from memstrata.steps.generate import (
    GenerationResult,
    MediaTaskGenerator,
    NullGenerator,
)
from memstrata.skills.intent_understanding.plan import PlanProducer
from memstrata.steps.intent import (
    INTENT_MODE_FAST,
    INTENT_MODE_PLAN,
    CompositionRequest,
    IntentInterpreter,
    IntentResolver,
)


def build_curator(
    bank: AssetBank,
    *,
    policy: MemoryPolicy,
    embedder: Any = None,
    angle_classifier: AngleClassifier | None = None,
    crop_attribute_classifier: CropAttributeClassifier | None = None,
    **overrides: Any,
) -> MemoryUpdater:
    """Construct a curator that actually carries ``policy``.

    Callers that need the curator *before* the pipeline (the production runner seeds the
    bank from the screenplay) must build it through here. Constructing ``MemoryUpdater``
    by hand is what silently dropped the whole production preset — including the
    attribute classifier, which left every stored representation at ``unknown``.
    """
    return MemoryUpdater(
        bank,
        embedder,
        policy=policy,
        angle_classifier=angle_classifier,
        crop_attribute_classifier=crop_attribute_classifier,
        **overrides,
    )


def build_decomposer(
    *,
    policy: MemoryPolicy,
    embedder: Any = None,
    cropper: Any = None,
    angle_classifier: AngleClassifier | None = None,
    discoverer: Discoverer | None = None,
    **overrides: Any,
) -> RoleAwareDecomposer:
    """Construct a decomposer that actually carries ``policy`` (see ``build_curator``)."""
    return RoleAwareDecomposer(
        embedder,
        cropper=cropper,
        angle_classifier=angle_classifier,
        crop_quality_gate=policy.crop_quality_gate,
        discoverer=discoverer if policy.discovery else None,
        **overrides,
    )


@dataclass(slots=True)
class SegmentResult:
    segment_id: int
    request: CompositionRequest
    context: ComposedContext
    generation: GenerationResult | None
    observations: list[Observation]
    touched_asset_ids: list[str]
    model_calls: int
    cohesion_report: list[dict[str, Any]] = field(default_factory=list)


class MemStrata:
    """One persistent asset bank + the four-step per-segment loop."""

    def __init__(
        self,
        bank: AssetBank | None = None,
        *,
        resolver: IntentResolver | None = None,
        # "" resolves from MEMSTRATA_INTENT_MODE (fast|slow|plan), defaulting to FAST, so the
        # plan-driven read path can be switched on per run without touching call sites.
        intent_mode: str = "",
        plan_producer: PlanProducer | None = None,
        generator: Any = None,
        decomposer: RoleAwareDecomposer | None = None,
        curator: MemoryUpdater | None = None,
        embedder: Any = None,
        angle_classifier: AngleClassifier | None = None,
        crop_attribute_classifier: CropAttributeClassifier | None = None,
        discoverer: Discoverer | None = None,
        policy: MemoryPolicy | None = None,
        crop_quality_gate: bool | None = None,
        relation_hops: int | None = None,
        max_total_representations: int | None = None,
        attributes_when_angles_known: bool | None = None,
        run_dir: str | Path | None = None,
        persist_path: str | Path | None = None,
        movie_id: str = "",
        long_video_path: str | Path | None = None,
        fps: float | None = None,
    ) -> None:
        # One policy object describes the whole write path; explicit kwargs override it.
        self.policy = policy or MemoryPolicy()
        overrides = {
            key: value
            for key, value in (
                ("crop_quality_gate", crop_quality_gate),
                ("relation_hops", relation_hops),
                ("max_total_representations", max_total_representations),
                ("attributes_when_angles_known", attributes_when_angles_known),
            )
            if value is not None
        }
        if overrides:
            self.policy = replace(self.policy, **overrides)

        self.bank = bank or AssetBank()
        # Crash-recovery: restore a previously persisted bank when the caller did
        # not hand in an explicit one.
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path is not None and self.persist_path.exists() and bank is None:
            self.bank = AssetBank.load(self.persist_path)
        intent_mode = (
            intent_mode or os.environ.get("MEMSTRATA_INTENT_MODE", "") or INTENT_MODE_FAST
        ).strip().lower()
        if intent_mode == INTENT_MODE_PLAN and plan_producer is None:
            # Imported lazily: only a plan-mode run pays for the MLLM transport module.
            from memstrata.mllm.planner import MllmPlanner
            from memstrata.skills.intent_understanding.plan import MllmPlanProducer

            plan_producer = MllmPlanProducer(MllmPlanner())
        self.interpreter = IntentInterpreter(
            self.bank,
            resolver=resolver,
            plan_producer=plan_producer,
            mode=intent_mode,
        )
        self.generator = generator or NullGenerator()
        self.relation_hops = max(0, int(self.policy.relation_hops))
        # Production closed loop: classify crop → [image + angle] before curate.
        # Default respects MEMSTRATA_ANGLE_CLASSIFIER (null|heuristic|vlm).
        self.angle_classifier = angle_classifier or build_angle_classifier()
        self.crop_attribute_classifier = (
            crop_attribute_classifier or build_crop_attribute_classifier()
        )
        runtime_embedder = embedder or build_role_routed_embedding_from_env()
        self.decomposer = decomposer or build_decomposer(
            policy=self.policy,
            embedder=runtime_embedder,
            angle_classifier=self.angle_classifier,
            discoverer=discoverer,
        )
        self.curator = curator or build_curator(
            self.bank,
            policy=self.policy,
            embedder=runtime_embedder,
            angle_classifier=self.angle_classifier,
            crop_attribute_classifier=self.crop_attribute_classifier,
        )
        self._warn_on_unapplied_policy(curator=curator, decomposer=decomposer)
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        # Identity of the produced film and the grown long_video.mp4 the memory snapshot's
        # timeline is anchored to; the producer sets long_video_path as it concatenates.
        self.movie_id = str(movie_id or "")
        self.fps = fps
        self.long_video_path = str(long_video_path) if long_video_path else None
        self.long_video_duration_sec: float | None = None
        self.segment_log: list[dict[str, Any]] = []

    def _warn_on_unapplied_policy(
        self, *, curator: MemoryUpdater | None, decomposer: RoleAwareDecomposer | None
    ) -> None:
        """Refuse to silently accept components that ignore this pipeline's policy.

        Handing in a pre-built curator/decomposer is legitimate (the production runner
        seeds the bank first), but only if it was built from the same policy. Previously a
        mismatch was invisible: the pipeline reported a production preset while the
        components ran with library defaults, which is exactly how the bank ended up
        unstratified. Build them via ``build_curator`` / ``build_decomposer``.
        """
        if self.policy.name == MemoryPolicy().name:
            return  # default policy: nothing meaningful to lose
        stale = []
        if curator is not None and getattr(curator, "policy", None) is not self.policy:
            stale.append("curator")
        if (
            decomposer is not None
            and bool(decomposer.crop_quality_gate) != bool(self.policy.crop_quality_gate)
        ):
            stale.append("decomposer")
        if stale:
            warnings.warn(
                f"policy {self.policy.name!r} is NOT applied to the supplied "
                f"{', '.join(stale)}; build them with memstrata.pipeline.build_curator / "
                "build_decomposer(policy=...) so the preset actually takes effect.",
                stacklevel=3,
            )

    @classmethod
    def for_production(
        cls,
        *,
        persist_path: str | Path,
        policy: MemoryPolicy | None = None,
        **kwargs: Any,
    ) -> "MemStrata":
        """Long-video production preset with the lossy/scene-dependent knobs turned on.

        Bundles the opt-in capabilities that are unsafe as global defaults: crash-recovery
        persistence, a bank-wide representation cap, deterministic crop-quality gating,
        one hop of structural relation expansion, per-type thresholds, and the per-segment
        cohesion self-audit. Any of these can be overridden via ``kwargs`` (forwarded to
        ``__init__``) or by passing a tuned ``policy``.
        """
        preset = policy or MemoryPolicy.production()
        return cls(persist_path=persist_path, policy=preset, **kwargs)

    def step1_compose(self, prompt: str, *, segment_id: int) -> tuple[CompositionRequest, ComposedContext, int]:
        request, model_calls = self.interpreter.interpret(prompt, segment_id=segment_id)
        request.relation_hops = self.relation_hops
        context = compose(self.bank, request, as_of_segment_id=segment_id)
        return request, context, model_calls

    def _entities_for_request(self, request: CompositionRequest) -> list[NamedEntity]:
        """Derive Step 3 targets from the intent-resolved, already-addressable records."""
        entities: list[NamedEntity] = []
        for ref in request.references:
            asset = self.bank.get_asset(ref.asset_id)
            if asset is None:
                continue
            latest = asset.representations[-1] if asset.representations else None
            entities.append(
                NamedEntity(
                    name=asset.name,
                    kind=asset.kind,
                    entity_id=asset.asset_id,
                    spatial_angle=ref.preferred_spatial or (
                        latest.spatial_angle if latest is not None else SpatialAngle.UNKNOWN
                    ),
                    state_angle=ref.preferred_state or (
                        latest.state_angle if latest is not None else StateAngle.UNKNOWN
                    ),
                )
            )
        return entities

    def run_segment(
        self,
        prompt: str,
        *,
        segment_id: int,
        named_entities: list[NamedEntity] | None = None,
        segment_video: str | None = None,
        skip_generate: bool = False,
        oracle_observations: list[Observation] | None = None,
        state_events: list[dict] | None = None,
        relations: list[dict] | None = None,
        generation_controls: dict[str, Any] | None = None,
    ) -> SegmentResult:
        """Full loop. Track A: skip_generate=True + oracle_observations."""
        request, context, model_calls = self.step1_compose(prompt, segment_id=segment_id)

        generation: GenerationResult | None = None
        if not skip_generate:
            gen = self.generator
            kwargs: dict[str, Any] = {"segment_id": segment_id}
            # MediaTaskGenerator accepts bank + controls.
            if isinstance(gen, MediaTaskGenerator) or hasattr(gen, "build_task"):
                kwargs["bank"] = self.bank
                kwargs["controls"] = generation_controls
            generation = gen.generate(
                context.enhanced_prompt or request.enhanced_prompt,
                context,
                **kwargs,
            )
            if segment_video is None and generation.video_path:
                segment_video = generation.video_path

        if oracle_observations is not None:
            observations = oracle_observations
        else:
            observations = self.decomposer.decompose(
                segment_id=segment_id,
                named_entities=(
                    named_entities if named_entities is not None else self._entities_for_request(request)
                ),
                segment_video=segment_video,
            )

        touched = self.curator.curate_observations(
            observations,
            segment_id=segment_id,
            state_events=state_events,
            relations=relations,
        )

        # Fourth, retroactive admission gate: the three admission gates only guard
        # *incoming* evidence, so they cannot repair an asset whose first rep was the
        # intruder. Sweeping after every segment stops a polluted identity from
        # conditioning the next generation. No-ops unless a floor is set (which requires
        # a semantic encoder), so the offline default is unchanged.
        cohesion_report: list[dict[str, Any]] = []
        if getattr(self.curator, "selfaudit_each_segment", False):
            cohesion_report = self.curator.audit_cohesion(isolate=True)

        result = SegmentResult(
            segment_id=segment_id,
            request=request,
            context=context,
            generation=generation,
            observations=observations,
            touched_asset_ids=touched,
            model_calls=model_calls,
            cohesion_report=cohesion_report,
        )
        self._log_segment(result)
        if self.persist_path is not None:
            self.bank.save(self.persist_path)
        return result

    def stratification(self) -> dict[str, Any]:
        """Current stratification-fill diagnostic for the bank (see the report docstring)."""
        return stratification_report(self.bank)

    def _log_segment(self, result: SegmentResult) -> None:
        entry = {
            "segment_id": result.segment_id,
            "model_calls": result.model_calls,
            "intent_mode_requested": result.request.requested_mode,
            "intent_mode_used": result.request.used_mode,
            "intent_fallback_reason": result.request.fallback_reason,
            # Which read path resolved the intent this segment: name | description | recency | mllm.
            # Aggregated across segments this quantifies how often stable name-anchoring resolves the
            # intent vs. falling back — the direct source for the paper's "slow-path fraction".
            "intent_resolution_source": getattr(
                result.request, "intent_resolution_source", "recency"
            ),
            # Plan-driven read path only (empty on FAST/SLOW): the generation route the planner
            # chose, what it ruled out, and what it named but the bank could not resolve (a
            # legitimate first sighting, or a hallucinated name — these must be told apart).
            "intent_route": getattr(result.request, "route", ""),
            "intent_forbidden_asset_ids": list(
                getattr(result.request, "forbidden_asset_ids", ())
            ),
            "intent_plan_unresolved_names": list(
                getattr(result.request, "plan_unresolved_names", ())
            ),
            "selected_assets": list(result.context.asset_ids),
            "functions": dict(result.context.functions),
            "requirements": dict(result.context.requirements),
            "exclusions": list(result.context.exclusions),
            "representation_ids": dict(result.context.representation_ids),
            "touched_asset_ids": list(result.touched_asset_ids),
            "bank_size": len(self.bank.assets),
            "bank_version": self.bank.version,
            # Direct evidence for the stratified-memory claim, per segment. If
            # ``spatial_known_ratio`` stays at 0 the attribute classifier is not wired and
            # no angle-stratification result may be reported from this run.
            "stratification": stratification_report(self.bank),
            "observation_sources": {
                source: sum(1 for o in result.observations if o.source == source)
                for source in sorted({o.source for o in result.observations})
            },
            "cohesion_selfaudit": list(result.cohesion_report),
            "generation": None,
        }
        if result.generation is not None:
            entry["generation"] = {
                "video_path": result.generation.video_path,
                "meta": result.generation.meta,
                "artifact": (
                    result.generation.artifact.to_dict()
                    if result.generation.artifact is not None
                    else None
                ),
                "task": (
                    result.generation.task.to_dict()
                    if result.generation.task is not None
                    else None
                ),
            }
        self.segment_log.append(entry)
        if self.run_dir is not None:
            segment_dir = self.run_dir / f"segment_{result.segment_id:03d}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            (segment_dir / "pipeline_record.json").write_text(
                json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (self.run_dir / "run_ledger.json").write_text(
                json.dumps({"segments": self.segment_log}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (segment_dir / "asset_bank_snapshot.json").write_text(
                json.dumps(self.bank.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.write_stratification_summary()
            # Curated, entity→states→visual memory product, refreshed every segment.
            self.write_memory_snapshot()

    def write_memory_snapshot(self) -> Path | None:
        """Emit the clean, human-readable ``memory.json`` (+ ``visual/`` tree).

        The per-segment ``asset_bank_snapshot.json`` is the raw internal dump; this is the
        curated, entity→states→visual view (schema ``memstrata-memory-1.0``) that mirrors
        the benchmark gt and is refreshed every segment so it stays a live, dynamically
        updated record. All ``sec`` values are on the grown ``long_video.mp4`` timeline
        (``self.long_video_path`` / ``self.long_video_duration_sec``, set by the producer).
        No-op without a ``run_dir``.
        """
        if self.run_dir is None:
            return None
        return export_memory_snapshot(
            self.bank,
            self.run_dir,
            movie_id=self.movie_id,
            fps=self.fps,
            video_path=self.long_video_path,
            video_duration_sec=self.long_video_duration_sec,
        )

    def write_stratification_summary(self, path: str | Path | None = None) -> Path | None:
        """Persist the stratification diagnostic (latest state + per-segment trend)."""
        target = Path(path) if path else (self.run_dir / "stratification.json" if self.run_dir else None)
        if target is None:
            return None
        payload = {
            "policy": self.policy.name,
            "angle_classifier": type(self.angle_classifier).__name__,
            "crop_attribute_classifier": type(self.crop_attribute_classifier).__name__,
            "embedder_is_semantic": getattr(self.curator, "embedder_is_semantic", None),
            "cohesion_floor": getattr(self.curator, "cohesion_floor", None),
            "latest": stratification_report(self.bank),
            "per_segment": [
                {"segment_id": row["segment_id"], **row.get("stratification", {})}
                for row in self.segment_log
            ],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target
