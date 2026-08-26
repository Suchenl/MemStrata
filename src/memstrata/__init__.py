"""MemStrata — active composition over role-aware assets (paper §3).

Package layout
--------------
bank/       Asset bank A_n
steps/      Four-step loop (intent, compose, generate, decompose, curate)
pipeline    MemStrata orchestrator
adapters/   Track A bench JSON surface
encoders/   Type-routed visual encoders (Step 3)
llm/        Intent resolver backend
lib/        Paths, weights, dedup, media helpers
extras/     Optional (shot boundary, …) — not on the paper hot path
"""

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetKind,
    AssetRepresentation,
    AssetStatus,
    AssetType,
    LifecycleStatus,
    ProductionAssetSpace,
)
from memstrata.steps import (
    ActiveComposer,
    AssetCurator,
    CompositionRequest,
    ComposedContext,
    EntityObservation,
    GenerationArtifact,
    GenerationResult,
    Generator,
    IntentInterpreter,
    InverseIngester,
    MediaGenerationTask,
    MediaGenerator,
    MediaTaskGenerator,
    MediaTaskType,
    MemoryUpdater,
    NamedEntity,
    NullGenerator,
    Observation,
    OracleBackend,
    OracleGenerator,
    RecordingBackend,
    RoleAwareDecomposer,
    build_video_backend,
    compose,
    composed_reference_images,
    list_video_backend_names,
)
from memstrata.pipeline import SegmentResult, MemStrata
from memstrata.mllm import MllmPlanner, build_angle_classifier
from memstrata.adapters import BenchReplayAdapter

__all__ = [
    "Asset",
    "AssetBank",
    "AssetKind",
    "AssetRepresentation",
    "AssetStatus",
    "AssetType",
    "LifecycleStatus",
    "ProductionAssetSpace",
    "CompositionRequest",
    "IntentInterpreter",
    "ActiveComposer",
    "ComposedContext",
    "compose",
    "GenerationArtifact",
    "GenerationResult",
    "Generator",
    "MediaGenerationTask",
    "MediaGenerator",
    "MediaTaskType",
    "MediaTaskGenerator",
    "NullGenerator",
    "OracleGenerator",
    "OracleBackend",
    "RecordingBackend",
    "build_video_backend",
    "list_video_backend_names",
    "composed_reference_images",
    "NamedEntity",
    "Observation",
    "RoleAwareDecomposer",
    "MemoryUpdater",
    "AssetCurator",
    "EntityObservation",
    "InverseIngester",
    "SegmentResult",
    "MemStrata",
    "MllmPlanner",
    "build_angle_classifier",
    "BenchReplayAdapter",
]
