"""Paper §3.2 four steps: intent → compose → generate → decompose → curate."""

from memstrata.steps.intent import CompositionRequest, IntentInterpreter
from memstrata.steps.compose import ActiveComposer, ComposedContext, compose
from memstrata.steps.generate import (
    GenerationArtifact,
    GenerationResult,
    Generator,
    MediaGenerationTask,
    MediaGenerator,
    MediaTaskGenerator,
    MediaTaskType,
    NullGenerator,
    OracleGenerator,
    composed_reference_images,
)
from memstrata.steps.generate.backends import (
    OracleBackend,
    RecordingBackend,
    build_video_backend,
    list_video_backend_names,
)
from memstrata.steps.decompose import NamedEntity, Observation, RoleAwareDecomposer
from memstrata.steps.curate import (
    AssetCurator,
    EntityObservation,
    InverseIngester,
    MemoryUpdater,
)

__all__ = [
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
    "composed_reference_images",
    "OracleBackend",
    "RecordingBackend",
    "build_video_backend",
    "list_video_backend_names",
    "NamedEntity",
    "Observation",
    "RoleAwareDecomposer",
    "MemoryUpdater",
    "AssetCurator",
    "EntityObservation",
    "InverseIngester",
]
