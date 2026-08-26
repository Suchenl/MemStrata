"""MLLM backends + role registry/runner for the MemStrata production loop.

Serves both text roles (intent resolution, asset selection, planning) and vision
roles (crop attribute / angle-state classification, ingest/dedup judging), hence
``mllm`` (multimodal) rather than ``llm``.
"""

from memstrata.mllm.angle_classifier import (
    AngleClassification,
    AngleClassifier,
    HeuristicAngleClassifier,
    NullAngleClassifier,
    VlmAngleClassifier,
    build_angle_classifier,
)
from memstrata.mllm.crop_attributes import (
    CropAttributeClassifier,
    CropAttributePack,
    HeuristicCropAttributeClassifier,
    Lighting,
    NullCropAttributeClassifier,
    Occlusion,
    ShotSize,
    VlmCropAttributeClassifier,
    build_crop_attribute_classifier,
)
from memstrata.mllm.planner import MllmPlanner
from memstrata.mllm.runner import (
    HttpTransport,
    MllmRoleRunner,
    ScriptedTransport,
    Transport,
    image_content_part,
)
from memstrata.mllm.roles import (
    DEFAULT_MODEL,
    OPTIMIZER_MODEL,
    ROLE_REGISTRY,
    Modality,
    Path,
    RoleSpec,
    Sampling,
    Status,
    Step,
    get_role,
    hot_path_roles,
    roles_by_status,
    roles_for_step,
    validate_registry,
)

__all__ = [
    "MllmPlanner",
    "MllmRoleRunner",
    "HttpTransport",
    "ScriptedTransport",
    "Transport",
    "image_content_part",
    "DEFAULT_MODEL",
    "OPTIMIZER_MODEL",
    "ROLE_REGISTRY",
    "Modality",
    "Path",
    "RoleSpec",
    "Sampling",
    "Status",
    "Step",
    "get_role",
    "hot_path_roles",
    "roles_by_status",
    "roles_for_step",
    "validate_registry",
    "AngleClassification",
    "AngleClassifier",
    "HeuristicAngleClassifier",
    "NullAngleClassifier",
    "VlmAngleClassifier",
    "build_angle_classifier",
    "CropAttributeClassifier",
    "CropAttributePack",
    "HeuristicCropAttributeClassifier",
    "Lighting",
    "NullCropAttributeClassifier",
    "Occlusion",
    "ShotSize",
    "VlmCropAttributeClassifier",
    "build_crop_attribute_classifier",
]
