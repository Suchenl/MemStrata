"""Decomposition skill: named entities → type-routed observations (paper Decompose step).

Canonical home for ``RoleAwareDecomposer`` + the ``Cropper`` / ``Discoverer`` protocols.
``memstrata.steps.decompose`` re-exports these for backward compatibility.
"""

from memstrata.skills.decomposition.decomposer import (
    SOURCE_DISCOVERED,
    SOURCE_REQUESTED,
    Cropper,
    DiscoveredEntity,
    Discoverer,
    NamedEntity,
    Observation,
    RoleAwareDecomposer,
)
from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer

__all__ = [
    "SOURCE_DISCOVERED",
    "SOURCE_REQUESTED",
    "Cropper",
    "DiscoveredEntity",
    "Discoverer",
    "NamedEntity",
    "Observation",
    "RoleAwareDecomposer",
    "VlmEntityDecomposer",
]
