"""Step 3: Role-aware Asset Decomposition — backward-compatible shim.

The implementation moved to the ``memstrata.skills.decomposition`` skill (reusable capability,
easier to migrate). This module re-exports the public API so existing
``from memstrata.steps.decompose import ...`` call sites keep working.
"""

from __future__ import annotations

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

__all__ = [
    "SOURCE_DISCOVERED",
    "SOURCE_REQUESTED",
    "Cropper",
    "DiscoveredEntity",
    "Discoverer",
    "NamedEntity",
    "Observation",
    "RoleAwareDecomposer",
]
