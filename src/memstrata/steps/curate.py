"""Step 4: Asset Memory & Management (落库) — backward-compatible shim.

The implementation moved to the ``memstrata.skills.memory_update`` skill (reusable
capability). This module re-exports the public API so existing
``from memstrata.steps.curate import ...`` call sites keep working.
"""

from __future__ import annotations

from memstrata.skills.memory_update.curator import (
    AssetCurator,
    EntityObservation,
    InverseIngester,
    MemoryPolicy,
    MemoryUpdater,
    stratification_report,
)

__all__ = [
    "MemoryUpdater",
    "AssetCurator",
    "EntityObservation",
    "InverseIngester",
    "MemoryPolicy",
    "stratification_report",
]
