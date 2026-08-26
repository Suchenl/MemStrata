# Deprecated: renamed to ``memstrata.skills.memory_update``; kept as a back-compat shim.
from memstrata.skills.memory_update import *  # noqa: F401,F403
from memstrata.skills.memory_update import (  # noqa: F401
    AssetCurator,
    EntityObservation,
    InverseIngester,
    MemoryPolicy,
    MemoryUpdater,
    export_memory_snapshot,
    stratification_report,
)
