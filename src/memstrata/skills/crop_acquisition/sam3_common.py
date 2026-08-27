"""Shared SAM3 transformers loader for refine + concept segmentation.

Vendored from bench S5 ``sam3_common``. The vendored-deps default is hard-resolved
via ``memstrata.skills.crop_acquisition._common.sam3_deps_dir`` to the upstream project
``models/vendor/sam3_transformers59`` dir (env ``MEMSTRATA_SAM3_DEPS`` overrides).
"""

from __future__ import annotations

import sys

from memstrata.skills.crop_acquisition._common import sam3_deps_dir


def vendored_deps_dir() -> str:
    """Vendored SAM3-capable transformers dir (env override, else upstream project vendor)."""
    return sam3_deps_dir()


def import_sam3_classes():
    """Import (Sam3Model, Sam3Processor) without hot-swapping transformers versions."""
    try:
        from transformers import Sam3Model, Sam3Processor

        return Sam3Model, Sam3Processor
    except ImportError:
        pass
    deps = vendored_deps_dir()
    if not deps:
        raise RuntimeError(
            "SAM3 requires transformers>=5.9 or MEMSTRATA_SAM3_DEPS "
            "(or models/vendor/sam3_transformers59)"
        )
    if "transformers" in sys.modules:
        raise RuntimeError(
            "SAM3 cannot load after an incompatible transformers import; "
            f"launch with PYTHONPATH={deps} prepended"
        )
    sys.path.insert(0, deps)
    from transformers import Sam3Model, Sam3Processor

    return Sam3Model, Sam3Processor
