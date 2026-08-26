"""Self-contained weight-path + vecmath helpers for the crop_acquisition subsystem.

Vendored (copy + hard-resolve) from ``vmem_bench.common.model_weights`` and
``vmem_bench.common.vecmath`` so the production ``memstrata`` package keeps its
hard rule of **zero imports from ``vmem_bench``** (see
``memstrata/docs/design_philosophy.md`` §5 and ``benchmarks/MemStrata/AGENTS.md`` rule 2).

Weight-path defaults are *hard-resolved to the Montage monorepo root* (not the
``benchmarks/MemStrata`` subdir), because the HF hub snapshots (GroundingDINO, etc.)
live under Montage ``models/model_weights/hub``. Every default is env-overridable.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# Hard-resolved Montage monorepo root default (env ``MONTAGE_WEIGHTS_ROOT`` overrides the
# weights root directly; ``MONTAGE_ROOT`` overrides the discovered repo root).
_MONTAGE_ROOT_FALLBACK = Path(".")
# Shared public checkpoints (SAM3, DINOv3) default location; env ``PUBLIC_MODELS_ROOT`` overrides.
_PUBLIC_MODELS_ROOT_FALLBACK = Path(os.environ["PUBLIC_MODELS_ROOT"]) if os.environ.get("PUBLIC_MODELS_ROOT") else Path(".")
# Vendored transformers>=5.9 for SAM3; env ``MEMSTRATA_SAM3_DEPS`` overrides.
_SAM3_DEPS_FALLBACK = _MONTAGE_ROOT_FALLBACK / "models" / "vendor" / "sam3_transformers59"


def repo_root() -> Path:
    """Montage monorepo root, hard-resolved to the root that owns ``models/model_weights``.

    Deliberately NOT the ``benchmarks/MemStrata`` subdir: the HF hub cache used by
    GroundingDINO/DINOv3 lives under the Montage root ``models/model_weights/hub``.
    """
    override = os.environ.get("MONTAGE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "montage").is_dir() and (parent / "models" / "model_weights").is_dir():
            return parent
    if _MONTAGE_ROOT_FALLBACK.is_dir():
        return _MONTAGE_ROOT_FALLBACK
    # crop_acquisition/_common.py -> .../memstrata/skills/crop_acquisition -> Montage is parents[6].
    return current.parents[6]


def weights_root() -> Path:
    override = os.environ.get("MONTAGE_WEIGHTS_ROOT")
    root = Path(override).expanduser().resolve() if override else repo_root() / "models" / "model_weights"
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_cache_dir() -> str:
    """Resolve (and create) the Montage HF hub cache; wire the HF_* env for offline nodes."""
    cache = weights_root() / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(weights_root()))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    return str(cache)


def public_models_root() -> Path:
    """Shared public checkpoint root (SAM3, DINOv3). Default ``${PUBLIC_MODELS_ROOT}``."""
    override = os.environ.get("PUBLIC_MODELS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return _PUBLIC_MODELS_ROOT_FALLBACK


def sam3_deps_dir() -> str:
    """Vendored SAM3-capable transformers>=5.9 dir (env override else hard-resolved default)."""
    deps = os.environ.get("MEMSTRATA_SAM3_DEPS")
    if deps:
        return deps
    return str(_SAM3_DEPS_FALLBACK) if _SAM3_DEPS_FALLBACK.is_dir() else ""


Vector = list[float]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must share the same dimensionality")
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    return dot / (norm_l * norm_r)


def cosine_distance(left: Vector, right: Vector) -> float:
    return 1.0 - cosine_similarity(left, right)
