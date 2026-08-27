"""Self-contained weight-path + vecmath helpers for the crop_acquisition subsystem.

Vendored (copy + hard-resolve) from ``vmem_bench.common.model_weights`` and
``vmem_bench.common.vecmath`` so the production ``memstrata`` package keeps its
hard rule of **zero imports from ``vmem_bench``** (see
``memstrata/docs/design_philosophy.md`` §5 and ``this repository/AGENTS.md`` rule 2).

Weight-path defaults are resolved inside this MemStrata checkout or through
``PUBLIC_MODELS_ROOT``. Every default is env-overridable, so the package does
not depend on a surrounding monorepo.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# This file lives at <repo>/src/memstrata/skills/crop_acquisition/_common.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
# Shared public checkpoints (SAM3, DINOv3) default location; env ``PUBLIC_MODELS_ROOT`` overrides.
_PUBLIC_MODELS_ROOT_FALLBACK = Path(os.environ["PUBLIC_MODELS_ROOT"]) if os.environ.get("PUBLIC_MODELS_ROOT") else Path(".")
# Vendored transformers>=5.9 for SAM3; env ``MEMSTRATA_SAM3_DEPS`` overrides.
_SAM3_DEPS_FALLBACK = _PROJECT_ROOT / "models" / "vendor" / "sam3_transformers59"


def repo_root() -> Path:
    """Return this standalone MemStrata repository root."""
    return _PROJECT_ROOT


def weights_root() -> Path:
    override = os.environ.get("MEMSTRATA_WEIGHTS_ROOT")
    root = Path(override).expanduser().resolve() if override else repo_root() / "models" / "model_weights"
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_cache_dir() -> str:
    """Resolve (and create) the upstream project HF hub cache; wire the HF_* env for offline nodes."""
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
