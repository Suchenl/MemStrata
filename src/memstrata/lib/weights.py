"""Single source of truth for where model weights live in MemStrata."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Absolute repository root, searching upwards for AGENTS.md or .git."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").is_file() or (parent / ".git").exists():
            return parent
    # Fallback to parents[5] based on: .../benchmarks/MemStrata/src/memstrata/lib/weights.py
    return current.parents[5]


def weights_root() -> Path:
    """Absolute path to the project weights directory (created if missing)."""
    override = os.environ.get("MEMSTRATA_WEIGHTS_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        root = repo_root() / "models" / "model_weights"
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_cache_dir() -> str:
    """Hugging Face cache directory under the project weights root."""
    cache = weights_root() / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(weights_root()))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    return str(cache)


def torch_hub_dir() -> str:
    """``torch.hub`` cache directory under the project weights root."""
    cache = weights_root() / "torch_hub"
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


def configure_torch_hub() -> str:
    """Point ``torch.hub`` (and its checkpoint downloads) at the project weights root."""
    directory = torch_hub_dir()
    os.environ.setdefault("TORCH_HOME", str(weights_root()))
    hf_cache_dir()  # ensure HF_HOME / HF_HUB_CACHE point into the project too
    try:
        import torch
        torch.hub.set_dir(directory)
    except ImportError:
        pass
    return directory


def public_models_root() -> Path:
    """Shared public model root, organized as ``<org>/<repo>``.

    Unset is allowed: CPU import / recording must not crash. Resolving a
    real checkpoint without the env var raises.
    """
    override = os.environ.get("PUBLIC_MODELS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    raise RuntimeError(
        "PUBLIC_MODELS_ROOT is not set; required only when loading generator/encoder weights"
    )


def resolve_model_reference(model: str) -> str:
    """Expand local model-root placeholders while preserving remote repo ids."""
    token = "${PUBLIC_MODELS_ROOT}"
    if model.startswith(token):
        return str(public_models_root() / model[len(token):].lstrip("/"))
    return os.path.expandvars(os.path.expanduser(model))
