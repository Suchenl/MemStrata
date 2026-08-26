"""Build a concrete image (keyframe) backend by config name.

Mirrors ``steps/generate/backends/factory.py`` for the image side. Self-contained:
only imports ``memstrata`` and loads model code / weights by path.
"""

from __future__ import annotations

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    import tomli as tomllib
from pathlib import Path
from typing import Any

from memstrata.steps.generate.backends._support import (
    ArtifactStore,
    RunContext,
    default_models_config_dir,
)


def _load_image_gen_config(name: str, models_config: Path) -> dict[str, Any]:
    path = Path(models_config) / "image_gen" / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No image-gen config for backend '{name}' at {path}. "
            f"Add a TOML under {Path(models_config) / 'image_gen'}."
        )
    data = tomllib.loads(path.read_text())
    models = data.get("models", {})
    if not models:
        raise ValueError(f"Image-gen config {path} has no [models.*] table")
    # Single backend per file: take the first (and typically only) model table.
    return next(iter(models.values()))


def build_image_backend(
    name: str,
    *,
    output_dir: str | Path,
    run_id: str = "memstrata_img",
    models_config: str | Path | None = None,
) -> Any:
    """Return an image backend with ``generate(task) -> GenerationArtifact``.

    ``name`` is the stem under ``configs/image_gen/<name>.toml`` (e.g.
    ``flux.2-klein-9b-kv-fp8``). Only the ``flux`` family is wired today.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg_root = Path(models_config) if models_config else default_models_config_dir()
    context = RunContext.create(out)
    store = ArtifactStore(context)

    config = _load_image_gen_config(name, cfg_root)
    family = str(config.get("family", "flux")).lower()

    if family == "flux":
        from memstrata.steps.generate.image_backends.flux_klein_backend import FluxKleinImageBackend
        return FluxKleinImageBackend.from_config(context, run_id, store, **config)

    raise ValueError(f"Unsupported image-gen family '{family}' for backend '{name}'")


def list_image_backend_names(*, models_config: str | Path | None = None) -> list[str]:
    cfg_root = Path(models_config) if models_config else default_models_config_dir()
    image_dir = cfg_root / "image_gen"
    if not image_dir.is_dir():
        return []
    return sorted(p.stem for p in image_dir.glob("*.toml"))
