"""Build a concrete video backend by config name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memstrata.steps.generate.backends._support import (
    ArtifactStore,
    RunContext,
    default_models_config_dir,
)
from memstrata.steps.generate.backends.oracle import OracleBackend
from memstrata.steps.generate.backends.recording import RecordingBackend


def build_video_backend(
    name: str,
    *,
    output_dir: str | Path,
    run_id: str = "memstrata_gen",
    models_config: str | Path | None = None,
) -> Any:
    """Return a backend with ``generate(task) -> GenerationArtifact``.

    ``name``:
      - ``recording`` / ``oracle`` — local lightweight backends
      - any stem under ``configs/video_gen/<name>.toml`` — real generators
        (diffusers / native_vace / ltx23 / longcat / helios / wan_lightx2v /
         wan22_turbo / magref / multishotmaster)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = name.lower().strip()

    if key == "recording":
        return RecordingBackend(out)
    if key == "oracle":
        return OracleBackend(out)

    cfg_root = Path(models_config) if models_config else default_models_config_dir()
    context = RunContext.create(out)
    store = ArtifactStore(context)

    # Peek provider from TOML without constructing the heavy class first.
    from memstrata.steps.generate.backends.diffusers_backend import _load_video_gen_config

    config = _load_video_gen_config(key, cfg_root)
    provider = str(config.get("provider", "diffusers")).lower()

    if provider in {"native_vace", "native_wan"}:
        from memstrata.steps.generate.backends.native_vace_backend import NativeVaceBackend
        return NativeVaceBackend.from_config(key, context, store, run_id, cfg_root)
    if provider == "ltx23":
        from memstrata.steps.generate.backends.ltx23_backend import Ltx23Backend
        return Ltx23Backend.from_config(key, context, store, run_id, cfg_root)
    if provider == "longcat":
        from memstrata.steps.generate.backends.longcat_backend import LongCatBackend
        return LongCatBackend.from_config(key, context, store, run_id, cfg_root)
    if provider == "helios":
        from memstrata.steps.generate.backends.helios_backend import HeliosBackend
        return HeliosBackend.from_config(key, context, store, run_id, cfg_root)
    if provider == "wan22_turbo":
        from memstrata.steps.generate.backends.wan22_turbo_backend import Wan22TurboBackend
        return Wan22TurboBackend.from_config(key, context, store, run_id, cfg_root)
    if provider in {"wan_lightx2v", "lightx2v"}:
        from memstrata.steps.generate.backends.wan22_lightx2v_backend import Wan22LightX2VBackend
        return Wan22LightX2VBackend.from_config(key, context, store, run_id, cfg_root)
    if provider == "magref":
        from memstrata.steps.generate.backends.magref_backend import MagRefBackend
        return MagRefBackend.from_config(key, context, store, run_id, cfg_root)
    if provider == "multishotmaster":
        from memstrata.steps.generate.backends.multishotmaster_backend import MultiShotMasterBackend
        return MultiShotMasterBackend.from_config(key, context, store, run_id, cfg_root)

    from memstrata.steps.generate.backends.diffusers_backend import DiffusersVideoBackend
    return DiffusersVideoBackend.from_config(key, context, store, run_id, cfg_root)


def list_video_backend_names(*, models_config: str | Path | None = None) -> list[str]:
    cfg_root = Path(models_config) if models_config else default_models_config_dir()
    names = ["recording", "oracle"]
    video_dir = cfg_root / "video_gen"
    if video_dir.is_dir():
        names.extend(sorted(p.stem for p in video_dir.glob("*.toml") if p.stem not in names))
    return names
