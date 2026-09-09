"""Explicit, versioned production behavior profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductionProfile:
    name: str
    write_naming: str
    embedder_provider: str
    read_slow_fallback: bool
    read_max_reps_per_asset: int
    wedetect_url: str
    require_wedetect: bool
    mllm_base_url: str
    mllm_model: str
    require_mllm: bool
    adaptive_location_memory: bool = False
    read_context_rep_budget: int | None = None
    location_storage_cap: int = 12
    location_read_max_refs: int = 4


PRODUCTION_DEFAULT = ProductionProfile(
    name="production",
    write_naming="mllm",
    embedder_provider="dinov3",
    read_slow_fallback=True,
    read_max_reps_per_asset=1,
    wedetect_url="http://127.0.0.1:8710",
    require_wedetect=False,
    mllm_base_url="http://127.0.0.1:8000/v1",
    mllm_model="Qwen3.5-9B-Instruct",
    require_mllm=False,
)

PAPER_TRACKA_202607 = ProductionProfile(
    name="paper_tracka_202607",
    write_naming="mllm",
    embedder_provider="dinov3",
    read_slow_fallback=True,
    read_max_reps_per_asset=1,
    wedetect_url="http://127.0.0.1:8710",
    require_wedetect=True,
    mllm_base_url="http://127.0.0.1:8000/v1",
    mllm_model="Qwen3.5-9B-Instruct",
    require_mllm=True,
)

LOCATION_ADAPTIVE_V1 = ProductionProfile(
    name="location_adaptive_v1",
    write_naming="mllm",
    embedder_provider="dinov3",
    read_slow_fallback=True,
    # Retained as the character/prop and legacy fallback ceiling. Location reads
    # use the explicit adaptive policy below.
    read_max_reps_per_asset=1,
    wedetect_url="http://127.0.0.1:8710",
    require_wedetect=True,
    mllm_base_url="http://127.0.0.1:8000/v1",
    mllm_model="Qwen3.5-9B-Instruct",
    require_mllm=True,
    adaptive_location_memory=True,
    read_context_rep_budget=16,
    location_storage_cap=12,
    location_read_max_refs=4,
)

_PROFILES = {
    PRODUCTION_DEFAULT.name: PRODUCTION_DEFAULT,
    PAPER_TRACKA_202607.name: PAPER_TRACKA_202607,
    LOCATION_ADAPTIVE_V1.name: LOCATION_ADAPTIVE_V1,
}


def resolve_production_profile(name: str | ProductionProfile) -> ProductionProfile:
    if isinstance(name, ProductionProfile):
        return name
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown production profile {name!r}; choose one of {sorted(_PROFILES)}"
        ) from exc


__all__ = [
    "LOCATION_ADAPTIVE_V1",
    "PAPER_TRACKA_202607",
    "PRODUCTION_DEFAULT",
    "ProductionProfile",
    "resolve_production_profile",
]
