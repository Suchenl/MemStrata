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


PRODUCTION_DEFAULT = ProductionProfile(
    name="production",
    write_naming="mllm",
    embedder_provider="dinov3",
    read_slow_fallback=True,
    read_max_reps_per_asset=1,
    wedetect_url="http://127.0.0.1:8710",
    require_wedetect=False,
)

PAPER_TRACKA_202607 = ProductionProfile(
    name="paper_tracka_202607",
    write_naming="mllm",
    embedder_provider="dinov3",
    read_slow_fallback=True,
    read_max_reps_per_asset=1,
    wedetect_url="http://127.0.0.1:8710",
    require_wedetect=True,
)

_PROFILES = {
    PRODUCTION_DEFAULT.name: PRODUCTION_DEFAULT,
    PAPER_TRACKA_202607.name: PAPER_TRACKA_202607,
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
    "PAPER_TRACKA_202607",
    "PRODUCTION_DEFAULT",
    "ProductionProfile",
    "resolve_production_profile",
]
