"""MemStrata production entrypoint: screenplay-driven long-video generation.

Importable runner (``from memstrata.production import run_production``) plus a CLI
(``python -m memstrata.production.run``). Wires the existing four-step loop
(``MemStrata.for_production`` + steps/skills) over a production_screenplay; no orchestration
logic lives in ``scripts/`` — those are thin bash launchers only.
"""

from __future__ import annotations

__all__ = ["build_pipeline", "main", "run_production"]


def __getattr__(name: str):
    # Lazy: `python -m memstrata.production.run` must not import run.py via this
    # package __init__, or runpy warns that the module is already in sys.modules.
    if name in __all__:
        from memstrata.production.run import build_pipeline, main, run_production

        return {
            "build_pipeline": build_pipeline,
            "main": main,
            "run_production": run_production,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
