"""MemStrata production entrypoint: screenplay-driven long-video generation.

Importable runner (``from memstrata.production import run_production``) plus a CLI
(``python -m memstrata.production.run``). Wires the existing four-step loop
(``MemStrata.for_production`` + steps/skills) over a production_screenplay; no orchestration
logic lives in ``scripts/`` — those are thin bash launchers only.
"""

from memstrata.production.run import build_pipeline, main, run_production

__all__ = ["build_pipeline", "main", "run_production"]
