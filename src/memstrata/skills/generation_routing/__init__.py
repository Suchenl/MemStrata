"""Generation-routing skill: per-segment generation-mode decision (role R2b).

Rules restrict the *feasible* generation modes (hard physical constraints); the MLLM then
picks among them. Modes map onto the ``MediaTaskGenerator`` / ``HeliosBackend`` seeding
branches (continue_ar / reanchor_lastframe / recompose_partial / recompose_keyframe).
"""

from memstrata.skills.generation_routing.router import (
    ALL_MODES,
    ROUTER_SCHEMA,
    GenMode,
    GenerationRouter,
    RouteDecision,
)

__all__ = ["ALL_MODES", "ROUTER_SCHEMA", "GenMode", "GenerationRouter", "RouteDecision"]
