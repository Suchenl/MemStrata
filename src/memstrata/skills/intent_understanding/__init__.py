"""Intent Understanding skill (paper Stage 3: Intent Interpretation).

Read-path front-end: resolve external intent ``g_n`` + asset space ``A_n`` into a
structured composition request ``q_n``. FAST mode is model-free (name/alias match via
``skills.memory_retrieval.name_match`` → description-overlap → recency); SLOW mode adds
an opt-in MLLM resolver; PLAN mode returns a typed ``IntentPlanV1`` (references + required
appearance state + forbidden entities + generation route). Dereferencing ``q_n`` into a
Composed Context is the separate ``skills.composition`` (Compose) stage.
"""

from __future__ import annotations

from memstrata.skills.intent_understanding.interpreter import (
    FUNCTION_BY_TYPE,
    INTENT_MODE_FAST,
    INTENT_MODE_PLAN,
    INTENT_MODE_SLOW,
    INTENT_MODES,
    AssetReference,
    CompositionRequest,
    IntentInterpreter,
    IntentResolver,
    MllmIntentResolver,
)
from memstrata.skills.intent_understanding.plan import (
    ROUTE_I2V_COMPOSED,
    ROUTE_I2V_CONTINUE,
    ROUTE_T2V,
    ROUTES,
    IntentPlanV1,
    MllmPlanProducer,
    PlanProducer,
    PlanReference,
    ResolvedPlan,
    parse_plan,
    resolve_plan,
)

__all__ = [
    "FUNCTION_BY_TYPE",
    "INTENT_MODE_FAST",
    "INTENT_MODE_PLAN",
    "INTENT_MODE_SLOW",
    "INTENT_MODES",
    "AssetReference",
    "CompositionRequest",
    "IntentInterpreter",
    "IntentResolver",
    "MllmIntentResolver",
    # Plan-driven read path (IntentPlanV1).
    "ROUTES",
    "ROUTE_I2V_COMPOSED",
    "ROUTE_I2V_CONTINUE",
    "ROUTE_T2V",
    "IntentPlanV1",
    "MllmPlanProducer",
    "PlanProducer",
    "PlanReference",
    "ResolvedPlan",
    "parse_plan",
    "resolve_plan",
]
