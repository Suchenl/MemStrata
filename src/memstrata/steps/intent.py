"""Step 1: Intent Interpretation (read path) — backward-compatible shim.

The implementation moved to the ``memstrata.skills.intent_understanding`` skill (paper
Stage 3, reusable capability, easier to migrate / tune). This module re-exports the public
API so existing ``from memstrata.steps.intent import ...`` call sites keep working.
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

__all__ = [
    "FUNCTION_BY_TYPE", "INTENT_MODE_FAST", "INTENT_MODE_PLAN", "INTENT_MODE_SLOW",
    "INTENT_MODES", "AssetReference", "CompositionRequest", "IntentInterpreter",
    "IntentResolver", "MllmIntentResolver",
]
