"""Backward-compatible shim — Intent Interpretation moved to a dedicated skill.

The read-path intent parser now lives in ``memstrata.skills.intent_understanding``
(paper Stage 3), separate from Compose (Stage 4). This module re-exports the public
API so existing ``from memstrata.skills.composition.intent import ...`` call sites keep
working unchanged.
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
]
