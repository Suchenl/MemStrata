"""MemStrata Compose step (read path) — reusable skill.

The paper's **Compose** step: turn external intent g_n + the addressable asset space A_n into a
structured composition request q_n (``IntentInterpreter``, model-free name/alias match by default,
optional slow MLLM resolver) and then dereference it into a Composed Context C_n
(``compose``, O(1) lookup — no similarity search, no per-read model call). Moved out of
``steps/`` (like ``decomposition`` / ``memory_update``) so it is reusable and the
agent-in-the-loop optimizer can point at a single ``registry.toml`` for the read-path knobs.
"""

from __future__ import annotations

# Intent Interpretation (Stage 3) moved to ``memstrata.skills.intent_understanding``.
# Re-exported here for backward compatibility with callers that imported the intent
# API from the composition package.
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
from memstrata.skills.composition.compose import (
    ActiveComposer,
    ComposedContext,
    compose,
    select_reps,
    select_reps_for_function,
    select_representations,
    select_representations_for_function,
    usable_representation_ids,
)

__all__ = [
    "FUNCTION_BY_TYPE", "INTENT_MODE_FAST", "INTENT_MODE_PLAN", "INTENT_MODE_SLOW",
    "INTENT_MODES",
    "AssetReference", "CompositionRequest", "IntentInterpreter", "IntentResolver",
    "MllmIntentResolver", "ActiveComposer", "ComposedContext", "compose", "select_reps",
    "select_reps_for_function", "select_representations", "select_representations_for_function",
    "usable_representation_ids",
]
