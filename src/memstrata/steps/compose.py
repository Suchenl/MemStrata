"""Step 1 (tail): model-free composition — backward-compatible shim.

The implementation moved to the ``memstrata.skills.composition`` skill (the paper's Compose
step). This module re-exports the public API so existing
``from memstrata.steps.compose import ...`` call sites keep working.
"""

from __future__ import annotations

from memstrata.skills.composition.compose import (
    ActiveComposer,
    ComposedContext,
    compose,
    is_usable,
    select_reps,
    select_reps_for_function,
    select_representations,
    select_representations_for_function,
    usable_representation_ids,
)

__all__ = [
    "ActiveComposer", "ComposedContext", "compose", "is_usable", "select_reps",
    "select_reps_for_function", "select_representations", "select_representations_for_function",
    "usable_representation_ids",
]
