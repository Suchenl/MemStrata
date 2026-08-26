"""Memory retrieval skill for MemStrata.

Deliberately separate from ``skills.composition``: composition is deterministic
address dereferencing over already-known assets, while this skill owns *recall*
over memory. It has two independent read paths:

- ``name_match``: model-free deterministic name/alias recall over stored assets —
  the fast read-path anchor used by ``composition.intent``.
- ``retrievers``: similarity search over historical frame/segment memory, used
  only for controls/ablations.

``name_match`` is light (only depends on the asset bank), so it is exported
eagerly. ``retrievers`` pulls the heavy encoder substrate, so it is exported
lazily to keep the name-match read path importable without those deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memstrata.skills.memory_retrieval.name_match import (
    NameHit,
    NameMatchCache,
    build_name_cache,
    match_cache,
    name_hits,
    name_match,
    unique_asset_ids,
)

if TYPE_CHECKING:
    from memstrata.skills.memory_retrieval.retrievers import (
        MemoryRetrievalStore,
        RetrievedRef,
        Retriever,
        RetrieverConfig,
        build_retriever,
    )

_LAZY_FROM_RETRIEVERS = frozenset(
    {"MemoryRetrievalStore", "RetrievedRef", "Retriever", "RetrieverConfig", "build_retriever"}
)


def __getattr__(name: str):
    if name in _LAZY_FROM_RETRIEVERS:
        from memstrata.skills.memory_retrieval import retrievers

        return getattr(retrievers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemoryRetrievalStore",
    "RetrievedRef",
    "Retriever",
    "RetrieverConfig",
    "build_retriever",
    "NameHit",
    "NameMatchCache",
    "build_name_cache",
    "match_cache",
    "name_hits",
    "name_match",
    "unique_asset_ids",
]
