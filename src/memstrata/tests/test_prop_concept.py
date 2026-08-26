"""Prop crop acquisition uses the decomposer's specific English ``category`` as the SAM3
concept, with the generic kind concepts kept only as fallback.

A live probe showed SAM3 concept="object" yields zero QA-passing prop candidates while a
specific noun ("red apple", "acorn", "vine rope") yields several. These tests pin the two
model-free pieces of that fix: (1) call1 parses ``category`` onto the NamedEntity, and
(2) the orchestrator resolves concepts as specific-first, kind-fallback, de-duplicated.
"""

from __future__ import annotations

from memstrata.bank import AssetType
from memstrata.skills.crop_acquisition.orchestrator import (
    CHARACTER_CONCEPTS,
    PROP_CONCEPTS,
    _resolve_concepts,
)
from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer


class _FakeRunner:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def run(self, role_key, *, instruction, images=None, schema=None):
        self.calls.append({"role": role_key, "instruction": instruction})
        return self._responses.pop(0)


def test_category_is_parsed_onto_named_entity() -> None:
    runner = _FakeRunner(
        [
            {
                "entities": [
                    {"kind": "prop", "label": "红色苹果", "category": "red apple",
                     "description": "a shiny red apple"},
                ]
            }
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=["/tmp/f.png"], prompt="红色苹果")
    assert ents[0].kind is AssetType.PROP
    assert ents[0].name == "红色苹果"
    assert ents[0].category == "red apple"


def test_missing_category_defaults_empty() -> None:
    runner = _FakeRunner(
        [{"entities": [{"kind": "prop", "label": "坚果", "description": "a nut"}]}]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=["/tmp/f.png"], prompt="坚果")
    assert ents[0].category == ""


def test_resolve_concepts_specific_first_kind_fallback() -> None:
    # Specific noun leads; generic kind concept ("object") stays as a safety net.
    assert _resolve_concepts("prop", ("red apple",)) == ("red apple", *PROP_CONCEPTS)
    assert _resolve_concepts("character", ("rabbit",)) == ("rabbit", *CHARACTER_CONCEPTS)


def test_resolve_concepts_empty_category_is_kind_only() -> None:
    assert _resolve_concepts("prop", None) == PROP_CONCEPTS
    assert _resolve_concepts("prop", ("",)) == PROP_CONCEPTS


def test_resolve_concepts_dedups() -> None:
    # A category that duplicates the kind concept must not appear twice.
    assert _resolve_concepts("prop", ("object",)) == ("object",)
