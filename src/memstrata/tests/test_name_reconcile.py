"""Fresh-context name reconciliation on the VLM write-path decomposer.

The decomposer's own VLM may translate a prompt-named entity ('紫色小鸟' -> 'purple bird'),
which breaks the deterministic read-side name recall. ``_reconcile_names`` runs a SECOND,
fresh model call (not a self-review) only on labels the matcher can't find in the prompt,
and re-binds them to the prompt's verbatim wording — but only after re-verifying the name is
actually in the prompt. These tests pin that behaviour with a scripted fake runner.
"""

from __future__ import annotations

from typing import Any

from memstrata.bank import AssetType
from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer


class _FakeRunner:
    """Returns queued responses per ``run`` call and records each call."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def run(self, role_key: str, *, instruction: str, images=None, schema=None) -> Any:
        self.calls.append(
            {"role": role_key, "instruction": instruction, "images": images, "schema": schema}
        )
        return self._responses.pop(0)


_FRAMES = ["/tmp/frame_0.png"]


def test_drifted_label_is_rebound_via_fresh_context_call() -> None:
    prompt = "开阔草地上，紫色小鸟站在树枝上鸣叫。"
    runner = _FakeRunner(
        [
            {
                "entities": [
                    {"kind": "character", "label": "purple bird", "description": "round purple bird"},
                    {"kind": "location", "label": "开阔草地", "description": "sunny meadow"},
                ]
            },
            {"resolved": [{"index": 0, "in_prompt": True, "prompt_name": "紫色小鸟"}]},
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)

    assert len(runner.calls) == 2  # decompose + one fresh reconciliation call
    names = {e.name for e in ents}
    assert names == {"紫色小鸟", "开阔草地"}  # drifted English label snapped to prompt wording
    # The reconciliation call only carries the unmatched label, not the already-matched one.
    recon = runner.calls[1]["instruction"]
    assert "purple bird" in recon
    assert "0. label='purple bird'" in recon


def test_all_labels_already_in_prompt_skips_second_call() -> None:
    prompt = "大兔子从兔子洞穴爬出，走向开阔草地。"
    runner = _FakeRunner(
        [
            {
                "entities": [
                    {"kind": "character", "label": "大兔子", "description": "big white rabbit"},
                    {"kind": "location", "label": "兔子洞穴", "description": "dark burrow"},
                ]
            }
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)

    assert len(runner.calls) == 1  # nothing drifted -> no auditor call, no extra cost
    assert {e.name for e in ents} == {"大兔子", "兔子洞穴"}


def test_genuine_discovery_is_kept_as_discovery() -> None:
    # An entity truly absent from the prompt: the auditor says in_prompt=false, and its
    # descriptive label must survive untouched (it stays a self-discovery, not forced to a hit).
    prompt = "紫色小鸟站在树枝上。"
    runner = _FakeRunner(
        [
            {
                "entities": [
                    {"kind": "character", "label": "紫色小鸟", "description": "purple bird"},
                    {"kind": "prop", "label": "old brass telescope", "description": "brass scope"},
                ]
            },
            # only 'old brass telescope' is unmatched; auditor confirms it is genuinely new.
            {"resolved": [{"index": 1, "in_prompt": False, "prompt_name": "old brass telescope"}]},
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)

    assert len(runner.calls) == 2
    names = {e.name for e in ents}
    assert names == {"紫色小鸟", "old brass telescope"}  # discovery kept, matched untouched
    # Only the unmatched label is listed for the auditor; the matched one is never sent.
    recon = runner.calls[1]["instruction"]
    assert "label='old brass telescope'" in recon
    assert "label='紫色小鸟'" not in recon


def test_hallucinated_prompt_name_is_rejected() -> None:
    prompt = "开阔草地上有一只鸟。"  # names no specific bird
    runner = _FakeRunner(
        [
            {"entities": [{"kind": "character", "label": "purple bird", "description": "a bird"}]},
            # auditor claims a name that is NOT actually in the prompt.
            {"resolved": [{"index": 0, "in_prompt": True, "prompt_name": "紫色小鸟"}]},
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)

    assert len(runner.calls) == 2
    assert ents[0].name == "purple bird"  # unverifiable rename rejected, original kept


def test_reconcile_can_be_disabled() -> None:
    prompt = "紫色小鸟在飞。"
    runner = _FakeRunner(
        [{"entities": [{"kind": "character", "label": "purple bird", "description": "a bird"}]}]
    )
    ents = VlmEntityDecomposer(runner=runner, reconcile_names=False).propose(
        frames=_FRAMES, prompt=prompt
    )

    assert len(runner.calls) == 1
    assert ents[0].name == "purple bird"


def test_entity_kinds_preserved_after_reconcile() -> None:
    prompt = "紫色小鸟站在树枝上。"
    runner = _FakeRunner(
        [
            {"entities": [{"kind": "character", "label": "purple bird", "description": "a bird"}]},
            {"resolved": [{"index": 0, "in_prompt": True, "prompt_name": "紫色小鸟"}]},
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)
    assert ents[0].kind is AssetType.CHARACTER
    assert ents[0].name == "紫色小鸟"
