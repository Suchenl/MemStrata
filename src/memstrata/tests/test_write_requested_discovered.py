"""Write-path requested/discovered anchoring + non-entity stoplist on the VLM decomposer.

Two fixes for bank fragmentation are pinned here:

* Requested vs discovered anchoring — an entity whose final label is verbatim in the prompt
  is *requested* and gets ``entity_id = name`` (a stable symbolic id so its reps group across
  segments). Anything else is *discovered* and left UNANCHORED (``entity_id is None``) so the
  curate step decides cross-segment identity by visual reconciliation (χ) instead of by its
  drifting descriptive label (which is what fragments one rabbit into 棕色兔子/棕色动物/…).
* Non-entity stoplist — backgrounds / end credits / on-screen text / the plain sky are never
  re-identifiable assets and are dropped before proposal, regardless of the model.
"""

from __future__ import annotations

from typing import Any

from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer


class _FakeRunner:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def run(self, role_key: str, *, instruction: str, images=None, schema=None) -> Any:
        self.calls.append({"role": role_key, "instruction": instruction})
        return self._responses.pop(0)


_FRAMES = ["/tmp/frame_0.png"]


def test_requested_label_is_anchored_discovered_is_unanchored() -> None:
    # '大兔子' is verbatim in the prompt (requested); the telescope is not (a genuine
    # discovery the auditor confirms in_prompt=false).
    prompt = "大兔子从洞穴里爬出来。"
    runner = _FakeRunner(
        [
            {
                "entities": [
                    {"kind": "character", "label": "大兔子", "category": "rabbit",
                     "description": "big rabbit"},
                    {"kind": "prop", "label": "old brass telescope", "category": "telescope",
                     "description": "brass scope"},
                ]
            },
            {"resolved": [{"index": 1, "in_prompt": False, "prompt_name": "old brass telescope"}]},
        ]
    )
    ents = {e.name: e for e in VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)}

    assert ents["大兔子"].entity_id == "大兔子"  # requested → name-anchored
    assert ents["old brass telescope"].entity_id is None  # discovered → visual reconcile


def test_drifted_label_becomes_requested_after_snap() -> None:
    # 'purple bird' drifts from '紫色小鸟'; after the auditor snaps it back it is verbatim in
    # the prompt, so it must be treated as REQUESTED (anchored), not discovered.
    prompt = "紫色小鸟站在树枝上。"
    runner = _FakeRunner(
        [
            {"entities": [{"kind": "character", "label": "purple bird", "category": "bird",
                           "description": "purple bird"}]},
            {"resolved": [{"index": 0, "in_prompt": True, "prompt_name": "紫色小鸟"}]},
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)
    assert len(ents) == 1
    assert ents[0].name == "紫色小鸟"
    assert ents[0].entity_id == "紫色小鸟"  # snapped → requested/anchored


def test_no_prompt_keeps_name_anchor() -> None:
    # Without a prompt we cannot tell requested from discovered → keep the historical anchor.
    runner = _FakeRunner(
        [{"entities": [{"kind": "prop", "label": "red apple", "category": "apple",
                        "description": "an apple"}]}]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt="")
    assert ents[0].entity_id == "red apple"


def test_non_entity_labels_are_dropped() -> None:
    prompt = "大兔子在开阔草地上。"
    runner = _FakeRunner(
        [
            {
                "entities": [
                    {"kind": "character", "label": "大兔子", "category": "rabbit", "description": "rabbit"},
                    {"kind": "location", "label": "森林背景", "category": "background", "description": "bg"},
                    {"kind": "prop", "label": "片尾曲", "category": "credits", "description": "end credits"},
                    {"kind": "location", "label": "the sky", "category": "sky", "description": "blue sky"},
                    {"kind": "prop", "label": "屏幕字幕", "category": "subtitle", "description": "text"},
                ]
            }
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)
    names = {e.name for e in ents}
    assert names == {"大兔子"}  # background / credits / sky / on-screen text all dropped


def test_noise_dropped_by_category_even_if_label_is_clean() -> None:
    # Label looks like a real entity, but the segmenter category marks it a background.
    prompt = "场景里有东西。"
    runner = _FakeRunner(
        [
            {
                "entities": [
                    {"kind": "location", "label": "远处山峦", "category": "backdrop", "description": "hills"},
                    {"kind": "prop", "label": "红色苹果", "category": "red apple", "description": "apple"},
                ]
            }
        ]
    )
    ents = VlmEntityDecomposer(runner=runner).propose(frames=_FRAMES, prompt=prompt)
    assert {e.name for e in ents} == {"红色苹果"}
