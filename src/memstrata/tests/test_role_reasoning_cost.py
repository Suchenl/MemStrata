"""Reasoning is a capability to hand out per role, never a default the hot path can drift into.

Measured on the Track B Qwen3.5-9B deployment (see
`.agents/skills/research/infrastructure/mllm-integrity/models/qwen3.5-9b-vllm.md`), the reasoning
toggle is a ~50x latency multiplier on free text — 37 tokens / 1.3s off versus 2048 tokens / 67.2s
on — and is neutralized to ~13 tokens under `json_schema`, because guided decoding binds every token
to the grammar and leaves the deliberation nowhere to go.

That combination is a trap rather than a happy accident. Several hot-path roles are declared
``thinking=True`` and are affordable today only because their grammar suppresses it. Switching any
of them to free-text output would silently re-arm the flag and multiply a per-shot call by ~50 — at
~2800 shots a run, that does not finish. Pin the pairing so the change has to be deliberate.
"""

from __future__ import annotations

from memstrata.mllm.roles import ROLE_REGISTRY


def test_thinking_on_the_hot_path_is_grammar_bound() -> None:
    armed = {
        key: spec.sampling.response_format
        for key, spec in ROLE_REGISTRY.items()
        if spec.hot_path and spec.sampling.thinking
        and spec.sampling.response_format != "json_schema"
    }
    assert not armed, (
        "a hot-path role with thinking=True and free-text output pays the full reasoning latency "
        f"(~50x on this deployment) on every shot: {armed}. Either set thinking=False, or move the "
        "role off the hot path, and re-measure before assuming the cost is acceptable."
    )


def test_reasoning_is_off_by_default() -> None:
    """A new role must not inherit reasoning by omission; opting in has to be a written choice."""
    from memstrata.mllm.roles import Sampling

    assert Sampling().thinking is False
