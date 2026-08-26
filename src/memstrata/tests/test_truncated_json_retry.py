"""A JSON role reply cut off by ``max_tokens`` must be re-asked with a wider budget.

Greedy decoding makes a truncated reply perfectly reproducible, so the plain retry the producer
already had could never clear it: story 0026 failed at the same character of the same shot on every
one of ~420 attempts, holding a card each time and leaving a permanent hole in an otherwise complete
114-shot sample. Only a retry that changes the token budget can end that loop.
"""

from __future__ import annotations

import json

import pytest

from memstrata.mllm.roles import ROLE_REGISTRY
from memstrata.mllm.runner import MllmRoleRunner


class _TruncateOnce:
    """First reply is cut mid-string, as vLLM returns when the reply hits ``max_tokens``."""

    def __init__(self, full: dict) -> None:
        self.full = full
        self.budgets: list[int] = []

    def chat(self, *, model, messages, sampling, schema, timeout):  # noqa: ANN001, ARG002
        self.budgets.append(sampling.max_tokens)
        body = json.dumps(self.full)
        return body if len(self.budgets) > 1 else body[: len(body) // 2]


def _json_role() -> str:
    for key, spec in ROLE_REGISTRY.items():
        if spec.sampling.response_format == "json_schema" and spec.schema_fields:
            return key
    pytest.skip("no JSON role registered")
    raise AssertionError


def test_truncated_reply_is_reasked_with_a_wider_budget() -> None:
    key = _json_role()
    role = ROLE_REGISTRY[key]
    full = {f: [] for f in role.schema_fields}
    transport = _TruncateOnce(full)
    runner = MllmRoleRunner(text_transport=transport, vision_transport=transport)

    assert runner.run(key, instruction="anything") == full
    assert len(transport.budgets) == 2, "a truncated reply must be re-asked exactly once"
    assert transport.budgets[1] > transport.budgets[0], (
        "the retry must widen the token budget; an identical re-ask reproduces the same "
        "truncation under greedy decoding"
    )


def test_structured_roles_budget_clears_a_busy_scene() -> None:
    """The retry is the safety net; the ceiling itself must clear a realistic worst case.

    Probed on the live Qwen3.5-9B endpoint, a 15-entity layout plan costs 784 completion tokens,
    and the Track B stories that lost shots were composing scenes roughly twice that busy. A
    ceiling that only fits the average scene turns every crowded shot into a coin flip against the
    retry, so pin enough headroom that truncation stays exceptional rather than routine.
    """
    tight = {k: s.sampling.max_tokens for k, s in ROLE_REGISTRY.items()
             if s.sampling.response_format == "json_schema" and s.sampling.max_tokens < 2048}
    assert not tight, (
        f"structured roles must leave room for a busy scene (>=2048 tokens); too tight: {tight}"
    )


def test_empty_reply_still_raises() -> None:
    """An empty body is a dead transport, not a budget problem, and must not be retried."""

    class _Empty:
        def __init__(self) -> None:
            self.n = 0

        def chat(self, *, model, messages, sampling, schema, timeout):  # noqa: ANN001, ARG002
            self.n += 1
            return "   "

    key = _json_role()
    transport = _Empty()
    runner = MllmRoleRunner(text_transport=transport, vision_transport=transport)
    with pytest.raises(json.JSONDecodeError):
        runner.run(key, instruction="anything")
    assert transport.n == 1, "an empty reply must not burn a second call"
