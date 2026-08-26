"""Fast/slow intent-path contract tests."""

from __future__ import annotations

from memstrata.bank import Asset, AssetBank, AssetType, LifecycleStatus
from memstrata.steps.intent import IntentInterpreter


class _Resolver:
    def __init__(self, output: list[str] | None = None, *, fail: bool = False) -> None:
        self.output = output or []
        self.fail = fail
        self.calls = 0

    def resolve(self, prompt: str, candidates: list[dict]) -> list[str]:
        del prompt, candidates
        self.calls += 1
        if self.fail:
            raise RuntimeError("resolver unavailable")
        return list(self.output)


def _bank() -> AssetBank:
    bank = AssetBank()
    bank.add_asset(
        Asset(
            asset_id="char_rabbit",
            kind=AssetType.CHARACTER,
            name="大兔子",
            status=LifecycleStatus.REUSABLE,
            description="体型巨大的白色兔子",
            metadata={"aliases": ["兔子主角"]},
        )
    )
    return bank


def test_fast_mode_never_calls_resolver_and_matches_alias() -> None:
    resolver = _Resolver(["char_rabbit"])
    request, model_calls = IntentInterpreter(
        _bank(),
        resolver=resolver,
        mode="fast",
    ).interpret("兔子主角走进森林", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_rabbit"]
    assert model_calls == 0
    assert resolver.calls == 0
    assert request.requested_mode == "fast"
    assert request.used_mode == "fast"
    assert request.fallback_reason == ""


def test_slow_mode_calls_resolver_and_filters_unknown_ids() -> None:
    resolver = _Resolver(["unknown", "char_rabbit", "char_rabbit"])
    request, model_calls = IntentInterpreter(
        _bank(),
        resolver=resolver,
        mode="slow",
    ).interpret("它重新出现了", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_rabbit"]
    assert model_calls == 1
    assert resolver.calls == 1
    assert request.used_mode == "slow"


def test_slow_mode_falls_back_to_fast_on_error() -> None:
    resolver = _Resolver(fail=True)
    request, model_calls = IntentInterpreter(
        _bank(),
        resolver=resolver,
        mode="slow",
    ).interpret("大兔子重新出现", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_rabbit"]
    assert model_calls == 1
    assert request.requested_mode == "slow"
    assert request.used_mode == "fast"
    assert request.fallback_reason == "resolver_error:RuntimeError"


def test_slow_on_miss_cascade_skips_resolver_when_fast_hits() -> None:
    # Fast name/alias match resolves the prompt -> no model call even with cascade armed.
    resolver = _Resolver(["char_rabbit"])
    request, model_calls = IntentInterpreter(
        _bank(),
        resolver=resolver,
        mode="fast",
        slow_on_miss=True,
    ).interpret("大兔子走进森林", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_rabbit"]
    assert model_calls == 0
    assert resolver.calls == 0
    assert request.intent_resolution_source == "name"


def test_slow_on_miss_cascade_fires_when_name_and_description_miss() -> None:
    # Cross-lingual reference that neither name/alias nor description overlap can bridge;
    # the cascade spends exactly one resolver call to recover the stored id.
    resolver = _Resolver(["char_rabbit"])
    request, model_calls = IntentInterpreter(
        _bank(),
        resolver=resolver,
        mode="fast",
        slow_on_miss=True,
    ).interpret("the giant white rabbit returns", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_rabbit"]
    assert model_calls == 1
    assert resolver.calls == 1
    assert request.intent_resolution_source == "mllm"
    assert request.used_mode == "slow"


def test_miss_without_cascade_stays_empty_and_model_free() -> None:
    resolver = _Resolver(["char_rabbit"])
    request, model_calls = IntentInterpreter(
        _bank(),
        resolver=resolver,
        mode="fast",  # slow_on_miss defaults False -> no cascade
    ).interpret("the giant white rabbit returns", segment_id=1)
    assert request.references == []
    assert model_calls == 0
    assert resolver.calls == 0
    # name + description both missed and no cascade → an empty selection is a genuine
    # miss, not "recency": the name-anchored FAST path never ranks by recency (that only
    # happens under disable_name_anchor), so labeling it "recency" overcounted misses.
    assert request.intent_resolution_source == "miss"


def test_read_budget_is_propagated_to_request() -> None:
    # Equal-budget read side: the adapter can let a matched asset contribute several stored
    # angles (a ceiling, not a fill target). The interpreter must forward the budget onto q_n.
    request, _ = IntentInterpreter(
        _bank(), mode="fast", max_reps_per_asset=3, context_rep_budget=12,
    ).interpret("兔子主角走进森林", segment_id=1)
    assert [ref.asset_id for ref in request.references] == ["char_rabbit"]
    assert request.max_reps_per_asset == 3
    assert request.context_rep_budget == 12


def test_read_budget_defaults_preserve_minimal_context() -> None:
    request, _ = IntentInterpreter(_bank(), mode="fast").interpret("兔子主角走进森林")
    assert request.max_reps_per_asset == 1
    assert request.context_rep_budget is None

