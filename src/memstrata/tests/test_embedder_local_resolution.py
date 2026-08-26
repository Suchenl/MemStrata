"""Image embedders must resolve weights locally, never reach for a gated hub id.

Regression guard for a silent-degradation bug: the dinov3 branch handed the bare id
``facebook/dinov3-vitb16-pretrain-lvd1689m`` to transformers. That repo is gated, so on a machine
without credentials every call 403'd. The production loop caught the exception per segment and
skipped the segment, so a Track B run completed all 87 segments with an empty memory bank — it
"succeeded" while measuring nothing. Resolving locally (like every other provider) turns that into
either a working encoder or an immediate, explicit error.
"""

from __future__ import annotations

import pytest

from memstrata.encoders import base


@pytest.mark.parametrize("backend", ["dinov3", "dinov2"])
def test_dinov3_resolves_through_the_local_resolver(monkeypatch, backend: str) -> None:
    seen: dict[str, object] = {}

    def _fake_resolve(*, provider, model, weights, default_rel, env_var):
        seen.update(
            provider=provider, model=model, weights=weights,
            default_rel=default_rel, env_var=env_var,
        )
        return "/local/snapshot/dinov3"

    monkeypatch.setattr(base, "_resolve_local_model", _fake_resolve)
    captured: dict[str, object] = {}

    class _Stub:
        def __init__(self, model_id: str) -> None:
            captured["model_id"] = model_id

    monkeypatch.setattr("memstrata.encoders.ssl.dinov3.DinoV3Embedding", _Stub)

    base._construct_image_embedding(backend, None, None)

    assert seen["default_rel"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert seen["env_var"] == "MEMSTRATA_DINOV3_WEIGHTS"
    # The bare hub id must never reach the model constructor.
    assert captured["model_id"] == "/local/snapshot/dinov3"


def test_missing_weights_raise_instead_of_falling_back_to_the_hub() -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        base._resolve_local_model(
            provider="dinov3",
            model=None,
            weights="/nonexistent/dinov3",
            default_rel="facebook/dinov3-vitb16-pretrain-lvd1689m",
            env_var="MEMSTRATA_DINOV3_WEIGHTS",
        )
    # The message must name the knob, otherwise the next person re-debugs a 403.
    assert "MEMSTRATA_DINOV3_WEIGHTS" in str(excinfo.value)
