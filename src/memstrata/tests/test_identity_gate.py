"""VLM-first write-path identity gate (paper §4.5).

Covers the χ-band routing in ``MemoryUpdater._reconcile_identity``:

* inactive (Null judge) → decision is bit-for-bit the old χ≥β_τ rule;
* encoder short-circuit / clear-reject skip the VLM entirely;
* gray-zone same/different is decided by the VLM verdict;
* the per-model θ gate and abstain both fall back to the encoder threshold;
* strong-blur crops in the gray zone are deferred, never hard-merged.

The judge is a call-counting stub so the tests need no model server, and
``identity_score`` is monkeypatched so each case lands in an exact χ band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
)
from memstrata.encoders import HashEmbedding
from memstrata.mllm.identity_judge import IdentityVerdict, NullIdentityJudge
from memstrata.skills.decomposition import SOURCE_DISCOVERED, Observation
from memstrata.skills.memory_update.curator import MemoryPolicy, MemoryUpdater

_CHAR = AssetType.CHARACTER
_BETA = 0.50
_SHORT = 0.15  # β + short → merge without VLM
_GRAY = 0.10  # β - gray  → new without VLM


class SpyJudge:
    """Records calls and returns a fixed verdict."""

    def __init__(self, same: bool | None, confidence: float = 0.95) -> None:
        self._verdict = IdentityVerdict(same=same, confidence=confidence, source="spy")
        self.calls = 0
        self.last_n_refs = 0

    def judge(self, crop, references, *, kind="", name_a="", name_b=""):  # noqa: ANN001
        self.calls += 1
        self.last_n_refs = 1 if isinstance(references, str) else len(references)
        return self._verdict


def _policy(**over) -> MemoryPolicy:
    base = dict(
        name="idgate",
        reconcile_threshold=_BETA,
        identity_vlm_enabled=True,
        identity_shortcircuit_margin=_SHORT,
        identity_gray_margin=_GRAY,
        identity_vlm_theta=0.90,
        identity_blur_defer=False,
    )
    base.update(over)
    return MemoryPolicy(**base)


def _sharp_crop(path: Path) -> str:
    """A high-sharpness crop (random noise → large Laplacian variance)."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


def _flat_crop(path: Path) -> str:
    """A solid-color crop → Laplacian variance 0 → below any blur floor."""
    Image.new("RGB", (48, 48), (120, 120, 120)).save(path)
    return str(path)


def _bank_with_candidate(ref_crop: str) -> tuple[AssetBank, MemoryUpdater, Asset]:
    bank = AssetBank()
    asset = Asset("char_hero", _CHAR, "Hero", LifecycleStatus.REUSABLE)
    asset.representations.append(
        AssetRepresentation(
            representation_id="char_hero@s000",
            asset_id="char_hero",
            object_uri=ref_crop,
            origin_segment_id=0,
        )
    )
    bank.add_asset(asset)
    return bank, None, asset  # updater built per-test with the right policy/judge


def _build(bank: AssetBank, policy: MemoryPolicy, judge) -> MemoryUpdater:  # noqa: ANN001
    return MemoryUpdater(bank, HashEmbedding(), policy=policy, identity_judge=judge)


def _obs(image_path: str, chi_target: float, updater: MemoryUpdater) -> Observation:
    obs = Observation(
        observation_id="o_disc",
        kind=_CHAR,
        name="",
        image_path=image_path,
        description="a bearded keeper",
        source=SOURCE_DISCOVERED,
    )
    updater.identity_score = lambda o, a, _c=chi_target: (_c, _c, _c)  # type: ignore[assignment]
    return obs


def test_inactive_judge_reproduces_encoder_rule(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    bank, _, _ = _bank_with_candidate(ref)
    # Null judge → gate inert even though the policy enables VLM.
    upd = _build(bank, _policy(), NullIdentityJudge())
    assert upd._identity_judge_active is False

    matched, meta = upd._reconcile_identity(_obs(ref, 0.60, upd))
    assert matched is not None and meta["decision"] == "merged" and meta["gate"] == "encoder"

    matched, meta = upd._reconcile_identity(_obs(ref, 0.40, upd))
    assert matched is None and meta["decision"] == "new_asset" and meta["gate"] == "encoder"


def test_encoder_shortcircuit_skips_vlm(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    bank, _, _ = _bank_with_candidate(ref)
    spy = SpyJudge(same=False)  # would say "different" if consulted
    upd = _build(bank, _policy(), spy)

    matched, meta = upd._reconcile_identity(_obs(ref, 0.70, upd))  # ≥ 0.50+0.15
    assert matched is not None and meta["gate"] == "encoder_shortcircuit"
    assert spy.calls == 0


def test_clear_reject_skips_vlm(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    bank, _, _ = _bank_with_candidate(ref)
    spy = SpyJudge(same=True)  # would say "same" if consulted
    upd = _build(bank, _policy(), spy)

    matched, meta = upd._reconcile_identity(_obs(ref, 0.35, upd))  # ≤ 0.50-0.10
    assert matched is None and meta["gate"] == "encoder_reject"
    assert spy.calls == 0


def test_gray_zone_vlm_same_merges(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    bank, _, _ = _bank_with_candidate(ref)
    spy = SpyJudge(same=True, confidence=0.95)
    upd = _build(bank, _policy(), spy)

    matched, meta = upd._reconcile_identity(_obs(ref, 0.55, upd))
    assert matched is not None and meta["gate"] == "vlm" and meta["decision"] == "merged"
    assert spy.calls == 1


def test_gray_zone_vlm_different_creates_new(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    bank, _, _ = _bank_with_candidate(ref)
    spy = SpyJudge(same=False, confidence=0.95)
    upd = _build(bank, _policy(), spy)

    matched, meta = upd._reconcile_identity(_obs(ref, 0.55, upd))
    assert matched is None and meta["gate"] == "vlm" and meta["decision"] == "new_asset"
    assert spy.calls == 1


def test_theta_gate_blocks_low_confidence_merge(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    bank, _, _ = _bank_with_candidate(ref)
    spy = SpyJudge(same=True, confidence=0.80)  # below θ=0.90
    upd = _build(bank, _policy(), spy)

    matched, meta = upd._reconcile_identity(_obs(ref, 0.55, upd))
    # Falls back to encoder threshold: χ=0.55 ≥ β=0.50 → merge, but flagged as fallback.
    assert matched is not None and meta["gate"] == "vlm_abstain_fallback"
    assert meta["vlm_same"] is True


def test_abstain_falls_back_to_encoder(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    bank, _, _ = _bank_with_candidate(ref)
    spy = SpyJudge(same=None)  # abstain
    upd = _build(bank, _policy(), spy)

    matched, meta = upd._reconcile_identity(_obs(ref, 0.45, upd))
    # χ=0.45 < β=0.50 → encoder fallback says new_asset.
    assert matched is None and meta["gate"] == "vlm_abstain_fallback"


def test_gray_zone_sends_multiple_references_in_one_call(tmp_path):
    refs = [_sharp_crop(tmp_path / f"ref{i}.png") for i in range(3)]
    bank = AssetBank()
    asset = Asset("char_hero", _CHAR, "Hero", LifecycleStatus.REUSABLE)
    for i, ref in enumerate(refs):
        asset.representations.append(
            AssetRepresentation(
                representation_id=f"char_hero@s{i:03d}",
                asset_id="char_hero",
                object_uri=ref,
                origin_segment_id=i,
            )
        )
    bank.add_asset(asset)
    spy = SpyJudge(same=True, confidence=0.95)
    upd = _build(bank, _policy(identity_max_references=4), spy)

    query = _sharp_crop(tmp_path / "query.png")
    matched, meta = upd._reconcile_identity(_obs(query, 0.55, upd))
    assert matched is not None and meta["gate"] == "vlm"
    assert spy.calls == 1  # still ONE call...
    assert spy.last_n_refs == 3 and meta["vlm_n_refs"] == 3  # ...carrying all 3 references


def test_strong_blur_is_deferred(tmp_path):
    ref = _sharp_crop(tmp_path / "ref.png")
    blurry = _flat_crop(tmp_path / "blurry.png")  # sharpness 0 < floor
    bank, _, _ = _bank_with_candidate(ref)
    spy = SpyJudge(same=True, confidence=0.99)
    upd = _build(bank, _policy(identity_blur_defer=True, identity_blur_min_sharpness=40.0), spy)

    matched, meta = upd._reconcile_identity(_obs(blurry, 0.55, upd))
    assert matched is None and meta["gate"] == "deferred_blur"
    assert meta["sharpness"] < 40.0
    assert spy.calls == 0  # never hard-judged under strong blur
