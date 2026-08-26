"""Write-side quality-gate enhancements (README ★ 写侧增强计划).

Assert-based and fully offline (Heuristic/Null classifiers + tiny generated PNGs):

- Part 1 deterministic overexposure gate (symmetric to the dark gate);
- Part 2 heavy occlusion downgrades identity_visible for admission (non-anchor, not rejected);
- Part 3 description upgrade fills an empty stable description but never overwrites one;
- Part 4 state-novelty re-ranking prefers an un-banked state as a tie-breaker on top of
  visual novelty, and is a no-op with no per-identity state history;
- Part 5 batch crop-attribute API returns one pack per item with matches_target set.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from memstrata.bank import Asset, AssetBank, AssetType, LifecycleStatus, SpatialAngle, StateAngle
from memstrata.encoders import HashEmbedding
from memstrata.lib.crop_quality import (
    is_dark_low_information,
    is_overexposed_low_information,
)
from memstrata.mllm.crop_attributes import (
    HeuristicCropAttributeClassifier,
    NullCropAttributeClassifier,
)
from memstrata.skills.crop_acquisition.crop_qa import audit_crop
from memstrata.skills.crop_acquisition.orchestrator import rank_acquisition_candidates
from memstrata.steps.curate import AssetCurator, EntityObservation
from memstrata.steps.decompose import Observation

_CHAR = AssetType.CHARACTER
_ANCHOR = "identity_anchor"


def _flat_png(directory: Path, name: str, value: int) -> str:
    path = directory / name
    Image.new("RGB", (48, 48), (value, value, value)).save(path)
    return str(path)


def _textured_png(directory: Path, name: str) -> str:
    """A mid-range, high-variance PNG: neither dark nor overexposed."""
    path = directory / name
    img = Image.new("RGB", (48, 48))
    img.putdata([((i * 7) % 256, (i * 13) % 256, (i * 29) % 256) for i in range(48 * 48)])
    img.save(path)
    return str(path)


# --- Part 1: deterministic overexposure gate ------------------------------------------

def test_overexposed_flat_crop_is_low_information() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        white = _flat_png(Path(tmp), "white.png", 250)  # near-white, flat
        assert is_overexposed_low_information(white) is True
        assert is_dark_low_information(white) is False


def test_normal_contrast_crop_passes_both_luminance_gates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = _textured_png(Path(tmp), "textured.png")
        assert is_overexposed_low_information(crop) is False
        assert is_dark_low_information(crop) is False


def test_dark_flat_crop_still_dark_not_overexposed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dark = _flat_png(Path(tmp), "dark.png", 5)  # near-black, flat
        assert is_dark_low_information(dark) is True
        assert is_overexposed_low_information(dark) is False


def test_audit_crop_flags_overexposure_for_non_location() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        white = Path(_flat_png(Path(tmp), "white.png", 250))
        qa = audit_crop(crop=white, bbox_norm=[100, 100, 500, 500], kind="character")
        assert "overexposed_low_information" in qa.reasons
        # A textured crop is not flagged as over/under exposed.
        textured = Path(_textured_png(Path(tmp), "textured.png"))
        qa2 = audit_crop(crop=textured, bbox_norm=[100, 100, 500, 500], kind="character")
        assert "overexposed_low_information" not in qa2.reasons
        assert "dark_low_information" not in qa2.reasons


def test_overexposed_crop_is_not_banked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        white = _flat_png(Path(tmp), "white.png", 250)
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), attributes_when_angles_known=False)
        cur.ingest_observation(
            EntityObservation(
                "o1", _CHAR, "Hero", white,
                spatial_angle=SpatialAngle.FRONT, state_angle=StateAngle.DEFAULT,
            ),
            segment_id=0,
        )
        asset = bank.find_by_name("Hero", kind=_CHAR)
        assert asset is not None and len(asset.representations) == 0


# --- Part 2: heavy occlusion → identity_visible False for admission -------------------

def test_heavy_occlusion_downgrades_to_non_anchor_but_is_kept() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Bright + front + heavy occlusion (filename drives the heuristic classifier).
        crop = _textured_png(Path(tmp), "hero_front_default_heavy.png")
        bank = AssetBank()
        cur = AssetCurator(
            bank, HashEmbedding(),
            crop_attribute_classifier=HeuristicCropAttributeClassifier(),
        )
        cur.ingest_observation(EntityObservation("o1", _CHAR, "Hero", crop), segment_id=0)
        asset = bank.find_by_name("Hero", kind=_CHAR)
        assert asset is not None and len(asset.representations) == 1  # kept for diversity
        rep = asset.representations[0]
        assert rep.annotations["crop_attributes"]["occlusion"] == "heavy"
        # Not hard-rejected, but downgraded off the identity anchor.
        assert rep.annotations["identity_anchor_eligible"] is False
        assert _ANCHOR not in rep.reference_aspects
        assert _ANCHOR in rep.excluded_aspects


def test_no_occlusion_crop_stays_an_anchor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crop = _textured_png(Path(tmp), "hero_front_default_none.png")
        bank = AssetBank()
        cur = AssetCurator(
            bank, HashEmbedding(),
            crop_attribute_classifier=HeuristicCropAttributeClassifier(),
        )
        cur.ingest_observation(EntityObservation("o1", _CHAR, "Hero", crop), segment_id=0)
        rep = bank.find_by_name("Hero", kind=_CHAR).representations[0]
        assert rep.annotations["crop_attributes"]["occlusion"] == "none"
        assert rep.annotations.get("identity_anchor_eligible", True) is True
        assert _ANCHOR in rep.reference_aspects


# --- Part 3: description upgrade ------------------------------------------------------

def test_empty_description_is_filled_from_observation() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)
    bank.add_asset(Asset("char_hero", _CHAR, "Hero", LifecycleStatus.REUSABLE))
    cur.curate_observations(
        [Observation(observation_id="o0", kind=_CHAR, name="Hero", entity_id="char_hero",
                     image_path="/nonexistent/o0.png", description="a bearded keeper")],
        segment_id=0,
    )
    asset = bank.get_asset("char_hero")
    assert asset.description == "a bearded keeper"
    assert asset.metadata["description"] == "a bearded keeper"


def test_non_empty_description_is_not_overwritten() -> None:
    bank = AssetBank()
    cur = AssetCurator(bank, HashEmbedding(), embed_on_ingest=False, dark_gate=False,
                       attributes_when_angles_known=False)
    bank.add_asset(Asset("char_hero", _CHAR, "Hero", LifecycleStatus.REUSABLE,
                         description="an original stable description"))
    cur.curate_observations(
        [Observation(observation_id="o0", kind=_CHAR, name="Hero", entity_id="char_hero",
                     image_path="/nonexistent/o0.png", description="a completely different one")],
        segment_id=0,
    )
    asset = bank.get_asset("char_hero")
    assert asset.description == "an original stable description"


def test_classifier_description_fills_empty_asset() -> None:
    """An empty-description asset gets its d from the classifier's observation description.

    The observation itself carries no description; the attribute classifier produces one,
    which the write-path adopts (new code path) only because the asset had none yet.
    """
    with tempfile.TemporaryDirectory() as tmp:
        crop = _textured_png(Path(tmp), "hero_front_default.png")
        bank = AssetBank()
        cur = AssetCurator(
            bank, HashEmbedding(),
            crop_attribute_classifier=HeuristicCropAttributeClassifier(),
        )
        cur.curate_observations(
            [Observation(observation_id="o1", kind=_CHAR, name="Hero",
                         entity_id="char_hero", image_path=crop)],  # no description
            segment_id=0,
        )
        asset = bank.get_asset("char_hero")
        assert asset.description  # filled from the classifier's observation_description
        assert asset.metadata.get("description")


# --- Part 4: state-novelty re-ranking -------------------------------------------------

def test_state_novelty_prefers_unbanked_state_at_equal_visual_novelty() -> None:
    banked = {"default"}
    same_state = {"identity_sim": 0.9, "novelty_score": 0.5, "state_angle": "default", "score": 1.0}
    new_state = {"identity_sim": 0.9, "novelty_score": 0.5, "state_angle": "changed", "score": 1.0}
    ranked = rank_acquisition_candidates([same_state, new_state], banked_states=banked)
    assert ranked[0] is new_state  # un-banked state wins the tie


def test_state_novelty_never_overrides_visual_novelty() -> None:
    banked = {"default"}
    high_visual_banked = {"identity_sim": 0.9, "novelty_score": 0.9, "state_angle": "default"}
    low_visual_novel = {"identity_sim": 0.9, "novelty_score": 0.1, "state_angle": "changed"}
    ranked = rank_acquisition_candidates([low_visual_novel, high_visual_banked], banked_states=banked)
    assert ranked[0] is high_visual_banked  # visual novelty still dominates


def test_state_novelty_is_a_noop_without_history() -> None:
    same_state = {"identity_sim": 0.9, "novelty_score": 0.5, "state_angle": "default", "score": 1.0}
    new_state = {"identity_sim": 0.9, "novelty_score": 0.5, "state_angle": "changed", "score": 1.0}
    # No banked-state history → stable order preserved (no reordering).
    ranked = rank_acquisition_candidates([same_state, new_state], banked_states=None)
    assert ranked[0] is same_state


# --- Part 5: batch crop-attribute API -------------------------------------------------

def test_heuristic_batch_returns_one_pack_per_item_with_matches_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        items = [
            {"image_path": str(root / "hero_red_hat_front.png"), "kind": "character", "name": "Hero"},
            {"image_path": str(root / "villain_blue_scarf.png"), "kind": "character", "name": "Villain"},
        ]
        clf = HeuristicCropAttributeClassifier()
        packs = clf.classify_batch(items, target_descriptions=["red hat", "red hat"])
        assert len(packs) == 2
        # Target words present in the stem → match; absent → no match.
        assert packs[0].extra.get("matches_target") is True
        assert packs[1].extra.get("matches_target") is False


def test_batch_without_targets_sets_no_matches_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        items = [{"image_path": str(root / "hero_front_default.png"), "kind": "character", "name": "Hero"}]
        packs = HeuristicCropAttributeClassifier().classify_batch(items)
        assert len(packs) == 1
        assert "matches_target" not in packs[0].extra
        # Per-item target of None also leaves matches_target unset.
        packs2 = HeuristicCropAttributeClassifier().classify_batch(items, target_descriptions=[None])
        assert "matches_target" not in packs2[0].extra


def test_null_batch_returns_one_pack_per_item() -> None:
    items = [
        {"image_path": "/nonexistent/a.png", "segment_id": 0},
        {"image_path": "/nonexistent/b.png", "segment_id": 1},
    ]
    packs = NullCropAttributeClassifier().classify_batch(items)
    assert len(packs) == 2
    assert [p.source for p in packs] == ["null", "null"]
    assert packs[0].segment_id == 0 and packs[1].segment_id == 1


# --- Part 6: curator uses ONE batched classify per segment -----------------------------


class _CountingCropAttributeClassifier:
    """Wraps an inner classifier and counts per-crop classify vs batched classify_batch."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.n_single = 0
        self.n_batch = 0

    def classify(self, image_path, *, kind="", name="", segment_id=None,
                 frame_index=None, seconds=None):
        self.n_single += 1
        return self.inner.classify(
            image_path, kind=kind, name=name, segment_id=segment_id,
            frame_index=frame_index, seconds=seconds,
        )

    def classify_batch(self, items, *, target_descriptions=None):
        self.n_batch += 1
        return self.inner.classify_batch(items, target_descriptions=target_descriptions)


def test_multi_crop_segment_triggers_one_batch_and_zero_per_crop_classify() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crops = {
            "char_hero": _textured_png(root, "hero_front_default_none.png"),
            "char_villain": _textured_png(root, "villain_side_changed_partial.png"),
            "prop_lantern": _textured_png(root, "lantern_front_default_close_up.png"),
        }
        inner = HeuristicCropAttributeClassifier()
        spy = _CountingCropAttributeClassifier(inner)
        bank = AssetBank()
        cur = AssetCurator(bank, HashEmbedding(), crop_attribute_classifier=spy)
        observations = [
            Observation(observation_id="o_hero", kind=_CHAR, name="Hero",
                        entity_id="char_hero", image_path=crops["char_hero"]),
            Observation(observation_id="o_villain", kind=_CHAR, name="Villain",
                        entity_id="char_villain", image_path=crops["char_villain"]),
            Observation(observation_id="o_lantern", kind=AssetType.PROP, name="Lantern",
                        entity_id="prop_lantern", image_path=crops["prop_lantern"]),
        ]
        cur.curate_observations(observations, segment_id=0)

        # Exactly ONE batched call and ZERO per-crop classify() for the whole segment.
        assert spy.n_batch == 1
        assert spy.n_single == 0

        # Packs/stratification match the per-crop path: each rep's crop_attributes equal
        # what the underlying classifier would produce for that crop on its own.
        for asset_id, path in crops.items():
            asset = bank.get_asset(asset_id)
            assert asset is not None and len(asset.representations) == 1
            rep = asset.representations[0]
            expected = inner.classify(
                path, kind=asset.kind.value, name=asset.name, segment_id=0
            ).to_dict()
            assert rep.annotations["crop_attributes"] == expected


class _BatchDisabledClassifier(HeuristicCropAttributeClassifier):
    """Heuristic classifier whose batch call returns the wrong count, forcing the
    curator to fall back to the per-crop ``classify`` path (behaviour-preservation ref)."""

    def classify_batch(self, items, *, target_descriptions=None):  # noqa: D401
        return []


def test_batched_and_per_crop_paths_produce_identical_reps() -> None:
    """The batched cache and the per-crop fallback must yield byte-identical reps."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = [
            _textured_png(root, "hero_front_default_none.png"),
            _textured_png(root, "villain_back_damaged_heavy.png"),
        ]
        names = [("char_hero", "Hero", _CHAR), ("char_villain", "Villain", _CHAR)]

        def _run(classifier) -> dict[str, dict]:
            bank = AssetBank()
            cur = AssetCurator(bank, HashEmbedding(), crop_attribute_classifier=classifier)
            cur.curate_observations(
                [
                    Observation(observation_id=f"o_{aid}", kind=kind, name=nm,
                                entity_id=aid, image_path=p)
                    for (aid, nm, kind), p in zip(names, paths)
                ],
                segment_id=0,
            )
            return {
                aid: bank.get_asset(aid).representations[0].annotations["crop_attributes"]
                for aid, _nm, _k in names
            }

        batched = _run(HeuristicCropAttributeClassifier())      # ONE classify_batch
        per_crop = _run(_BatchDisabledClassifier())             # per-crop classify()
        assert batched == per_crop


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"{fn.__name__} passed")
    print(f"all {len(fns)} write-side enhancement tests passed")
    sys.exit(0)
