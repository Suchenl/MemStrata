"""Write-side naming: a first appearance must enter memory under the prompt's own name.

Regression guard for a measured failure. With naming absent, the only path into memory for an
entity the read side did not already select is discovery, which never infers names — so a Track B
run banked 23 entities over 21 segments, every one of them under a synthetic label
(`character_disc_c000_0`), and the name-authoritative read path returned an empty selection while
the bank was full. Descriptions were fine; the names were the problem.

The namer's requested/discovered split is the contract under test: a label verbatim in the prompt
is anchored by that name (retrievable next shot), anything merely visible stays unanchored so
identity is settled by visual reconciliation rather than a drifting descriptive label.
"""

from __future__ import annotations

from pathlib import Path

from memstrata.bank import AssetType
from memstrata.skills.decomposition.decomposer import (
    SOURCE_DISCOVERED,
    SOURCE_REQUESTED,
    NamedEntity,
    RoleAwareDecomposer,
)


class _Cropper:
    """Hands back a real file per entity so observation building can proceed."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.calls: list[str] = []

    def crop(self, segment_video: str, entity: NamedEntity, *, segment_id: int):
        del segment_video, segment_id
        self.calls.append(entity.name)
        out = self.tmp / f"{entity.kind.value}_{len(self.calls)}.png"
        out.write_bytes(b"crop")
        return str(out)


class _Namer:
    def __init__(self, entities: list[NamedEntity] | None = None, *, fail: bool = False) -> None:
        self.entities = entities or []
        self.fail = fail
        self.prompts: list[str] = []
        self.frames: list[list[str]] = []

    def propose(self, *, frames, prompt: str = "") -> list[NamedEntity]:
        self.prompts.append(prompt)
        self.frames.append(list(frames) if not isinstance(frames, str) else [frames])
        if self.fail:
            raise RuntimeError("namer endpoint down")
        return list(self.entities)


def _decomposer(tmp: Path, namer: _Namer | None, **kw) -> RoleAwareDecomposer:
    return RoleAwareDecomposer(
        cropper=_Cropper(tmp),
        entity_namer=namer,
        namer_frame_dir=tmp,
        **kw,
    )


def _fake_frames(monkeypatch, paths: list[str]) -> None:
    monkeypatch.setattr(
        "memstrata.lib.media.sample_video_frames",
        lambda *a, **k: list(paths),
    )


def test_prompt_named_entity_is_anchored_by_that_name(tmp_path, monkeypatch) -> None:
    _fake_frames(monkeypatch, [str(tmp_path / "f0.png")])
    namer = _Namer([
        NamedEntity(
            name="Elias", kind=AssetType.CHARACTER,
            description="elderly man with a white beard",
            entity_id="Elias",  # the namer sets this when the label is verbatim in the prompt
        )
    ])
    obs = _decomposer(tmp_path, namer).decompose(
        segment_id=3, named_entities=[], segment_video="seg.mp4",
        prompt="Elias climbs the lighthouse stairs",
    )
    assert [(o.name, o.entity_id, o.source) for o in obs] == [
        ("Elias", "Elias", SOURCE_REQUESTED)
    ]
    assert namer.prompts == ["Elias climbs the lighthouse stairs"]


def test_visible_only_entity_stays_unanchored_for_visual_reconciliation(tmp_path, monkeypatch) -> None:
    """An unanchored observation is what lets one entity under drifting labels collapse into
    one record; anchoring it by its descriptive label would fragment identity instead."""
    _fake_frames(monkeypatch, [str(tmp_path / "f0.png")])
    namer = _Namer([
        NamedEntity(name="a white seabird", kind=AssetType.CHARACTER, description="gull, wings out")
    ])
    obs = _decomposer(tmp_path, namer).decompose(
        segment_id=4, named_entities=[], segment_video="seg.mp4", prompt="Elias climbs the stairs",
    )
    assert [(o.entity_id, o.source) for o in obs] == [(None, SOURCE_DISCOVERED)]


def test_namer_description_survives_when_no_angle_description(tmp_path, monkeypatch) -> None:
    _fake_frames(monkeypatch, [str(tmp_path / "f0.png")])
    namer = _Namer([
        NamedEntity(name="brass lantern", kind=AssetType.PROP, description="dented brass lantern")
    ])
    obs = _decomposer(tmp_path, namer).decompose(
        segment_id=5, named_entities=[], segment_video="seg.mp4", prompt="he lifts the lantern",
    )
    assert obs[0].description == "dented brass lantern"


def test_read_side_selection_is_not_observed_twice(tmp_path, monkeypatch) -> None:
    _fake_frames(monkeypatch, [str(tmp_path / "f0.png")])
    namer = _Namer([
        NamedEntity(name="Elias", kind=AssetType.CHARACTER, entity_id="Elias"),
        NamedEntity(name="brass lantern", kind=AssetType.PROP, entity_id="brass lantern"),
    ])
    obs = _decomposer(tmp_path, namer).decompose(
        segment_id=6,
        named_entities=[NamedEntity(name="Elias", kind=AssetType.CHARACTER, entity_id="char_elias")],
        segment_video="seg.mp4",
        prompt="Elias lifts the brass lantern",
    )
    # Elias came from the read side; only the lantern is new.
    assert [o.name for o in obs] == ["Elias", "brass lantern"]
    assert [o.entity_id for o in obs] == ["char_elias", "brass lantern"]


def test_namer_failure_degrades_to_no_named_observations(tmp_path, monkeypatch) -> None:
    _fake_frames(monkeypatch, [str(tmp_path / "f0.png")])
    obs = _decomposer(tmp_path, _Namer(fail=True)).decompose(
        segment_id=7, named_entities=[], segment_video="seg.mp4", prompt="Elias climbs",
    )
    assert obs == []


def test_namer_is_skipped_without_frames_or_prompt(tmp_path, monkeypatch) -> None:
    namer = _Namer([NamedEntity(name="Elias", kind=AssetType.CHARACTER, entity_id="Elias")])
    dec = _decomposer(tmp_path, namer)

    _fake_frames(monkeypatch, [])
    assert dec.decompose(segment_id=8, named_entities=[], segment_video="seg.mp4",
                         prompt="Elias climbs") == []

    _fake_frames(monkeypatch, [str(tmp_path / "f0.png")])
    # No prompt => nothing to bind names to, so the namer must not be consulted at all.
    assert dec.decompose(segment_id=9, named_entities=[], segment_video="seg.mp4", prompt="") == []
    assert namer.prompts == []


def test_no_namer_keeps_the_historical_two_part_behaviour(tmp_path) -> None:
    obs = _decomposer(tmp_path, None).decompose(
        segment_id=10,
        named_entities=[NamedEntity(name="Ana", kind=AssetType.CHARACTER, entity_id="char_ana")],
        segment_video="seg.mp4",
        prompt="Ana walks the pier",
    )
    assert [(o.name, o.source) for o in obs] == [("Ana", SOURCE_REQUESTED)]
