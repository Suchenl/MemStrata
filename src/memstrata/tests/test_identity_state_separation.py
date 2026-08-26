"""A transient qualifier is a STATE of one identity, never a second identity.

Measured on Track B story 0001: the bank held 'young Mara' and 'Mara' as two characters, and
'indigo Petrel' and 'Petrel' as two props. The namer had no field to put a qualifier in and was
told to copy the prompt verbatim, so it folded the condition into the name and identity split
once per appearance.

Identity now lives in ``name`` and the condition in ``state_modifier``, which also marks the
representation non-default — the signal the curator's cohesion gate needs to ADMIT a
legitimately different-looking appearance of a known entity rather than reject it.
"""

from __future__ import annotations

from memstrata.bank import AssetType, StateAngle
from memstrata.skills.decomposition.decomposer import NamedEntity, RoleAwareDecomposer
from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer, _is_default_state


class _Transport:
    """Returns one canned decomposition, then echoes reconcile requests as genuine discoveries."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def run(self, role: str, *, instruction: str, images, schema):
        del role, images, schema
        if "naming auditor" in instruction:
            return {"resolved": []}
        return {"entities": self.rows}


def _propose(rows: list[dict], prompt: str) -> list[NamedEntity]:
    namer = VlmEntityDecomposer(runner=_Transport(rows))
    return namer.propose(frames=["f0.png"], prompt=prompt)


def _row(**kw) -> dict:
    row = {"kind": "character", "label": "Mara", "state_modifier": "", "category": "woman",
           "description": "woman in a grey coat"}
    row.update(kw)
    return row


def test_qualifier_goes_to_state_not_into_the_name() -> None:
    ents = _propose(
        [_row(label="Mara", state_modifier="young")],
        "young Mara runs along the pier",
    )
    assert [(e.name, e.state_modifier, e.state_angle) for e in ents] == [
        ("Mara", "young", StateAngle.CHANGED)
    ]
    # Anchored by the bare identity, so the next shot saying just "Mara" resolves to it.
    assert ents[0].entity_id == "Mara"


def test_default_condition_leaves_the_state_unset() -> None:
    ents = _propose([_row(label="Mara", state_modifier="")], "Mara runs along the pier")
    assert (ents[0].state_modifier, ents[0].state_angle) == ("", StateAngle.UNKNOWN)


def test_a_model_that_writes_none_for_no_qualifier_is_not_treated_as_a_state() -> None:
    """Asked for a qualifier and given none, models fill the field with 'none' / 'normal';
    honouring that would mark every representation CHANGED and void the state stratum."""
    for filler in ("none", "None", "normal", "default", "intact", "无", "正常"):
        assert _is_default_state(filler)
        ents = _propose([_row(state_modifier=filler)], "Mara runs along the pier")
        assert (ents[0].state_modifier, ents[0].state_angle) == ("", StateAngle.UNKNOWN)


def test_a_distinguishing_qualifier_kept_by_the_model_stays_a_separate_identity() -> None:
    """When the prompt uses a qualifier to tell two similar entities apart, the label keeps it
    and they must remain two identities — the lookalike-disambiguation case."""
    ents = _propose(
        [
            _row(kind="prop", label="indigo Petrel", state_modifier="", category="boat"),
            _row(kind="prop", label="red Petrel", state_modifier="", category="boat"),
        ],
        "the indigo Petrel passes the red Petrel",
    )
    assert [e.name for e in ents] == ["indigo Petrel", "red Petrel"]
    assert [e.entity_id for e in ents] == ["indigo Petrel", "red Petrel"]


def test_a_bare_architectural_surface_is_not_banked_as_a_prop() -> None:
    ents = _propose(
        [
            _row(kind="prop", label="window", category="window"),
            _row(kind="prop", label="Fresnel lens", category="lens"),
        ],
        "Elias polishes the Fresnel lens by the window",
    )
    assert [e.name for e in ents] == ["Fresnel lens"]


def test_a_qualified_surface_is_still_a_real_prop() -> None:
    """Its segmenter category is the bare noun, so the filter must look at the label only."""
    ents = _propose(
        [_row(kind="prop", label="stained-glass window", category="window")],
        "light falls through the stained-glass window",
    )
    assert [e.name for e in ents] == ["stained-glass window"]


def test_a_room_can_still_be_the_setting() -> None:
    ents = _propose(
        [_row(kind="location", label="lantern room", category="room")],
        "inside the lantern room",
    )
    assert [e.name for e in ents] == ["lantern room"]


class _Cropper:
    def __init__(self, tmp) -> None:
        self.tmp = tmp

    def crop(self, segment_video: str, entity: NamedEntity, *, segment_id: int):
        del segment_video, segment_id
        out = self.tmp / f"{entity.kind.value}.png"
        out.write_bytes(b"crop")
        return str(out)


def test_the_modifier_reaches_the_observation_for_state_aware_reads(tmp_path) -> None:
    dec = RoleAwareDecomposer(cropper=_Cropper(tmp_path))
    obs = dec.decompose(
        segment_id=2,
        named_entities=[
            NamedEntity(name="Mara", kind=AssetType.CHARACTER, entity_id="Mara",
                        description="woman in a grey coat", state_modifier="young",
                        state_angle=StateAngle.CHANGED)
        ],
        segment_video="seg.mp4",
    )
    assert obs[0].angle_meta["state_modifier"] == "young"
    assert obs[0].state_angle == StateAngle.CHANGED
    # The literal wording is what a later shot asking for the young appearance matches on.
    assert "young" in obs[0].description.lower()


def test_a_modifier_already_in_the_description_is_not_duplicated(tmp_path) -> None:
    dec = RoleAwareDecomposer(cropper=_Cropper(tmp_path))
    obs = dec.decompose(
        segment_id=3,
        named_entities=[
            NamedEntity(name="Mara", kind=AssetType.CHARACTER, entity_id="Mara",
                        description="young woman with braided hair", state_modifier="young",
                        state_angle=StateAngle.CHANGED)
        ],
        segment_video="seg.mp4",
    )
    assert obs[0].description == "young woman with braided hair"
