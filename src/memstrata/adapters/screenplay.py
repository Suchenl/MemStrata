"""Production-screenplay adapter (the Montage-native flow of interface_flows.md §D).

Reads a ``production_screenplay`` JSON (see data/Screenplay/products/<lang>/<id>.json)
and turns it into the two things the MemStrata closed loop consumes:

  * ``seed_packet`` — a ``MemoryUpdater.ingest_packet`` payload seeded from ``main_entities``
    (identity + name + kind + appearance description; no pixels yet — the first shot's
    keyframe bootstraps each entity's visual, then decompose banks real crops).
  * ``iter_shots`` — one :class:`ShotPlan` per ``production_screenplay.shots`` entry: the
    segment prompt (actions with ``(E#)`` tags resolved to names), transition, referenced
    entities, and required/forbidden asset directives from ``planned_assets``.

Deterministic and model-free: the heavy lifting (compose/generate/decompose/curate) stays in
``memstrata.*``; this only maps the approved screenplay onto per-segment run inputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENTITY_TAG = re.compile(r"\s*[（(]\s*(E\d+)\s*[）)]")  # full-width or ASCII parens

_KIND_BY_TYPE = {
    "character": "character",
    "object": "prop",
    "prop": "prop",
    "location": "location",
}


@dataclass(slots=True)
class ShotPlan:
    """One production shot mapped to a MemStrata segment."""

    segment_id: int
    shot_id: str
    scene_id: str
    prompt: str
    transition: str                      # "cut" | "continue"
    duration_sec: float
    referenced_entities: list[str] = field(default_factory=list)  # entity_ids
    required_ids: list[str] = field(default_factory=list)
    forbidden_ids: list[str] = field(default_factory=list)         # operation avoid/deprecate
    is_scene_start: bool = False


def load_screenplay(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "production_screenplay" not in data:
        raise ValueError(f"{path}: not a production screenplay (missing 'production_screenplay')")
    return data


def entity_names(screenplay: dict[str, Any]) -> dict[str, str]:
    return {e["entity_id"]: e["name"] for e in screenplay.get("main_entities", [])}


def _resolve_tags(text: str, names: dict[str, str]) -> str:
    """Resolve ``(E#)`` traceability tags to the registered entity name.

    The prose often uses a *short form* of a name (e.g. ``泥碑（E2）`` while the bank stores
    ``楔形泥碑``, or ``向导（E4）`` for ``向导哈桑``). The read-path anchor (``memory_retrieval.name_match``) is a
    verbatim substring test, so stripping the tag would drop those entities from selection. By
    substituting the tag with the canonical ``（name）`` we guarantee the anchor name appears in
    the prompt exactly once per tagged mention — this is the screenplay's own authoring intent
    (the tag declares which entity the mention refers to) and also grounds generation. Unknown
    ids are dropped."""
    def _sub(match: "re.Match[str]") -> str:
        name = names.get(match.group(1))
        return f"（{name}）" if name else ""

    return _ENTITY_TAG.sub(_sub, text).strip()


def seed_packet(screenplay: dict[str, Any]) -> dict[str, Any]:
    """An ingest_packet seeding the bank from ``main_entities`` (name-anchored identity,
    appearance as description, empty crop_path -> visual is bootstrapped by the first shot)."""
    observations = []
    for ent in screenplay.get("main_entities", []):
        kind = _KIND_BY_TYPE.get(str(ent.get("entity_type", "")).lower(), "character")
        eid = str(ent["entity_id"])
        observations.append({
            "entity_id": eid,
            "kind": kind,
            "name": str(ent.get("name", eid)),
            "description": str(ent.get("appearance", "")),
            "crop_path": "",                       # no pixels yet; bootstrap on first appearance
            "representation_id": f"{eid}@seed",
        })
    return {"segment_id": -1, "observations": observations, "state_events": []}


def iter_shots(screenplay: dict[str, Any]) -> list[ShotPlan]:
    names = entity_names(screenplay)
    shots = screenplay["production_screenplay"].get("shots", [])
    plans: list[ShotPlan] = []
    seen_scenes: set[str] = set()
    for i, shot in enumerate(shots):
        actions = shot.get("visual_track", {}).get("actions", [])
        prompt = " ".join(_resolve_tags(a, names) for a in actions).strip()
        scene_id = str(shot.get("scene_id", ""))
        required, forbidden = [], []
        for pa in shot.get("planned_assets", []):
            pid = str(pa.get("planned_asset_id", ""))
            op = str(pa.get("operation", "preserve")).lower()
            if op in {"avoid", "deprecate"}:
                forbidden.append(pid)
            elif pa.get("required") or op in {"preserve", "transform"}:
                required.append(pid)
        refs = list(dict.fromkeys(
            [str(e) for e in shot.get("active_characters", [])] + required
        ))
        plans.append(ShotPlan(
            segment_id=i,
            shot_id=str(shot.get("shot_id", f"shot_{i:04d}")),
            scene_id=scene_id,
            prompt=prompt,
            transition=str(shot.get("transition", "cut")).lower(),
            duration_sec=float(shot.get("duration_sec", 5.0)),
            referenced_entities=refs,
            required_ids=required,
            forbidden_ids=forbidden,
            is_scene_start=(scene_id not in seen_scenes),
        ))
        seen_scenes.add(scene_id)
    return plans
