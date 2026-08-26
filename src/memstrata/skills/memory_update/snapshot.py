"""Human-readable, dynamically-updatable memory snapshot exporter.

Writes ``<out_dir>/memory.json`` — a clean, deterministic view of the stratified
``AssetBank`` that mirrors the benchmark gt entity schema (``entities`` keyed by id,
each with ``states`` / appearance text and an ``initial_state``) and ADDS the two
things a live memory bank has that the gt does not: on-disk visual evidence paths and
per-state appearance timestamps.

Design constraints (deliberately dependency-light so it runs anywhere the bank does):
only the standard library — ``json`` / ``os`` / ``re`` / ``shutil`` / ``pathlib`` /
``math``. No numpy / PIL / torch. Fully deterministic ordering so two exports of the
same bank produce byte-identical JSON. The JSON write is atomic (temp file + replace).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from memstrata.bank.schema import NON_USABLE

SCHEMA = "memstrata-memory-1.0"

# τ → directory segment for the on-disk visual tree.
_KIND_PLURAL = {
    "character": "characters",
    "prop": "props",
    "location": "locations",
}

# Annotation keys that may carry an absolute source-video second for a representation.
_SECONDS_KEYS = ("source_seconds", "seconds", "frame_position")

# Keep CJK unified ideographs so Chinese entity names stay human-readable in the
# on-disk visual tree (大兔子/灰飞鼠/…); everything unsafe (spaces, slashes, punctuation)
# still collapses to ``_``.
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.\-\u4e00-\u9fff\u3400-\u4dbf]")


def _slug(value: Any) -> str:
    """Filesystem-safe token: keep ``[A-Za-z0-9_.-]`` and CJK, replace the rest with ``_``."""
    text = str(value if value is not None else "")
    return _SLUG_RE.sub("_", text) or "_"


def _kind_plural(kind_value: str) -> str:
    return _KIND_PLURAL.get(kind_value, f"{kind_value}s")


def _rep_seconds(rep: Any, rep_seconds: dict[str, Any] | None) -> float | None:
    """Absolute source seconds for a rep: explicit mapping first, then annotations."""
    if rep_seconds and rep.representation_id in rep_seconds:
        value = rep_seconds[rep.representation_id]
    else:
        value = None
        annotations = getattr(rep, "annotations", {}) or {}
        for key in _SECONDS_KEYS:
            if annotations.get(key) is not None:
                value = annotations[key]
                break
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _state_key(rep: Any) -> str:
    """State label for a rep: explicit annotation first, else the state-angle value."""
    annotations = getattr(rep, "annotations", {}) or {}
    for key in ("state_label", "state"):
        label = annotations.get(key)
        if label:
            return str(label)
    return rep.state_angle.value


def _rep_description(rep: Any) -> str:
    annotations = getattr(rep, "annotations", {}) or {}
    for key in ("description", "observation_description"):
        text = annotations.get(key)
        if text:
            return str(text)
    return ""


def export_memory_snapshot(
    bank,
    out_dir,
    *,
    movie_id: str = "",
    fps: float | None = None,
    updated_sec: float | None = None,
    rep_seconds: dict[str, Any] | None = None,
    copy_images: bool = True,
    video_path: str | None = None,
    video_duration_sec: float | None = None,
) -> Path:
    """Export ``bank`` to ``<out_dir>/memory.json`` and (optionally) copy visual evidence.

    Parameters
    ----------
    bank:
        An ``AssetBank``. Only usable assets (status not in ``NON_USABLE``) and
        non-deprecated representations are exported.
    out_dir:
        Destination directory. Created if missing. ``memory.json`` and, when
        ``copy_images`` is set, a ``visual/`` tree are written under it.
    movie_id, fps, updated_sec:
        Recorded verbatim in the top-level header. ``updated_sec`` defaults to the
        maximum known representation second (or ``None`` when no rep carries one).
    rep_seconds:
        Optional ``{representation_id: absolute_source_seconds}`` mapping. Takes
        precedence over per-rep annotations.
    copy_images:
        When True, copy each representation's image into a deterministic
        ``visual/<kind_plural>/<asset_id>/states/<state>/<rep_id><ext>`` layout and
        record the POSIX path relative to ``out_dir``. Missing source files are
        skipped (their image is omitted; export continues).
    video_path, video_duration_sec:
        The sibling grown film (``<out_dir>/long_video.mp4``) that each new segment
        is appended to. When given, recorded under the top-level ``video`` header as
        ``{"path": <posix rel to out_dir>, "duration_sec": ...}``. All ``sec`` /
        ``first_seen_sec`` values in ``entities`` are on THIS film's timeline.

    Returns
    -------
    Path
        Absolute path to the written ``memory.json``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entities: dict[str, Any] = {}
    max_sec: float | None = None

    for asset_id in sorted(bank.assets):
        asset = bank.assets[asset_id]
        if asset.status in NON_USABLE:
            continue

        kind_value = asset.kind.value
        kind_plural = _kind_plural(kind_value)
        asset_slug = _slug(asset_id)

        # state_key -> aggregation record; insertion order is the fallback tie-break.
        states: dict[str, dict[str, Any]] = {}
        insertion_order: list[str] = []

        for rep in asset.representations:
            if rep.deprecated:
                continue
            key = _state_key(rep)
            sec = _rep_seconds(rep, rep_seconds)
            segment = int(getattr(rep, "origin_segment_id", 0) or 0)
            if sec is not None and (max_sec is None or sec > max_sec):
                max_sec = sec

            if key not in states:
                states[key] = {
                    "description": "",
                    "appearances": [],
                    "images": [],
                    "source_frames": [],
                    "secs": [],
                    "segments": [],
                }
                insertion_order.append(key)
            record = states[key]

            if not record["description"]:
                desc = _rep_description(rep)
                if desc:
                    record["description"] = desc

            record["appearances"].append({"sec": sec, "segment": segment})
            if sec is not None:
                record["secs"].append(sec)
            record["segments"].append(segment)

            if copy_images:
                source = str(getattr(rep, "object_uri", "") or "")
                if source and os.path.isfile(source):
                    ext = Path(source).suffix
                    rel_dir = Path("visual") / kind_plural / asset_slug / "states" / _slug(key)
                    target_dir = out_path / rel_dir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{_slug(rep.representation_id)}{ext}"
                    target = target_dir / filename
                    shutil.copyfile(source, target)
                    record["images"].append((rel_dir / filename).as_posix())

                    # Design C: keep the FULL source frame beside the crop so the bank
                    # supports crop↔frame provenance self-audit and frame-level retrieval.
                    frame_src = str((getattr(rep, "annotations", {}) or {}).get("source_frame", "") or "")
                    if frame_src and os.path.isfile(frame_src) and os.path.abspath(frame_src) != os.path.abspath(source):
                        frame_ext = Path(frame_src).suffix
                        frame_dir = target_dir / "frames"
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        frame_name = f"{_slug(rep.representation_id)}__frame{frame_ext}"
                        shutil.copyfile(frame_src, frame_dir / frame_name)
                        record["source_frames"].append((rel_dir / "frames" / frame_name).as_posix())

        # Finalize per-state fields and ordering.
        def _state_first_seen(rec: dict[str, Any]) -> float | None:
            return min(rec["secs"]) if rec["secs"] else None

        def _state_first_segment(rec: dict[str, Any]) -> int:
            return min(rec["segments"]) if rec["segments"] else 0

        ordered_keys = sorted(
            states,
            key=lambda k: (
                _state_first_seen(states[k]) if _state_first_seen(states[k]) is not None else math.inf,
                _state_first_segment(states[k]),
                k,
            ),
        )

        state_objects: dict[str, Any] = {}
        entity_secs: list[float] = []
        for key in ordered_keys:
            rec = states[key]
            first_seen = _state_first_seen(rec)
            if first_seen is not None:
                entity_secs.append(first_seen)
            appearances = sorted(
                rec["appearances"],
                key=lambda a: (a["sec"] if a["sec"] is not None else math.inf, a["segment"]),
            )
            state_objects[key] = {
                "description": rec["description"],
                "first_seen_sec": first_seen,
                "appearances": appearances,
                "images": sorted(rec["images"]),
            }
            if rec["source_frames"]:
                state_objects[key]["source_frames"] = sorted(rec["source_frames"])

        entity_first_seen = min(entity_secs) if entity_secs else None

        # initial_state: explicit metadata wins; else the earliest state; else the first
        # inserted state; else None when the asset has no representations.
        initial_state = asset.metadata.get("initial_state")
        if not initial_state:
            if ordered_keys:
                initial_state = ordered_keys[0]
            elif insertion_order:
                initial_state = insertion_order[0]
            else:
                initial_state = None

        aliases = asset.metadata.get("aliases", [])
        entities[asset_id] = {
            "name": asset.name,
            "kind": kind_value,
            "description": asset.d,
            "aliases": list(aliases) if isinstance(aliases, list) else [],
            "lifecycle": asset.status.value,
            "initial_state": initial_state,
            "first_seen_sec": entity_first_seen,
            "states": state_objects,
        }

    resolved_updated = updated_sec if updated_sec is not None else max_sec

    snapshot: dict[str, Any] = {
        "schema": SCHEMA,
        "movie_id": movie_id,
        "fps": fps,
        "updated_sec": resolved_updated,
    }
    if video_path is not None:
        # Record the grown film as a path relative to out_dir when it sits inside it;
        # fall back to the given path verbatim otherwise.
        try:
            rel_video = Path(video_path).resolve().relative_to(out_path.resolve()).as_posix()
        except ValueError:
            rel_video = Path(video_path).as_posix()
        video_obj: dict[str, Any] = {"path": rel_video}
        if video_duration_sec is not None:
            try:
                video_obj["duration_sec"] = float(video_duration_sec)
            except (TypeError, ValueError):
                video_obj["duration_sec"] = None
        snapshot["video"] = video_obj
    snapshot["entities"] = entities

    memory_path = out_path / "memory.json"
    tmp_path = out_path / "memory.json.tmp"
    tmp_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp_path, memory_path)
    return memory_path
