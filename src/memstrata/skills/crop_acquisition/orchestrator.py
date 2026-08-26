"""Targeted, novelty-first per-entity crop acquisition (production adaptation of S5).

This is the perception core of the bench S5 ``propose_and_pick`` route, rewritten for
the production ``memstrata`` write-path:

* bench S5 acquires crops for a WHOLE segment of entities at once and uses a VLM to pick
  which proposal is which (closed-set), banking whatever best matches each entity;
* production here acquires ONE named entity at a time and, crucially, selects the
  proposal that records **new** appearance content (novelty-first), not the proposal
  most similar to what the bank already holds.

The flow (faithful to S5 perception building blocks — SAM3 concept segmentation,
GroundingDINO detection, DINOv3 exemplar identity, deterministic mask/crop QA):

  1. Propose candidate regions for this one entity (SAM3 concept + GDINO phrase).
  2. DINOv3-embed each masked candidate crop.
  3. IDENTITY GATE (correctness): keep candidates whose max cosine-sim to
     ``exemplar_vectors`` >= ``identity_threshold``. Empty exemplars (first sighting)
     ⇒ skip the gate and accept the best-QA proposal.
  4. NOVELTY SELECTION: among identity-OK candidates pick the one MAXIMIZING
     ``novelty = 1 - max cosine-sim to existing_rep_vectors``; tie-break by crop QA
     sharpness and mask fill ratio. (Picking the exemplar-nearest crop would re-record
     old content — that is exactly what we must NOT do.)
  5. crop QA reject (dark / low-information) as a final guard.
  6. Save the chosen masked crop; return metadata (or ``None`` if nothing survives).

Heavy model objects (SAM3 / GroundingDINO / DINOv3) are **not** imported here — they are
passed in already-constructed (built once by ``crop_server``), so this module imports
cleanly without ``torch``/``transformers``/``sam3``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from memstrata.skills.crop_acquisition.crop_io import materialize_crop
from memstrata.skills.crop_acquisition.crop_qa import audit_crop, bbox_area_fraction
from memstrata.skills.crop_acquisition.discovery import DISCOVERY_CONCEPTS
from memstrata.skills.crop_acquisition.exemplar_identity import max_cosine_to
from memstrata.skills.crop_acquisition.geometry import (
    dedup_by_iou,
    mask_to_bbox_norm,
    px_to_norm_bbox,
)
from memstrata.skills.crop_acquisition.mask_quality import assess_mask_quality

# Concept vocab per production asset kind (mirrors S5 CHARACTER/PROP concepts).
CHARACTER_CONCEPTS = ("person", "animal")
PROP_CONCEPTS = ("object",)
LOCATION_CONCEPTS = DISCOVERY_CONCEPTS["location"]

# Cosine floor to confirm "this is our entity". Kept deliberately LENIENT: its only job is
# to drop obvious background strangers, NOT to demand high similarity to a possibly cross-
# domain seed exemplar (seed portrait vs a masked back-view generated crop score low in
# DINOv3 even for the same person). Too high a floor rejects exactly the novel-pose crops we
# want to record. When nothing clears it, the right outcome is a miss for this segment, not
# writing an unrelated person into the bank.
DEFAULT_IDENTITY_THRESHOLD = 0.25
_MAX_CHARACTER_BBOX_AREA = 1.0     # close-ups are valid; near-full-frame is caught by QA
_MIN_MASK_FILL = 0.18              # adaptive floor; thin poses/long props get a lower gate
_MIN_SIDE_PX = 16
_IOU_DEDUP = 0.7


def _candidate_identity(cand: dict[str, Any]) -> float:
    return float(cand.get("identity_sim") or 0.0)


def _candidate_fill(cand: dict[str, Any]) -> float:
    mq = cand.get("mask_quality") or {}
    return float(mq.get("fg_area", 0)) if mq else 0.0


def _candidate_state_novelty(cand: dict[str, Any], banked_states: set[str] | None) -> int:
    """State-aware preference: 1 when the candidate records a state NOT yet banked.

    A pure tie-breaker layered on top of visual novelty (never replaces it). It is a
    graceful no-op whenever there is no per-identity state history (``banked_states``
    empty/None) or the candidate carries no known ``state_angle`` — so the offline /
    unclassified path behaves exactly as before.
    """
    if not banked_states:
        return 0
    state = str(cand.get("state_angle") or "").strip().lower()
    if not state or state == "unknown":
        return 0
    return 1 if state not in banked_states else 0


def rank_acquisition_candidates(
    kept: list[dict[str, Any]],
    *,
    banked_states: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank identity-OK candidates best→worst for the final QA walk.

    Order: identity ⟶ visual novelty ⟶ state novelty (tie-breaker) ⟶ detection score
    ⟶ mask fill. State novelty sits directly BELOW visual novelty, so among candidates
    of equal visual novelty the one in an un-banked state wins, without ever overriding
    the visual-novelty ranking itself.
    """
    return sorted(
        kept,
        key=lambda c: (
            _candidate_identity(c),
            float(c.get("novelty_score") or 0.0),
            _candidate_state_novelty(c, banked_states),
            float(c.get("score") or 0.0),
            _candidate_fill(c),
        ),
        reverse=True,
    )


def _concepts_for(kind: str) -> tuple[str, ...]:
    if kind == "character":
        return CHARACTER_CONCEPTS
    if kind == "prop":
        return PROP_CONCEPTS
    if kind == "location":
        return LOCATION_CONCEPTS
    return ()


def _resolve_concepts(kind: str, concepts: tuple[str, ...] | None) -> tuple[str, ...]:
    """Entity-specific concepts first, then the generic kind concepts as fallback.

    SAM3-concept is open-vocabulary and works far better with a concrete noun ('red apple',
    'acorn') than the generic kind word ('object') — the generic-'object'-only prop path
    finds nothing. Callers pass the decomposer's per-entity ``category`` here; the kind
    concepts stay appended as a safety net so a bad/empty category never removes coverage.
    """
    specific = tuple(c.strip() for c in (concepts or ()) if c and c.strip())
    return tuple(dict.fromkeys([*specific, *_concepts_for(kind)]))


def _compact_description(description: str, *, max_chars: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(description or "").strip().lower())
    text = re.sub(r"[^\w\s,\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,-")
    return text[:max_chars].strip(" ,-")


# Generic kind words carry no describe->box signal for a referring grounder; when the
# decomposer gave no real description and the name is one of these, skip WeDetect and let
# the SAM3-concept path handle it (the perception fallback names entities "character"/"prop").
_GENERIC_NAMES = frozenset({
    "character", "characters", "prop", "props", "object", "objects", "person", "people",
    "animal", "animals", "scene", "location", "background", "entity",
})


def _grounding_query(name: str, description: str) -> str:
    """Referring query for WeDetect-Ref: prefer the rich description, else a real name.

    WeDetect-Ref handles long multilingual phrases, so we pass the decomposer's own
    ``description`` verbatim (no GDINO-style 'person/object' prefixing, which biases a
    describe->box model). Falls back to ``name`` only when it is a concrete label, never a
    generic kind word — returning "" then signals the caller to use the SAM3 path.
    """
    desc = re.sub(r"\s+", " ", str(description or "").strip())
    if desc:
        return desc
    label = str(name or "").strip()
    if label and label.casefold() not in _GENERIC_NAMES:
        return label
    return ""


def _grounding_phrases_for(kind: str, description: str) -> tuple[str, ...]:
    """Use category/appearance phrases for GDINO, never a proper-name-only prompt."""
    concepts = list(_concepts_for(kind))
    desc = _compact_description(description)
    phrases: list[str] = []
    if desc:
        if kind == "character":
            animal_terms = ("animal", "rabbit", "dog", "cat", "horse", "bird", "bear")
            head = "animal" if any(term in desc for term in animal_terms) else "person"
            phrases.append(f"{head} {desc}")
        elif kind == "prop":
            phrases.append(f"object {desc}")
        elif kind == "location":
            phrases.append(f"scene {desc}")
    phrases.extend(concepts)
    # Preserve order while removing duplicates and empty phrases.
    return tuple(dict.fromkeys(p.strip() for p in phrases if p and p.strip()))


def _adaptive_mask_fill_floor(bbox_norm: list[int], base: float) -> float:
    """Relax the fill gate for elongated/large poses where masks naturally fill less bbox."""
    if len(bbox_norm) != 4:
        return float(base)
    y0, x0, y1, x1 = bbox_norm
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    aspect = max(w / h, h / w)
    area = bbox_area_fraction(bbox_norm)
    if aspect >= 2.2:
        return min(float(base), 0.10)
    if area >= 0.25:
        return min(float(base), 0.12)
    return float(base)


def _mask_fill_ratio(mask: object, bbox_norm: list[int], height: int, width: int) -> float:
    import numpy as np

    y0, x0, y1, x1 = bbox_norm
    top = max(0, min(height - 1, round(y0 / 1000 * height)))
    left = max(0, min(width - 1, round(x0 / 1000 * width)))
    bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
    right = max(left + 1, min(width, round(x1 / 1000 * width)))
    array = np.asarray(mask, dtype=bool)[top:bottom, left:right]
    if array.size == 0:
        return 0.0
    return float(array.mean())


def _save_mask_png(mask: object, bbox_norm: list[int], out_path: Path) -> Path | None:
    """Write the bbox-cropped boolean mask as an 8-bit L PNG; return path or None."""
    import numpy as np
    from PIL import Image

    array = np.asarray(mask, dtype=bool)
    height, width = array.shape[:2]
    y0, x0, y1, x1 = bbox_norm
    top = max(0, min(height - 1, round(y0 / 1000 * height)))
    left = max(0, min(width - 1, round(x0 / 1000 * width)))
    bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
    right = max(left + 1, min(width, round(x1 / 1000 * width)))
    crop = array[top:bottom, left:right]
    if crop.size == 0:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((crop.astype(np.uint8) * 255), mode="L").save(out_path)
    return out_path


def _propose_candidates(
    *,
    frame_path: Path,
    entity_name: str,
    entity_kind: str,
    entity_description: str,
    scratch_dir: Path,
    segmenter: Any | None,
    detector: Any | None,
    max_character_bbox_area: float,
    min_mask_fill: float,
    min_side_px: int,
    iou_threshold: float,
    concepts: tuple[str, ...] | None = None,
    grounder: Any | None = None,
) -> list[dict[str, Any]]:
    """Collect candidate crops for ONE entity.

    When a WeDetect-Ref ``grounder`` is available and the entity carries a real
    description/name, the description-grounded boxes are **authoritative** (they fix the
    SAM3-concept "most-salient-wins" mis-crop) and the SAM3/GDINO path is skipped. When the
    grounder yields nothing (service down, empty query, no box) we fall back to the
    SAM3 concept + GDINO phrase proposals.
    """
    from PIL import Image

    scratch_dir.mkdir(parents=True, exist_ok=True)
    pil = Image.open(frame_path).convert("RGB")
    width, height = pil.size
    raw: list[dict[str, Any]] = []

    # --- WeDetect-Ref referring grounding (describe -> bbox, authoritative) ---
    query = _grounding_query(entity_name, entity_description) if grounder is not None else ""
    if grounder is not None and query:
        try:
            hits = grounder.ground(frame_path, query, kind=entity_kind)
        except Exception:
            hits = []
        g_raw: list[dict[str, Any]] = []
        for ordinal, (bbox_norm, score) in enumerate(hits):
            if not bbox_norm or len(bbox_norm) != 4:
                continue
            y0, x0, y1, x1 = bbox_norm
            left = max(0, min(width - 1, round(x0 / 1000 * width)))
            top = max(0, min(height - 1, round(y0 / 1000 * height)))
            right = max(left + 1, min(width, round(x1 / 1000 * width)))
            bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
            if (right - left) < min_side_px or (bottom - top) < min_side_px:
                continue
            if (
                entity_kind == "character"
                and max_character_bbox_area < 1.0
                and bbox_area_fraction(list(bbox_norm)) > max_character_bbox_area
            ):
                continue
            crop_path = scratch_dir / f"wedetect_{ordinal}.png"
            crop_path = materialize_crop(
                frame=frame_path, bbox_norm=list(bbox_norm), out_path=crop_path
            )
            g_raw.append({
                "bbox_norm": list(bbox_norm),
                "score": float(score),
                "mask": None,
                "crop_path": crop_path,
                "source": "wedetect_ref",
                "grounding_query": query,
                "quality_profile": "referring_bbox_no_mask",
                "mask_quality": {"available": False, "reason": "wedetect_bbox_only"},
            })
        if g_raw:
            # Description-grounded boxes win: do NOT also run the salience-ranked SAM3 path,
            # whose most-salient proposal would re-introduce the wrong-entity crop.
            return dedup_by_iou(g_raw, iou_threshold=iou_threshold)

    # --- SAM3 concept proposals (masked) ---
    if segmenter is not None:
        concepts = list(_resolve_concepts(entity_kind, concepts))
        by_concept = segmenter.segment_multi(frame_path, concepts) if concepts else {}
        for concept, instances in by_concept.items():
            for ordinal, (bbox_px, score, mask) in enumerate(instances):
                if mask is None:
                    continue
                bbox_norm = mask_to_bbox_norm(mask)
                if bbox_norm is None or len(bbox_norm) != 4:
                    continue
                x0, y0, x1, y1 = (int(v) for v in bbox_px)
                if (x1 - x0) < min_side_px or (y1 - y0) < min_side_px:
                    continue
                quality = assess_mask_quality(mask)
                if not quality.ok:
                    continue
                if entity_kind == "character":
                    if (
                        max_character_bbox_area < 1.0
                        and bbox_area_fraction(bbox_norm) > max_character_bbox_area
                    ):
                        continue
                    fill = _mask_fill_ratio(mask, bbox_norm, height, width)
                    if fill < _adaptive_mask_fill_floor(bbox_norm, min_mask_fill):
                        continue
                crop_path = scratch_dir / f"sam3_{concept}_{ordinal}.png"
                crop_path = materialize_crop(
                    frame=frame_path, bbox_norm=bbox_norm, out_path=crop_path, mask=mask
                )
                raw.append({
                    "bbox_norm": bbox_norm,
                    "score": float(score),
                    "mask": mask,
                    "crop_path": crop_path,
                    "source": "sam3_concept",
                    "concept": concept,
                    "mask_quality": quality.to_dict(),
                    "quality_profile": "masked_sam3_quality_gated",
                })

    # --- GroundingDINO phrase proposals (unmasked bbox) ---
    if detector is not None:
        for phrase in _grounding_phrases_for(entity_kind, entity_description):
            for ordinal, (bbox_norm, score) in enumerate(detector.detect_all(frame_path, phrase)):
                y0, x0, y1, x1 = bbox_norm
                left = max(0, min(width - 1, round(x0 / 1000 * width)))
                top = max(0, min(height - 1, round(y0 / 1000 * height)))
                right = max(left + 1, min(width, round(x1 / 1000 * width)))
                bottom = max(top + 1, min(height, round(y1 / 1000 * height)))
                if (right - left) < min_side_px or (bottom - top) < min_side_px:
                    continue
                if (
                    entity_kind == "character"
                    and max_character_bbox_area < 1.0
                    and bbox_area_fraction(list(bbox_norm)) > max_character_bbox_area
                ):
                    continue
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", phrase)[:24] or "phrase"
                crop_path = scratch_dir / f"gdino_{slug}_{ordinal}.png"
                crop_path = materialize_crop(
                    frame=frame_path, bbox_norm=list(bbox_norm), out_path=crop_path
                )
                raw.append({
                    "bbox_norm": list(bbox_norm),
                    "score": float(score),
                    "mask": None,
                    "crop_path": crop_path,
                    "source": "grounding_dino",
                    "grounding_phrase": phrase,
                    "quality_profile": "bbox_high_recall_no_mask",
                    "mask_quality": {"available": False, "reason": "gdino_bbox_only"},
                })

    return dedup_by_iou(raw, iou_threshold=iou_threshold)


def acquire_entity_crop(
    frame_path: str | Path,
    *,
    frame_paths: list[str | Path] | None = None,
    frame_positions: list[float] | None = None,
    entity_name: str,
    entity_kind: str,
    entity_description: str = "",
    concepts: tuple[str, ...] | None = None,
    exemplar_vectors: list[list[float]],
    existing_rep_vectors: list[list[float]],
    out_dir: str | Path,
    banked_states: set[str] | None = None,
    segmenter: Any | None = None,
    detector: Any | None = None,
    grounder: Any | None = None,
    embedder: Any | None = None,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    max_candidates: int = 8,
    max_character_bbox_area: float = _MAX_CHARACTER_BBOX_AREA,
    min_mask_fill: float = _MIN_MASK_FILL,
    min_side_px: int = _MIN_SIDE_PX,
    iou_threshold: float = _IOU_DEDUP,
) -> dict[str, Any] | None:
    """Acquire the most NOVEL identity-correct crop for one named entity.

    Returns ``{crop_path, bbox, mask_path, identity_sim, novelty_score, source}`` or
    ``None`` when no identity-OK, QA-passing candidate exists.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "_candidates"

    resolved_frame_paths = [Path(p) for p in (frame_paths or [frame_path]) if p]
    if not resolved_frame_paths:
        return None
    candidates: list[dict[str, Any]] = []
    for frame_index, candidate_frame in enumerate(resolved_frame_paths):
        frame_scratch = scratch_dir / f"frame_{frame_index:02d}"
        frame_candidates = _propose_candidates(
            frame_path=candidate_frame,
            entity_name=entity_name,
            entity_kind=entity_kind,
            entity_description=entity_description,
            scratch_dir=frame_scratch,
            segmenter=segmenter,
            detector=detector,
            grounder=grounder,
            max_character_bbox_area=max_character_bbox_area,
            min_mask_fill=min_mask_fill,
            min_side_px=min_side_px,
            iou_threshold=iou_threshold,
            concepts=concepts,
        )
        for cand in frame_candidates:
            cand["frame_path"] = candidate_frame
            cand["frame_index"] = frame_index
            if frame_positions and frame_index < len(frame_positions):
                cand["frame_position"] = float(frame_positions[frame_index])
        candidates.extend(frame_candidates)
    if not candidates:
        return None
    candidate_budget = max(1, max_candidates) * max(1, len(resolved_frame_paths))
    candidates = sorted(candidates, key=lambda c: float(c.get("score") or 0.0), reverse=True)
    candidates = candidates[:candidate_budget]

    # DINOv3-embed each masked candidate crop (white-composited internally).
    vecs: list[list[float]] = []
    if embedder is not None:
        try:
            vecs = embedder.embed_batch([Path(c["crop_path"]) for c in candidates])
        except Exception:
            vecs = []
    if vecs and len(vecs) == len(candidates):
        for cand, vec in zip(candidates, vecs):
            cand["_vec"] = vec

    have_embeddings = all("_vec" in c for c in candidates) and len(candidates) > 0
    if exemplar_vectors and not have_embeddings:
        return None
    use_identity_gate = bool(exemplar_vectors)

    # 3) IDENTITY GATE — confirm "this is our entity" (correctness only). If exemplars
    # exist and nothing clears the floor, this segment is a miss; recording a stranger is worse.
    identity_gate: str = "off"
    kept: list[dict[str, Any]] = []
    below: list[dict[str, Any]] = []
    for cand in candidates:
        if use_identity_gate:
            sim = float(max_cosine_to(cand["_vec"], exemplar_vectors))
            cand["identity_sim"] = sim
            (kept if sim >= identity_threshold else below).append(cand)
        else:
            # First sighting (no exemplars) or no embedder: skip the gate.
            cand["identity_sim"] = (
                float(max_cosine_to(cand["_vec"], exemplar_vectors))
                if (exemplar_vectors and "_vec" in cand)
                else None
            )
            kept.append(cand)
    if use_identity_gate:
        if not kept:
            return None
        identity_gate = "applied"
    if not kept:
        return None

    # 4) NOVELTY SELECTION — prefer the candidate recording the MOST NEW content.
    def _novelty(cand: dict[str, Any]) -> float:
        if have_embeddings and existing_rep_vectors:
            return 1.0 - max_cosine_to(cand["_vec"], existing_rep_vectors)
        return 1.0  # nothing to be redundant with yet ⇒ everything is fully novel

    for cand in kept:
        cand["novelty_score"] = _novelty(cand)

    # Sort by identity desc first; visual novelty chooses among identity-compatible
    # crops, and state novelty is a tie-breaker that prefers an un-banked state.
    ranked = rank_acquisition_candidates(kept, banked_states=banked_states)

    # 5) crop QA (dark / low-information) as a final guard; walk best→worst.
    for cand in ranked:
        crop_path = Path(cand["crop_path"])
        qa = audit_crop(crop=crop_path, bbox_norm=cand["bbox_norm"], kind=entity_kind)
        if not qa.accepted:
            continue

        # 6) Save the chosen masked crop into out_dir (stable name). Keep CJK/word chars so
        # two differently-named entities in the same out_dir don't collide on "crop__.png"
        # (a bare a-zA-Z0-9 slug maps every Chinese name to the same empty slug -> overwrite).
        slug = re.sub(r"[^\w]+", "_", entity_name.strip().lower()).strip("_")[:32] or "entity"
        final_crop = out_dir / f"crop_{slug}.png"
        final_crop = materialize_crop(
            frame=Path(cand.get("frame_path") or resolved_frame_paths[0]),
            bbox_norm=cand["bbox_norm"],
            out_path=final_crop,
            mask=cand.get("mask"),
        )
        mask_path: str | None = None
        if cand.get("mask") is not None:
            saved = _save_mask_png(cand["mask"], cand["bbox_norm"], out_dir / f"crop_{slug}_mask.png")
            mask_path = str(saved) if saved is not None else None

        return {
            "crop_path": str(final_crop),
            "bbox": list(cand["bbox_norm"]),
            "mask_path": mask_path,
            "identity_sim": cand.get("identity_sim"),
            "identity_gate": identity_gate,
            "novelty_score": float(cand["novelty_score"]),
            "source": cand["source"],
            "source_detail": {
                "concept": cand.get("concept"),
                "grounding_phrase": cand.get("grounding_phrase"),
                "grounding_query": cand.get("grounding_query"),
                "quality_profile": cand.get("quality_profile"),
                "frame_index": cand.get("frame_index"),
                "frame_position": cand.get("frame_position"),
            },
            "frame_path": str(cand.get("frame_path") or resolved_frame_paths[0]),
            "frame_position": cand.get("frame_position"),
            "candidate_count": len(candidates),
            "identity_threshold": float(identity_threshold),
            "min_side_px": int(min_side_px),
            "max_character_bbox_area": float(max_character_bbox_area),
            "min_mask_fill": float(min_mask_fill),
            "qa": qa.to_dict(),
        }

    return None


__all__ = [
    "acquire_entity_crop",
    "rank_acquisition_candidates",
    "CHARACTER_CONCEPTS",
    "PROP_CONCEPTS",
    "LOCATION_CONCEPTS",
    "DEFAULT_IDENTITY_THRESHOLD",
    "_MIN_SIDE_PX",
]
