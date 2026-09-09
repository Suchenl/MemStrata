"""Cached, model-honest evidence for location scene candidates.

The producer lives in the persistent crop server so all location names observed in
one segment share foreground detections and temporal frame embeddings.  GroundingDINO
provides bounding-box *estimates*, never masks; DINOv3 provides generic temporal visual
consistency, never place identity or resolver merge evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from memstrata.skills.location_scene_validity import (
    COVERAGE_COMPLETE,
    COVERAGE_LOWER_BOUND,
    COVERAGE_MISSING,
    SCENE_EVIDENCE_SCHEMA_VERSION,
)

SUBJECT_PROMPT_VERSION = "dynamic-subject-v1"
SUBJECT_PROMPT = "person. human body. legs. feet. animal. bird."


def normalized_bbox_area(bbox: Sequence[int | float] | None) -> float | None:
    """Exact area fraction for a normalized ``[y0, x0, y1, x1]`` box."""

    box = _valid_box(bbox)
    if box is None:
        return None
    y0, x0, y1, x1 = box
    return max(0.0, min(1.0, (y1 - y0) * (x1 - x0) / 1_000_000.0))


def bbox_coverage(
    candidate_bbox: Sequence[int | float],
    subject_bboxes: Sequence[Sequence[int | float]],
    *,
    content_bbox: Sequence[int | float] | None = None,
) -> tuple[float, float]:
    """Return max-instance and rectangle-union coverage inside a candidate.

    Coverage is relative to the candidate's intersection with the reliable content
    box when one is supplied.  The geometry is explicitly rectangle/bbox based.
    """

    candidate = _valid_box(candidate_bbox)
    if candidate is None:
        return 0.0, 0.0
    denominator_box = candidate
    content = _valid_box(content_bbox)
    if content is not None:
        clipped = _intersection(candidate, content)
        if clipped is not None:
            denominator_box = clipped
    denominator = _box_area(denominator_box)
    if denominator <= 0.0:
        return 0.0, 0.0

    clipped_subjects = []
    for raw in subject_bboxes:
        subject = _valid_box(raw)
        overlap = _intersection(denominator_box, subject) if subject is not None else None
        if overlap is not None:
            clipped_subjects.append(overlap)
    if not clipped_subjects:
        return 0.0, 0.0
    maximum = max(_box_area(box) for box in clipped_subjects) / denominator
    union = _rect_union_area(clipped_subjects) / denominator
    return min(1.0, maximum), min(1.0, union)


def content_box_normalized(frame_path: str | Path) -> tuple[list[int] | None, str]:
    """Conservatively strip uniform dark/bright edge bars.

    Returns ``(None, "unknown")`` on unreadable media rather than pretending that the
    full image is verified content.
    """

    try:
        rgb = np.asarray(Image.open(Path(frame_path)).convert("RGB"), dtype=np.float64)
    except (OSError, ValueError):
        return None, "unknown"
    if rgb.ndim != 3 or rgb.shape[0] < 16 or rgb.shape[1] < 16:
        return [0, 0, 1000, 1000], "full_frame"

    gray = rgb[:, :, :3] @ np.asarray([0.299, 0.587, 0.114])
    height, width = gray.shape
    row_mean, row_std = gray.mean(axis=1), gray.std(axis=1)
    col_mean, col_std = gray.mean(axis=0), gray.std(axis=0)
    row_bar = ((row_mean < 24.0) | (row_mean > 248.0)) & (row_std < 8.0)
    col_bar = ((col_mean < 24.0) | (col_mean > 248.0)) & (col_std < 8.0)
    top = _leading_true(row_bar)
    bottom = _leading_true(row_bar[::-1])
    left = _leading_true(col_bar)
    right = _leading_true(col_bar[::-1])
    if height - top - bottom < int(0.25 * height):
        top = bottom = 0
    if width - left - right < int(0.25 * width):
        left = right = 0
    normalized = [
        int(round(top / height * 1000)),
        int(round(left / width * 1000)),
        int(round((height - bottom) / height * 1000)),
        int(round((width - right) / width * 1000)),
    ]
    status = "detected" if normalized != [0, 0, 1000, 1000] else "full_frame"
    return normalized, status


class CachedSegmentSceneEvidenceProducer:
    """Reuse loaded detector/embedder work across all names sharing sampled frames."""

    def __init__(
        self,
        *,
        detector: Any | None,
        embedder: Any | None,
        subject_prompt: str = SUBJECT_PROMPT,
        subject_prompt_version: str = SUBJECT_PROMPT_VERSION,
        temporal_support_threshold: float = 0.80,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.subject_prompt = str(subject_prompt)
        self.subject_prompt_version = str(subject_prompt_version)
        self.temporal_support_threshold = float(temporal_support_threshold)
        self._foreground_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._known_subject_cache: dict[
            tuple[Any, ...], list[list[float]]
        ] = {}
        self._embedding_cache: dict[tuple[Any, ...], list[float] | None] = {}
        self._temporal_cache: dict[tuple[tuple[Any, ...], ...], dict[str, Any]] = {}
        self.detector_frame_calls = 0
        self.detector_batch_calls = 0
        self.embedding_frame_calls = 0
        self.embedding_batch_calls = 0

    def collect_candidates(
        self,
        *,
        frame_paths: Sequence[str | Path],
        candidates: Sequence[Mapping[str, Any]],
        lower_bound_bboxes: Mapping[Any, Sequence[Sequence[int | float]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return one schema-versioned evidence mapping per candidate."""

        paths = [Path(path).resolve() for path in frame_paths]
        frame_keys = [self._frame_key(path) for path in paths]
        foreground = self._foreground_for_frames(paths, frame_keys)
        temporal = self._temporal_for_frames(paths, frame_keys)
        lower = lower_bound_bboxes or {}
        out: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                frame_index = int(candidate.get("frame_index", 0) or 0)
            except (TypeError, ValueError):
                frame_index = 0
            bbox = _valid_box(candidate.get("bbox_norm"))
            if bbox is None or not paths or not 0 <= frame_index < len(paths):
                out.append(self._missing_candidate_evidence(candidate, frame_index, temporal))
                continue

            frame_fg = foreground[frame_index]
            detector_boxes = list(frame_fg.get("boxes") or [])
            lower_rows = lower.get(
                frame_index,
                lower.get(str(frame_index), ()),
            )
            known_boxes = [
                box
                for box in (
                    _valid_box(raw)
                    for raw in (
                        *lower_rows,
                        *self._known_subject_cache.get(frame_keys[frame_index], ()),
                    )
                )
                if box is not None
            ]
            detector_status = str(frame_fg.get("status") or "missing")
            if detector_status == "success":
                semantics = COVERAGE_COMPLETE
                subject_boxes = [*detector_boxes, *known_boxes]
                source = f"grounding_dino:{self.subject_prompt_version}"
            elif known_boxes:
                semantics = COVERAGE_LOWER_BOUND
                subject_boxes = known_boxes
                source = "same_segment_subject_bbox"
            else:
                semantics = COVERAGE_MISSING
                subject_boxes = []
                source = ""

            content_box, content_status = content_box_normalized(paths[frame_index])
            if semantics == COVERAGE_MISSING:
                maximum = union = None
            else:
                maximum, union = bbox_coverage(
                    bbox,
                    subject_boxes,
                    content_bbox=content_box,
                )
            exact_area = normalized_bbox_area(bbox)
            content_area = _content_area_fraction(bbox, content_box)
            evidence = {
                "schema_version": SCENE_EVIDENCE_SCHEMA_VERSION,
                "coverage_semantics": semantics,
                "coverage_geometry": "bbox",
                "foreground_max_coverage": maximum,
                "foreground_union_coverage": union,
                "foreground_source": source,
                "foreground_prompt_version": self.subject_prompt_version,
                "foreground_detector_status": detector_status,
                "foreground_detector_error": str(frame_fg.get("error") or "")[:200],
                "crop_area_fraction": exact_area,
                "content_area_fraction": content_area,
                "content_box_status": content_status,
                "content_bbox": content_box,
                "candidate": {
                    "bbox": list(bbox),
                    "frame_index": frame_index,
                    "frame_path": str(paths[frame_index]),
                    "frame_position": candidate.get("frame_position"),
                    "source": str(candidate.get("source") or ""),
                    "score": _optional_float(candidate.get("score")),
                },
                **temporal,
            }
            out.append(evidence)
        return out

    def record_subject_bbox(
        self,
        *,
        frame_path: str | Path,
        bbox_norm: Sequence[int | float],
    ) -> None:
        """Record an already-acquired character as lower-bound foreground evidence."""

        path = Path(frame_path).resolve()
        box = _valid_box(bbox_norm)
        if box is None:
            return
        rows = self._known_subject_cache.setdefault(self._frame_key(path), [])
        if box not in rows:
            rows.append(box)

    def cache_stats(self) -> dict[str, int]:
        return {
            "detector_frame_calls": self.detector_frame_calls,
            "detector_batch_calls": self.detector_batch_calls,
            "embedding_frame_calls": self.embedding_frame_calls,
            "embedding_batch_calls": self.embedding_batch_calls,
            "foreground_cache_entries": len(self._foreground_cache),
            "embedding_cache_entries": len(self._embedding_cache),
            "temporal_cache_entries": len(self._temporal_cache),
            "known_subject_frame_entries": len(self._known_subject_cache),
        }

    def _foreground_for_frames(
        self,
        paths: list[Path],
        keys: list[tuple[Any, ...]],
    ) -> list[dict[str, Any]]:
        missing = [
            (index, path, keys[index])
            for index, path in enumerate(paths)
            if keys[index] not in self._foreground_cache
        ]
        if missing and self.detector is not None:
            detect_batch = getattr(self.detector, "detect_batch", None)
            if callable(detect_batch):
                self.detector_batch_calls += 1
                self.detector_frame_calls += len(missing)
                try:
                    rows = detect_batch([path for _, path, _ in missing], self.subject_prompt)
                    if not isinstance(rows, Sequence) or len(rows) != len(missing):
                        raise ValueError("detector batch result length mismatch")
                    for (_, _, key), row in zip(missing, rows):
                        self._foreground_cache[key] = {
                            "status": "success",
                            "boxes": _boxes_from_hits(row),
                        }
                except Exception as exc:  # noqa: BLE001 - missing must remain distinguishable
                    for _, _, key in missing:
                        self._foreground_cache[key] = {
                            "status": "missing",
                            "boxes": [],
                            "error": repr(exc),
                        }
            else:
                for _, path, key in missing:
                    self.detector_frame_calls += 1
                    try:
                        hits = self.detector.detect_all(path, self.subject_prompt)
                        self._foreground_cache[key] = {
                            "status": "success",
                            "boxes": _boxes_from_hits(hits),
                        }
                    except Exception as exc:  # noqa: BLE001
                        self._foreground_cache[key] = {
                            "status": "missing",
                            "boxes": [],
                            "error": repr(exc),
                        }
        elif missing:
            for _, _, key in missing:
                self._foreground_cache[key] = {
                    "status": "missing",
                    "boxes": [],
                    "error": "detector_unavailable",
                }
        return [self._foreground_cache[key] for key in keys]

    def _temporal_for_frames(
        self,
        paths: list[Path],
        keys: list[tuple[Any, ...]],
    ) -> dict[str, Any]:
        set_key = tuple(keys)
        cached = self._temporal_cache.get(set_key)
        if cached is not None:
            return dict(cached)

        missing = [
            (path, key)
            for path, key in zip(paths, keys)
            if key not in self._embedding_cache
        ]
        if missing and self.embedder is not None:
            self.embedding_batch_calls += 1
            self.embedding_frame_calls += len(missing)
            try:
                vectors = self.embedder.embed_batch([path for path, _ in missing])
                if not isinstance(vectors, Sequence) or len(vectors) != len(missing):
                    raise ValueError("embedding batch result length mismatch")
                for (_, key), vector in zip(missing, vectors):
                    self._embedding_cache[key] = [float(value) for value in vector]
            except Exception:
                for _, key in missing:
                    self._embedding_cache[key] = None
        elif missing:
            for _, key in missing:
                self._embedding_cache[key] = None

        vectors = [self._embedding_cache[key] for key in keys]
        summary = _temporal_summary(
            vectors,
            threshold=self.temporal_support_threshold,
            source=str(getattr(self.embedder, "name", "") or ""),
            observation_count=len(paths),
        )
        self._temporal_cache[set_key] = summary
        return dict(summary)

    @staticmethod
    def _frame_key(path: Path) -> tuple[Any, ...]:
        try:
            stat = path.stat()
            return (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            return (str(path), -1, -1)

    def _missing_candidate_evidence(
        self,
        candidate: Mapping[str, Any],
        frame_index: int,
        temporal: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": SCENE_EVIDENCE_SCHEMA_VERSION,
            "coverage_semantics": COVERAGE_MISSING,
            "coverage_geometry": "bbox",
            "foreground_max_coverage": None,
            "foreground_union_coverage": None,
            "foreground_source": "",
            "foreground_prompt_version": self.subject_prompt_version,
            "foreground_detector_status": "missing",
            "crop_area_fraction": normalized_bbox_area(candidate.get("bbox_norm")),
            "content_area_fraction": None,
            "content_box_status": "unknown",
            "content_bbox": None,
            "candidate": {
                "bbox": candidate.get("bbox_norm"),
                "frame_index": frame_index,
                "source": str(candidate.get("source") or ""),
                "score": _optional_float(candidate.get("score")),
            },
            **temporal,
        }


def _temporal_summary(
    vectors: Sequence[Sequence[float] | None],
    *,
    threshold: float,
    source: str,
    observation_count: int,
) -> dict[str, Any]:
    valid = [
        (index, _unit(vector))
        for index, vector in enumerate(vectors)
        if vector is not None
    ]
    valid = [(index, vector) for index, vector in valid if vector is not None]
    if not valid:
        return {
            "temporal_visual_status": "missing",
            "temporal_visual_support_count": None,
            "temporal_visual_observation_count": observation_count,
            "temporal_visual_consistency": None,
            "temporal_visual_min_similarity": None,
            "temporal_visual_medoid_index": None,
            "temporal_visual_threshold": threshold,
            "temporal_source": "",
            "temporal_semantics": "generic_visual_not_vpr",
        }

    similarities = [
        [float(np.dot(a, b)) for _, b in valid]
        for _, a in valid
    ]
    means = [sum(row) / len(row) for row in similarities]
    medoid_local = max(range(len(valid)), key=means.__getitem__)
    medoid_index = valid[medoid_local][0]
    medoid_sims = similarities[medoid_local]
    support = sum(sim >= threshold for sim in medoid_sims)
    status = "available" if len(valid) == observation_count else "partial"
    return {
        "temporal_visual_status": status,
        "temporal_visual_support_count": support,
        "temporal_visual_observation_count": observation_count,
        "temporal_visual_consistency": round(sum(medoid_sims) / len(medoid_sims), 6),
        "temporal_visual_min_similarity": round(min(medoid_sims), 6),
        "temporal_visual_medoid_index": medoid_index,
        "temporal_visual_threshold": threshold,
        "temporal_source": source,
        "temporal_semantics": "generic_visual_not_vpr",
    }


def _unit(vector: Sequence[float]) -> np.ndarray | None:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return arr / norm if arr.ndim == 1 and arr.size and norm > 1e-12 else None


def _boxes_from_hits(hits: Any) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not isinstance(hits, Sequence):
        return boxes
    for hit in hits:
        raw = hit[0] if isinstance(hit, Sequence) and len(hit) >= 1 else None
        box = _valid_box(raw)
        if box is not None:
            boxes.append(box)
    return boxes


def _content_area_fraction(
    bbox: Sequence[int | float],
    content_bbox: Sequence[int | float] | None,
) -> float | None:
    content = _valid_box(content_bbox)
    candidate = _valid_box(bbox)
    if content is None or candidate is None:
        return None
    overlap = _intersection(candidate, content)
    denominator = _box_area(content)
    return (
        0.0
        if overlap is None or denominator <= 0.0
        else min(1.0, _box_area(overlap) / denominator)
    )


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_box(raw: Any) -> list[float] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    try:
        y0, x0, y1, x1 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    y0, x0, y1, x1 = (
        min(1000.0, max(0.0, y0)),
        min(1000.0, max(0.0, x0)),
        min(1000.0, max(0.0, y1)),
        min(1000.0, max(0.0, x1)),
    )
    return [y0, x0, y1, x1] if y1 > y0 and x1 > x0 else None


def _intersection(a: Sequence[float], b: Sequence[float] | None) -> list[float] | None:
    if b is None:
        return None
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    return [y0, x0, y1, x1] if y1 > y0 and x1 > x0 else None


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _rect_union_area(boxes: Sequence[Sequence[float]]) -> float:
    xs = sorted({box[1] for box in boxes} | {box[3] for box in boxes})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (box[0], box[2])
            for box in boxes
            if box[1] < right and box[3] > left
        )
        covered = 0.0
        start = end = None
        for low, high in intervals:
            if start is None:
                start, end = low, high
            elif low > end:
                covered += end - start
                start, end = low, high
            else:
                end = max(end, high)
        if start is not None and end is not None:
            covered += end - start
        area += (right - left) * covered
    return area


def _leading_true(flags: np.ndarray) -> int:
    count = 0
    for flag in flags:
        if not bool(flag):
            break
        count += 1
    return count


__all__ = [
    "COVERAGE_COMPLETE",
    "COVERAGE_LOWER_BOUND",
    "COVERAGE_MISSING",
    "CachedSegmentSceneEvidenceProducer",
    "SCENE_EVIDENCE_SCHEMA_VERSION",
    "SUBJECT_PROMPT",
    "SUBJECT_PROMPT_VERSION",
    "bbox_coverage",
    "content_box_normalized",
    "normalized_bbox_area",
]
