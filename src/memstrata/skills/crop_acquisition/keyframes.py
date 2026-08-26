"""Diverse keyframe selection for the write-side decomposer (DINO farthest-point).

Given several candidate frames uniformly sampled from ONE realized segment, pick a
small, visually diverse subset so the entity decomposer sees representative views (and
within-segment state changes) without paying for near-duplicate frames. Selection is
deterministic: seed with the most representative frame (highest mean similarity to the
pool), then farthest-point sampling by cosine distance over DINOv3 embeddings. The first
returned path is the representative frame (a good default crop source).

Pure Python over already-normalized embedding vectors (no numpy); the candidate pool is
tiny (a few-second clip at fps 2-4), so the O(n^2) selection is negligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence


class _BatchEmbedder(Protocol):
    def embed_batch(self, images: list[Path]) -> list[list[float]]: ...


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def select_diverse_keyframes(
    frame_paths: Sequence[str | Path],
    embedder: _BatchEmbedder,
    *,
    k: int,
) -> list[str]:
    """Return up to ``k`` diverse keyframes (representative frame first).

    Falls back to the first ``k`` inputs if embedding is unavailable or fails, so the
    caller always gets a usable, non-empty subset when candidates exist.
    """
    paths = [str(p) for p in frame_paths if p]
    if k <= 0 or not paths:
        return []
    if len(paths) <= k:
        return paths
    try:
        vecs = embedder.embed_batch([Path(p) for p in paths])
    except Exception:
        return paths[:k]
    if len(vecs) != len(paths) or not vecs:
        return paths[:k]

    n = len(paths)
    sim = [[_dot(vecs[i], vecs[j]) for j in range(n)] for i in range(n)]
    mean_sim = [sum(sim[i]) / n for i in range(n)]
    seed = max(range(n), key=lambda i: mean_sim[i])  # most representative frame
    selected = [seed]
    while len(selected) < k:
        # farthest-point step: add the frame whose NEAREST already-selected frame is the
        # least similar (i.e. maximize distance to the closest selected keyframe).
        best: int | None = None
        best_nearest_sim: float | None = None
        for i in range(n):
            if i in selected:
                continue
            nearest_sim = max(sim[i][s] for s in selected)
            if best_nearest_sim is None or nearest_sim < best_nearest_sim:
                best_nearest_sim, best = nearest_sim, i
        if best is None:
            break
        selected.append(best)
    return [paths[i] for i in selected]
