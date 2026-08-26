"""Stdlib-only HTTP client for the isolated WeDetect-Ref grounding service.

WeDetect (WeChatCV/WeDetect, GPL-v3) runs as a SEPARATE process/env — see
``scripts/memstrata/servers/serve_wedetect.py`` — and this client speaks to it over
HTTP using only the Python standard library, so the GPL code is **never imported
into the memstrata process**. It exposes a referring-expression *grounder*: given a
frame path and a natural-language description, return normalized bounding boxes.

Why this exists: the SAM3-concept proposer ranks candidates by salience, so a query
for a non-salient entity (e.g. "squirrel") returns the most salient one instead
(the rabbit) — the memory bank then stores the wrong crop. WeDetect-Ref grounds the
box directly from the *description*, which fixes that mis-crop. This client is a
drop-in describe->box backend; when the service is unset/down it returns nothing and
the caller falls back to the SAM3 path (no hard dependency, offline-safe).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


class WeDetectRefGrounder:
    """Thin HTTP handle to the WeDetect-Ref service (referring-expression grounding)."""

    def __init__(
        self,
        url: str,
        *,
        score_thre: float = 0.25,
        topk: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.score_thre = float(score_thre)
        self.topk = int(topk)
        self.timeout = float(timeout)

    @classmethod
    def from_env(cls) -> "WeDetectRefGrounder | None":
        """Build from ``MEMSTRATA_WEDETECT_URL`` (+ optional thresholds), or None.

        Returns None when the env var is unset OR the service does not answer a health
        check — so callers can always do ``grounder = WeDetectRefGrounder.from_env()``
        and treat None as "fall back to SAM3".
        """
        url = os.environ.get("MEMSTRATA_WEDETECT_URL", "").strip()
        if not url:
            return None
        try:
            score = float(os.environ.get("MEMSTRATA_WEDETECT_SCORE_THRE", "0.25") or 0.25)
        except ValueError:
            score = 0.25
        try:
            topk = int(os.environ.get("MEMSTRATA_WEDETECT_TOPK", "5") or 5)
        except ValueError:
            topk = 5
        grounder = cls(url, score_thre=score, topk=topk)
        return grounder if grounder.healthy() else None

    def healthy(self) -> bool:
        try:
            req = urllib.request.Request(self.url + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310 (trusted local URL)
                return 200 <= int(resp.status) < 300
        except Exception:
            return False

    def ground(
        self, frame_path: str | Path, query: str, *, kind: str = ""
    ) -> list[tuple[list[int], float]]:
        """Ground ``query`` in ``frame_path``.

        Returns ``[([y0, x0, y1, x1] on a 0-1000 grid, score), ...]`` sorted by score,
        or ``[]`` on any failure / empty query (caller falls back to SAM3).
        """
        q = str(query or "").strip()
        if not q:
            return []
        payload = json.dumps(
            {
                "image_path": str(Path(frame_path).resolve()),
                "query": q,
                "score_thre": self.score_thre,
                "topk": self.topk,
                "kind": str(kind or ""),
            }
        ).encode("utf-8")
        try:
            req = urllib.request.Request(
                self.url + "/ground",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []
        out: list[tuple[list[int], float]] = []
        boxes = data.get("boxes") or []
        scores = data.get("scores") or []
        for box, score in zip(boxes, scores):
            if not box or len(box) != 4:
                continue
            out.append(([int(box[0]), int(box[1]), int(box[2]), int(box[3])], float(score)))
        return out


__all__ = ["WeDetectRefGrounder"]
