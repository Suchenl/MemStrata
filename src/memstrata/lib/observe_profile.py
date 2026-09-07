"""Low-overhead structured timing for the production observe path."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass
class ObserveProfile:
    kind: str
    segment_id: int | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    status: str = "ok"
    wall_ms: float = 0.0
    subphases: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def add(self, name: str, elapsed_ms: float, *, calls: int = 1) -> None:
        row = self.subphases.setdefault(name, {"calls": 0, "total_ms": 0.0, "max_ms": 0.0})
        row["calls"] = int(row["calls"]) + int(calls)
        row["total_ms"] = float(row["total_ms"]) + float(elapsed_ms)
        row["max_ms"] = max(float(row["max_ms"]), float(elapsed_ms))

    def merge(self, payload: dict[str, Any] | None, *, prefix: str = "") -> None:
        for name, raw in ((payload or {}).get("subphases") or {}).items():
            if not isinstance(raw, dict):
                continue
            calls = int(raw.get("calls") or 0)
            if calls <= 0:
                continue
            row = self.subphases.setdefault(
                f"{prefix}{name}",
                {"calls": 0, "total_ms": 0.0, "max_ms": 0.0},
            )
            row["calls"] = int(row["calls"]) + calls
            row["total_ms"] = float(row["total_ms"]) + float(raw.get("total_ms") or 0.0)
            row["max_ms"] = max(float(row["max_ms"]), float(raw.get("max_ms") or 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "segment_id": self.segment_id,
            "started_at": self.started_at,
            "status": self.status,
            "wall_ms": round(self.wall_ms, 3),
            "timing_semantics": "inclusive; nested subphases may overlap",
            "subphases": {
                name: {
                    "calls": int(row["calls"]),
                    "total_ms": round(float(row["total_ms"]), 3),
                    "max_ms": round(float(row["max_ms"]), 3),
                }
                for name, row in sorted(self.subphases.items())
            },
        }


_ACTIVE: ContextVar[ObserveProfile | None] = ContextVar("memstrata_observe_profile", default=None)


@contextmanager
def capture_profile(
    kind: str,
    *,
    segment_id: int | None = None,
    output_path: str | Path | None = None,
) -> Iterator[ObserveProfile]:
    profile = ObserveProfile(kind=kind, segment_id=segment_id)
    token = _ACTIVE.set(profile)
    started = time.perf_counter()
    try:
        yield profile
    except BaseException:
        profile.status = "error"
        raise
    finally:
        profile.wall_ms = (time.perf_counter() - started) * 1000.0
        _ACTIVE.reset(token)
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(profile.to_dict(), ensure_ascii=False) + "\n"
            fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)


@contextmanager
def profile_span(name: str) -> Iterator[None]:
    profile = _ACTIVE.get()
    if profile is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        profile.add(name, (time.perf_counter() - started) * 1000.0)


def merge_profile(payload: dict[str, Any] | None, *, prefix: str = "") -> None:
    profile = _ACTIVE.get()
    if profile is not None:
        profile.merge(payload, prefix=prefix)


__all__ = ["ObserveProfile", "capture_profile", "merge_profile", "profile_span"]
