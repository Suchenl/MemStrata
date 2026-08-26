#!/usr/bin/env python3
"""Smoke the Wan2.2-TI2V-5B-Turbo backend through the real MemStrata backend path.

Exercises both routes the benchmark needs on ONE resident model:
  * i2v — a reference image in ``controls['composed_references']`` (the composed-keyframe path)
  * t2v — no reference at all (the generator-floor control row)

Run with an explicit card so it never lands on a busy GPU:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/memstrata/_smoke_wan22_turbo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from memstrata.steps.generate.backends.factory import build_video_backend
from memstrata.steps.generate.schemas import MediaGenerationTask, MediaTaskType

BACKEND = "wan22_ti2v5b_turbo"
REF_IMAGE = Path(
    "./models/vendor/wan22_ti2v5b_turbo/examples/images/cat.JPG"
)
PROMPT = "a fluffy cat on a surfboard drifting over calm turquoise water, cinematic"


def _task(segment: str, *, reference: Path | None) -> MediaGenerationTask:
    controls: dict = {}
    if reference is not None:
        controls["composed_references"] = [{"image": str(reference)}]
    return MediaGenerationTask(
        task_id=f"smoke_{segment}",
        segment_id=segment,
        task_type=MediaTaskType.VIDEO_SEGMENT,
        plan_version=1,
        model_name=BACKEND,
        prompt=PROMPT,
        controls=controls,
    )


def main() -> int:
    out_dir = Path("/tmp/turbo_backend_smoke")
    backend = build_video_backend(BACKEND, output_dir=out_dir, run_id="turbo_smoke")

    cases = [("s0000", REF_IMAGE), ("s0001", None)]
    failures = 0
    for segment, reference in cases:
        label = "i2v" if reference is not None else "t2v"
        t0 = time.time()
        try:
            artifact = backend.generate(_task(segment, reference=reference))
        except Exception as exc:  # noqa: BLE001 - a smoke should report, not traceback out
            print(f"[{label}] FAILED after {time.time() - t0:.1f}s: {exc!r}", flush=True)
            failures += 1
            continue
        # The first case pays the ~5 min weight load; the second is the warm per-segment cost.
        print(
            f"[{label}] ok in {time.time() - t0:.1f}s  route={_route(artifact)}  "
            f"video={artifact.object_uri}",
            flush=True,
        )
    return 1 if failures else 0


def _route(artifact) -> str:
    for note in artifact.degradation_notes:
        if note.startswith("route="):
            return note.split("=", 1)[1]
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
