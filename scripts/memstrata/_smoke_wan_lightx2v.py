"""One-chunk live smoke for the LightX2V-framework backend (provider=wan_lightx2v / lightx2v).

Mirrors _smoke_wan_distill.py (native_wan) but drives the wan22_i2v_a14b_lightx2v_4step backend,
so the two can be A/B compared on the SAME keyframe (pink_1.png) for speed + quality. Uses the real
LightX2VPipeline (model_cls=wan2.2_moe_distill, flash_attn2, 4-step CFG-free, no offload).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

MS = Path(".")
sys.path.insert(0, str(MS / "src"))

from memstrata.steps.generate.backends.factory import build_video_backend
from memstrata.steps.generate.schemas import MediaGenerationTask, MediaTaskType


def main() -> int:
    keyframe = sys.argv[1] if len(sys.argv) > 1 else str(
        MS.parents[1] / "submodules/_2_generation/backends/frames-to-video/examples/pink_1.png"
    )
    backend_name = sys.argv[2] if len(sys.argv) > 2 else "wan22_i2v_a14b_lightx2v_4step"
    import os
    prompt = os.environ.get("SMOKE_PROMPT") or (
        "Cinematic shot. The subject moves naturally and turns slightly; the camera slowly pushes "
        "in. Soft natural light, people walking in the background. Smooth realistic motion."
    )
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = MS / "production/outputs/_backend_smoke" / backend_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] backend={backend_name}", flush=True)
    print(f"[smoke] keyframe={keyframe}", flush=True)
    print(f"[smoke] out_dir={out_dir}", flush=True)

    backend = build_video_backend(
        backend_name, output_dir=out_dir, run_id="wan_lightx2v_smoke",
        models_config=str(MS / "configs"))
    task = MediaGenerationTask(
        task_id="smoke_task", chunk_id="chunk_0000", task_type=MediaTaskType.VIDEO_CHUNK,
        plan_version=1, model_name=backend_name, prompt=prompt,
        controls={"composed_references": [{"image": keyframe}], "gen_mode": "recompose_keyframe"})
    t0 = time.time()
    art = backend.generate(task)
    dt = time.time() - t0
    print(f"[smoke] DONE in {dt:.1f}s", flush=True)
    print(f"[smoke] object_uri={art.object_uri}", flush=True)
    print(f"[smoke] notes={art.degradation_notes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
