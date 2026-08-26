"""Generate FLUX.2-klein face keyframes for the LoRA face-stress test.

Produces photoreal keyframes with clear HUMAN FACES at 832x480 (landscape, matching the
lightx2v distill output pins) so we can check whether SVI / morphic break faces under i2v,
and provide a start+end pair for first-last-frame interpolation (frames-to-video). Writes to
production/outputs/_lora_face_test/keyframes/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

MS = Path(".")
sys.path.insert(0, str(MS / "src"))

from memstrata.steps.generate.image_backends.factory import build_image_backend
from memstrata.steps.generate.schemas import MediaGenerationTask, MediaTaskType

# (name, prompt). Same person across the pair so it doubles as an interpolation start/end.
KEYFRAMES = [
    ("keeper_front",
     "Photorealistic cinematic close-up portrait of a weathered elderly lighthouse keeper, "
     "deep facial wrinkles, short grey beard, calm blue eyes looking directly at the camera, "
     "wearing a dark green oilskin raincoat, soft overcast coastal daylight, sharp skin detail, "
     "shallow depth of field, 35mm film"),
    ("keeper_profile",
     "Photorealistic cinematic medium shot of the same weathered elderly lighthouse keeper, "
     "three-quarter profile, turning his head to gaze out toward a stormy grey sea, dramatic "
     "side lighting on his face, short grey beard, dark green oilskin raincoat, sharp facial detail"),
    ("woman_front",
     "Photorealistic cinematic close-up portrait of a young woman fisher in her twenties, "
     "clear symmetric face, freckles, wet dark hair, a deep-blue headscarf, looking at the camera, "
     "cold coastal daylight, sharp skin and eye detail, shallow depth of field, 35mm film"),
]


def main() -> int:
    out_dir = MS / "production/outputs/_lora_face_test"
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    backend = build_image_backend(
        "flux.2-klein-9b-kv-fp8", output_dir=out_dir, run_id="flux_faces",
        models_config=str(MS / "configs"))
    for i, (name, prompt) in enumerate(KEYFRAMES):
        t0 = time.time()
        task = MediaGenerationTask(
            task_id=f"kf_{name}", chunk_id=f"kf_{i:02d}", task_type=MediaTaskType.KEYFRAME,
            plan_version=1, model_name="flux.2-klein-9b-kv-fp8", prompt=prompt,
            controls={"height": 480, "width": 832, "seed": 2026 + i})
        art = backend.generate(task)
        dst = kf_dir / f"{name}.png"
        try:
            import shutil
            src = Path(art.object_uri)
            if src.is_file():
                shutil.copy2(src, dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[flux] copy {name} failed: {exc!r}", flush=True)
        print(f"[flux] {name}: {art.object_uri} ({time.time()-t0:.1f}s) -> {dst}", flush=True)
    print(f"[flux] DONE. keyframes in {kf_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
