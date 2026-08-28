#!/usr/bin/env python3
"""Check that this MemStrata checkout has the pieces a GPU production run needs locally.

Does not download weights. Set PUBLIC_MODELS_ROOT and see MODELS.md.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCREENPLAY = ROOT / "production/screenplay/products/en/0000_detective_mystery.json"


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    if not (ROOT / "src" / "memstrata").is_dir():
        errors.append(f"missing package at {ROOT / 'src' / 'memstrata'}")
    if not SCREENPLAY.is_file():
        errors.append(f"missing default screenplay {SCREENPLAY}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        errors.append("ffmpeg not on PATH and imageio-ffmpeg is unavailable")
    if shutil.which("python3") is None:
        errors.append("python3 not on PATH")
    models_root = os.environ.get("PUBLIC_MODELS_ROOT", "").strip()
    required_models = (
        ("DINOv3", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
        ("Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
        ("FLUX.2 Klein", "black-forest-labs/FLUX.2-klein-9b-kv"),
        ("Wan2.2 LightX2V", "Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step"),
    )
    if not models_root:
        errors.append("PUBLIC_MODELS_ROOT is unset")
    else:
        for label, relative in required_models:
            if not (Path(models_root).expanduser() / relative).is_dir():
                errors.append(f"missing {label} weights at {Path(models_root).expanduser() / relative}")
    wedetect_url = os.environ.get("MEMSTRATA_WEDETECT_URL", "").strip()
    wedetect_healthy = False
    if wedetect_url:
        parsed = urllib.parse.urlparse(wedetect_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append("MEMSTRATA_WEDETECT_URL must be an http(s) URL")
        else:
            try:
                request = urllib.request.Request(wedetect_url.rstrip("/") + "/health", method="GET")
                with urllib.request.urlopen(request, timeout=5) as response:
                    wedetect_healthy = 200 <= int(response.status) < 300
            except Exception:
                warnings.append("WeDetect-Ref is configured but its /health check failed; SAM3 fallback will be used")

    sam3_deps = os.environ.get("MEMSTRATA_SAM3_DEPS", "").strip()
    if not sam3_deps:
        sam3_deps = str(ROOT / "models" / "vendor" / "sam3_transformers59")
    sam3_ready = bool(
        models_root
        and (Path(models_root).expanduser() / "facebook/sam3").is_dir()
        and Path(sam3_deps).expanduser().is_dir()
    )
    if not wedetect_healthy and not sam3_ready:
        errors.append(
            "crop acquisition has neither a healthy MEMSTRATA_WEDETECT_URL nor "
            "the SAM3 fallback (weights + MEMSTRATA_SAM3_DEPS)"
        )

    for label, env_key in (
        ("LightX2V", "MEMSTRATA_LIGHTX2V_PYTHON"),
        ("FLUX", "MEMSTRATA_FLUX_PYTHON"),
    ):
        interpreter = os.environ.get(env_key, "").strip() or "python3"
        resolved = Path(interpreter).expanduser()
        if resolved.is_absolute():
            executable = resolved.is_file() and os.access(resolved, os.X_OK)
        else:
            executable = shutil.which(interpreter) is not None
        if not executable:
            errors.append(f"{label} interpreter not executable: {interpreter} (set {env_key})")
    if errors:
        print("MemStrata doctor: FAIL")
        for item in errors:
            print(f"  - {item}")
        for item in warnings:
            print(f"  ! {item}")
        print("GPU run: export PUBLIC_MODELS_ROOT=...  (see MODELS.md)")
        print("         bash scripts/memstrata/run_production.sh")
        return 1
    print("MemStrata doctor: OK")
    print(f"  root={ROOT}")
    print("  next: export PUBLIC_MODELS_ROOT=...  (see MODELS.md)")
    print("        bash scripts/memstrata/run_production.sh")
    print("  WeDetect-Ref: set MEMSTRATA_WEDETECT_URL=http://127.0.0.1:8710 when the service is running.")
    for item in warnings:
        print(f"  ! {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
