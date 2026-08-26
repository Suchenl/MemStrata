#!/usr/bin/env python3
"""Check that this MemStrata checkout has the pieces a GPU production run needs locally.

Does not download weights. Set PUBLIC_MODELS_ROOT and see MODELS.md.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCREENPLAY = ROOT / "production/screenplay/products/en/0000_detective_mystery.json"


def main() -> int:
    errors: list[str] = []
    if not (ROOT / "src" / "memstrata").is_dir():
        errors.append(f"missing package at {ROOT / 'src' / 'memstrata'}")
    if not SCREENPLAY.is_file():
        errors.append(f"missing default screenplay {SCREENPLAY}")
    if shutil.which("ffmpeg") is None:
        errors.append("ffmpeg not on PATH")
    if shutil.which("python3") is None:
        errors.append("python3 not on PATH")
    if errors:
        print("MemStrata doctor: FAIL")
        for item in errors:
            print(f"  - {item}")
        print("GPU run: export PUBLIC_MODELS_ROOT=...  (see MODELS.md)")
        print("         bash scripts/memstrata/run_production.sh")
        return 1
    print("MemStrata doctor: OK")
    print(f"  root={ROOT}")
    print("  next: export PUBLIC_MODELS_ROOT=...  (see MODELS.md)")
    print("        bash scripts/memstrata/run_production.sh")
    if not os.environ.get("PUBLIC_MODELS_ROOT"):
        print("  PUBLIC_MODELS_ROOT is unset; the production run will not find weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
