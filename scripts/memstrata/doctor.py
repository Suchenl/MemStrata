#!/usr/bin/env python3
"""Check that this MemStrata checkout can run the CPU demo."""

from __future__ import annotations

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
        print("CPU demo: bash scripts/memstrata/cpu_demo.sh")
        return 1
    print("MemStrata doctor: OK")
    print(f"  root={ROOT}")
    print("  next: bash scripts/memstrata/cpu_demo.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
