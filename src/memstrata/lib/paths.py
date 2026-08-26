"""Helper to resolve paths relative to the MemStrata project root."""

from __future__ import annotations

from pathlib import Path


def memstrata_root() -> Path:
    """Return the absolute path to the MemStrata project root (benchmarks/MemStrata)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "MemStrata" and (parent / "src").is_dir():
            return parent
    # Fallback to parents[3] based on: benchmarks/MemStrata/src/memstrata/lib/paths.py
    return current.parents[3]
