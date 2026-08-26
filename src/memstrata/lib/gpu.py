"""Rule-based free-GPU selection for launching resident model services.

Persistent services (VACE generator, future perception/MLLM daemons) should land on the
*freest* card(s) rather than always defaulting to GPU 0 and colliding. This queries
``nvidia-smi`` and returns the indices with the most free memory.

ponytail: parses ``nvidia-smi`` CSV (no pynvml dependency). Ceiling: it reads a one-shot
snapshot, so two launches racing within the same instant could pick the same card; the
upgrade path is a file-lock reservation if that ever bites. Returns ``None`` on any failure
(no nvidia-smi, parse error) so callers fall back to inherited ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def _rank_free_gpus(csv_text: str, count: int, min_free_mib: int) -> list[int] | None:
    """Parse ``nvidia-smi`` ``index,memory.free`` CSV into the ``count`` freest indices.

    Pure/testable core: returns ``None`` on a malformed line, else the eligible indices
    (>= ``min_free_mib`` free) sorted by most-free-first, truncated to ``count``.
    """

    free: list[tuple[int, int]] = []
    for line in csv_text.strip().splitlines():
        idx_str, _, free_str = line.partition(",")
        try:
            idx, free_mib = int(idx_str.strip()), int(free_str.strip())
        except ValueError:
            return None  # unexpected format -> let caller fall back
        if free_mib >= min_free_mib:
            free.append((free_mib, idx))
    free.sort(reverse=True)  # most-free first
    return [idx for _, idx in free[:count]]


def pick_free_gpu_ids(count: int = 1, *, min_free_mib: int = 4000) -> list[int] | None:
    """Return ``count`` GPU indices with the most free memory (descending).

    Only cards with at least ``min_free_mib`` free are eligible; returns ``None`` when
    ``nvidia-smi`` is unavailable/unparseable, and an empty list when no card qualifies.
    """

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return _rank_free_gpus(out, count, min_free_mib)


def gpu_free_memory_mib() -> dict[int, int] | None:
    """Return physical GPU index -> free memory MiB, or ``None`` when unavailable."""

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    free: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx_str, _, free_str = line.partition(",")
        try:
            free[int(idx_str.strip())] = int(free_str.strip())
        except ValueError:
            return None
    return free


def visible_devices_have_min_free(visible: str, *, min_free_mib: int) -> bool | None:
    """Check whether every numeric CUDA_VISIBLE_DEVICES entry has enough free memory.

    ``None`` means the check could not be performed, usually because the visible string uses
    UUID/MIG syntax or ``nvidia-smi`` is unavailable.
    """

    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not tokens:
        return None
    try:
        ids = [int(token) for token in tokens]
    except ValueError:
        return None
    free = gpu_free_memory_mib()
    if free is None:
        return None
    return all(free.get(idx, -1) >= min_free_mib for idx in ids)


def cuda_visible_devices_for(count: int = 1, *, min_free_mib: int = 4000) -> str | None:
    """Return a ``CUDA_VISIBLE_DEVICES`` string for the ``count`` freest cards, or ``None``.

    ``None`` means "could not choose" (no nvidia-smi / parse error) OR "not enough free
    cards"; the caller should then leave the environment untouched.
    """

    ids = pick_free_gpu_ids(count, min_free_mib=min_free_mib)
    if not ids or len(ids) < count:
        return None
    return ",".join(str(i) for i in ids)


def ensure_cuda_visible_devices_have_min_free(env: dict[str, str] | None = None, *, min_free_mib: int = 4000) -> None:
    """Raise when an explicit numeric CUDA_VISIBLE_DEVICES does not meet the free-memory floor."""

    env = env or os.environ
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return
    ok = visible_devices_have_min_free(visible, min_free_mib=min_free_mib)
    if ok is False:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible} does not satisfy min_free_mib={min_free_mib}; "
            "pick an emptier GPU or move to another node before starting the service"
        )
