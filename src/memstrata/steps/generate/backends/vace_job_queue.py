"""File-based job queue shared by the persistent VACE server and its backend client.

Single-producer (the production loop) / single-consumer (one server per run), so the protocol stays
deliberately tiny: requests land in ``inbox/`` via atomic rename, results come back in ``outbox/``.
ponytail: file queue, not a socket/broker -- fine for one sequential producer; upgrade to a socket or
a real broker only if we ever need concurrent producers or cross-host serving.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)  # rename is atomic on POSIX => readers never see a partial file


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def server_dirs(server_dir: Path) -> tuple[Path, Path]:
    inbox = server_dir / "inbox"
    outbox = server_dir / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    return inbox, outbox


def server_ready(server_dir: Path) -> bool:
    return (Path(server_dir) / "ready").exists()


def request_stop(server_dir: Path) -> None:
    (Path(server_dir) / "stop").touch()


# --- client side -------------------------------------------------------------------------------


def submit_job(server_dir: Path, request: dict, *, timeout: float = 3600.0, poll: float = 1.0) -> dict:
    """Submit one job and block until the server writes its result (or ``timeout``)."""

    server_dir = Path(server_dir)
    inbox, outbox = server_dirs(server_dir)
    job_id = str(request.get("job_id") or uuid.uuid4().hex)
    request = {**request, "job_id": job_id}
    result_path = outbox / f"{job_id}.json"
    _atomic_write_json(inbox / f"{job_id}.json", request)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if result_path.exists():
            result = _read_json(result_path)
            result_path.unlink(missing_ok=True)
            return result
        time.sleep(poll)
    raise TimeoutError(f"VACE server job {job_id} timed out after {timeout}s")


# --- server side -------------------------------------------------------------------------------


def next_job(inbox: Path) -> Path | None:
    jobs = sorted(p for p in Path(inbox).glob("*.json"))
    return jobs[0] if jobs else None


def write_result(outbox: Path, job_id: str, result: dict) -> None:
    _atomic_write_json(Path(outbox) / f"{job_id}.json", result)
