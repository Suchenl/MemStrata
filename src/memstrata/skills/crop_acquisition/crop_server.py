"""Persistent crop-acquisition inference server (file-queue protocol).

Mirrors the file-queue pattern of
``memstrata/steps/generate/backends/helios_persistent_server.py``: the ~model cold start
(GroundingDINO + SAM3 + DINOv3) is paid ONCE, then the process loops over pending job
JSONs and writes result JSONs, so the production loop never reloads models per entity.

Protocol (single-producer / single-consumer):
  ``<server_dir>/pending/<job_id>.json``  request  (written atomically by the client)
  ``<server_dir>/done/<job_id>.json``     result   (written atomically by the server)
  ``<server_dir>/ready``                  sentinel (PID, written AFTER models load)
  ``<server_dir>/stop``                   sentinel (touch to stop the server)

Request JSON (``job_kind="acquire"``, the default — one named entity):
  {job_id, frame_path, entity_name, entity_kind,
   exemplar_vectors, existing_rep_vectors, out_dir, <optional acquire kwargs>}
Request JSON (``job_kind="discover"`` — type-constrained discovery, O_disc):
  {job_id, job_kind: "discover", frame_path, kinds: [...], out_dir,
   exclude_bboxes: [[y0,x0,y1,x1], ...], <optional discovery kwargs>}
Result JSON:
  {job_id, status: "ok"|"error",
   result: <acquire_entity_crop dict | {"discovered": [...]} | null>, error?}

Heavy imports (torch / transformers / SAM3) happen only inside ``build_models`` /
``serve`` — importing this module is cheap and does NOT require transformers>=5.9.

LAUNCH (SAM3 needs the vendored transformers>=5.9 prepended on PYTHONPATH):

  PYTHONPATH=${MONTAGE_ROOT}/models/vendor/sam3_transformers59:\
benchmarks/MemStrata/src:src \
  python3 \
    -m memstrata.skills.crop_acquisition.crop_server --server_dir <dir>

(or use scripts/memstrata/servers/serve_crop_acq.sh, which sets this up for you; in the production loop
crop_client auto-starts this server, so the manual launcher is only for debugging.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)  # rename is atomic on POSIX => readers never see a partial file


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _server_dirs(server_dir: Path) -> tuple[Path, Path]:
    pending = server_dir / "pending"
    done = server_dir / "done"
    pending.mkdir(parents=True, exist_ok=True)
    done.mkdir(parents=True, exist_ok=True)
    return pending, done


def _next_job(pending: Path) -> Path | None:
    jobs = sorted(p for p in Path(pending).glob("*.json") if not p.name.endswith(".tmp"))
    return jobs[0] if jobs else None


def _normalize_device(device: str | None) -> str | None:
    """Accept "", "3", "cuda:3", "cpu" -> a torch-valid device string (or None for auto).

    A bare GPU index (e.g. "3") is not a valid torch device string; map it to "cuda:3".
    """
    if device is None:
        return None
    d = str(device).strip()
    if not d:
        return None
    if d.isdigit():
        return f"cuda:{d}"
    return d


class _Models:
    """Lazily-constructed bundle of the three perception models (built once)."""

    def __init__(self, *, device: str | None = None) -> None:
        # Imported here (not at module top) so the client/orchestrator import stays light.
        from memstrata.skills.crop_acquisition.grounding_dino import GroundingDinoProposer
        from memstrata.skills.crop_acquisition.sam3_concept import Sam3ConceptSegmenter
        from memstrata.skills.crop_acquisition.embedding import DinoV3Embedder

        device = _normalize_device(device)

        logging.info("[crop_acq] loading GroundingDINO ...")
        self.detector = None
        try:
            detector = GroundingDinoProposer(device=device)
            detector._ensure_loaded()
            self.detector = detector
        except Exception as exc:  # noqa: BLE001 - missing offline weights should degrade
            logging.warning(
                "[crop_acq] GroundingDINO unavailable; continuing with SAM3+DINOv3 only (%r)",
                exc,
            )
        logging.info("[crop_acq] loading SAM3 concept segmenter ...")
        self.segmenter = Sam3ConceptSegmenter(device=device)
        self.segmenter._ensure_loaded()
        logging.info("[crop_acq] loading DINOv3 embedder ...")
        self.embedder = DinoV3Embedder(device=device)
        self.embedder._ensure_loaded()


def _run_discovery_job(models: _Models, request: dict[str, Any]) -> dict[str, Any]:
    """``job_kind="discover"`` — type-constrained proposals for one frame (O_disc).

    Runs on the SAME loaded models as the requested path, so discovery costs extra
    proposals per frame but no extra model load.
    """
    from memstrata.skills.crop_acquisition.discovery import discover_entities

    extra: dict[str, Any] = {}
    for key in ("max_per_kind", "min_mask_fill", "min_side_px", "min_area_fraction",
                "iou_threshold"):
        if key in request:
            extra[key] = request[key]
    found = discover_entities(
        request["frame_path"],
        kinds=tuple(str(k) for k in request.get("kinds") or ()),
        out_dir=request["out_dir"],
        segmenter=models.segmenter,
        embedder=models.embedder,
        exclude_bboxes=list(request.get("exclude_bboxes") or []),
        **extra,
    )
    return {"discovered": found}


def _run_job(models: _Models, request: dict[str, Any]) -> dict[str, Any] | None:
    if str(request.get("job_kind", "acquire")) == "discover":
        return _run_discovery_job(models, request)

    from memstrata.skills.crop_acquisition.orchestrator import acquire_entity_crop

    extra: dict[str, Any] = {}
    for key in (
        "identity_threshold",
        "max_candidates",
        "max_character_bbox_area",
        "min_mask_fill",
        "min_side_px",
        "iou_threshold",
        "frame_paths",
        "frame_positions",
        "entity_description",
    ):
        if key in request:
            extra[key] = request[key]

    # Identity/novelty MUST be scored in the SAME (DINOv3) space as the candidates. The
    # pipeline embedder (e.g. HashEmbedding) is a different space, so we do NOT trust
    # caller-sent vectors for that; instead the caller passes the entity's existing crop
    # IMAGE paths and we re-embed them here with our DINOv3. (Raw vectors are still accepted
    # as a fallback for callers that already embed in DINOv3 space.)
    from pathlib import Path as _P

    def _embed_paths(paths: list[str]) -> list[list[float]]:
        imgs = [_P(p) for p in paths if p and _P(p).is_file()]
        return models.embedder.embed_batch(imgs) if imgs else []

    exemplar_vectors = list(request.get("exemplar_vectors") or [])
    existing_rep_vectors = list(request.get("existing_rep_vectors") or [])
    exemplar_vectors += _embed_paths(list(request.get("exemplar_image_paths") or []))
    existing_rep_vectors += _embed_paths(list(request.get("existing_rep_image_paths") or []))

    return acquire_entity_crop(
        request["frame_path"],
        entity_name=str(request["entity_name"]),
        entity_kind=str(request["entity_kind"]),
        exemplar_vectors=exemplar_vectors,
        existing_rep_vectors=existing_rep_vectors,
        out_dir=request["out_dir"],
        segmenter=models.segmenter,
        detector=models.detector,
        embedder=models.embedder,
        **extra,
    )


def serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    server_dir = Path(args.server_dir).resolve()
    pending, done = _server_dirs(server_dir)

    models = _Models(device=args.device or None)
    (server_dir / "ready").write_text(str(os.getpid()))
    logging.info("[crop_acq] server ready at %s (pid=%s)", server_dir, os.getpid())

    idle_timeout = float(args.idle_timeout)
    last_job = time.time()
    while True:
        if (server_dir / "stop").exists():
            logging.info("[crop_acq] stop sentinel found; exiting")
            break
        job_path = _next_job(pending)
        if job_path is None:
            if idle_timeout > 0 and time.time() - last_job > idle_timeout:
                logging.info("[crop_acq] idle timeout reached; exiting")
                break
            time.sleep(1.0)
            continue
        try:
            request = _read_json(job_path)
        except Exception:  # noqa: BLE001 - partial/corrupt file; skip
            time.sleep(0.2)
            continue
        job_path.unlink(missing_ok=True)
        job_id = str(request.get("job_id") or job_path.stem)
        t0 = time.time()
        try:
            result = _run_job(models, request)
            _atomic_write_json(done / f"{job_id}.json", {"job_id": job_id, "status": "ok", "result": result})
            logging.info("[crop_acq] job %s done in %.2fs (result=%s)",
                         job_id, time.time() - t0, "hit" if result else "none")
        except Exception as exc:  # noqa: BLE001 - propagate failure through result file
            logging.exception("[crop_acq] job failed: %s", job_id)
            _atomic_write_json(done / f"{job_id}.json", {"job_id": job_id, "status": "error", "error": repr(exc)})
        last_job = time.time()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent crop-acquisition server")
    parser.add_argument("--server_dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--idle_timeout", type=float, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    serve(_parse_args())
