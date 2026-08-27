"""Production ``Cropper`` that drives the persistent crop-acquisition server.

``ProposeIdentifyCropper`` plugs into ``memstrata.steps.decompose.RoleAwareDecomposer``
as the ``Cropper`` protocol (``crop(segment_video, entity, *, segment_id) -> str | None``),
but instead of a single VLM grounding call it runs the S5-derived
SAM3+GroundingDINO+DINOv3 propose/identify/novelty perception (see ``orchestrator``) out
of process, so the ~model cold start is paid once by ``crop_server``.

Per crop() call:
  1. Sample several frames (default 0.2/0.5/0.8) from the realized segment video
     (imageio.v3), so one occluded/back-facing instant does not permanently miss.
  2. Look up the entity's exemplar + existing representation vectors from the bank
     (``rep.annotations['embedding']``) via ``entity.entity_id``.
  3. Submit a job to the server's ``pending/`` queue; poll ``done/`` for the result.
  4. Return the acquired crop_path (or None).

ALL paths handed to the server are resolved to ABSOLUTE (work_dir / server_dir /
frame_path) — the server subprocess may run with a different cwd, and a relative path
would make it write/read under the wrong dir (this exact bug bit HeliosBackend).

Imports stay light (no torch/transformers): the model classes live behind the server
process; imageio/PIL are imported lazily inside frame sampling.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memstrata.skills.crop_acquisition.orchestrator import (
    DEFAULT_IDENTITY_THRESHOLD,
    _MIN_SIDE_PX,
)
from memstrata.skills.crop_acquisition._common import sam3_deps_dir

if TYPE_CHECKING:  # avoid heavy / cyclic imports at runtime
    from memstrata.bank import AssetBank
    from memstrata.steps.decompose import NamedEntity

logger = logging.getLogger(__name__)

_DEFAULT_MONTAGE_ROOT = "."
# The vendored sam3_transformers59 bundle is compiled for CPython 3.11 (transformers 5.9 +
# hf_hub/pydantic/regex/tiktoken .so are all cp311). The server subprocess therefore MUST
# run under CPython 3.11 + torch. Override with MEMSTRATA_PYTHON. The *client* can still
# run under any interpreter (it only samples frames + talks to the queue).
_DEFAULT_PYTHON = os.environ.get("MEMSTRATA_PYTHON") or "python3"
_DEFAULT_FRAME_POSITIONS = (0.2, 0.5, 0.8)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ProposeIdentifyCropper:
    """S5-derived propose/identify/novelty cropper backed by a persistent server."""

    def __init__(
        self,
        bank: "AssetBank",
        *,
        server_dir: str | Path,
        work_dir: str | Path = "/tmp/memstrata_crop_acq",
        montage_root: str | Path = _DEFAULT_MONTAGE_ROOT,
        python: str | Path = _DEFAULT_PYTHON,
        sam3_deps: str | Path | None = None,
        public_models_root: str | Path | None = None,
        frame_pos: float = 0.8,
        frame_positions: tuple[float, ...] | list[float] | None = None,
        auto_start: bool = True,
        identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
        job_timeout: float = 1800.0,
        server_ready_timeout: float = 1200.0,
        device: str = "",
        extra_acquire_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.bank = bank
        # Absolute paths — the server subprocess runs with a different cwd (see class docstring).
        self.server_dir = Path(server_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.montage_root = Path(montage_root).resolve()
        self.python = str(python)
        configured_sam3_deps = str(sam3_deps) if sam3_deps else sam3_deps_dir()
        self.sam3_deps = Path(configured_sam3_deps).resolve() if configured_sam3_deps else None
        self.public_models_root = str(public_models_root or "${PUBLIC_MODELS_ROOT}")
        self.frame_pos = min(max(float(frame_pos), 0.0), 1.0)
        positions = frame_positions if frame_positions is not None else _DEFAULT_FRAME_POSITIONS
        cleaned = [min(max(float(p), 0.0), 1.0) for p in positions]
        self.frame_positions = tuple(dict.fromkeys(cleaned)) or (self.frame_pos,)
        self.auto_start = bool(auto_start)
        self.identity_threshold = float(identity_threshold)
        self.job_timeout = float(job_timeout)
        self.server_ready_timeout = float(server_ready_timeout)
        self.device = str(device)
        self.extra_acquire_kwargs = dict(extra_acquire_kwargs or {})
        self._proc: subprocess.Popen | None = None
        self._stats_by_entity: dict[str, dict[str, Any]] = {}

    # --- server lifecycle -------------------------------------------------------------

    def _server_ready(self) -> bool:
        ready = self.server_dir / "ready"
        if not ready.exists():
            return False
        try:
            pid = int(ready.read_text().strip())
        except (OSError, ValueError):
            return False
        return _pid_alive(pid)

    def _ensure_server(self) -> None:
        if self._server_ready():
            return
        if not self.auto_start:
            raise RuntimeError(
                f"crop-acquisition server not ready at {self.server_dir} and auto_start=False"
            )
        self.server_dir.mkdir(parents=True, exist_ok=True)
        (self.server_dir / "stop").unlink(missing_ok=True)
        (self.server_dir / "ready").unlink(missing_ok=True)
        log = open(self.server_dir / "server.log", "ab")  # noqa: SIM115 - owned by child
        self._proc = subprocess.Popen(
            self._server_command(),
            cwd=str(self.montage_root),
            env=self._server_env(),
            stdout=log,
            stderr=log,
        )
        deadline = time.time() + self.server_ready_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"crop-acquisition server exited during startup "
                    f"(see {self.server_dir / 'server.log'})"
                )
            if self._server_ready():
                return
            time.sleep(2.0)
        raise TimeoutError(
            f"crop-acquisition server did not become ready within {self.server_ready_timeout:g}s "
            f"(see {self.server_dir / 'server.log'})"
        )

    def _server_command(self) -> list[str]:
        return [
            self.python,
            "-m",
            "memstrata.skills.crop_acquisition.crop_server",
            "--server_dir",
            str(self.server_dir),
            "--device",
            self.device,
        ]

    def _server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Vendored transformers>=5.9 first when available, then this standalone repo's src.
        pythonpath = os.pathsep.join(
            ([str(self.sam3_deps)] if self.sam3_deps else [])
            + [
                str(self.montage_root / "src"),
                env.get("PYTHONPATH", ""),
            ],
        )
        env["PYTHONPATH"] = pythonpath
        env["MONTAGE_ROOT"] = str(self.montage_root)
        env["PUBLIC_MODELS_ROOT"] = self.public_models_root
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        return env

    # --- bank vectors -----------------------------------------------------------------

    def _entity_images(self, entity_id: str) -> tuple[list[str], list[str]]:
        """Return (exemplar_image_paths, existing_rep_image_paths) for an entity from the bank.

        We pass the existing crop IMAGE paths (rep.object_uri), NOT the bank's stored
        embeddings: the pipeline embedder (e.g. HashEmbedding) lives in a different space
        than the crop server's DINOv3, so the server re-embeds these images with its own
        DINOv3 to keep identity/novelty scoring in one space.

        exemplar_image_paths: identity-anchor-eligible, non-deprecated reps
          (used only to confirm "this is our entity").
        existing_rep_image_paths: ALL non-deprecated reps
          (used to score novelty so we record NEW content, not old).
        """
        exemplar: list[str] = []
        existing: list[str] = []
        asset = self.bank.get_asset(entity_id) if entity_id else None
        if asset is None:
            return exemplar, existing
        for rep in asset.representations:
            if getattr(rep, "deprecated", False):
                continue
            uri = getattr(rep, "object_uri", None)
            if not uri:
                continue
            existing.append(str(uri))
            if (rep.annotations or {}).get("identity_anchor_eligible") is not False:
                exemplar.append(str(uri))
        return exemplar, existing

    # --- frame sampling ---------------------------------------------------------------

    def _sample_frame(self, segment_video: str, out_path: Path, *, frame_pos: float | None = None) -> bool:
        sampled = self._sample_frames(
            segment_video,
            [(self.frame_pos if frame_pos is None else float(frame_pos), out_path)],
        )
        return bool(sampled)

    def _sample_frames(self, segment_video: str, targets: list[tuple[float, Path]]) -> list[Path]:
        try:
            import imageio.v3 as iio
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            logger.warning("ProposeIdentifyCropper: imageio/PIL unavailable (%s)", exc)
            return []
        try:
            frames = iio.imread(segment_video, index=None)  # (T, H, W, 3)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ProposeIdentifyCropper: cannot read %s (%s)", segment_video, exc)
            return []
        if frames is None or len(frames) == 0:
            return []
        saved: list[Path] = []
        for pos, out_path in targets:
            idx = min(
                len(frames) - 1,
                int(round(min(max(float(pos), 0.0), 1.0) * (len(frames) - 1))),
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frames[idx]).convert("RGB").save(out_path)
            saved.append(out_path)
        return saved

    # --- job submission ---------------------------------------------------------------

    def _submit_and_wait(self, request: dict[str, Any]) -> dict[str, Any]:
        pending = self.server_dir / "pending"
        done = self.server_dir / "done"
        pending.mkdir(parents=True, exist_ok=True)
        done.mkdir(parents=True, exist_ok=True)
        job_id = str(request.get("job_id") or uuid.uuid4().hex)
        request = {**request, "job_id": job_id}
        result_path = done / f"{job_id}.json"
        _atomic_write_json(pending / f"{job_id}.json", request)
        deadline = time.time() + self.job_timeout
        while time.time() < deadline:
            if result_path.exists():
                result = json.loads(result_path.read_text())
                result_path.unlink(missing_ok=True)
                return result
            time.sleep(0.5)
        raise TimeoutError(f"crop-acquisition job {job_id} timed out after {self.job_timeout}s")

    # --- Cropper protocol -------------------------------------------------------------

    def frame_for_segment(self, segment_video: str, *, segment_id: int) -> Path | None:
        """Sample (once per segment) the frame both acquisition and discovery work from.

        Cached on disk so enabling discovery does not re-decode the segment, and so both
        paths reason about the *same* pixels — otherwise "already explained by a requested
        crop" would compare boxes from two different frames.
        """
        frame_path = (self.work_dir / f"segment_{segment_id:03d}" / "frame.jpg").resolve()
        if frame_path.is_file() and frame_path.stat().st_size > 0:
            return frame_path
        return frame_path if self._sample_frame(segment_video, frame_path) else None

    @staticmethod
    def _frame_tag(frame_pos: float) -> str:
        return f"p{int(round(frame_pos * 1000)):03d}"

    def frame_paths_for_segment(self, segment_video: str, *, segment_id: int) -> tuple[list[Path], list[float]]:
        """Sample/cache the multi-timepoint candidate pool for requested acquisition."""
        segment_dir = (self.work_dir / f"segment_{segment_id:03d}").resolve()
        targets = [
            (pos, segment_dir / f"frame_{self._frame_tag(pos)}.jpg")
            for pos in self.frame_positions
        ]
        cached = {
            path
            for _, path in targets
            if path.is_file() and path.stat().st_size > 0
        }
        missing = [(pos, path) for pos, path in targets if path not in cached]
        if missing:
            self._sample_frames(segment_video, missing)
        paths: list[Path] = []
        positions: list[float] = []
        for pos, path in targets:
            if path.is_file() and path.stat().st_size > 0:
                paths.append(path)
                positions.append(pos)
        if paths:
            return paths, positions
        fallback = self.frame_for_segment(segment_video, segment_id=segment_id)
        return ([fallback], [self.frame_pos]) if fallback is not None else ([], [])

    # --- diagnostics / manifest --------------------------------------------------------

    def _record_attempt(self, entity_id: str, *, hit: bool, payload: dict[str, Any] | None) -> None:
        row = self._stats_by_entity.setdefault(
            entity_id,
            {"attempts": 0, "hits": 0, "misses": 0, "identity_sims": [], "sources": {}},
        )
        row["attempts"] += 1
        if hit:
            row["hits"] += 1
            source = str((payload or {}).get("source") or "unknown")
            row["sources"][source] = int(row["sources"].get(source, 0)) + 1
            sim = (payload or {}).get("identity_sim")
            if sim is not None:
                row["identity_sims"].append(float(sim))
        else:
            row["misses"] += 1
        self._write_summary()

    def _write_summary(self) -> None:
        entities: dict[str, dict[str, Any]] = {}
        for entity_id, row in self._stats_by_entity.items():
            attempts = max(1, int(row["attempts"]))
            sims = list(row.get("identity_sims") or [])
            entities[entity_id] = {
                **row,
                "miss_rate": float(row["misses"]) / attempts,
                "identity_sim_min": min(sims) if sims else None,
                "identity_sim_mean": (sum(sims) / len(sims)) if sims else None,
            }
        summary = {
            "config": {
                "identity_threshold": self.identity_threshold,
                "frame_positions": list(self.frame_positions),
                "min_side_px": int(self.extra_acquire_kwargs.get("min_side_px", _MIN_SIDE_PX)),
                "max_character_bbox_area": self.extra_acquire_kwargs.get("max_character_bbox_area", 1.0),
                "min_mask_fill": self.extra_acquire_kwargs.get("min_mask_fill", 0.18),
            },
            "entities": entities,
        }
        _atomic_write_json(self.work_dir / "crop_acquisition_summary.json", summary)

    def crop(self, segment_video: str, entity: "NamedEntity", *, segment_id: int) -> dict | None:
        entity_id = getattr(entity, "entity_id", None) or f"{getattr(entity.kind, 'value', entity.kind)}_{entity.name}"
        out_dir = (self.work_dir / f"segment_{segment_id:03d}" / str(entity_id)).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths, frame_positions = self.frame_paths_for_segment(segment_video, segment_id=segment_id)
        if not frame_paths:
            self._record_attempt(str(entity_id), hit=False, payload=None)
            return None

        exemplar_image_paths, existing_rep_image_paths = self._entity_images(str(entity_id))
        try:
            self._ensure_server()
        except Exception as exc:  # noqa: BLE001 - server failure must not abort the loop
            logger.warning("ProposeIdentifyCropper: server unavailable (%s)", exc)
            self._record_attempt(str(entity_id), hit=False, payload=None)
            return None

        request = {
            "frame_path": str(frame_paths[0].resolve()),
            "frame_paths": [str(path.resolve()) for path in frame_paths],
            "frame_positions": frame_positions,
            "entity_name": str(entity.name),
            "entity_kind": str(getattr(entity.kind, "value", entity.kind)),
            "entity_description": str(getattr(entity, "description", "") or ""),
            "exemplar_image_paths": exemplar_image_paths,
            "existing_rep_image_paths": existing_rep_image_paths,
            "out_dir": str(out_dir),
            "identity_threshold": self.identity_threshold,
            **self.extra_acquire_kwargs,
        }
        try:
            result = self._submit_and_wait(request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ProposeIdentifyCropper: job failed for %s (%s)", entity_id, exc)
            self._record_attempt(str(entity_id), hit=False, payload=None)
            return None

        if result.get("status") != "ok":
            logger.info("ProposeIdentifyCropper: server error for %s: %s", entity_id, result.get("error"))
            self._record_attempt(str(entity_id), hit=False, payload=None)
            return None
        payload = result.get("result")
        if not payload:
            logger.info("ProposeIdentifyCropper: no identity-OK/novel crop for %s in segment %s",
                        entity_id, segment_id)
            self._record_attempt(str(entity_id), hit=False, payload=None)
            return None
        crop_path = payload.get("crop_path")
        logger.info(
            "ProposeIdentifyCropper: segment %s %s -> %s (identity_sim=%s novelty=%s source=%s)",
            segment_id, entity_id, crop_path,
            payload.get("identity_sim"), payload.get("novelty_score"), payload.get("source"),
        )
        if not crop_path:
            self._record_attempt(str(entity_id), hit=False, payload=payload)
            return None
        self._record_attempt(str(entity_id), hit=True, payload=payload)
        # Report the bbox too: discovery needs it to tell "region already acquired for a
        # named entity" from "genuinely new region".
        return {
            "crop_path": str(crop_path),
            "bbox": payload.get("bbox"),
            "meta": {
                "crop_acquisition": {
                    "identity_threshold": payload.get("identity_threshold", self.identity_threshold),
                    "identity_sim": payload.get("identity_sim"),
                    "identity_gate": payload.get("identity_gate"),
                    "novelty_score": payload.get("novelty_score"),
                    "source": payload.get("source"),
                    "source_detail": payload.get("source_detail"),
                    "frame_position": payload.get("frame_position"),
                    "candidate_count": payload.get("candidate_count"),
                    "min_side_px": payload.get("min_side_px"),
                    "max_character_bbox_area": payload.get("max_character_bbox_area"),
                    "min_mask_fill": payload.get("min_mask_fill"),
                }
            },
        }

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Submit an arbitrary job to this cropper's server (used by discovery)."""
        self._ensure_server()
        return self._submit_and_wait(request)


class ServerConceptDiscoverer:
    """``Discoverer`` backed by the same crop-acquisition server as the cropper.

    Sharing the server matters: the perception models are the expensive part, so
    discovery adds proposals per frame rather than a second model load. It also
    guarantees both paths see the identical cached frame, which is what makes the
    ``covered`` bbox comparison meaningful.
    """

    def __init__(
        self,
        cropper: ProposeIdentifyCropper,
        *,
        work_dir: str | Path,
        max_per_kind: int = 3,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.cropper = cropper
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_per_kind = int(max_per_kind)
        self.extra_kwargs = dict(extra_kwargs or {})

    def discover(self, segment_video: str, *, segment_id: int, kinds: tuple) -> list:
        from memstrata.skills.decomposition import DiscoveredEntity
        from memstrata.bank import AssetType

        frame_path = self.cropper.frame_for_segment(segment_video, segment_id=segment_id)
        if frame_path is None:
            return []
        out_dir = (self.work_dir / f"segment_{segment_id:03d}").resolve()
        request = {
            "job_kind": "discover",
            "frame_path": str(frame_path),
            "kinds": [str(getattr(k, "value", k)) for k in kinds],
            "out_dir": str(out_dir),
            "exclude_bboxes": self._requested_bboxes(segment_id),
            "max_per_kind": self.max_per_kind,
            **self.extra_kwargs,
        }
        try:
            result = self.cropper.submit(request)
        except Exception as exc:  # noqa: BLE001 - discovery must never fail a segment
            logger.warning("ServerConceptDiscoverer: job failed for segment %s (%s)", segment_id, exc)
            return []
        if result.get("status") != "ok":
            logger.info("ServerConceptDiscoverer: server error: %s", result.get("error"))
            return []
        rows = (result.get("result") or {}).get("discovered") or []
        out = []
        for row in rows:
            try:
                kind = AssetType(str(row["kind"]))
            except (KeyError, ValueError):
                continue
            out.append(
                DiscoveredEntity(
                    kind=kind,
                    crop_path=str(row["crop_path"]),
                    bbox_norm=list(row.get("bbox") or []) or None,
                    quality=float((row.get("qa") or {}).get("area_fraction", 1.0) or 1.0),
                    meta={
                        "discovery_concept": row.get("concept", ""),
                        "discovery_score": row.get("score"),
                    },
                )
            )
        logger.info("ServerConceptDiscoverer: segment %s discovered %s regions", segment_id, len(out))
        return out

    def _requested_bboxes(self, segment_id: int) -> list[list[int]]:
        """Boxes already acquired for named entities in this segment (best-effort)."""
        boxes: list[list[int]] = []
        segment_dir = self.cropper.work_dir / f"segment_{segment_id:03d}"
        if not segment_dir.is_dir():
            return boxes
        for asset in self.cropper.bank.assets.values():
            for rep in asset.representations:
                if int(getattr(rep, "origin_segment_id", -1)) != segment_id:
                    continue
                bbox = (rep.annotations or {}).get("bbox")
                if bbox and len(bbox) == 4:
                    boxes.append([int(v) for v in bbox])
        return boxes


__all__ = ["ProposeIdentifyCropper", "ServerConceptDiscoverer"]
