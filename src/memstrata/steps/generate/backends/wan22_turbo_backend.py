"""Wan2.2-TI2V-5B-Turbo backend (4-step, CFG-free) via a persistent server.

Selected with ``provider = "wan22_turbo"`` in ``configs/video_gen/<name>.toml``. The model is
a distilled DiT that only runs under its upstream Self-Forcing pipeline, vendored at
``models/vendor/wan22_ti2v5b_turbo``; ``wan22_turbo_server.py`` keeps it resident because
loading costs ~5 min against ~7.5 s per 480x832/81f generation.

TI2V-5B is natively text+image conditioned, so a single resident model serves every route
MemStrata needs -- which route is used is decided by what the caller puts in ``task.controls``:
  * ``composed_references`` (the composed FLUX keyframe) -> I2V, the reference-conditioned path
  * ``continue_from_frame`` (previous segment's last frame) -> I2V continuation
  * neither -> T2V, i.e. no visual memory at all (also the generator-floor control row)
Unlike the I2V-only LightX2V backend, a missing reference is therefore a legitimate route here
rather than an error.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from memstrata.lib.prompt_standardizer import standardize_prompt
from memstrata.steps.generate.backends._support import ArtifactStore as ProjectService
from memstrata.steps.generate.backends._support import RunContext as ProjectContext
from memstrata.steps.generate.backends._support import new_id, repo_root, resolve_model_reference
from memstrata.steps.generate.backends.diffusers_backend import _load_video_gen_config
from memstrata.steps.generate.backends.vace_job_queue import server_ready, submit_job
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)


class Wan22TurboBackend:
    """Persistent-server adapter for Wan2.2-TI2V-5B-Turbo (4-step distilled)."""

    def __init__(
        self,
        context: ProjectContext,
        *,
        params: dict[str, Any],
        run_id: str,
        project_service: ProjectService | None = None,
    ) -> None:
        self.context = context
        self.params = params
        self.run_id = run_id
        self.project_service = project_service or ProjectService()
        self.model_name = resolve_model_reference(
            str(params.get("model") or "Wan-AI/Wan2.2-TI2V-5B-Turbo")
        )

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "Wan22TurboBackend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("Wan22TurboBackend only supports video_segment tasks")

        work_dir = (
            self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id
        ).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        server_dir = (
            self.context.workspace_path / "runs" / self.run_id / "wan22_turbo_server"
        ).resolve()
        self._ensure_server(server_dir)

        out_path = work_dir / "out_video.mp4"
        request = self._build_job_request(task, out_path)
        result = submit_job(
            server_dir, request, timeout=float(self.params.get("server_job_timeout", 1800))
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"wan22_turbo persistent server job failed: {result.get('error')}")

        digest, object_path = self.project_service.import_object(
            self.context, Path(result["out_video"])
        )
        notes = [
            "provider=wan22_turbo",
            "model=wan2.2_ti2v_5b_turbo",
            f"route={result.get('mode', 'unknown')}",
            "distill=4step_self_forcing",
            "cfg=off",
            f"frames={self.params.get('num_frames', 81)}",
            f"render={self.params.get('render_width', 1280)}x{self.params.get('render_height', 704)}",
            f"size={self.params.get('width', 832)}x{self.params.get('height', 480)}",
        ]
        return GenerationArtifact(
            artifact_id=new_id("artifact"),
            task_id=task.task_id,
            segment_id=task.segment_id,
            media_type="video",
            object_hash=digest,
            object_uri=str(object_path),
            model_name=self.model_name,
            degradation_notes=notes,
        )

    def _build_job_request(self, task: MediaGenerationTask, out_path: Path) -> dict[str, Any]:
        request: dict[str, Any] = {
            "prompt": standardize_prompt(task.prompt, "wan"),
            "save_file": str(out_path),
            "seed": int(self.params.get("base_seed", 2026)),
            # Render native, deliver at the benchmark's size (see the server's module docstring).
            "height": int(self.params.get("render_height", 704)),
            "width": int(self.params.get("render_width", 1280)),
            "out_height": int(self.params.get("height", 480)),
            "out_width": int(self.params.get("width", 832)),
            "num_frames": int(self.params.get("num_frames", 81)),
        }
        first_frame = self._first_frame(task)
        if first_frame is not None:
            request["first_frame_path"] = str(first_frame.resolve())
        return request

    def _first_frame(self, task: MediaGenerationTask) -> Path | None:
        """Composed references win over a raw continuation frame.

        A composed keyframe carries the resolved identities for this beat, so it is the
        stronger conditioning signal; ``continue_from_frame`` is the fallback for beats that
        continue a scene without re-composing. Neither present => T2V.
        """
        refs = _reference_image_paths(task)
        if refs:
            return refs[0]
        raw = task.controls.get("continue_from_frame")
        if raw:
            path = Path(str(raw))
            if path.is_file():
                return path
        return None

    def _ensure_server(self, server_dir: Path) -> None:
        ready = server_dir / "ready"
        if server_ready(server_dir) and _pid_alive(_read_pid(ready)):
            return
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / "stop").unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        log = open(server_dir / "server.log", "ab")  # noqa: SIM115 - owned by child process
        proc = subprocess.Popen(
            self._server_command(server_dir),
            env=self._env(),
            stdout=log,
            stderr=log,
        )
        ready_timeout = float(self.params.get("server_ready_timeout", 1800))
        poll_interval = float(self.params.get("server_ready_poll_interval", 2.0))
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"wan22_turbo server exited during startup (see {server_dir / 'server.log'})"
                )
            if server_ready(server_dir):
                return
            time.sleep(min(max(poll_interval, 0.1), max(deadline - time.time(), 0.1)))
        if proc.poll() is None:
            raise TimeoutError(
                "wan22_turbo server is still starting but did not become ready within "
                f"{ready_timeout:g}s (see {server_dir / 'server.log'})."
            )
        raise RuntimeError(
            f"wan22_turbo server exited during startup (see {server_dir / 'server.log'})"
        )

    def _server_command(self, server_dir: Path) -> list[str]:
        return [
            _resolve_python(str(self.params.get("python"))),
            str(Path(__file__).resolve().parent / "wan22_turbo_server.py"),
            "--repo", str(self._repo_path()),
            "--checkpoint_folder", str(self._checkpoint_folder()),
            "--config_path", str(self._config_path()),
            "--server_dir", str(server_dir),
            "--idle_timeout", str(self.params.get("server_idle_timeout", 1800)),
            "--height", str(self.params.get("render_height", 704)),
            "--width", str(self.params.get("render_width", 1280)),
            "--out_height", str(self.params.get("height", 480)),
            "--out_width", str(self.params.get("width", 832)),
            "--num_frames", str(self.params.get("num_frames", 81)),
            "--fps", str(self.params.get("fps", 24)),
            "--seed", str(self.params.get("base_seed", 2026)),
        ]

    def _abs_param(self, key: str, what: str) -> Path:
        raw = str(self.params.get(key, "")).strip()
        if not raw:
            raise ValueError(f"wan22_turbo backend requires `{key}` ({what})")
        path = Path(resolve_model_reference(raw))
        return path if path.is_absolute() else repo_root() / path

    def _repo_path(self) -> Path:
        return self._abs_param("repo", "the vendored Wan2.2-TI2V-5B-Turbo checkout")

    def _checkpoint_folder(self) -> Path:
        return self._abs_param("model", "the dir holding the Turbo model.pt")

    def _config_path(self) -> Path:
        raw = str(self.params.get("config_path", "")).strip()
        if not raw:
            return self._repo_path() / "configs" / "inference" / "wan22.yaml"
        path = Path(resolve_model_reference(raw))
        return path if path.is_absolute() else self._repo_path() / path

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        ff = str(self.params.get("ffmpeg", "")).strip()
        if ff and Path(ff).exists():
            env["IMAGEIO_FFMPEG_EXE"] = ff
        min_free_mib = int(self.params.get("min_free_mib", 40000))
        visible = env.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            from memstrata.lib.gpu import (
                ensure_cuda_visible_devices_have_min_free,
                freest_visible_device,
            )

            # The server occupies exactly ONE card, so when several are visible it should take the
            # emptiest rather than demand the floor from all of them. That distinction matters on a
            # packed node: a story is given its own generation card plus the node's shared services
            # card (so the small crop server can reach it), and the shared card is by design nearly
            # full. Requiring every visible card to clear the floor failed every such story.
            narrowed = freest_visible_device(visible, min_free_mib=min_free_mib)
            if narrowed is not None:
                env["CUDA_VISIBLE_DEVICES"] = narrowed
            else:
                ensure_cuda_visible_devices_have_min_free(env, min_free_mib=min_free_mib)
        elif self.params.get("auto_pick_gpu", True):
            from memstrata.lib.gpu import cuda_visible_devices_for

            picked = cuda_visible_devices_for(1, min_free_mib=min_free_mib)
            if picked is None:
                raise RuntimeError(
                    f"No GPU has enough free memory for wan22_turbo (min_free_mib={min_free_mib})."
                )
            env["CUDA_VISIBLE_DEVICES"] = picked
        return env


def _reference_image_paths(task: MediaGenerationTask) -> list[Path]:
    refs = task.controls.get("composed_references") or []
    paths: list[Path] = []
    for ref in refs:
        uri = ref.get("image") if isinstance(ref, dict) else ref
        if not uri:
            continue
        path = Path(str(uri))
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            paths.append(path)
    return paths


def _resolve_python(raw: str) -> str:
    if not raw or raw == "None":
        raise ValueError(
            "wan22_turbo backend requires `python` (an interpreter with torch + flash-attn, "
            "e.g. torch + flash-attn)"
        )
    path = Path(raw)
    return str(path if path.is_absolute() else repo_root() / path)


def _read_pid(ready_file: Path) -> int:
    try:
        return int(ready_file.read_text().strip())
    except (OSError, ValueError):
        return -1


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
