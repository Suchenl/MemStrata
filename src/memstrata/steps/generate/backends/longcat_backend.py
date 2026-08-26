"""LongCat-Video backend adapter.

LongCat-Video is not available as an upstream Diffusers pipeline yet, so MemStrata
talks to the official checkout through a small persistent file-queue server.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from memstrata.steps.generate.backends._support import ArtifactStore as ProjectService
from memstrata.steps.generate.backends._support import repo_root, resolve_model_reference
from memstrata.lib.prompt_standardizer import standardize_prompt
from memstrata.steps.generate.backends._support import RunContext as ProjectContext, new_id
from memstrata.steps.generate.backends.diffusers_backend import _load_video_gen_config
from memstrata.steps.generate.backends.vace_job_queue import server_ready, submit_job
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)


class LongCatBackend:
    """Persistent-server adapter for Meituan LongCat-Video."""

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
        self.model_name = resolve_model_reference(str(params.get("model") or "meituan-longcat/LongCat-Video"))

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "LongCatBackend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("LongCatBackend only supports video_segment tasks")

        work_dir = self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id
        work_dir.mkdir(parents=True, exist_ok=True)
        if not self.params.get("serve_persistent", True):
            raise ValueError("LongCatBackend requires serve_persistent=true to avoid per-segment reloads")

        server_dir = self.context.workspace_path / "runs" / self.run_id / "longcat_server"
        self._ensure_server(server_dir)
        out_path = work_dir / "out_video.mp4"
        request = self._build_job_request(task, out_path)
        result = submit_job(
            server_dir,
            request,
            timeout=float(self.params.get("server_job_timeout", 3600)),
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"LongCat persistent server job failed: {result.get('error')}")

        digest, object_path = self.project_service.import_object(self.context, Path(result["out_video"]))
        notes = ["provider=longcat", "serve=persistent", f"mode={request['mode']}"]
        if request["mode"] == "vc":
            notes.append(f"continuation_conditioning={request['continuation_conditioning']}")
            notes.append(f"num_cond_frames={request['num_cond_frames']}")
            notes.append(f"min_condition_duration_sec={request['min_condition_duration_sec']}")
        if self.params.get("use_distill", True):
            notes.append("distill=cfg_step_lora")
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
        continuation = task.controls.get("continuation")
        refs = _reference_image_paths(task)
        mode = "t2v"
        source_video = None
        first_frame_path = None
        if isinstance(continuation, dict) and continuation.get("source_video"):
            mode = "vc"
            source_video = str(continuation["source_video"])
        elif refs:
            mode = "i2v"
            first_frame_path = str(refs[0])

        return {
            "prompt": standardize_prompt(task.prompt, "longcat"),
            "negative_prompt": str(self.params.get("negative_prompt", _DEFAULT_NEGATIVE_PROMPT)),
            "save_file": str(out_path),
            "mode": mode,
            "source_video": source_video,
            "first_frame_path": first_frame_path,
            "continuation_conditioning": "tail",
            "height": int(self.params.get("height", 480)),
            "width": int(self.params.get("width", 832)),
            "num_frames": int(self.params.get("frame_num", 49)),
            "fps": int(self.params.get("fps", 15)),
            "num_cond_frames": int(self.params.get("num_cond_frames", 9)),
            "min_condition_duration_sec": float(self.params.get("min_condition_duration_sec", 1.0)),
            "num_inference_steps": int(self.params.get("num_inference_steps", 16)),
            "guidance_scale": float(self.params.get("guidance_scale", 1.0)),
            "seed": int(self.params.get("base_seed", 42)),
            "use_distill": bool(self.params.get("use_distill", True)),
            "use_kv_cache": bool(self.params.get("use_kv_cache", True)),
            "offload_kv_cache": bool(self.params.get("offload_kv_cache", False)),
            "enhance_hf": bool(self.params.get("enhance_hf", False)),
        }

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
            cwd=str(repo_root()),
            env=self._env(),
            stdout=log,
            stderr=log,
        )
        ready_timeout = float(self.params.get("server_ready_timeout", 1200))
        poll_interval = float(self.params.get("server_ready_poll_interval", 2.0))
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"LongCat persistent server exited during startup (see {server_dir / 'server.log'})"
                )
            if server_ready(server_dir):
                return
            time.sleep(min(max(poll_interval, 0.1), max(deadline - time.time(), 0.1)))
        if proc.poll() is None:
            raise TimeoutError(
                "LongCat persistent server is still starting but did not become ready "
                f"within {ready_timeout:g}s (see {server_dir / 'server.log'})."
            )
        raise RuntimeError(f"LongCat persistent server exited during startup (see {server_dir / 'server.log'})")

    def _server_command(self, server_dir: Path) -> list[str]:
        python = _resolve_python(str(self.params.get("python", sys.executable)))
        nproc = int(self.params.get("nproc_per_node", 1))
        if nproc != 1:
            raise ValueError("LongCatBackend currently supports nproc_per_node=1 only")
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "1",
            str(Path(__file__).resolve().parent / "longcat_persistent_server.py"),
            "--checkpoint_dir",
            str(self.model_name),
            "--server_dir",
            str(server_dir),
            "--idle_timeout",
            str(self.params.get("server_idle_timeout", 1800)),
        ]
        if self.params.get("enable_compile"):
            command.append("--enable_compile")
        return command

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        roots = [
            str(_required_path(self.params, "code_root")),
            str(repo_root() / "src"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(roots + [env.get("PYTHONPATH", "")])
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        min_free_mib = int(self.params.get("min_free_mib", self.params.get("required_free_mib", 4000)))
        if env.get("CUDA_VISIBLE_DEVICES"):
            from memstrata.lib.gpu import ensure_cuda_visible_devices_have_min_free

            ensure_cuda_visible_devices_have_min_free(env, min_free_mib=min_free_mib)
        elif self.params.get("auto_pick_gpu", True):
            from memstrata.lib.gpu import cuda_visible_devices_for

            picked = cuda_visible_devices_for(1, min_free_mib=min_free_mib)
            if picked is not None:
                env["CUDA_VISIBLE_DEVICES"] = picked
            else:
                raise RuntimeError(
                    f"No GPU has enough free memory for LongCat service (min_free_mib={min_free_mib}). "
                    "Use a freer GPU/node instead of squeezing onto a busy card."
                )
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


def _required_path(params: dict[str, Any], key: str) -> Path:
    raw = str(params.get(key, "")).strip()
    if not raw:
        raise ValueError(f"LongCat backend requires `{key}` in video_gen config")
    path = Path(resolve_model_reference(raw))
    return path if path.is_absolute() else repo_root() / path


def _resolve_python(raw: str) -> str:
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


_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)
