"""LTX-2.3 video generation backend.

This backend interfaces with the official LTX-2.3 codebase located in
``submodules/_2_generation/reference/LTX-2`` to run inference.
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
from memstrata.steps.generate.backends._support import RunContext as ProjectContext, new_id
from memstrata.lib.prompt_standardizer import standardize_prompt
from memstrata.steps.generate.backends.diffusers_backend import _load_video_gen_config
from memstrata.steps.generate.backends.vace_job_queue import server_ready, submit_job
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)


class Ltx23Backend:
    """Shell adapter for official LTX-2.3 inference code."""

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
        self.model_name = resolve_model_reference(str(params.get("model") or "ltx23"))

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "Ltx23Backend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("Ltx23Backend only supports video_segment tasks")

        work_dir = self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id
        work_dir.mkdir(parents=True, exist_ok=True)

        out_path = work_dir / "out_video.mp4"
        notes: list[str] = ["provider=ltx23"]

        # Transform prompt into optimal natural language caption for LTX-2.3
        transformed_prompt = self._transform_prompt(task.prompt)
        notes.append(f"transformed_prompt={transformed_prompt}")

        if self.params.get("serve_persistent"):
            out_path, server_notes = self._generate_via_server(task, work_dir, transformed_prompt)
            notes.extend(server_notes)
        else:
            # Handle continuation or reference image conditioning (fallback)
            first_frame_path = self._prepare_first_frame_condition(task, work_dir)
            command = self._command(task, out_path, first_frame_path, transformed_prompt)
            subprocess.run(command, cwd=work_dir, env=self._env(), check=True)

        digest, object_path = self.project_service.import_object(self.context, out_path)
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

    def _prepare_first_frame_condition(self, task: MediaGenerationTask, work_dir: Path) -> Path | None:
        """Extract the last frame of the previous video if continuing, or use the first reference image."""
        continuation = task.controls.get("continuation")
        if isinstance(continuation, dict) and continuation.get("source_video"):
            prev_video = Path(str(continuation["source_video"]))
            if prev_video.is_file():
                import cv2  # noqa: PLC0415
                cap = cv2.VideoCapture(str(prev_video))
                last_frame = None
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    last_frame = frame
                cap.release()
                if last_frame is not None:
                    # Keep conditioning frames lossless and avoid interpolation blur when possible.
                    width = int(self.params.get("width", 832))
                    height = int(self.params.get("height", 480))
                    if (last_frame.shape[1], last_frame.shape[0]) != (width, height):
                        interpolation = _cv2_resize_interpolation(
                            str(self.params.get("image_conditioning_resize_interpolation", "nearest_exact"))
                        )
                        last_frame = cv2.resize(last_frame, (width, height), interpolation=interpolation)
                    first_frame_path = work_dir / "first_frame_cond.png"
                    cv2.imwrite(str(first_frame_path), last_frame)
                    return first_frame_path

        # Fallback to the first composed reference image
        refs = task.controls.get("composed_references") or []
        for ref in refs:
            uri = ref.get("image") if isinstance(ref, dict) else ref
            if uri:
                path = Path(str(uri))
                if path.is_file():
                    return path

        return None

    def _command(self, task: MediaGenerationTask, out_path: Path, first_frame_path: Path | None, transformed_prompt: str) -> list[str]:
        python_val = self.params.get("python", sys.executable)
        if python_val.startswith("/"):
            python = python_val
        else:
            python = str((repo_root() / python_val).absolute())

        entrypoint_val = self.params.get("entrypoint")
        if entrypoint_val.startswith("/"):
            entrypoint = entrypoint_val
        else:
            entrypoint = str((repo_root() / entrypoint_val).absolute())

        command = [
            python,
            entrypoint,
            "--checkpoint-path", str(self.params["model"]),
            "--gemma-root", str(self.params["gemma_root"]),
            "--prompt", transformed_prompt,
            "--height", str(self.params.get("height", 480)),
            "--width", str(self.params.get("width", 832)),
            "--num-frames", str(self.params.get("frame_num", 49)),
            "--frame-rate", str(self.params.get("fps", 16)),
            "--output-path", str(out_path),
            "--num-inference-steps", str(self.params.get("num_inference_steps", 8)),
            "--video-cfg-guidance-scale", str(self.params.get("video_cfg_guidance_scale", 1.0)),
            "--video-stg-guidance-scale", str(self.params.get("video_stg_guidance_scale", 0.0)),
        ]

        if first_frame_path:
            command += ["--image", str(first_frame_path), "0", "1.0", str(self._image_conditioning_crf())]

        return command

    def _transform_prompt(self, structured_prompt: str) -> str:
        """Transform a structured screenplay prompt into a high-quality natural language caption for LTX-2.3."""
        return standardize_prompt(structured_prompt, "ltx23")

    def _image_conditioning_crf(self) -> int:
        crf = int(self.params.get("image_conditioning_crf", 0))
        if not 0 <= crf <= 51:
            raise ValueError("image_conditioning_crf must be in [0, 51]")
        return crf

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        ltx_root = (repo_root() / "submodules" / "_2_generation" / "reference" / "LTX-2").resolve()
        venv_path = ltx_root / ".venv"
        env["VIRTUAL_ENV"] = str(venv_path)
        env["PATH"] = os.pathsep.join([str(venv_path / "bin"), env.get("PATH", "")])

        pythonpath = [
            str(ltx_root / "packages" / "ltx-core" / "src"),
            str(ltx_root / "packages" / "ltx-pipelines" / "src"),
            str(repo_root() / "src"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(pythonpath + [env.get("PYTHONPATH", "")])
        return env

    # -- persistent server path ---------------------------------------------------------------

    def _generate_via_server(self, task: MediaGenerationTask, work_dir: Path, transformed_prompt: str) -> tuple[Path, list[str]]:
        server_dir = self.context.workspace_path / "runs" / self.run_id / "ltx23_server"
        self._ensure_server(server_dir)
        out_path = work_dir / "out_video.mp4"
        request = self._build_job_request(task, work_dir, out_path, transformed_prompt)
        result = submit_job(
            server_dir,
            request,
            timeout=float(self.params.get("server_job_timeout", 3600)),
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"LTX-2.3 persistent server job failed: {result.get('error')}")
        return Path(result["out_video"]), ["serve=persistent"]

    def _build_job_request(self, task: MediaGenerationTask, work_dir: Path, out_path: Path, transformed_prompt: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "prompt": transformed_prompt,
            "save_file": str(out_path),
            "height": int(self.params.get("height", 480)),
            "width": int(self.params.get("width", 832)),
            "num_frames": int(self.params.get("frame_num", 49)),
            "frame_rate": float(self.params.get("fps", 16.0)),
            "num_inference_steps": int(self.params.get("num_inference_steps", 8)),
            "video_cfg_guidance_scale": float(self.params.get("video_cfg_guidance_scale", 1.0)),
            "video_stg_guidance_scale": float(self.params.get("video_stg_guidance_scale", 0.0)),
            "seed": int(self.params.get("base_seed", 42)),
            "image_conditioning_crf": self._image_conditioning_crf(),
        }

        # Handle continuation or reference image conditioning
        first_frame_path = self._prepare_first_frame_condition(task, work_dir)
        if first_frame_path:
            request["first_frame_path"] = str(first_frame_path)

        # Handle true Video-to-Video and Audio-to-Audio continuation (prefix conditioning)
        continuation = task.controls.get("continuation")
        if isinstance(continuation, dict) and continuation.get("source_video"):
            prev_video = Path(str(continuation["source_video"]))
            if prev_video.is_file():
                request["video_prefix_path"] = str(prev_video)
                request["audio_prefix_path"] = str(prev_video)

        return request

    def _ensure_server(self, server_dir: Path) -> None:
        ready = server_dir / "ready"
        if server_ready(server_dir) and _pid_alive(_read_pid(ready)):
            return
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / "stop").unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        log = open(server_dir / "server.log", "ab")  # noqa: SIM115
        proc = subprocess.Popen(self._server_command(server_dir), cwd=str(server_dir), env=self._env(), stdout=log, stderr=log)
        ready_timeout = float(self.params.get("server_ready_timeout", 1200))
        poll_interval = float(self.params.get("server_ready_poll_interval", 2.0))
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"LTX-2.3 persistent server exited during startup (see {server_dir / 'server.log'})"
                )
            if server_ready(server_dir):
                return
            time.sleep(min(max(poll_interval, 0.1), max(deadline - time.time(), 0.1)))
        if proc.poll() is None:
            raise TimeoutError(
                "LTX-2.3 persistent server is still starting but did not become ready "
                f"within {ready_timeout:g}s (see {server_dir / 'server.log'})."
            )
        raise RuntimeError(f"LTX-2.3 persistent server exited during startup (see {server_dir / 'server.log'})")

    def _server_command(self, server_dir: Path) -> list[str]:
        python_val = self.params.get("python", sys.executable)
        if python_val.startswith("/"):
            python = python_val
        else:
            python = str((repo_root() / python_val).absolute())

        return [
            python,
            "-m",
            "memstrata.steps.generate.backends.ltx23_persistent_server",
            "--checkpoint_path", str(self.params["model"]),
            "--gemma_root", str(self.params["gemma_root"]),
            "--server_dir", str(server_dir),
            "--idle_timeout", str(self.params.get("server_idle_timeout", 1800)),
        ]


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


def _cv2_resize_interpolation(name: str) -> int:
    import cv2  # noqa: PLC0415

    normalized = name.lower().replace("-", "_")
    if normalized == "nearest_exact":
        return getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    modes = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    try:
        return modes[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported image_conditioning_resize_interpolation: {name}") from exc
