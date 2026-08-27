"""Wan2.2-I2V-A14B (MoE) 4-step distilled backend via LightX2V.

Drop-in replacement for HeliosBackend with much higher fidelity: a dual-expert 14B model
run in 4 CFG-free steps (2 high + 2 low). Same MemStrata contract -- it animates the composed
FLUX keyframe (``task.controls["composed_references"][0].image``) into a segment video through a
persistent file-queue server (wan22_lightx2v_server.py) so the two 14B experts load only once.

Selected with ``provider = "wan_lightx2v"`` in ``configs/video_gen/<name>.toml``. See
README_wan22_lightx2v.md for the one-time weight layout + env setup (blocked on the distilled
weight paths, which the user is downloading).
"""

from __future__ import annotations

import os
import shutil
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


class Wan22LightX2VBackend:
    """Persistent-server adapter for the Wan2.2-I2V-A14B MoE 4-step distilled model."""

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
            str(params.get("model") or "lightx2v/Wan2.2-I2V-A14B-Moe-Distill")
        )

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "Wan22LightX2VBackend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("Wan22LightX2VBackend only supports video_segment tasks")
        if not self.params.get("serve_persistent", True):
            raise ValueError("Wan22LightX2VBackend requires serve_persistent=true (14B experts are costly to reload)")

        work_dir = (self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        server_dir = (self.context.workspace_path / "runs" / self.run_id / "wan22_lightx2v_server").resolve()
        self._ensure_server(server_dir)

        out_path = work_dir / "out_video.mp4"
        request = self._build_job_request(task, out_path)
        result = submit_job(
            server_dir, request, timeout=float(self.params.get("server_job_timeout", 3600))
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"wan22_lightx2v persistent server job failed: {result.get('error')}")

        digest, object_path = self.project_service.import_object(self.context, Path(result["out_video"]))
        notes = [
            "provider=wan_lightx2v",
            "model=wan2.2_i2v_a14b_moe",
            "engine=lightx2v",
            "distill=4step_2high_2low",
            "cfg=off",
            f"steps={self.params.get('infer_steps', 4)}",
            f"frames={self.params.get('num_frames', 81)}",
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
        # Wan2.2 A14B i2v is seeded from a single first frame -- the composed FLUX keyframe.
        # The MemStrata loop runs --force-recompose (a fresh keyframe every beat), so i2v is
        # the only mode needed here; continue_ar/v2v continuation is a documented TODO
        # (lightx2v supports --video_path v2v, wire it if AR chaining is revisited).
        gen_mode = str(task.controls.get("gen_mode", "recompose_keyframe"))
        if gen_mode in {"continue_ar", "reanchor_lastframe"}:
            raise ValueError(
                f"Wan22LightX2VBackend currently supports i2v (recompose_*) only, not {gen_mode!r}. "
                "Run the loop with --force-recompose (fresh keyframe per beat)."
            )
        refs = _reference_image_paths(task)
        if not refs:
            raise ValueError(
                "Wan22LightX2VBackend needs a first-frame keyframe in "
                "task.controls['composed_references'] (the composed FLUX keyframe)."
            )
        request = {
            "prompt": standardize_prompt(task.prompt, "wan"),
            "negative_prompt": str(self.params.get("negative_prompt", "")),
            "first_frame_path": str(refs[0].resolve()),
            "save_file": str(out_path),
            "seed": int(self.params.get("base_seed", 2026)),
        }
        # Per-shot morphic-LoRA toggle. Producers set task.controls["use_lora"] to run a
        # shot WITH (true) or WITHOUT (false) the morphic interpolation LoRA on the same
        # persistent server. Absent => the server falls back to the backend config's
        # natural state (morphic backend => on; plain backend => off), so default runs are
        # unchanged.
        if "use_lora" in task.controls:
            request["use_lora"] = bool(task.controls["use_lora"])
        return request

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
                    f"wan22_lightx2v server exited during startup (see {server_dir / 'server.log'})"
                )
            if server_ready(server_dir):
                return
            time.sleep(min(max(poll_interval, 0.1), max(deadline - time.time(), 0.1)))
        if proc.poll() is None:
            raise TimeoutError(
                "wan22_lightx2v server is still starting but did not become ready "
                f"within {ready_timeout:g}s (see {server_dir / 'server.log'})."
            )
        raise RuntimeError(f"wan22_lightx2v server exited during startup (see {server_dir / 'server.log'})")

    def _server_command(self, server_dir: Path) -> list[str]:
        model_path = self._model_path()
        cmd = [
            _resolve_python(str(self.params.get("python"))),
            str(Path(__file__).resolve().parent / "wan22_lightx2v_server.py"),
            "--model_path", str(model_path),
            "--model_cls", str(self.params.get("model_cls", "wan2.2_moe_distill")),
            "--server_dir", str(server_dir),
            "--idle_timeout", str(self.params.get("server_idle_timeout", 1800)),
            "--infer_steps", str(self.params.get("infer_steps", 4)),
            "--height", str(self.params.get("height", 720)),
            "--width", str(self.params.get("width", 1280)),
            "--num_frames", str(self.params.get("num_frames", 81)),
            "--sample_shift", str(self.params.get("sample_shift", 5.0)),
            "--guidance_high", str(self.params.get("guidance_high", 1.0)),
            "--guidance_low", str(self.params.get("guidance_low", 1.0)),
            "--attn_mode", str(self.params.get("attn_mode", "sage_attn2")),
        ]
        cfg_json = str(self.params.get("config_json", "")).strip()
        if cfg_json:
            path = Path(resolve_model_reference(cfg_json))
            cmd += ["--config_json", str(path if path.is_absolute() else repo_root() / path)]
        if bool(self.params.get("offload", False)):
            cmd.append("--offload")
        return cmd

    def _model_path(self) -> Path:
        raw = str(self.params.get("model", "")).strip()
        if not raw:
            raise ValueError(
                "wan22_lightx2v backend requires `model` = the LightX2V-layout dir "
                "(high_noise_model/ + low_noise_model/ + base components). See README_wan22_lightx2v.md."
            )
        path = Path(resolve_model_reference(raw))
        return path if path.is_absolute() else repo_root() / path

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        ff = str(self.params.get("ffmpeg", "")).strip()
        if ff and Path(ff).exists():
            env["IMAGEIO_FFMPEG_EXE"] = ff
        # GPU pinning: two 14B experts need a lot of VRAM; default to a single freer card.
        min_free_mib = int(self.params.get("min_free_mib", 70000))
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
                    f"No GPU has enough free memory for wan22_lightx2v (min_free_mib={min_free_mib}). "
                    "Use a freer GPU/node, or set offload=true in the config for a smaller card."
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


def _resolve_python(raw: str) -> str:
    # Env override wins: point at your lightx2v-env interpreter without editing shipped configs.
    env = os.environ.get("MEMSTRATA_LIGHTX2V_PYTHON", "").strip()
    if env:
        return env
    if not raw or raw == "None":
        raise ValueError(
            "wan22_lightx2v backend needs the lightx2v-env interpreter: set the config `python` "
            "(a bare command resolved on PATH, or an absolute path) or export "
            "MEMSTRATA_LIGHTX2V_PYTHON=/path/to/lightx2v-env/bin/python"
        )
    # Bare command (e.g. "python3"): resolve on PATH so a shipped default config stays portable.
    if "/" not in raw and os.sep not in raw:
        found = shutil.which(raw)
        if found:
            return found
        raise ValueError(
            f"wan22_lightx2v backend: `{raw}` not found on PATH; set an absolute `python` in the "
            "config or export MEMSTRATA_LIGHTX2V_PYTHON=/path/to/lightx2v-env/bin/python"
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
