"""Helios-Distilled i2v backend adapter.

Helios is a fast (CFG-distilled, 4-6 step) autoregressive i2v model with only a bounded
rolling-window history (no long-range memory of its own) -- which is exactly why it pairs
well with MemStrata: the *external* stratified memory supplies long-range identity via the
FLUX keyframe, Helios just animates it quickly. This adapter feeds Helios the composed FLUX
keyframe (``task.controls["composed_references"][0].image``) as the i2v first frame and
continues the segment, through a persistent file-queue server (see helios_persistent_server.py).
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

# HF `kernels` FA2 revision alias shim for the helios env on A800 (see probe / kernel note).
_SITECUSTOMIZE = '''\
"""FA2 kernel-version alias shim for the Helios env on A800 (HF_HUB_OFFLINE=1)."""
import kernels._versions as _kernel_versions
import kernels.utils as _kernel_utils

_ORIGINAL_SELECT = _kernel_versions.select_revision_or_version
_FLASH_ATTN2_V1 = "be3acec1c49820c5058b7c75b9b11ad27eb2fbd6"


def _select_revision_or_version(repo_id, *, revision, version):
    if repo_id == "kernels-community/flash-attn2" and revision is None and version == 1:
        return _FLASH_ATTN2_V1
    return _ORIGINAL_SELECT(repo_id, revision=revision, version=version)


_kernel_versions.select_revision_or_version = _select_revision_or_version
_kernel_utils.select_revision_or_version = _select_revision_or_version
'''


class HeliosBackend:
    """Persistent-server adapter for the Helios-Distilled i2v model."""

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
            str(params.get("model") or "BestWishYsh/Helios-Distilled")
        )

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "HeliosBackend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("HeliosBackend only supports video_segment tasks")
        if not self.params.get("serve_persistent", True):
            raise ValueError("HeliosBackend requires serve_persistent=true to avoid per-segment reloads")

        # Absolute paths: the server subprocess runs with cwd=helios_root (see _ensure_server),
        # so any relative server_dir/out_path would create inbox/outbox/ready and write the video
        # under the Helios checkout instead of the run dir -> the driver would never see `ready`.
        work_dir = (self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        server_dir = (self.context.workspace_path / "runs" / self.run_id / "helios_server").resolve()
        self._ensure_server(server_dir)

        out_path = work_dir / "out_video.mp4"
        request = self._build_job_request(task, out_path)
        result = submit_job(
            server_dir, request, timeout=float(self.params.get("server_job_timeout", 3600))
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"Helios persistent server job failed: {result.get('error')}")

        digest, object_path = self.project_service.import_object(self.context, Path(result["out_video"]))
        notes = [
            "provider=helios",
            "serve=persistent",
            f"mode={request.get('mode', 'i2v')}",
            "distill=cfg_4step",
            f"num_frames={request['num_frames']}",
            f"steps={request['num_inference_steps']}",
            f"is_skip_first_segment={request['is_skip_first_segment']}",
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
        # Generation mode (from the R2b router). Decides how Helios is seeded:
        #   continue_ar -> video=[style_anchor + prior window]  (server reads history_video)
        #   reanchor    -> image=<prior segment's last frame>   (server reads history_video)
        #   i2v/default -> image=<composed FLUX keyframe / composite>
        gen_mode = str(task.controls.get("gen_mode", "recompose_keyframe"))
        seed: dict[str, Any] = {}
        if gen_mode == "continue_ar":
            history_video = _continuation_video(task)
            if not history_video:
                raise ValueError("continue_ar needs task.controls['history_video'] (prior segment)")
            seed = {"mode": "continue_ar", "history_video": str(Path(history_video).resolve())}
            anchor = task.controls.get("style_anchor")
            if anchor and Path(str(anchor)).is_file():
                seed["style_anchor_path"] = str(Path(str(anchor)).resolve())
        elif gen_mode == "reanchor_lastframe":
            history_video = _continuation_video(task)
            if not history_video:
                raise ValueError("reanchor_lastframe needs task.controls['history_video'] (prior segment)")
            seed = {"mode": "reanchor", "history_video": str(Path(history_video).resolve())}
        else:  # recompose_keyframe / recompose_partial -> a single first-frame image
            refs = _reference_image_paths(task)
            if not refs:
                raise ValueError(
                    "HeliosBackend needs a first-frame keyframe in task.controls['composed_references'] "
                    "(the composed FLUX keyframe). Run the keyframe/Crop2Image step before generate."
                )
            # Absolute: the server runs with cwd=helios_root, so a relative keyframe path
            # would fail load_image() ("not a valid path").
            seed = {"mode": "i2v", "first_frame_path": str(refs[0].resolve())}
        return {
            "prompt": standardize_prompt(task.prompt, "wan"),
            "negative_prompt": "",  # no-op for the CFG-distilled model; kept for API parity
            "save_file": str(out_path),
            **seed,
            "height": int(self.params.get("height", 480)),
            "width": int(self.params.get("width", 832)),
            "num_frames": int(self.params.get("num_frames", 121)),
            "num_inference_steps": int(self.params.get("num_inference_steps", 6)),
            "pyramid": list(self.params.get("pyramid", [2, 2, 2])),
            "guidance_scale": float(self.params.get("guidance_scale", 1.0)),
            "fps": int(self.params.get("fps", 24)),
            "seed": int(self.params.get("base_seed", 2026)),
            "sigma_min": float(self.params.get("sigma_min", 0.25)),
            "sigma_max": float(self.params.get("sigma_max", 0.4)),
            "is_skip_first_segment": bool(self.params.get("is_skip_first_segment", False)),
            "is_amplify_first_segment": bool(self.params.get("is_amplify_first_segment", True)),
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
            cwd=str(self._helios_root()),
            env=self._env(server_dir),
            stdout=log,
            stderr=log,
        )
        ready_timeout = float(self.params.get("server_ready_timeout", 1200))
        poll_interval = float(self.params.get("server_ready_poll_interval", 2.0))
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Helios persistent server exited during startup (see {server_dir / 'server.log'})"
                )
            if server_ready(server_dir):
                return
            time.sleep(min(max(poll_interval, 0.1), max(deadline - time.time(), 0.1)))
        if proc.poll() is None:
            raise TimeoutError(
                "Helios persistent server is still starting but did not become ready "
                f"within {ready_timeout:g}s (see {server_dir / 'server.log'})."
            )
        raise RuntimeError(f"Helios persistent server exited during startup (see {server_dir / 'server.log'})")

    def _server_command(self, server_dir: Path) -> list[str]:
        if int(self.params.get("nproc_per_node", 1)) != 1:
            raise ValueError("HeliosBackend currently supports nproc_per_node=1 only")
        return [
            _resolve_python(str(self.params.get("python"))),
            str(Path(__file__).resolve().parent / "helios_persistent_server.py"),
            "--model",
            str(self.model_name),
            "--server_dir",
            str(server_dir),
            "--idle_timeout",
            str(self.params.get("server_idle_timeout", 1800)),
        ]

    def _helios_root(self) -> Path:
        raw = str(self.params.get("code_root", "")).strip()
        if not raw:
            raise ValueError("Helios backend requires `code_root` (path to the Helios checkout)")
        path = Path(resolve_model_reference(raw))
        return path if path.is_absolute() else repo_root() / path

    def _env(self, server_dir: Path) -> dict[str, str]:
        env = dict(os.environ)
        # Write the FA2 kernel-alias shim into the server dir and put it first on PYTHONPATH,
        # alongside the Helios checkout (so `import helios...` resolves).
        meta = server_dir / "_meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(meta), str(self._helios_root()), env.get("PYTHONPATH", "")]
        )
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        # HF/kernels caches MUST hold the Helios weights + the shimmed FA2 kernel snapshot.
        # Force-set (not setdefault): these default to the *MemStrata* repo root, whose
        # models/model_weights/hub lacks the FA2 kernel; the real cache is the upstream project root.
        # `this repository` -> upstream project is two parents up.
        default_mw = repo_root().parent.parent / "models" / "model_weights"
        hf_home = str(self.params.get("hf_home") or default_mw)
        hub_cache = str(self.params.get("kernels_cache") or (Path(hf_home) / "hub"))
        env["HF_HOME"] = hf_home
        env["HF_HUB_CACHE"] = str(self.params.get("hf_hub_cache") or hub_cache)
        env["KERNELS_CACHE"] = hub_cache
        env["MONTAGE_HELIOS_DEFER_KERNEL_LOAD"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # diffusers export_to_video -> imageio-ffmpeg; borrow a real ffmpeg if configured.
        ff = str(self.params.get("ffmpeg", "")).strip()
        if ff and Path(ff).exists():
            env["IMAGEIO_FFMPEG_EXE"] = ff

        min_free_mib = int(self.params.get("min_free_mib", 60000))
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
                    f"No GPU has enough free memory for Helios service (min_free_mib={min_free_mib}). "
                    "Use a freer GPU/node instead of squeezing onto a busy card."
                )
        return env


def _continuation_video(task: MediaGenerationTask) -> str | None:
    """Prior segment path for continue_ar / reanchor. Prefer an explicit history_video, else
    fall back to the ``continuation.source_video`` that MediaTaskGenerator already tracks."""
    hv = task.controls.get("history_video")
    if hv and Path(str(hv)).is_file():
        return str(hv)
    cont = task.controls.get("continuation") or {}
    src = cont.get("source_video") if isinstance(cont, dict) else None
    return str(src) if src and Path(str(src)).is_file() else None


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
        raise ValueError("Helios backend requires `python` (path to the helios-env interpreter)")
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
