"""FLUX.2 [klein] 9B-KV image generation and editing backend.

Vendored from Montage ``src/montage/models/image_generation/flux_klein_backend.py``
(import paths rewritten to stay self-contained under ``benchmarks/MemStrata`` — no
``montage`` imports; the model code / weights are still loaded by path). This is the
keyframe (Layout Anchor) image generator for the keyframe-first main pipeline:
FLUX produces the start / end / mid keyframes, then Wan2.2-I2V-A14B (+ SVI / Morphic
LoRA) expands them into video.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from memstrata.steps.generate.backends._support import ArtifactStore as ProjectService
from memstrata.steps.generate.backends._support import RunContext as ProjectContext, new_id
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)
from .base import apply_photographic_grain
from memstrata.lib.prompt_standardizer import standardize_prompt


def _resolve_flux_python(raw: object) -> str:
    """Interpreter for the persistent FLUX server.

    Precedence: ``MEMSTRATA_FLUX_PYTHON`` env override > an explicit path in the
    config (absolute or relative) > the current interpreter. A *bare* command
    name (e.g. the shipped default ``python = "python3"``) falls back to
    ``sys.executable``, which is the env that has ``diffusers`` /
    ``Flux2KleinKVPipeline`` installed after ``pip install -e``. This keeps the
    default turnkey while still honoring an explicitly configured env.
    """
    env = os.environ.get("MEMSTRATA_FLUX_PYTHON", "").strip()
    if env:
        return env
    s = str(raw).strip() if raw is not None else ""
    if not s or s == "None":
        return sys.executable
    if "/" in s or os.sep in s:
        return s
    return sys.executable

logger = logging.getLogger(__name__)


class FluxKleinImageBackend:
    """Image generation and editing backend using Flux2KleinKVPipeline."""

    def __init__(
        self,
        context: ProjectContext,
        run_id: str,
        project_service: ProjectService | None = None,
        model_path: str = "${PUBLIC_MODELS_ROOT}/black-forest-labs/FLUX.2-klein-9b-kv",
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.context = context
        self.run_id = run_id
        self.project_service = project_service or ProjectService()
        self.model_path = model_path
        self.device = device
        self.params = kwargs
        self._pipe: Any = None

    @classmethod
    def from_config(
        cls,
        context: ProjectContext,
        run_id: str,
        project_service: ProjectService,
        **kwargs: Any,
    ) -> FluxKleinImageBackend:
        """Create backend from configuration."""
        model_path = (
            kwargs.get("config")
            or kwargs.get("model_path")
            or "${PUBLIC_MODELS_ROOT}/black-forest-labs/FLUX.2-klein-9b-kv"
        )
        device = kwargs.get("device", "cuda")
        return cls(
            context, run_id=run_id, project_service=project_service, model_path=model_path, device=device, **kwargs
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        """Generate or edit an image using the pipeline."""
        if task.task_type not in {
            MediaTaskType.REFERENCE_IMAGE,
            MediaTaskType.KEYFRAME,
            MediaTaskType.IMAGE_EDIT,
        }:
            raise ValueError(f"FluxKleinImageBackend does not support task type: {task.task_type}")

        work_dir = self.context.workspace_path / "runs" / self.run_id / "gen_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)

        if self.params.get("serve_persistent"):
            out_path, notes = self._generate_via_server(task, work_dir)
        else:
            # Lazy load heavy dependencies
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
            from diffusers import Flux2KleinKVPipeline  # noqa: PLC0415

            pipe = self._ensure_pipeline(Flux2KleinKVPipeline, torch)

            # Parse task controls
            height = int(task.controls.get("height", 1024))
            width = int(task.controls.get("width", 1024))
            steps = int(task.controls.get("steps", 4))
            seed = task.controls.get("seed")
            quality_preset = task.controls.get("quality_preset")
            post_process_grain = task.controls.get("post_process_grain", False)

            # De-AI / Photographic quality prompt enhancement
            prompt = standardize_prompt(task.prompt, "flux_klein", quality_preset)

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))

            # Check for reference images in composed_references
            refs = task.controls.get("composed_references") or []
            ref_paths: list[Path] = []
            for ref in refs:
                uri = ref.get("image") if isinstance(ref, dict) else ref
                if uri:
                    path = Path(str(uri))
                    if path.is_file():
                        ref_paths.append(path)

            notes = []
            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "generator": generator,
            }

            # Run pipeline
            if ref_paths:
                logger.info("Running FLUX.2-klein-9b-kv in IMAGE-TO-IMAGE mode")
                ref_image = Image.open(ref_paths[0]).convert("RGB")
                kwargs["image"] = ref_image
                notes.append(f"conditioned_on={ref_paths[0].name}")
            else:
                logger.info("Running FLUX.2-klein-9b-kv in TEXT-TO-IMAGE mode")

            result = pipe(**kwargs)
            image = result.images[0]

            # Apply old-school photographic grain post-processing to shatter VAE CG gloss
            if post_process_grain or quality_preset == "raw_film":
                logger.info("Applying photographic film grain post-processing")
                image = apply_photographic_grain(image)

            out_path = work_dir / f"{task.segment_id}.png"
            image.save(out_path)

        # Import object to project database
        digest, object_path = self.project_service.import_object(self.context, out_path)
        if out_path.exists():
            out_path.unlink(missing_ok=True)

        return GenerationArtifact(
            artifact_id=new_id("artifact"),
            task_id=task.task_id,
            segment_id=task.segment_id,
            media_type="image",
            object_hash=digest,
            object_uri=str(object_path),
            model_name="FLUX.2-klein-9b-kv",
            degradation_notes=notes,
        )

    def _generate_via_server(self, task: MediaGenerationTask, work_dir: Path) -> tuple[Path, list[str]]:
        global_server_dir = self.context.workspace_path / "services" / "image_generator"
        from memstrata.steps.generate.backends.vace_job_queue import server_ready, submit_job
        if server_ready(global_server_dir) and _pid_alive(_read_pid(global_server_dir / "ready")):
            server_dir = global_server_dir
            logger.info("Reusing Stage 0 / global persistent FLUX server")
        else:
            server_dir = self.context.workspace_path / "runs" / self.run_id / "flux_server"
            self._ensure_server(server_dir)

        out_path = work_dir / f"{task.segment_id}.png"
        request = {
            "prompt": task.prompt,
            "height": int(task.controls.get("height", 1024)),
            "width": int(task.controls.get("width", 1024)),
            "steps": int(task.controls.get("steps", 4)),
            "seed": task.controls.get("seed"),
            "quality_preset": task.controls.get("quality_preset"),
            "post_process_grain": task.controls.get("post_process_grain", False),
            "composed_references": [str(p) for p in _reference_image_paths(task)] or None,
            "save_file": str(out_path),
        }
        result = submit_job(server_dir, request, timeout=300.0)
        if result.get("status") != "ok":
            raise RuntimeError(f"FLUX persistent server job failed: {result.get('error')}")
        return Path(result["out_image"]), ["provider=diffusers", "serve=persistent"]

    def _ensure_server(self, server_dir: Path) -> None:
        from memstrata.steps.generate.backends.vace_job_queue import server_ready
        ready = server_dir / "ready"
        if server_ready(server_dir) and _pid_alive(_read_pid(ready)):
            return
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / "stop").unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        log_path = server_dir / "server.log"
        log = open(log_path, "ab")  # noqa: SIM115 - handed to the child process

        # Build command
        python = _resolve_flux_python(self.params.get("python"))
        command = [
            python,
            "-m",
            "memstrata.steps.generate.image_backends.flux_persistent_server",
            "--model_path", os.path.expandvars(os.path.expanduser(self.model_path)),
            "--server_dir", str(server_dir),
            "--device", self.device,
            "--idle_timeout", str(self.params.get("server_idle_timeout", 1800)),
        ]

        env = dict(os.environ)
        if self.params.get("auto_pick_gpu", True) and not env.get("CUDA_VISIBLE_DEVICES"):
            from memstrata.lib.gpu import cuda_visible_devices_for
            picked = cuda_visible_devices_for(1)
            if picked is not None:
                env["CUDA_VISIBLE_DEVICES"] = picked

        logger.info(f"Launching FLUX.2 Klein persistent server locally with command: {' '.join(command)}...")
        proc = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=log, stderr=log, start_new_session=True)
        ready_timeout = float(self.params.get("server_ready_timeout", 300))
        poll_interval = 1.0
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"FLUX persistent server exited during startup (see {log_path})"
                )
            if server_ready(server_dir):
                return
            time.sleep(poll_interval)
        if proc.poll() is None:
            raise TimeoutError(
                f"FLUX persistent server did not become ready within {ready_timeout}s"
            )

    def _ensure_pipeline(self, pipeline_cls: Any, torch: Any) -> Any:
        if self._pipe is None:
            model_path = os.path.expandvars(os.path.expanduser(self.model_path))
            logger.info(f"Loading Flux2KleinKVPipeline from: {model_path}")
            self._pipe = pipeline_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
            self._pipe = self._pipe.to(self.device)
        return self._pipe


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
