"""MAGREF video generation backend adapter."""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Any

from memstrata.steps.generate.backends._support import ArtifactStore as ProjectService
from memstrata.steps.generate.backends._support import resolve_model_reference
from memstrata.steps.generate.backends._support import RunContext as ProjectContext, new_id
from memstrata.steps.generate.backends.diffusers_backend import _load_video_gen_config, _reference_image_paths
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)


def _required_path(params: dict[str, Any], key: str) -> Path:
    val = params.get(key)
    if not val:
        raise ValueError(f"MAGREF parameter {key} is required")
    return Path(str(val))


class MagRefBackend:
    """Shell adapter for MAGREF-Video/MAGREF multi-reference subject-to-video model."""

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
        self.model_name = resolve_model_reference(str(params.get("model") or "MAGREF-Video/MAGREF"))

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "MagRefBackend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("MagRefBackend only supports video_segment tasks")

        work_dir = self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id
        work_dir.mkdir(parents=True, exist_ok=True)

        code_root = _required_path(self.params, "code_root")
        model_dir = _required_path(self.params, "model")

        # 1) Get reference images
        refs = _reference_image_paths(task)
        if not refs:
            raise ValueError("MagRefBackend requires at least one reference image")

        # 2) Write prompt/reference input file for MAGREF
        input_file = work_dir / "input.txt"
        input_line = f"{task.segment_id}@@{task.prompt}@@" + "@@".join(str(r) for r in refs)
        input_file.write_text(input_line + "\n", encoding="utf-8")

        raw_save_dir = work_dir / "raw"
        raw_save_dir.mkdir(parents=True, exist_ok=True)

        # 3) Build and execute command
        python = str(self.params.get("python", sys.executable))
        cmd = [
            python,
            "generate.py",
            "--ckpt_dir",
            str(model_dir),
            "--save_dir",
            str(raw_save_dir),
            "--prompt_path",
            str(input_file),
            "--base_seed",
            str(self.params.get("base_seed", 20260703)),
            "--frame_num",
            str(self.params.get("frame_num", 81)),
            "--offload_model",
            str(self.params.get("offload_model", True)),
        ]

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(code_root), env.get("PYTHONPATH", "")])
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        subprocess.run(cmd, cwd=str(code_root), env=env, check=True)

        # 4) Locate raw output and optionally convert to compatible H.264 MP4
        raw_video = raw_save_dir / f"{task.segment_id}.mp4"
        if not raw_video.is_file():
            raise FileNotFoundError(f"MAGREF completed but output was not found at {raw_video}")

        final_video = work_dir / "out_h264.mp4"
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            try:
                subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-i",
                        str(raw_video),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(final_video),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            except Exception:
                final_video = raw_video
        else:
            final_video = raw_video

        # Import object into workspace
        digest, object_path = self.project_service.import_object(self.context, final_video)

        return GenerationArtifact(
            artifact_id=new_id("artifact"),
            task_id=task.task_id,
            segment_id=task.segment_id,
            media_type="video",
            object_hash=digest,
            object_uri=str(object_path),
            model_name=self.model_name,
            degradation_notes=["provider=magref"],
        )
