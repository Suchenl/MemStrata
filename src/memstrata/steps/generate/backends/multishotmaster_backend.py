"""MultiShotMaster video generation backend adapter."""

from __future__ import annotations

import csv
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any

from memstrata.steps.generate.backends._support import ArtifactStore as ProjectService
from memstrata.steps.generate.backends._support import repo_root, resolve_model_reference
from memstrata.steps.generate.backends._support import RunContext as ProjectContext, new_id
from memstrata.steps.generate.backends.diffusers_backend import _load_video_gen_config
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)


class MultiShotMasterBackend:
    """Shell adapter for KlingAI's MultiShotMaster video consistency model."""

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
        self.model_name = resolve_model_reference(str(params.get("model") or "multishotmaster-1.3B"))

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "MultiShotMasterBackend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("MultiShotMasterBackend only supports video_segment tasks")

        work_dir = self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id
        work_dir.mkdir(parents=True, exist_ok=True)

        code_root = _required_path(self.params, "code_root")

        # 1) Prepare inputs for MultiShotMaster (CSV and JSON)
        # Determine the shot group length and prompt.
        # MultiShotMaster expects a list of shots like [[0, 49]].
        frame_num = int(self.params.get("frame_num", 49))
        shot_groups = [[0, frame_num]]

        # Write caption JSON
        caption_data = {
            "global_caption": task.prompt,
            "shot0": task.prompt,
        }
        caption_path = work_dir / "caption.json"
        caption_path.write_text(json.dumps(caption_data, ensure_ascii=False, indent=4), encoding="utf-8")

        # Write test CSV
        csv_path = work_dir / "test.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["shot_groups", "gemini_caption"])
            writer.writerow([str(shot_groups), str(caption_path)])

        # 2) Build command and run
        command = self._command(task, work_dir, csv_path)
        # We run the command inside the code_root because MultiShotMaster relies on local modules (util.py)
        subprocess.run(command, cwd=str(code_root), env=self._env(), check=True)

        # 3) Locate output
        # Output is saved by MultiShotMaster as output/<segment_id>/0.mp4 under code_root.
        out_path = code_root / "output" / task.segment_id / "0.mp4"
        if not out_path.is_file():
            # Fall back to searching for any generated video if the index or pattern differs
            outputs = list((code_root / "output" / task.segment_id).glob("**/*.mp4"))
            if not outputs:
                raise FileNotFoundError(
                    f"MultiShotMaster completed but output was not found at {out_path} or matching glob"
                )
            out_path = max(outputs, key=lambda p: p.stat().st_mtime)

        # Import object into workspace
        digest, object_path = self.project_service.import_object(self.context, out_path)

        return GenerationArtifact(
            artifact_id=new_id("artifact"),
            task_id=task.task_id,
            segment_id=task.segment_id,
            media_type="video",
            object_hash=digest,
            object_uri=str(object_path),
            model_name=self.model_name,
            degradation_notes=["provider=multishotmaster"],
        )

    def _command(self, task: MediaGenerationTask, work_dir: Path, csv_path: Path) -> list[str]:
        code_root = _required_path(self.params, "code_root")
        python = str(self.params.get("python", sys.executable))
        nproc_per_node = int(self.params.get("nproc_per_node", 1))
        use_usp = bool(self.params.get("use_usp", False))

        infer_py = code_root / "infer_multishot.py"
        if not infer_py.is_file():
            raise FileNotFoundError(f"MultiShotMaster inference script not found at {infer_py}")

        command = []
        if use_usp and nproc_per_node > 1:
            command += [python, "-m", "torch.distributed.run", f"--nproc_per_node={nproc_per_node}"]
        else:
            command += [python]

        model_json_name = str(self.params.get("model_path_json", "checkpoints/model_configs/model_path_1.3B.json"))
        model_path_json = code_root / model_json_name

        command += [
            str(infer_py),
            "--test_csv_path", str(csv_path),
            "--output_name", task.segment_id,
            "--model_path_json", str(model_path_json),
            "--target_width", str(int(self.params.get("width", 832))),
            "--target_height", str(int(self.params.get("height", 480))),
        ]
        if use_usp:
            command += ["--use_usp", "True"]
        return command

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        roots = []
        for key in ("code_root", "wan_code_root"):
            if self.params.get(key):
                roots.append(str(_required_path(self.params, key)))
        extra = self.params.get("pythonpath", [])
        if isinstance(extra, str):
            roots.append(str(_resolve_code_path(extra)))
        elif isinstance(extra, list):
            roots.extend(str(_resolve_code_path(str(path))) for path in extra)
        if roots:
            env["PYTHONPATH"] = os.pathsep.join(roots + [env.get("PYTHONPATH", "")])

        # Rule-based free-GPU selection: picks the freest card(s)
        if self.params.get("auto_pick_gpu", True) and not env.get("CUDA_VISIBLE_DEVICES"):
            from memstrata.lib.gpu import cuda_visible_devices_for

            picked = cuda_visible_devices_for(int(self.params.get("nproc_per_node", 1)))
            if picked is not None:
                env["CUDA_VISIBLE_DEVICES"] = picked
        return env


def _required_path(params: dict[str, Any], key: str) -> Path:
    raw = str(params.get(key, "")).strip()
    if not raw:
        raise ValueError(f"MultiShotMaster backend requires `{key}` in video_gen config")
    return _resolve_code_path(raw)


def _resolve_code_path(raw: str) -> Path:
    path = Path(resolve_model_reference(raw))
    return path if path.is_absolute() else repo_root() / path
