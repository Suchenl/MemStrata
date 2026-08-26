"""Native Wan/VACE runner backend.

This backend is for the official Wan/VACE checkpoint layout and inference code. It complements
``DiffusersVideoBackend``: Diffusers layouts still use ``provider = "diffusers"``, while raw
Wan/VACE layouts use ``provider = "native_vace"`` and point at a VACE code checkout.
"""

from __future__ import annotations

import os
import shlex
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


class NativeVaceBackend:
    """Shell adapter for official Wan/VACE inference code."""

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
        self.model_name = resolve_model_reference(str(params.get("model") or params.get("ckpt_dir") or "vace"))

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "NativeVaceBackend":
        return cls(
            context,
            params=_load_video_gen_config(name, models_config),
            run_id=run_id,
            project_service=project_service,
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("NativeVaceBackend only supports video_segment tasks")

        work_dir = self.context.workspace_path / "runs" / self.run_id / "gen_tmp" / task.segment_id
        work_dir.mkdir(parents=True, exist_ok=True)
        # Persistent serving (project rule): build the model once and reuse across segments. Falls back
        # to a fresh per-segment subprocess only when serve_persistent is disabled (debugging).
        if self.params.get("serve_persistent"):
            out_path, notes = self._generate_via_server(task, work_dir)
        else:
            command = self._command(task, work_dir)
            subprocess.run(command, cwd=work_dir, env=self._env(), check=True)
            out_path = self._find_output(work_dir)
            notes = ["provider=native_vace"]
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

    # -- persistent server path ---------------------------------------------------------------

    def _generate_via_server(self, task: MediaGenerationTask, work_dir: Path) -> tuple[Path, list[str]]:
        global_server_dir = self.context.workspace_path / "services" / "video_generator"
        if server_ready(global_server_dir) and _pid_alive(_read_pid(global_server_dir / "ready")):
            server_dir = global_server_dir
            import logging
            logging.getLogger(__name__).info("Reusing Stage 0 / global persistent VACE server")
        else:
            server_dir = self.context.workspace_path / "runs" / self.run_id / "vace_server"
            self._ensure_server(server_dir)
        out_path = work_dir / "out_video.mp4"
        request = self._build_job_request(task, work_dir, out_path)
        result = submit_job(
            server_dir,
            request,
            timeout=float(self.params.get("server_job_timeout", 3600)),
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"VACE persistent server job failed: {result.get('error')}")
        return Path(result["out_video"]), ["provider=native_vace", "serve=persistent"]

    def _build_job_request(self, task: MediaGenerationTask, work_dir: Path, out_path: Path) -> dict[str, Any]:
        refs = _reference_image_paths(task)
        request: dict[str, Any] = {
            "prompt": standardize_prompt(task.prompt, "native_vace"),
            "size": f"{int(self.params.get('width', 832))}*{int(self.params.get('height', 480))}",
            "frame_num": int(self.params.get("frame_num", 49)),
            "src_ref_images": [str(p) for p in refs] or None,
            "src_video": None,
            "src_mask": None,
            "sample_solver": str(self.params.get("sample_solver", "unipc")),
            "sample_steps": int(self.params.get("sample_steps", 18)),
            "sample_shift": float(self.params.get("sample_shift", 8)),
            "sample_guide_scale": float(self.params.get("sample_guide_scale", 5.0)),
            "base_seed": int(self.params.get("base_seed", 2025)),
            "save_file": str(out_path),
        }
        continuation = task.controls.get("continuation")
        if isinstance(continuation, dict) and continuation.get("source_video"):
            src_video, src_mask = _build_continuation_source(
                Path(str(continuation["source_video"])),
                work_dir,
                width=int(self.params.get("width", 832)),
                height=int(self.params.get("height", 480)),
                frame_num=int(self.params.get("frame_num", 49)),
                cond_frames=int(self.params.get("continuation_cond_frames", 13)),
                fps=int(self.params.get("fps", 16)),
            )
            request["src_video"] = str(src_video)
            request["src_mask"] = str(src_mask)
        return request

    def _ensure_server(self, server_dir: Path) -> None:
        ready = server_dir / "ready"
        if server_ready(server_dir) and _pid_alive(_read_pid(ready)):
            return
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / "stop").unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        log = open(server_dir / "server.log", "ab")  # noqa: SIM115 - handed to the child process
        proc = subprocess.Popen(self._server_command(server_dir), cwd=str(server_dir), env=self._env(), stdout=log, stderr=log)
        ready_timeout = float(self.params.get("server_ready_timeout", 1200))
        poll_interval = float(self.params.get("server_ready_poll_interval", 2.0))
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"VACE persistent server exited during startup (see {server_dir / 'server.log'})"
                )
            if server_ready(server_dir):
                return
            time.sleep(min(max(poll_interval, 0.1), max(deadline - time.time(), 0.1)))
        if proc.poll() is None:
            raise TimeoutError(
                "VACE persistent server is still starting but did not become ready "
                f"within {ready_timeout:g}s (see {server_dir / 'server.log'}). "
                "On slow remote storage, increase `server_ready_timeout` or prewarm the server."
            )
        raise RuntimeError(f"VACE persistent server exited during startup (see {server_dir / 'server.log'})")

    def _server_command(self, server_dir: Path) -> list[str]:
        python = str(self.params.get("python", sys.executable))
        nproc = int(self.params.get("nproc_per_node", 1))
        code_root = _required_path(self.params, "code_root")
        command = [python]
        if nproc > 1:
            command += ["-m", "torch.distributed.run", "--nproc_per_node", str(nproc)]
        command += [
            "-m",
            "memstrata.steps.generate.backends.vace_persistent_server",
            "--vace_module_root", str(code_root if (code_root / "wan").is_dir() else (code_root / "vace")),
            "--server_dir", str(server_dir),
            "--ckpt_dir", str(self.model_name),
            "--model_name", str(self.params.get("model_name", "vace-1.3B")),
            "--size", f"{int(self.params.get('width', 832))}*{int(self.params.get('height', 480))}",
            "--idle_timeout", str(self.params.get("server_idle_timeout", 1800)),
        ]
        if self.params.get("offload_model"):
            command += ["--offload_model", "true"]
        if self.params.get("t5_cpu", False):
            command += ["--t5_cpu"]
        if self.params.get("convert_model_dtype", False):
            command += ["--convert_model_dtype"]
        # Distributed flags (dit_fsdp / ulysses_size / ring_size / t5_fsdp) ride along via extra_args,
        # exactly as the one-shot subprocess path passes them.
        extra_args = self.params.get("extra_args", [])
        if isinstance(extra_args, str):
            command += shlex.split(extra_args)
        elif isinstance(extra_args, list):
            command += [str(arg) for arg in extra_args]
        return command

    def _command(self, task: MediaGenerationTask, work_dir: Path) -> list[str]:
        code_root = _required_path(self.params, "code_root")
        generate_py = code_root / str(self.params.get("entrypoint", "generate.py"))
        if not generate_py.is_file():
            raise FileNotFoundError(
                f"VACE entrypoint not found: {generate_py}. Set VACE_CODE_ROOT or config code_root."
            )

        family = self.params.get("family")
        uses_multiframe_i2v_entrypoint = family in {"wan_i2v_morphic", "wan_i2v_svi"}

        refs = _reference_image_paths(task)
        continuation = task.controls.get("continuation")
        if uses_multiframe_i2v_entrypoint and isinstance(continuation, dict) and continuation.get("source_video"):
            refs = [
                _extract_tail_reference_frame(
                    Path(str(continuation["source_video"])),
                    work_dir,
                    width=int(self.params.get("width", 832)),
                    height=int(self.params.get("height", 480)),
                    )
            ] + refs
        if uses_multiframe_i2v_entrypoint and not refs and task.controls.get("source_video"):
            refs = [
                _extract_source_reference_frame(
                    Path(str(task.controls["source_video"])),
                    work_dir,
                    width=int(self.params.get("width", 832)),
                    height=int(self.params.get("height", 480)),
                    start_sec=float(task.controls.get("source_start_sec", 0.0)),
                )
            ]
        model_name = str(self.params.get("model_name", self.params.get("task_name", "vace-1.3B")))
        size = f"{int(self.params.get('width', 832))}*{int(self.params.get('height', 480))}"
        out_path = work_dir / "out_video.mp4"
        command = self._entrypoint_command(generate_py, code_root)
        if uses_multiframe_i2v_entrypoint:
            command += [
                "--task", model_name,
                "--size", size,
                "--ckpt_dir", self.model_name,
                "--prompt", standardize_prompt(task.prompt, "native_vace"),
                "--save_file", str(out_path),
            ]
        else:
            command += [
                "--model_name", model_name,
                "--size", size,
                "--ckpt_dir", self.model_name,
                "--prompt", standardize_prompt(task.prompt, "native_vace"),
                "--save_file", str(out_path),
            ]
        if "frame_num" in self.params:
            command += ["--frame_num", str(self.params["frame_num"])]
        if refs:
            if uses_multiframe_i2v_entrypoint:
                command += ["--image", str(refs[0])]
                if len(refs) > 1:
                    command += ["--img_end", str(refs[-1])]
                if len(refs) > 2:
                    command += ["--middle_images"] + [str(p) for p in refs[1:-1]]
                    timestamps = [round(i / (len(refs) - 1), 2) for i in range(1, len(refs) - 1)]
                    command += ["--middle_images_timestamps"] + [str(t) for t in timestamps]
            else:
                command += ["--src_ref_images", ",".join(str(p) for p in refs)]
        # Plan-driven continuation: when the loop hands us the previous segment's video (transition ==
        # CONTINUE), build VACE V2V inputs (tail frames known + gray frames to generate) and pass
        # them ALONGSIDE the reference images -- VACE natively supports masked video-to-video with
        # references simultaneously (see wan_vace.vace_encode_frames). A cut/jump omits this.
        if (not uses_multiframe_i2v_entrypoint) and isinstance(continuation, dict) and continuation.get("source_video"):
            src_video, src_mask = _build_continuation_source(
                Path(str(continuation["source_video"])),
                work_dir,
                width=int(self.params.get("width", 832)),
                height=int(self.params.get("height", 480)),
                frame_num=int(self.params.get("frame_num", 49)),
                cond_frames=int(self.params.get("continuation_cond_frames", 13)),
                fps=int(self.params.get("fps", 16)),
            )
            command += ["--src_video", str(src_video), "--src_mask", str(src_mask)]
        if "offload_model" in self.params:
            command += ["--offload_model", str(bool(self.params["offload_model"]))]
        if self.params.get("t5_cpu", False):
            command += ["--t5_cpu"]
        if self.params.get("convert_model_dtype", False):
            command += ["--convert_model_dtype"]
        if "low_noise_lora_weights_path" in self.params:
            command += ["--low_noise_lora_weights_path", str(self.params["low_noise_lora_weights_path"])]
        if "high_noise_lora_weights_path" in self.params:
            command += ["--high_noise_lora_weights_path", str(self.params["high_noise_lora_weights_path"])]
        for key, flag in (
            ("lora_rank", "--lora_rank"),
            ("lora_alpha", "--lora_alpha"),
        ):
            if key in self.params:
                command += [flag, str(self.params[key])]
        if "use_prompt_extend" in self.params:
            command += ["--use_prompt_extend", str(self.params["use_prompt_extend"])]
        for key, flag in (
            ("sample_steps", "--sample_steps"),
            ("sample_shift", "--sample_shift"),
            ("sample_guide_scale", "--sample_guide_scale"),
        ):
            if key in self.params:
                command += [flag, str(self.params[key])]
        extra_args = self.params.get("extra_args", [])
        if isinstance(extra_args, str):
            command += shlex.split(extra_args)
        elif isinstance(extra_args, list):
            command += [str(arg) for arg in extra_args]
        # The official script writes into the current working directory by default; keep it isolated.
        work_dir.mkdir(parents=True, exist_ok=True)
        return command

    def _entrypoint_command(self, generate_py: Path, code_root: Path) -> list[str]:
        python = str(self.params.get("python", sys.executable))
        nproc_per_node = int(self.params.get("nproc_per_node", 1))
        vace_module_root = code_root if (code_root / "wan").is_dir() else (code_root / "vace")
        if self.params.get("shim_annotators", True):
            if nproc_per_node > 1:
                return [
                    python,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node",
                    str(nproc_per_node),
                    "-m",
                    "memstrata.steps.generate.backends.vace_wan_entrypoint",
                    "--vace_script",
                    str(generate_py),
                    "--vace_module_root",
                    str(vace_module_root),
                ]
            return [
                python,
                "-m",
                "memstrata.steps.generate.backends.vace_wan_entrypoint",
                "--vace_script",
                str(generate_py),
                "--vace_module_root",
                str(vace_module_root),
            ]
        if nproc_per_node > 1:
            return [python, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc_per_node), str(generate_py)]
        return [python, str(generate_py)]

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

        curr_pythonpath = env.get("PYTHONPATH", "")
        curr_parts = []
        if curr_pythonpath:
            for part in curr_pythonpath.split(os.pathsep):
                if part:
                    curr_parts.append(str(Path(part).resolve()))
        else:
            from memstrata.steps.generate.backends._support import repo_root
            curr_parts.append(str(repo_root() / "src"))

        if roots:
            env["PYTHONPATH"] = os.pathsep.join(roots + curr_parts)
        else:
            env["PYTHONPATH"] = os.pathsep.join(curr_parts)

        # Rule-based free-GPU pick: land the (persistent) service on the freest card(s) instead
        # of always defaulting to GPU 0. Respect an explicit CUDA_VISIBLE_DEVICES and an opt-out.
        if self.params.get("auto_pick_gpu", True) and not env.get("CUDA_VISIBLE_DEVICES"):
            from memstrata.lib.gpu import cuda_visible_devices_for

            picked = cuda_visible_devices_for(int(self.params.get("nproc_per_node", 1)))
            if picked is not None:
                env["CUDA_VISIBLE_DEVICES"] = picked
        return env

    def _find_output(self, work_dir: Path) -> Path:
        output_glob = str(self.params.get("output_glob", "**/*.mp4"))
        outputs = [p for p in work_dir.glob(output_glob) if p.is_file()]
        if not outputs:
            raise FileNotFoundError(
                f"Native VACE command completed but no output matched {output_glob!r} under {work_dir}"
            )
        return max(outputs, key=lambda p: p.stat().st_mtime)


def _required_path(params: dict[str, Any], key: str) -> Path:
    raw = str(params.get(key, "")).strip()
    if not raw:
        raise ValueError(f"Native VACE backend requires `{key}` in video_gen config")
    return _resolve_code_path(raw)


def _resolve_code_path(raw: str) -> Path:
    path = Path(resolve_model_reference(raw))
    return path if path.is_absolute() else repo_root() / path


def _read_pid(ready_file: Path) -> int:
    try:
        return int(ready_file.read_text().strip())
    except (OSError, ValueError):
        return -1


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0: existence/permission probe, does not actually signal
    except OSError:
        return False
    return True


def _assemble_continuation(tail_frames: list, frame_num: int, ref_color: int = 127):
    """Assemble VACE V2V continuation frames + masks from a known tail clip.

    Mirrors VACE's ``FrameRefExpandAnnotator`` ``firstclip`` mode: the known tail frames come first
    with an all-zero (inactive => kept) mask, followed by ``frame_num - len(tail)`` gray frames with
    an all-255 (reactive => generate) mask. Returns ``(out_frames, out_masks)`` as BGR / single-mask
    numpy arrays. Pure (no I/O) so the mask logic is unit-testable without codecs.
    """

    import numpy as np

    if not tail_frames:
        raise ValueError("continuation needs at least one known tail frame")
    tail = tail_frames[: max(1, min(len(tail_frames), frame_num - 1))]
    expand = frame_num - len(tail)
    h, w = tail[0].shape[:2]
    gray = np.full((h, w, 3), ref_color, dtype=np.uint8)
    black = np.zeros((h, w, 3), dtype=np.uint8)
    white = np.full((h, w, 3), 255, dtype=np.uint8)
    out_frames = list(tail) + [gray] * expand
    out_masks = [black] * len(tail) + [white] * expand
    return out_frames, out_masks


def _build_continuation_source(
    prev_video: Path,
    work_dir: Path,
    *,
    width: int,
    height: int,
    frame_num: int,
    cond_frames: int,
    fps: int,
) -> tuple[Path, Path]:
    """Read the previous segment's tail and write VACE ``--src_video`` / ``--src_mask`` mp4s."""

    import cv2

    cap = cv2.VideoCapture(str(prev_video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, height)))
    cap.release()
    if not frames:
        raise RuntimeError(f"continuation source has no readable frames: {prev_video}")
    tail = frames[-max(1, min(cond_frames, frame_num - 1)):]
    out_frames, out_masks = _assemble_continuation(tail, frame_num)
    src_video = work_dir / "cont_src_video.mp4"
    src_mask = work_dir / "cont_src_mask.mp4"
    _write_video(src_video, out_frames, fps)
    _write_video(src_mask, out_masks, fps)
    return src_video, src_mask


def _extract_tail_reference_frame(prev_video: Path, work_dir: Path, *, width: int, height: int) -> Path:
    import cv2

    work_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(prev_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"continuation source has no readable frames: {prev_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read continuation tail frame: {prev_video}")
    frame = cv2.resize(frame, (width, height))
    out = work_dir / "cont_tail_ref.png"
    cv2.imwrite(str(out), frame)
    return out


def _extract_source_reference_frame(source_video: Path, work_dir: Path, *, width: int, height: int, start_sec: float) -> Path:
    import cv2

    work_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"source video is not readable for I2V reference: {source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(start_sec * fps))))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec * 1000.0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read source reference frame at {start_sec:.3f}s: {source_video}")
    frame = cv2.resize(frame, (width, height))
    out = work_dir / "source_ref.png"
    cv2.imwrite(str(out), frame)
    return out


def _write_video(path: Path, frames_bgr: list, fps: int) -> None:
    import cv2

    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open a writer for {path} (missing mp4v codec?)")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


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
