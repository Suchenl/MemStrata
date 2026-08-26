"""Frame-number keyframe continuation for the Wan frames-to-video (Morphic) backend.

This module holds *our* orchestration logic, kept in MemStrata and distinct from the
vendored model code. The idea (per user request):

  * Lay the previous clip's tail frames at consecutive frame indices ``0,1,2,...,P-1``
    as a real *motion prefix* (not a single static anchor), so the generator continues
    the actual motion.
  * Insert FLUX-generated future keyframes at sparse frame indices (e.g. ``40, 60, -1``).
  * Drive the vendored ``generate.py --keyframes/--keyframe_frames`` entrypoint through
    the ``vace_wan_entrypoint`` shim (so the *local* ``wan`` module is imported, not a
    globally-installed one).

Frame-number semantics (matches ``wan.image2video.resolve_keyframe_slots``):
  * ``0`` = start frame, ``-1`` = last frame, negatives count back from the end.
  * indices are used EXACTLY (no VAE-stride snapping); two keyframes on the same
    resolved frame is a hard error (silent overwrite is what caused start flicker).

The mask / latent injection itself lives in the backend ``wan/image2video.py``; here we
only build the frame layout and the subprocess command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _ff(name: str) -> str:
    """Resolve an ffmpeg-family binary (portable): PATH first, then the interpreter's bin dir."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.is_file() else name

from memstrata.steps.generate.backends._support import repo_root
from memstrata.steps.generate.backends.diffusers_backend import _load_video_gen_config


def resolve_frames(frames: Sequence[int], frame_num: int) -> list[int]:
    """Resolve raw frame indices to exact positions, fail-fast (mirror of the backend).

    Kept as a tiny local copy so callers can validate a layout *before* paying the
    GPU/subprocess cost; the authoritative check also runs inside the backend.
    ponytail: 12-line mirror of ``wan.image2video.resolve_keyframe_slots`` to avoid
    importing the torch-heavy backend just to validate. Keep the two in sync.
    """
    occupied: set[int] = set()
    resolved: list[int] = []
    for raw in frames:
        idx = raw if raw >= 0 else frame_num + raw
        if not 0 <= idx < frame_num:
            raise ValueError(f"frame {raw} resolves to {idx}, out of range [0, {frame_num})")
        if idx in occupied:
            raise ValueError(f"frame {raw} resolves to {idx}, which collides with another keyframe")
        occupied.add(idx)
        resolved.append(idx)
    return resolved


def build_keyframe_layout(
    prefix_frames: Sequence[Path],
    future_frames: Sequence[Path],
    future_frame_indices: Sequence[int],
    frame_num: int,
) -> tuple[list[str], list[int]]:
    """Place ``prefix_frames`` at 0,1,...,P-1 and ``future_frames`` at the given indices.

    Returns ``(keyframes, keyframe_frames)`` -- two 1:1 lists ready for
    ``--keyframes`` / ``--keyframe_frames``. Validates the whole layout (range +
    collisions) so a bad layout fails before launching the model.
    """
    if not prefix_frames:
        raise ValueError("need at least one prefix frame (it becomes the start frame 0)")
    if len(future_frames) != len(future_frame_indices):
        raise ValueError(
            f"future_frames ({len(future_frames)}) and future_frame_indices "
            f"({len(future_frame_indices)}) must have equal length"
        )
    prefix_indices = list(range(len(prefix_frames)))
    keyframes = [str(p) for p in prefix_frames] + [str(p) for p in future_frames]
    keyframe_frames = prefix_indices + list(future_frame_indices)
    resolve_frames(keyframe_frames, frame_num)  # validate; raises on range/collision
    return keyframes, keyframe_frames


def extract_prefix_frames(
    video: Path,
    out_dir: Path,
    count: int,
    width: int,
    height: int,
) -> list[Path]:
    """Extract the LAST ``count`` frames of ``video`` (resized to ``width``x``height``).

    Uses ffmpeg (robust across codecs; cv2 seek is unreliable) and returns the frame
    paths in temporal order, so ``[..., tail_{n-1}, tail_n]`` -> indices 0..count-1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = _count_frames(video)
    start = max(0, total - count)
    pattern = out_dir / "prefix_%03d.png"
    cmd = [
        _ff("ffmpeg"), "-y", "-loglevel", "error", "-i", str(video),
        "-vf", f"select=gte(n\\,{start}),scale={width}:{height}",
        "-vsync", "0", "-start_number", "0", str(pattern),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("prefix_*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no prefix frames from {video}")
    return frames


def _count_frames(video: Path) -> int:
    out = subprocess.run(
        [
            _ff("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    try:
        return int(out)
    except ValueError as exc:
        raise RuntimeError(f"could not count frames of {video} (ffprobe said {out!r})") from exc


def generate(
    *,
    config_name: str,
    prompt: str,
    keyframes: Sequence[str],
    keyframe_frames: Sequence[int],
    out_path: Path,
    models_config: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> Path:
    """Run the Wan frames-to-video backend with an explicit frame-number keyframe layout.

    Loads the video_gen config (``config_name``), builds the ``--keyframes`` command,
    and launches it through the ``vace_wan_entrypoint`` shim. Returns ``out_path``.
    """
    # _load_video_gen_config appends "video_gen/<name>.toml", so pass the parent dir.
    models_config = models_config or (repo_root() / "models" / "model_configs")
    params = _load_video_gen_config(config_name, models_config)

    code_root = _resolve_path(str(params["code_root"]))
    generate_py = code_root / str(params.get("entrypoint", "generate.py"))
    if not generate_py.is_file():
        raise FileNotFoundError(f"backend entrypoint not found: {generate_py}")
    python = str(params.get("python", sys.executable))
    size = f"{int(params.get('width', 832))}*{int(params.get('height', 480))}"
    frame_num = int(params.get("frame_num", 81))
    resolve_frames(list(keyframe_frames), frame_num)  # validate before spawning

    command = [
        python, "-m", "memstrata.steps.generate.backends.vace_wan_entrypoint",
        "--vace_script", str(generate_py),
        "--vace_module_root", str(code_root),
        "--task", str(params.get("model_name", "i2v-A14B")),
        "--size", size,
        "--ckpt_dir", str(params["model"]),
        "--prompt", prompt,
        "--save_file", str(out_path),
        "--frame_num", str(frame_num),
        "--keyframes", *[str(p) for p in keyframes],
        "--keyframe_frames", *[str(i) for i in keyframe_frames],
    ]
    for key, flag in (
        ("high_noise_lora_weights_path", "--high_noise_lora_weights_path"),
        ("low_noise_lora_weights_path", "--low_noise_lora_weights_path"),
        ("lora_rank", "--lora_rank"),
        ("lora_alpha", "--lora_alpha"),
        ("sample_steps", "--sample_steps"),
        ("sample_shift", "--sample_shift"),
        ("sample_guide_scale", "--sample_guide_scale"),
    ):
        if key in params:
            command += [flag, str(params[key])]
    for key, flag in (("offload_model", "--offload_model"),):
        if key in params:
            command += [flag, str(bool(params[key]))]
    if params.get("t5_cpu", False):
        command += ["--t5_cpu"]
    if params.get("convert_model_dtype", False):
        command += ["--convert_model_dtype"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=str(code_root), env=_build_env(code_root, extra_env), check=True)
    if not out_path.is_file():
        raise FileNotFoundError(f"backend finished but output missing: {out_path}")
    return out_path


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root() / path


def _build_env(code_root: Path, extra_env: dict[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(repo_root() / "src"), str(code_root)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if not env.get("CUDA_VISIBLE_DEVICES"):
        from memstrata.lib.gpu import cuda_visible_devices_for

        picked = cuda_visible_devices_for(1)
        if picked is not None:
            env["CUDA_VISIBLE_DEVICES"] = picked
    if extra_env:
        env.update(extra_env)
    return env


def _self_check() -> None:
    """Assert-based check of the pure layout logic (no model / GPU)."""
    from pathlib import Path as _P

    F = 81
    assert resolve_frames([-1], F) == [80]
    assert resolve_frames([1, 2, 3, 4, 40, 60, -1], F) == [1, 2, 3, 4, 40, 60, 80]
    for bad in ([0, 0], [81], [-82]):
        try:
            resolve_frames(bad, F)
            raise AssertionError(f"expected error for {bad}")
        except ValueError:
            pass

    prefix = [_P(f"p{i}.png") for i in range(5)]  # -> frames 0,1,2,3,4
    future = [_P("f0.png"), _P("f1.png"), _P("f2.png")]
    kfs, frames = build_keyframe_layout(prefix, future, [40, 60, -1], F)
    assert frames == [0, 1, 2, 3, 4, 40, 60, -1]
    assert kfs[0] == "p0.png" and kfs[-1] == "f2.png"
    # A future keyframe colliding with the prefix range must fail fast.
    try:
        build_keyframe_layout(prefix, [_P("x.png")], [3], F)
        raise AssertionError("expected collision between prefix frame 3 and future keyframe 3")
    except ValueError:
        pass
    print("keyframes2video layout self-check PASSED")


if __name__ == "__main__":
    _self_check()
