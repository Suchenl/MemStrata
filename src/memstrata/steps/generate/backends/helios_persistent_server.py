"""Persistent Helios-Distilled i2v inference server.

Launched once per MemStrata run by :class:`HeliosBackend`. Loads the official Helios
pipeline once (the ~5min cold start is paid a single time) and serves sequential
``image_to_video`` segment jobs through the shared ``vace_job_queue`` file protocol
(``inbox/`` -> ``outbox/``, ``ready``/``stop`` files).

The pipeline setup + i2v call faithfully mirror
``experiments/probe/helios_distilled_i2v_from_flux`` (which mirrors the official
``infer_helios.py``). It is dependency-light and imports **no** ``memstrata`` code, so it
runs cleanly inside the ``helios`` conda env. The env quirks (HF ``kernels`` FA2 revision
alias via ``sitecustomize``, ``MONTAGE_HELIOS_DEFER_KERNEL_LOAD``, offline HF caches,
``HELIOS_ROOT`` on ``PYTHONPATH``, ``IMAGEIO_FFMPEG_EXE``) are set by ``HeliosBackend._env``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image, load_video

# Helios rolling-window: history_sizes=[16,2,1] latent frames x temporal scale 4
# => a 73-pixel-frame window (+ always-on first-frame style anchor via keep_first_frame).
_HISTORY_PIXEL_WINDOW = 73

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)


def _pin_flash_attn2_offline() -> None:
    """Force the HF ``kernels`` lib to resolve ``flash-attn2`` to a locally-cached commit.

    On A800 the pipeline calls ``set_attention_backend('flash_hub')`` which asks
    ``kernels.get_kernel`` to resolve a revision; under ``HF_HUB_OFFLINE=1`` the default
    resolution enumerates Hub refs over the network and crashes. We pin the revision to a
    commit that is already in ``KERNELS_CACHE`` so resolution stays fully offline. Applied
    here (import time, before ``build_pipe``) instead of relying on ``sitecustomize`` auto
    import, which proved flaky under the 3-model closed loop.
    """
    pinned = "be3acec1c49820c5058b7c75b9b11ad27eb2fbd6"  # cached flash-attn2 v1 snapshot
    try:
        import kernels._versions as _kv
        import kernels.utils as _ku
    except Exception:  # noqa: BLE001 - kernels not present / different layout
        return
    original = _kv.select_revision_or_version

    def _patched(repo_id, *, revision=None, version=None):
        if repo_id == "kernels-community/flash-attn2":
            return pinned
        return original(repo_id, revision=revision, version=version)

    _kv.select_revision_or_version = _patched
    _ku.select_revision_or_version = _patched


_pin_flash_attn2_offline()

# Helios-Distilled is CFG-distilled and runs at guidance_scale=1.0, so the uncond branch
# never runs and negative_prompt is a NO-OP here. Motion is controlled by image_noise_sigma_*
# and the positive prompt (which must describe the moving crowd), not by a negative prompt.
DEFAULT_NEGATIVE = ""


def build_pipe(model_dir: str, device: torch.device) -> HeliosPipeline:
    dtype = torch.bfloat16
    logging.info("[helios] loading transformer ...")
    transformer = HeliosTransformer3DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=dtype
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()
    if torch.cuda.get_device_capability()[0] >= 9:
        try:
            transformer.set_attention_backend("_flash_3_hub")
        except Exception:  # noqa: BLE001
            transformer.set_attention_backend("flash_hub")
    else:
        transformer.set_attention_backend("flash_hub")  # A100/A800 => FA2
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(model_dir, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        model_dir, transformer=transformer, vae=vae, scheduler=scheduler, torch_dtype=dtype
    )
    return pipe.to(device)


def _seed_condition(request: dict[str, Any], w: int, h: int) -> tuple[Any, Any]:
    """Resolve the (image, video) conditioning for one job from its ``mode``.

    Returns ``(image, video)`` with exactly one populated (the Helios pipeline forbids both):

      * ``i2v``         -> image = the given first frame (FLUX keyframe / composite / anchor).
      * ``reanchor``    -> image = the LAST frame of the prior segment (soft continuity).
      * ``continue_ar`` -> video = [style_anchor] + last ~73 frames of the prior segment(s);
                           Helios keeps frame 0 as the style anchor (keep_first_frame=True)
                           and samples [16,2,1] long/mid/short history from the rest. This is
                           the ONLY faithful continuation: a single last frame (reanchor) can
                           carry appearance but NOT motion state, so actions jump-cut.
    """
    mode = str(request.get("mode", "i2v"))
    if mode == "continue_ar":
        history_video = request.get("history_video")
        if not history_video:
            raise ValueError("continue_ar requires history_video (the prior segment path)")
        frames = load_video(str(history_video))
        recent = frames[-_HISTORY_PIXEL_WINDOW:] if len(frames) > _HISTORY_PIXEL_WINDOW else frames
        anchor_path = request.get("style_anchor_path")
        anchor = load_image(str(anchor_path)) if anchor_path else recent[0]
        video = [anchor.resize((w, h))] + [f.resize((w, h)) for f in recent]
        return None, video
    if mode == "reanchor":
        history_video = request.get("history_video")
        if not history_video:
            raise ValueError("reanchor requires history_video (the prior segment path)")
        frames = load_video(str(history_video))
        return frames[-1].resize((w, h)), None
    # default i2v: a single given first frame
    image_path = request.get("first_frame_path")
    if not image_path:
        raise ValueError("i2v job requires first_frame_path (a FLUX keyframe / composite)")
    return load_image(str(image_path)).resize((w, h)), None


@torch.inference_mode()
def run_job(pipe: HeliosPipeline, request: dict[str, Any]) -> Path:
    out_path = Path(str(request["save_file"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = int(request.get("width", 832)), int(request.get("height", 480))
    fps = int(request.get("fps", 24))
    image, video = _seed_condition(request, w, h)
    is_skip_first_segment = bool(request.get("is_skip_first_segment", False))
    output = pipe(
        prompt=str(request["prompt"]),
        negative_prompt=request.get("negative_prompt") or DEFAULT_NEGATIVE,
        height=h,
        width=w,
        num_frames=int(request.get("num_frames", 121)),
        num_inference_steps=int(request.get("num_inference_steps", 6)),
        guidance_scale=float(request.get("guidance_scale", 1.0)),
        generator=torch.Generator(device="cuda").manual_seed(int(request.get("seed", 2026))),
        history_sizes=[16, 2, 1],
        num_latent_frames_per_segment=9,
        keep_first_frame=True,
        is_enable_stage2=True,
        pyramid_num_inference_steps_list=list(request.get("pyramid", [2, 2, 2])),
        is_skip_first_segment=is_skip_first_segment,
        is_amplify_first_segment=bool(request.get("is_amplify_first_segment", True)),
        use_zero_init=False,
        zero_steps=1,
        image=image,
        image_noise_sigma_min=float(request.get("sigma_min", 0.25)),
        image_noise_sigma_max=float(request.get("sigma_max", 0.4)),
        video=video,
        use_interpolate_prompt=False,
        interpolation_steps=3,
        interpolate_time_list=None,
    ).frames[0]
    export_to_video(output, str(out_path), fps=fps)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return out_path


# --- file-queue protocol (kept in sync with memstrata.steps.generate.backends.vace_job_queue) ---


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _server_dirs(server_dir: Path) -> tuple[Path, Path]:
    inbox, outbox = server_dir / "inbox", server_dir / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    return inbox, outbox


def _next_job(inbox: Path) -> Path | None:
    jobs = sorted(p for p in Path(inbox).glob("*.json"))
    return jobs[0] if jobs else None


def serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    server_dir = Path(args.server_dir)
    inbox, outbox = _server_dirs(server_dir)

    pipe = build_pipe(args.model, torch.device("cuda"))
    (server_dir / "ready").write_text(str(os.getpid()))
    logging.info("[helios] server ready at %s", server_dir)

    idle_timeout = float(args.idle_timeout)
    last_job = time.time()
    while True:
        if (server_dir / "stop").exists():
            break
        job_path = _next_job(inbox)
        if job_path is None:
            if time.time() - last_job > idle_timeout:
                logging.info("[helios] idle timeout reached; exiting")
                break
            time.sleep(1.0)
            continue
        request = _read_json(job_path)
        job_path.unlink(missing_ok=True)
        job_id = str(request["job_id"])
        t0 = time.time()
        try:
            out_path = run_job(pipe, request)
            _atomic_write_json(outbox / f"{job_id}.json", {"status": "ok", "out_video": str(out_path)})
            logging.info("[helios] job %s done in %.1fs", job_id, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - propagate failure through result file
            logging.exception("[helios] job failed: %s", job_id)
            _atomic_write_json(outbox / f"{job_id}.json", {"status": "error", "error": repr(exc)})
        last_job = time.time()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--server_dir", required=True)
    parser.add_argument("--idle_timeout", type=float, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    serve(_parse_args())
