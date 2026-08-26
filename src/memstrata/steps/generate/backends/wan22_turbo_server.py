"""Persistent server for Wan2.2-TI2V-5B-Turbo (4-step, CFG-free) segment generation.

The Turbo checkpoint is a bare distilled DiT that only runs under its upstream repo's
Self-Forcing pipeline (``Wan22FewstepInferencePipeline``), vendored at
``models/vendor/wan22_ti2v5b_turbo``. Loading it costs ~5 min (base 5B shards + the 19.9 GB
``model.pt`` + T5 + VAE off shared storage) while a 4-step 480x832/81f generation costs
~25 s at the native 704x1280, so the weights must stay resident: this server loads once and then
serves segments over the same tiny inbox/outbox file queue as the VACE / LightX2V servers.

Rendering happens at the checkpoint's native 704x1280 and the clip is downscaled once, on the
decoded tensor, to whatever size the caller asks to be delivered. Rendering smaller is not the
saving it looks like -- ``seq_len`` is hardcoded to the 720p sequence length upstream, so the DiT
runs the same width regardless -- and off-native renders visibly shake.

Unlike the LightX2V server this one covers BOTH routes with one resident model, since
TI2V-5B is natively text+image conditioned:
  * ``first_frame_path`` present -> I2V (composed keyframe, or previous segment's last frame)
  * ``first_frame_path`` absent  -> T2V (no visual memory; also the generator-floor control)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)  # atomic on POSIX => the client never reads a partial result


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _server_dirs(server_dir: Path) -> tuple[Path, Path]:
    inbox = server_dir / "inbox"
    outbox = server_dir / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    return inbox, outbox


def _next_job(inbox: Path) -> Path | None:
    jobs = sorted(p for p in inbox.glob("*.json"))
    return jobs[0] if jobs else None


def build_pipe(args: argparse.Namespace):
    """Load the Turbo pipeline once (mirrors upstream ``wan2.2_fewstep.py``)."""
    # The upstream wrappers resolve every base component through the RELATIVE path
    # ``wan_models/<model_name>/`` (T5 .pth, VAE .pth, transformer shards), so the vendored
    # repo must be both importable AND the process cwd.
    repo = Path(args.repo).resolve()
    if not (repo / "pipeline" / "wan22_fewstep_inference.py").is_file():
        raise SystemExit(f"not a Wan2.2-TI2V-5B-Turbo checkout: {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import torch
    from omegaconf import OmegaConf
    from pipeline.wan22_fewstep_inference import Wan22FewstepInferencePipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    t0 = time.time()
    pipe = Wan22FewstepInferencePipeline(OmegaConf.load(args.config_path))
    state = torch.load(Path(args.checkpoint_folder) / "model.pt", map_location="cpu")
    clean: dict = {}
    for raw_key, value in state.items():
        key = raw_key
        for junk in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
            key = key.replace(junk, "")
        clean[key] = value
    _, unexpected = pipe.generator.load_state_dict(clean, strict=False)
    if unexpected:
        # Upstream asserts on this too: unexpected keys mean the checkpoint does not match
        # the pipeline's generator, so silently continuing would generate noise.
        raise RuntimeError(f"unexpected keys in Turbo checkpoint: {unexpected[:8]}")
    pipe = pipe.to(device="cuda", dtype=torch.bfloat16)
    logging.info("[wan22_turbo] weights resident after %.1fs", time.time() - t0)
    return pipe


def run_job(pipe, request: dict, args: argparse.Namespace) -> tuple[Path, str]:
    import torch
    import torchvision.transforms.functional as TF
    from diffusers.utils import export_to_video
    from PIL import Image

    height = int(request.get("height") or args.height)
    width = int(request.get("width") or args.width)
    out_height = int(request.get("out_height") or args.out_height or height)
    out_width = int(request.get("out_width") or args.out_width or width)
    num_frames = int(request.get("num_frames") or args.num_frames)
    if num_frames % 4 != 1:
        raise ValueError(f"num_frames must be one more than a multiple of 4, got {num_frames}")
    if height % 16 or width % 16:
        raise ValueError(f"height/width must be multiples of 16, got {height}x{width}")
    out_path = Path(request["save_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_latent = None
    mode = "t2v"
    ref = str(request.get("first_frame_path") or "").strip()
    if ref:
        img = Image.open(ref).convert("RGB").resize((width, height), Image.LANCZOS)
        tensor = (
            TF.to_tensor(img)
            .sub_(0.5)
            .div_(0.5)
            .to("cuda")
            .unsqueeze(1)
            .to(dtype=torch.bfloat16)
        )
        image_latent = pipe.vae.encode_to_latent(tensor.unsqueeze(0))
        mode = "i2v"

    noise = torch.randn(
        1,
        (num_frames - 1) // 4 + 1,
        48,
        height // 16,
        width // 16,
        generator=torch.Generator(device="cuda").manual_seed(int(request.get("seed", args.seed))),
        dtype=torch.bfloat16,
        device="cuda",
    )
    torch.cuda.synchronize()
    t0 = time.time()
    video = pipe.inference(
        noise=noise,
        text_prompts=[str(request.get("prompt") or "")],
        wan22_image_latent=image_latent,
    )[0]
    torch.cuda.synchronize()
    denoise_s = time.time() - t0
    if (out_height, out_width) != (height, width):
        # Generate at the checkpoint's native size, deliver at the benchmark's size. Off its
        # native 704x1280 this model produces per-frame micro-jitter (measured: accel 0.49 px /
        # ratio 1.13 at 480x832 versus 0.17 px / ratio 0.09 at 704x1280 on the same keyframe,
        # prompt and seed), and because ``seq_len`` is hardcoded to the 720p sequence length the
        # smaller render barely saves compute anyway. Resizing the decoded tensor keeps this to a
        # single encode.
        video = torch.nn.functional.interpolate(
            video, size=(out_height, out_width), mode="bilinear",
            align_corners=False, antialias=True,
        )
    export_to_video(video.permute(0, 2, 3, 1).cpu().numpy(), str(out_path), fps=int(args.fps))
    logging.info(
        "[wan22_turbo] %s render %dx%d -> deliver %dx%d %df: denoise+decode %.1fs",
        mode, width, height, out_width, out_height, num_frames, denoise_s,
    )
    return out_path, mode


def serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    server_dir = Path(args.server_dir)
    inbox, outbox = _server_dirs(server_dir)

    pipe = build_pipe(args)
    (server_dir / "ready").write_text(str(os.getpid()))
    logging.info("[wan22_turbo] server ready at %s", server_dir)

    idle_timeout = float(args.idle_timeout)
    last_job = time.time()
    while True:
        if (server_dir / "stop").exists():
            break
        job_path = _next_job(inbox)
        if job_path is None:
            if time.time() - last_job > idle_timeout:
                logging.info("[wan22_turbo] idle timeout reached; exiting")
                break
            time.sleep(1.0)
            continue
        request = _read_json(job_path)
        job_path.unlink(missing_ok=True)
        job_id = str(request["job_id"])
        t0 = time.time()
        try:
            out_path, mode = run_job(pipe, request, args)
            _atomic_write_json(
                outbox / f"{job_id}.json",
                {"status": "ok", "out_video": str(out_path), "mode": mode},
            )
            logging.info("[wan22_turbo] job %s (%s) done in %.1fs", job_id, mode, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - propagate failure through the result file
            logging.exception("[wan22_turbo] job failed: %s", job_id)
            _atomic_write_json(outbox / f"{job_id}.json", {"status": "error", "error": repr(exc)})
        last_job = time.time()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="vendored Wan2.2-TI2V-5B-Turbo checkout")
    p.add_argument("--checkpoint_folder", required=True, help="dir holding the Turbo model.pt")
    p.add_argument("--config_path", required=True, help="upstream inference yaml (4-step schedule)")
    p.add_argument("--server_dir", required=True)
    p.add_argument("--idle_timeout", type=float, default=1800)
    p.add_argument("--height", type=int, default=704, help="render height (checkpoint native)")
    p.add_argument("--width", type=int, default=1280, help="render width (checkpoint native)")
    p.add_argument("--out_height", type=int, default=0, help="delivered height (0 = same as render)")
    p.add_argument("--out_width", type=int, default=0, help="delivered width (0 = same as render)")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


if __name__ == "__main__":
    serve(_parse_args())
