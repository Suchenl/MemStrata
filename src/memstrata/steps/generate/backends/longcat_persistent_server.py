"""Persistent LongCat-Video inference server.

Launched once per MemStrata production run. It loads the official LongCat-Video
pipeline once and serves sequential segment jobs through the shared file queue.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from diffusers.utils import load_video
from torchvision.io import write_video
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline


class LongCatServer:
    def __init__(self, checkpoint_dir: str, *, enable_compile: bool = False) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.enable_compile = enable_compile
        self.rank = int(os.environ.get("RANK", "0"))
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 0:
            raise RuntimeError("LongCat-Video requires CUDA")
        self.local_rank = self.rank % num_gpus
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=24))
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.world_size != 1:
            raise RuntimeError("LongCat persistent server currently supports one process/GPU only")

        init_context_parallel(
            context_parallel_size=self.world_size,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )
        cp_split_hw = context_parallel_util.get_optimal_split(context_parallel_util.get_cp_size())

        logging.info("Loading LongCat tokenizer/text encoder/VAE/scheduler/DiT...")
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLWan.from_pretrained(
            checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
        )
        dit = LongCatVideoTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder="dit",
            cp_split_hw=cp_split_hw,
            torch_dtype=torch.bfloat16,
        )
        if enable_compile:
            dit = torch.compile(dit)
        self.pipe = LongCatVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
        )
        self.pipe.to(self.local_rank)
        self._distill_loaded = False

    def ensure_distill(self) -> None:
        if self._distill_loaded:
            return
        lora_path = Path(self.checkpoint_dir) / "lora" / "cfg_step_lora.safetensors"
        self.pipe.dit.load_lora(str(lora_path), "cfg_step_lora")
        self.pipe.dit.enable_loras(["cfg_step_lora"])
        self._distill_loaded = True

    @torch.inference_mode()
    def run_job(self, request: dict[str, Any]) -> Path:
        if bool(request.get("use_distill", True)):
            self.ensure_distill()

        mode = str(request.get("mode") or "t2v")
        out_path = Path(str(request["save_file"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=self.local_rank).manual_seed(
            int(request.get("seed", 42)) + self.global_rank
        )
        fps = int(request.get("fps", 15))
        common = {
            "prompt": str(request["prompt"]),
            "num_frames": int(request.get("num_frames", 49)),
            "num_inference_steps": int(request.get("num_inference_steps", 16)),
            "use_distill": bool(request.get("use_distill", True)),
            "guidance_scale": float(request.get("guidance_scale", 1.0)),
            "generator": generator,
        }
        negative_prompt = request.get("negative_prompt")
        if negative_prompt and not common["use_distill"]:
            common["negative_prompt"] = str(negative_prompt)

        if mode == "vc":
            frames = self._generate_vc(request, common)
        elif mode == "i2v":
            frames = self._generate_i2v(request, common)
        elif mode == "t2v":
            frames = self._generate_t2v(request, common)
        else:
            raise ValueError(f"Unsupported LongCat mode: {mode}")
        _write_video(out_path, frames, fps)
        _torch_gc()
        return out_path

    def _generate_t2v(self, request: dict[str, Any], common: dict[str, Any]) -> np.ndarray:
        return self.pipe.generate_t2v(
            height=int(request.get("height", 480)),
            width=int(request.get("width", 832)),
            **common,
        )[0]

    def _generate_i2v(self, request: dict[str, Any], common: dict[str, Any]) -> np.ndarray:
        image_path = request.get("first_frame_path")
        if not image_path:
            raise ValueError("LongCat i2v job requires first_frame_path")
        from memstrata.lib.media import load_crop_rgb_for_model

        image = load_crop_rgb_for_model(image_path)
        return self.pipe.generate_i2v(
            image=image,
            resolution="480p",
            **common,
        )[0]

    def _generate_vc(self, request: dict[str, Any], common: dict[str, Any]) -> np.ndarray:
        source = request.get("source_video")
        if not source:
            raise ValueError("LongCat vc job requires source_video")
        video_path = Path(str(source))
        video = load_video(str(video_path))
        fps = int(request.get("fps", 15))
        stride = max(1, round(_fps(video_path) / fps))
        sampled = video[::stride]
        min_cond_frames = max(
            1,
            math.ceil(fps * float(request.get("min_condition_duration_sec", 1.0))),
        )
        num_cond_frames = min(
            len(sampled),
            max(int(request.get("num_cond_frames", 9)), min_cond_frames),
        )
        condition_video = sampled[-num_cond_frames:]
        output = self.pipe.generate_vc(
            video=condition_video,
            resolution="480p",
            num_cond_frames=num_cond_frames,
            use_kv_cache=bool(request.get("use_kv_cache", True)),
            offload_kv_cache=bool(request.get("offload_kv_cache", False)),
            enhance_hf=bool(request.get("enhance_hf", False)),
            **common,
        )[0]
        prefix = np.array([np.asarray(frame) / 255.0 for frame in condition_video])
        return np.concatenate([prefix, output[num_cond_frames:]], axis=0)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def server_dirs(server_dir: Path) -> tuple[Path, Path]:
    inbox = server_dir / "inbox"
    outbox = server_dir / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    return inbox, outbox


def next_job(inbox: Path) -> Path | None:
    jobs = sorted(p for p in Path(inbox).glob("*.json"))
    return jobs[0] if jobs else None


def write_result(outbox: Path, job_id: str, result: dict) -> None:
    _atomic_write_json(Path(outbox) / f"{job_id}.json", result)


def serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    server_dir = Path(args.server_dir)
    inbox, outbox = server_dirs(server_dir)
    server = LongCatServer(args.checkpoint_dir, enable_compile=args.enable_compile)
    if server.global_rank == 0:
        (server_dir / "ready").write_text(str(os.getpid()))
        logging.info("LongCat server ready at %s", server_dir)

    idle_timeout = float(args.idle_timeout)
    last_job = time.time()
    while True:
        if (server_dir / "stop").exists():
            break
        job_path = next_job(inbox)
        if job_path is None:
            if time.time() - last_job > idle_timeout:
                logging.info("Idle timeout reached; exiting")
                break
            time.sleep(1.0)
            continue
        request = _read_json(job_path)
        job_path.unlink(missing_ok=True)
        job_id = str(request["job_id"])
        try:
            out_path = server.run_job(request)
            write_result(outbox, job_id, {"status": "ok", "out_video": str(out_path)})
        except Exception as exc:  # noqa: BLE001 - propagate failure through result file
            logging.exception("LongCat job failed: %s", job_id)
            write_result(outbox, job_id, {"status": "error", "error": repr(exc)})
        last_job = time.time()

    if dist.is_initialized():
        dist.destroy_process_group()


def _fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    cap.release()
    return fps


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    processed = [(frame * 255).clip(0, 255).astype(np.uint8) for frame in frames]
    tensor = torch.from_numpy(np.array(processed))
    write_video(str(path), tensor, fps=fps, video_codec="libx264", options={"crf": "18"})


def _torch_gc() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--server_dir", required=True)
    parser.add_argument("--idle_timeout", type=float, default=1800)
    parser.add_argument("--enable_compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Ensure repo src is importable when launched from a run directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    serve(_parse_args())
