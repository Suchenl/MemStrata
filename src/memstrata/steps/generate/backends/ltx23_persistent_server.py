"""Persistent LTX-2.3 inference server: build the LTX-2.3 model once, then serve segment jobs.

Launched once per production run. It builds the pipeline a single time and loops over a file job queue,
so the production loop stops paying the Gemma-3 + Transformer reload cost on every segment.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch

# Ensure package importable when launched as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memstrata.steps.generate.backends.vace_job_queue import (
    _read_json,
    next_job,
    server_dirs,
    write_result,
)

# Import LTX-2.3 components
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.tools import LatentTools
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.helpers import audio_latent_from_file, video_latent_from_file
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class AudioConditionByPrefixLatent(ConditioningItem):
    """Conditions audio generation by injecting latents at the beginning of the audio sequence."""

    def __init__(self, latent: torch.Tensor, strength: float = 1.0):
        self.latent = latent
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)
        num_cond_tokens = tokens.shape[1]

        latent_state = latent_state.clone()
        start = 0
        stop = start + num_cond_tokens

        latent_state.clean_latent[:, start:stop] = tokens
        latent_state.denoise_mask[:, start:stop] = 1.0 - self.strength

        return latent_state


class VideoConditionByLatentIndex(ConditioningItem):
    """Conditions video generation by injecting latents at a specific latent frame index."""

    def __init__(self, latent: torch.Tensor, strength: float, latent_idx: int):
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)
        start_token = latent_tools.patchifier.get_token_count(
            latent_tools.target_shape._replace(frames=self.latent_idx)
        )
        stop_token = start_token + tokens.shape[1]

        latent_state = latent_state.clone()
        latent_state.clean_latent[:, start_token:stop_token] = tokens
        latent_state.denoise_mask[:, start_token:stop_token] = 1.0 - self.strength

        return latent_state


class LTX23ContinuationPipeline:
    """Custom LTX-2.3 pipeline that supports true video and audio prefix continuation."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        device: torch.device | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ):
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self._scheduler = LTX2Scheduler()

        logging.info("Loading PromptEncoder (Gemma-3)...")
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            offload_mode=offload_mode,
        )

        logging.info("Loading ImageConditioner...")
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
        )

        logging.info("Loading AudioConditioner...")
        self.audio_conditioner = AudioConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
        )

        logging.info("Loading DiffusionStage (Transformer)...")
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            offload_mode=offload_mode,
        )

        logging.info("Loading VideoDecoder...")
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
        )

        logging.info("Loading AudioDecoder...")
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[Any],
        video_prefix_path: str | None = None,
        audio_prefix_path: str | None = None,
        max_batch_size: int = 1,
        return_latents: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio] | tuple[torch.Tensor, torch.Tensor, torch.Generator]:
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        logging.info("Encoding prompts...")
        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Build video conditionings
        logging.info("Encoding video conditionings...")

        def build_video_conds(enc):
            conds = combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
            if video_prefix_path:
                meta = get_videostream_metadata(video_prefix_path)
                fps = meta.fps
                total_frames = meta.frames
                # Condition on the last 17 frames of the previous segment (3 latent frames)
                num_prefix_frames = min(17, total_frames)
                start_time = max(0.0, (total_frames - num_prefix_frames) / fps)
                max_duration = num_prefix_frames / fps

                logging.info(
                    f"Loading video prefix from {video_prefix_path} (frames={num_prefix_frames}, start={start_time:.2f}s)"
                )
                prefix_latent = video_latent_from_file(
                    video_encoder=enc,
                    file_path=video_prefix_path,
                    output_shape=VideoPixelShape(batch=1, frames=num_prefix_frames, height=height, width=width, fps=fps),
                    device=self.device,
                    dtype=dtype,
                    start_time=start_time,
                    max_duration=max_duration,
                )
                prefix_cond = VideoConditionByLatentIndex(
                    latent=prefix_latent,
                    strength=1.0,
                    latent_idx=0,
                )
                conds.append(prefix_cond)
            return conds

        stage_1_conditionings = self.image_conditioner(build_video_conds)

        # Build audio conditionings
        audio_conditionings = []
        if audio_prefix_path:
            logging.info("Encoding audio conditionings...")

            def build_audio_conds(enc):
                meta = get_videostream_metadata(audio_prefix_path)
                fps = meta.fps
                total_frames = meta.frames
                # Condition on the last 1.0s of the previous segment's audio
                prev_duration = total_frames / fps
                start_time = max(0.0, prev_duration - 1.0)
                max_duration = 1.0

                logging.info(
                    f"Loading audio prefix from {audio_prefix_path} (start={start_time:.2f}s, duration={max_duration:.2f}s)"
                )
                prefix_latent = audio_latent_from_file(
                    audio_encoder=enc,
                    file_path=audio_prefix_path,
                    output_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
                    device=self.device,
                    dtype=dtype,
                    start_time=start_time,
                    max_duration=max_duration,
                )
                if prefix_latent is not None:
                    return [AudioConditionByPrefixLatent(latent=prefix_latent, strength=1.0)]
                return []

            audio_conditionings = self.audio_conditioner(build_audio_conds)

        sigmas = self._scheduler.execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        from ltx_core.components.guiders import create_multimodal_guider_factory

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        logging.info("Running diffusion stage...")
        video_state, audio_state = self.stage(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                conditionings=audio_conditionings,
            ),
            max_batch_size=max_batch_size,
        )

        if return_latents:
            return video_state.latent, audio_state.latent, generator

        logging.info("Decoding video and audio...")
        decoded_video = self.video_decoder(video_state.latent, None, generator=generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Persistent LTX-2.3 inference server")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--gemma_root", required=True)
    p.add_argument("--server_dir", required=True)
    p.add_argument("--idle_timeout", type=float, default=1800.0)
    return p


def main() -> None:
    args = _parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    device = get_device()
    logging.info(f"Building LTX23ContinuationPipeline once (persistent server) on device {device}...")
    pipeline = LTX23ContinuationPipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        device=device,
        offload_mode=OffloadMode.CPU,
    )

    server_dir = Path(args.server_dir)
    inbox, outbox = server_dirs(server_dir)

    (server_dir / "ready").write_text(str(os.getpid()))
    logging.info("LTX-2.3 server ready; polling %s", inbox)

    last_active = time.time()
    try:
        while True:
            req = None
            job_path = next_job(inbox)
            if job_path is not None:
                try:
                    req = _read_json(job_path)
                except Exception:
                    req = None
                job_path.unlink(missing_ok=True)
            elif (server_dir / "stop").exists() or (time.time() - last_active) > args.idle_timeout:
                req = {"__stop__": True}

            if req is None:
                time.sleep(0.5)
                continue
            if req.get("__stop__"):
                break

            last_active = time.time()
            _run_job(pipeline, req, outbox)
    finally:
        (server_dir / "ready").unlink(missing_ok=True)


@torch.inference_mode()
def _run_job(pipeline: LTX23ContinuationPipeline, req: dict[str, Any], outbox: Path) -> None:
    job_id = req["job_id"]
    save_file = req["save_file"]
    logging.info(f"Running job {job_id} -> {save_file}")

    try:
        # Parse images/conditioning inputs
        images = []
        image_conditioning_crf = int(req.get("image_conditioning_crf", 0))
        if req.get("keyframes"):
            from ltx_pipelines.utils.args import ImageConditioningInput
            for kf in req["keyframes"]:
                images.append(ImageConditioningInput(
                    path=kf["path"],
                    frame_idx=int(kf.get("frame_idx", 0)),
                    strength=float(kf.get("strength", 1.0)),
                    crf=int(kf.get("crf", image_conditioning_crf)),
                ))
        elif req.get("first_frame_path"):
            from ltx_pipelines.utils.args import ImageConditioningInput

            images.append(
                ImageConditioningInput(
                    path=req["first_frame_path"],
                    frame_idx=0,
                    strength=1.0,
                    crf=image_conditioning_crf,
                )
            )

        async_decode = req.get("async_decode", False)

        if async_decode:
            logging.info(f"Running job {job_id} with async decoding enabled")
            video_latent, audio_latent, generator = pipeline(
                prompt=req["prompt"],
                negative_prompt=req.get("negative_prompt", ""),
                seed=int(req.get("seed", 42)),
                height=int(req.get("height", 480)),
                width=int(req.get("width", 832)),
                num_frames=int(req.get("num_frames", 49)),
                frame_rate=float(req.get("frame_rate", 16.0)),
                num_inference_steps=int(req.get("num_inference_steps", 8)),
                video_guider_params=MultiModalGuiderParams(
                    cfg_scale=float(req.get("video_cfg_guidance_scale", 1.0)),
                    stg_scale=float(req.get("video_stg_guidance_scale", 0.0)),
                ),
                audio_guider_params=MultiModalGuiderParams(
                    cfg_scale=float(req.get("audio_cfg_guidance_scale", 1.0)),
                    stg_scale=float(req.get("audio_stg_guidance_scale", 0.0)),
                ),
                images=images,
                video_prefix_path=req.get("video_prefix_path"),
                audio_prefix_path=req.get("audio_prefix_path"),
                return_latents=True,
            )

            import threading
            def decode_and_encode_task():
                try:
                    logging.info(f"Background thread starting VAE decode for job {job_id}")
                    with torch.inference_mode():
                        decoded_video = pipeline.video_decoder(video_latent, None, generator=generator)
                        decoded_audio = pipeline.audio_decoder(audio_latent)
                        
                        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
                        encode_video(
                            video=decoded_video,
                            fps=float(req.get("frame_rate", 16.0)),
                            audio=decoded_audio,
                            output_path=save_file,
                            video_segments_number=1,
                        )
                    logging.info(f"Background thread completed decode and encode for job {job_id}")
                except Exception as exc:
                    logging.exception(f"Background thread failed for job {job_id}")

            threading.Thread(target=decode_and_encode_task, daemon=True).start()
            logging.info(f"Job {job_id} diffusion finished, async decode thread spawned")
            write_result(outbox, job_id, {"status": "ok", "out_video": save_file})
        else:
            video, audio = pipeline(
                prompt=req["prompt"],
                negative_prompt=req.get("negative_prompt", ""),
                seed=int(req.get("seed", 42)),
                height=int(req.get("height", 480)),
                width=int(req.get("width", 832)),
                num_frames=int(req.get("num_frames", 49)),
                frame_rate=float(req.get("frame_rate", 16.0)),
                num_inference_steps=int(req.get("num_inference_steps", 8)),
                video_guider_params=MultiModalGuiderParams(
                    cfg_scale=float(req.get("video_cfg_guidance_scale", 1.0)),
                    stg_scale=float(req.get("video_stg_guidance_scale", 0.0)),
                ),
                audio_guider_params=MultiModalGuiderParams(
                    cfg_scale=float(req.get("audio_cfg_guidance_scale", 1.0)),
                    stg_scale=float(req.get("audio_stg_guidance_scale", 0.0)),
                ),
                images=images,
                video_prefix_path=req.get("video_prefix_path"),
                audio_prefix_path=req.get("audio_prefix_path"),
                return_latents=False,
            )

            Path(save_file).parent.mkdir(parents=True, exist_ok=True)
            encode_video(
                video=video,
                fps=float(req.get("frame_rate", 16.0)),
                audio=audio,
                output_path=save_file,
                video_segments_number=1,
            )

            write_result(outbox, job_id, {"status": "ok", "out_video": save_file})
            logging.info(f"Job {job_id} completed successfully")
    except Exception as exc:
        logging.exception(f"Job {job_id} failed")
        write_result(outbox, job_id, {"status": "error", "error": repr(exc)})


if __name__ == "__main__":
    main()
