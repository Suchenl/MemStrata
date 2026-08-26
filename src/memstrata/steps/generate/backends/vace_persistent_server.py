"""Persistent VACE inference server: build the Wan/VACE model once, then serve segment jobs.

Launched once per production run (single GPU, or via ``torch.distributed.run`` for multi-GPU). It
builds ``WanVace`` a single time and loops over a file job queue, so the production loop stops paying
the T5+VAE+DiT reload cost on every segment (see ``.cursor/rules/persistent-model-serving.mdc``).

Run (single GPU):
    python -m memstrata.steps.generate.backends.vace_persistent_server \
        --vace_module_root <VACE>/vace --server_dir <dir> --ckpt_dir <ckpt> --model_name vace-1.3B \
        --size 832*480

The distributed loop mirrors ``vace_wan_inference.main``: rank 0 owns the queue and broadcasts each
job dict to the other ranks, all ranks run ``generate`` collectively, rank 0 writes the result.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from memstrata.steps.generate.backends.vace_job_queue import (
    _read_json,
    next_job,
    server_dirs,
    write_result,
)
from memstrata.steps.generate.backends.vace_wan_entrypoint import (
    _install_plain_prompt_annotator_shim,
    _install_sdpa_attention_fallback,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Persistent VACE inference server")
    p.add_argument("--vace_module_root", required=True)
    p.add_argument("--server_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--model_name", default="vace-1.3B")
    p.add_argument("--size", default="832*480", help="Default size key (jobs may override).")
    p.add_argument("--offload_model", default="false")
    p.add_argument("--t5_cpu", action="store_true")
    p.add_argument("--convert_model_dtype", action="store_true")
    p.add_argument("--t5_fsdp", action="store_true")
    p.add_argument("--dit_fsdp", action="store_true")
    p.add_argument("--ulysses_size", type=int, default=1)
    p.add_argument("--ring_size", type=int, default=1)
    p.add_argument("--idle_timeout", type=float, default=1800.0, help="Exit after this many idle seconds.")
    return p


def main() -> None:
    args = _parser().parse_args()

    module_root = str(Path(args.vace_module_root).resolve())
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
    _install_plain_prompt_annotator_shim()
    _install_sdpa_attention_fallback()

    import torch
    import torch.distributed as dist

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        if args.ulysses_size > 1 or args.ring_size > 1:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=args.ring_size,
                ulysses_degree=args.ulysses_size,
            )

    is_wan_i2v = "i2v" in args.model_name
    if is_wan_i2v:
        import wan
        from wan.configs import SIZE_CONFIGS, WAN_CONFIGS, MAX_AREA_CONFIGS
        try:
            from wan.utils.utils import cache_video
        except ImportError:
            from wan.utils.utils import save_video as cache_video
        cfg = WAN_CONFIGS[args.model_name]
        logging.info("Building WanI2V once (persistent server)...")
        import inspect
        sig = inspect.signature(wan.WanI2V)
        wan_kwargs = {
            "config": cfg,
            "checkpoint_dir": args.ckpt_dir,
            "device_id": device,
            "rank": rank,
            "t5_fsdp": args.t5_fsdp,
            "dit_fsdp": args.dit_fsdp,
            "t5_cpu": args.t5_cpu,
        }
        sp_val = (args.ulysses_size > 1 or args.ring_size > 1)
        if "use_sp" in sig.parameters:
            wan_kwargs["use_sp"] = sp_val
        else:
            wan_kwargs["use_usp"] = sp_val
        if "convert_model_dtype" in sig.parameters:
            wan_kwargs["convert_model_dtype"] = args.convert_model_dtype
        wan_vace = wan.WanI2V(**wan_kwargs)
    else:
        from models.wan import WanVace
        from models.wan.configs import SIZE_CONFIGS, WAN_CONFIGS
        try:
            from wan.utils.utils import cache_video
        except ImportError:
            from wan.utils.utils import save_video as cache_video
        cfg = WAN_CONFIGS[args.model_name]
        logging.info("Building WanVace once (persistent server)...")
        wan_vace = WanVace(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
    offload = str(args.offload_model).lower() in ("1", "true", "yes")

    server_dir = Path(args.server_dir)
    inbox, outbox = server_dirs(server_dir)
    if rank == 0:
        (server_dir / "ready").write_text(str(os.getpid()))
        logging.info("VACE server ready; polling %s", inbox)

    last_active = time.time()
    try:
        while True:
            req = None
            if rank == 0:
                job_path = next_job(inbox)
                if job_path is not None:
                    try:
                        req = _read_json(job_path)
                    except Exception:  # noqa: BLE001 - malformed request, drop it
                        req = None
                    job_path.unlink(missing_ok=True)
                elif (server_dir / "stop").exists() or (time.time() - last_active) > args.idle_timeout:
                    req = {"__stop__": True}
            if world_size > 1:
                box = [req]
                dist.broadcast_object_list(box, src=0)
                req = box[0]
            if req is None:
                time.sleep(0.5)
                continue
            if req.get("__stop__"):
                break
            last_active = time.time()
            _run_job(wan_vace, cfg, SIZE_CONFIGS, cache_video, req, device, offload, rank, outbox, is_wan_i2v=is_wan_i2v)
    finally:
        if rank == 0:
            (server_dir / "ready").unlink(missing_ok=True)
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_job(wan_vace, cfg, SIZE_CONFIGS, cache_video, req, device, offload, rank, outbox, is_wan_i2v=False) -> None:
    size = SIZE_CONFIGS[req["size"]]
    try:
        if is_wan_i2v:
            from wan.configs import MAX_AREA_CONFIGS
            from memstrata.lib.media import load_crop_rgb_for_model

            img = load_crop_rgb_for_model(req["src_ref_images"][0])
            video = wan_vace.generate(
                req["prompt"],
                img,
                max_area=MAX_AREA_CONFIGS[req["size"]],
                frame_num=int(req["frame_num"]),
                shift=float(req["sample_shift"]),
                sample_solver=req.get("sample_solver", "unipc"),
                sampling_steps=int(req["sample_steps"]),
                guide_scale=float(req["sample_guide_scale"]),
                seed=int(req["base_seed"]),
                offload_model=offload,
            )
        else:
            src_video, src_mask, src_ref = wan_vace.prepare_source(
                [req.get("src_video")],
                [req.get("src_mask")],
                [req.get("src_ref_images") or None],
                int(req["frame_num"]),
                size,
                device,
            )
            video = wan_vace.generate(
                req["prompt"],
                src_video,
                src_mask,
                src_ref,
                size=size,
                frame_num=int(req["frame_num"]),
                shift=float(req["sample_shift"]),
                sample_solver=req.get("sample_solver", "unipc"),
                sampling_steps=int(req["sample_steps"]),
                guide_scale=float(req["sample_guide_scale"]),
                seed=int(req["base_seed"]),
                offload_model=offload,
            )
        if rank == 0:
            save_file = req["save_file"]
            Path(save_file).parent.mkdir(parents=True, exist_ok=True)
            cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            write_result(outbox, req["job_id"], {"status": "ok", "out_video": save_file})
            logging.info("job %s -> %s", req["job_id"], save_file)
    except Exception as exc:  # noqa: BLE001 - report failure to the client, keep serving
        if rank == 0:
            write_result(outbox, req["job_id"], {"status": "error", "error": repr(exc)})
        logging.exception("job %s failed", req.get("job_id"))


if __name__ == "__main__":
    main()
