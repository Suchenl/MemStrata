"""Persistent FLUX.2 Klein inference server: build the model once, then serve image jobs.

Self-contained under the ``memstrata`` package. Launched
once per production run (single GPU) or Stage 0, it builds ``Flux2KleinKVPipeline`` once
and loops over a file job queue. This avoids reloading weights on every image task.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add MemStrata ``src`` to sys.path so ``memstrata.*`` resolves when run as a module.
# module lives at src/memstrata/steps/generate/image_backends/ -> parents[4] == src
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from memstrata.steps.generate.backends.vace_job_queue import (
    _read_json,
    next_job,
    server_dirs,
    write_result,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Persistent FLUX.2 Klein inference server")
    p.add_argument("--model_path", default="${PUBLIC_MODELS_ROOT}/black-forest-labs/FLUX.2-klein-9b-kv")
    p.add_argument("--server_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--idle_timeout", type=float, default=1800.0)
    return p


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    import torch
    from PIL import Image
    from diffusers import Flux2KleinKVPipeline
    from memstrata.steps.generate.image_backends.base import apply_photographic_grain
    from memstrata.lib.prompt_standardizer import standardize_prompt

    # Expand ${PUBLIC_MODELS_ROOT} / ~ so the default token works out of the box.
    model_path = os.path.expandvars(os.path.expanduser(args.model_path))
    logging.info("Building Flux2KleinKVPipeline once (persistent server) from %s...", model_path)
    pipe = Flux2KleinKVPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(args.device)

    server_dir = Path(args.server_dir)
    inbox, outbox = server_dirs(server_dir)
    (server_dir / "ready").write_text(str(os.getpid()))
    logging.info("FLUX.2 Klein server ready; polling %s", inbox)

    last_active = time.time()
    try:
        while True:
            job_path = next_job(inbox)
            req = None
            if job_path is not None:
                try:
                    req = _read_json(job_path)
                except Exception:
                    req = None
                job_path.unlink(missing_ok=True)
            elif (server_dir / "stop").exists() or (time.time() - last_active) > args.idle_timeout:
                break

            if req is None:
                time.sleep(0.5)
                continue

            last_active = time.time()
            try:
                height = int(req.get("height", 1024))
                width = int(req.get("width", 1024))
                steps = int(req.get("steps", 4))
                seed = req.get("seed")
                quality_preset = req.get("quality_preset")
                post_process_grain = req.get("post_process_grain", False)

                prompt = standardize_prompt(req["prompt"], "flux_klein", quality_preset)
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=args.device).manual_seed(int(seed))

                ref_paths = req.get("composed_references") or []
                kwargs = {
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": steps,
                    "generator": generator,
                }
                if ref_paths:
                    ref_image = Image.open(ref_paths[0]).convert("RGB")
                    kwargs["image"] = ref_image

                result = pipe(**kwargs)
                image = result.images[0]

                if post_process_grain or quality_preset == "raw_film":
                    image = apply_photographic_grain(image)

                save_file = req["save_file"]
                Path(save_file).parent.mkdir(parents=True, exist_ok=True)
                image.save(save_file)

                write_result(outbox, req["job_id"], {"status": "ok", "out_image": save_file})
                logging.info("job %s -> %s", req["job_id"], save_file)
            except Exception as exc:
                write_result(outbox, req["job_id"], {"status": "error", "error": repr(exc)})
                logging.exception("job %s failed", req.get("job_id"))
    finally:
        (server_dir / "ready").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
