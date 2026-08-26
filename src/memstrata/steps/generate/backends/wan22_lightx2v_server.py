"""Persistent Wan2.2-I2V-A14B (MoE, 4-step distilled) inference server via LightX2V.

Wan2.2-I2V-A14B is a *dual-expert* (high-noise + low-noise) MoE diffusion model. The
lightx2v step-distilled release runs it in **4 steps total** (2 high + 2 low) with **no
CFG** (guidance_scale=1.0, Euler scheduler, sample_shift=5.0) -- near-real-time i2v that is
much higher fidelity than a 4-6 step Helios. It has no long-range memory of its own, so it
pairs with MemStrata exactly like Helios: the external stratified memory supplies identity
through the FLUX-fused keyframe, this model animates it.

Both 14B experts stay resident (~2x28GB BF16 / ~2x15GB FP8), so we serve it persistently and
talk to it over the same tiny file queue as the Helios / VACE servers (inbox/outbox + ready).

The engine is the official ``lightx2v.LightX2VPipeline`` (model_cls="wan2.2_moe", task="i2v").
``--model_path`` must be a directory laid out the LightX2V way (see README_wan22_lightx2v.md)::

    <model_path>/
      high_noise_model/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p.safetensors
      low_noise_model/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p.safetensors
      google/  models_t5_*.pth  Wan2.1_VAE.pth  config.json  ...   (from base Wan2.2-I2V-A14B)

Run inside an env with lightx2v installed (see README). Generation params (steps/res/frames/
shift/guidance) are fixed at generator-creation time; per-job we only vary image/prompt/seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _server_dirs(server_dir: Path) -> tuple[Path, Path]:
    inbox = server_dir / "inbox"
    outbox = server_dir / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    return inbox, outbox


def _next_job(inbox: Path) -> Path | None:
    jobs = sorted(p for p in Path(inbox).glob("*.json"))
    return jobs[0] if jobs else None


def _iter_experts(runner) -> list:
    """Return the transformer model objects that may carry reversible LoRA deltas.

    For the Wan2.2 MoE distill runner this is ``[high_noise_model, low_noise_model]``
    (``runner.model.model``); for single-model runners it is ``[runner.model]``.
    """
    model = getattr(runner, "model", None)
    if model is None:
        return []
    inner = getattr(model, "model", None)
    if isinstance(inner, (list, tuple)):
        return [m for m in inner if m is not None]
    return [model]


class LoraToggle:
    """Reverses / re-applies the *statically merged* morphic LoRA on one persistent server.

    The LoRA is merged additively at load time (``W' = W + strength * B @ A``) and the
    exact per-tensor deltas are recorded on each expert (``model._lora_delta_records``)
    when ``lora_reversible`` is set. Those recorded tensors share storage with the resident
    weights (the weight containers hold ``.t()`` views on the non-offload GPU path), so
    mutating them in place mutates the resident model -- no reload, no second weight copy.

    We snapshot both states ONCE at init: the merged/ON weights as loaded, and the base/OFF
    weights (``merged - delta``). Toggling is then a plain ``copy_`` from the right snapshot,
    which is exact and drift-free (a repeated ``sub_``/``add_`` in bf16 is not bit-exact and
    would slowly drift the weights over many per-shot toggles). The ON snapshot equals the
    weights the model loaded with, so the LoRA-on default stays byte-identical to before.
    Snapshots live on CPU to avoid holding extra VRAM; state is tracked for idempotency.
    """

    def __init__(self, runner) -> None:
        import torch

        self._snaps: list[tuple] = []  # (param, base_cpu, merged_cpu)
        with torch.no_grad():
            for expert in _iter_experts(runner):
                recs = getattr(expert, "_lora_delta_records", None)
                if not recs:
                    continue
                for param, delta in recs:
                    merged_cpu = param.detach().to("cpu", copy=True)
                    base_cpu = merged_cpu - delta.to(merged_cpu.dtype)
                    self._snaps.append((param, base_cpu, merged_cpu))
                # Snapshots (base + merged) fully capture both states; drop the recorded
                # deltas so we don't keep a third CPU copy of the affected weights around.
                try:
                    expert._lora_delta_records = None
                except Exception:  # noqa: BLE001 - best-effort memory release
                    pass
        # LoRA starts merged (ON) right after create_generator().
        self._enabled = True

    @property
    def available(self) -> bool:
        return bool(self._snaps)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set(self, want_on: bool) -> bool:
        """Bring LoRA state to ``want_on``. Returns True if a toggle happened."""
        if not self._snaps:
            return False
        if bool(want_on) == self._enabled:
            return False
        import torch

        with torch.no_grad():
            for param, base_cpu, merged_cpu in self._snaps:
                src = merged_cpu if want_on else base_cpu
                param.data.copy_(src.to(param.device))
        torch.cuda.empty_cache()
        self._enabled = bool(want_on)
        return True


def build_pipe(args: argparse.Namespace):
    """Init the LightX2V Wan2.2 MoE i2v pipeline once. Kept resident for the server's life."""
    from lightx2v import LightX2VPipeline  # imported lazily: only the server env has lightx2v

    pipe = LightX2VPipeline(
        model_path=args.model_path,
        model_cls=args.model_cls,
        task="i2v",
    )
    # Optional VRAM offload (consumer GPUs). On A100/A800 leave it off for max speed.
    if args.offload:
        pipe.enable_offload(cpu_offload=True, offload_granularity="block")

    gen_kwargs: dict[str, Any] = dict(
        attn_mode=args.attn_mode,
        infer_steps=args.infer_steps,           # 4 = 2 high-noise + 2 low-noise
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        # MoE expects a per-expert list; distilled models run CFG-free -> [1.0, 1.0].
        guidance_scale=[args.guidance_high, args.guidance_low],
        sample_shift=args.sample_shift,
    )
    # A hand-authored lightx2v config JSON (which pins the 4-step 2+2 boundary + denoising list)
    # takes precedence when provided; otherwise rely on model_path auto-discovery of
    # high_noise_model/ + low_noise_model/.
    #
    # NB: set_config() only re-loads the JSON (the branch that injects target_video_length /
    # target_height / boundary_step_index / denoising_step_list into the runner config) when the
    # pipeline carries a ``config_json`` attribute holding the path. create_generator(config_json=)
    # populates attrs but does NOT set that attribute, so we set it explicitly here — otherwise the
    # WanDistillRunner scheduler raises KeyError('target_video_length').
    if args.config_json and Path(args.config_json).is_file():
        pipe.config_json = args.config_json
        pipe.create_generator(config_json=args.config_json, **gen_kwargs)
    else:
        pipe.create_generator(**gen_kwargs)
    return pipe


def _config_has_lora(config_json: str) -> bool:
    """True if the hand-authored lightx2v config JSON declares any lora_configs."""
    if not config_json or not Path(config_json).is_file():
        return False
    try:
        cfg = json.loads(Path(config_json).read_text())
    except (OSError, ValueError):
        return False
    return bool(cfg.get("lora_configs"))


def run_job(pipe, request: dict[str, Any], *, lora: "LoraToggle | None" = None, natural_lora: bool = True) -> Path:
    """One i2v generation: animate the composed keyframe (first_frame_path) with the prompt.

    ``use_lora`` (per-job, optional) toggles the morphic LoRA on/off for this shot. When
    absent it defaults to ``natural_lora`` (the backend config's natural state), so
    existing behavior is unchanged.
    """
    want_lora = bool(request.get("use_lora", natural_lora))
    if lora is not None and lora.available:
        toggled = lora.set(want_lora)
        logging.info(
            "[wan22_lightx2v] job %s lora=%s (state=%s%s)",
            request.get("job_id"), want_lora, "on" if lora.enabled else "off",
            ", toggled" if toggled else "",
        )
    elif "use_lora" in request and want_lora != natural_lora:
        # Requested a non-default state but no reversible LoRA is available to honor it.
        logging.warning(
            "[wan22_lightx2v] job %s requested use_lora=%s but no reversible LoRA is loaded; "
            "running in natural state (lora=%s)",
            request.get("job_id"), want_lora, natural_lora,
        )
    image_path = request.get("first_frame_path") or request.get("image_path")
    if not image_path or not Path(image_path).is_file():
        raise ValueError(f"wan22_lightx2v job needs a valid first_frame_path (got {image_path!r})")
    save_path = Path(request["save_file"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pipe.generate(
        seed=int(request.get("seed", 42)),
        image_path=str(Path(image_path).resolve()),
        prompt=str(request.get("prompt", "")),
        negative_prompt=str(request.get("negative_prompt", "")),
        save_result_path=str(save_path),
    )
    if not save_path.is_file():
        raise RuntimeError(f"lightx2v produced no output at {save_path}")
    return save_path


def serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    server_dir = Path(args.server_dir)
    inbox, outbox = _server_dirs(server_dir)

    pipe = build_pipe(args)
    natural_lora = _config_has_lora(args.config_json)
    lora = LoraToggle(pipe.runner) if natural_lora else None
    if lora is not None and lora.available:
        logging.info(
            "[wan22_lightx2v] reversible LoRA ready: %d delta tensors, per-job use_lora "
            "toggle enabled (natural state=on)", len(lora._records),
        )
    elif natural_lora:
        logging.warning(
            "[wan22_lightx2v] config declares lora_configs but no reversible deltas were "
            "recorded (set 'lora_reversible': true in the config JSON to enable per-job toggling)"
        )
    (server_dir / "ready").write_text(str(os.getpid()))
    logging.info("[wan22_lightx2v] server ready at %s", server_dir)

    idle_timeout = float(args.idle_timeout)
    last_job = time.time()
    while True:
        if (server_dir / "stop").exists():
            break
        job_path = _next_job(inbox)
        if job_path is None:
            if time.time() - last_job > idle_timeout:
                logging.info("[wan22_lightx2v] idle timeout reached; exiting")
                break
            time.sleep(1.0)
            continue
        request = _read_json(job_path)
        job_path.unlink(missing_ok=True)
        job_id = str(request["job_id"])
        t0 = time.time()
        try:
            out_path = run_job(pipe, request, lora=lora, natural_lora=natural_lora)
            _atomic_write_json(outbox / f"{job_id}.json", {"status": "ok", "out_video": str(out_path)})
            logging.info("[wan22_lightx2v] job %s done in %.1fs", job_id, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - propagate failure through result file
            logging.exception("[wan22_lightx2v] job failed: %s", job_id)
            _atomic_write_json(outbox / f"{job_id}.json", {"status": "error", "error": repr(exc)})
        last_job = time.time()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="LightX2V-layout dir (high_noise_model/ + low_noise_model/ + base t5/vae/clip/config)")
    p.add_argument("--model_cls", default="wan2.2_moe_distill",
                   help="LightX2V runner class. wan2.2_moe_distill = 4-step CFG-free distilled experts "
                        "(auto-discovers high/low_noise_model from model_path); wan2.2_moe = full 40-step MoE.")
    p.add_argument("--server_dir", required=True)
    p.add_argument("--config_json", default="", help="optional hand-authored lightx2v config JSON")
    p.add_argument("--idle_timeout", type=float, default=1800)
    p.add_argument("--infer_steps", type=int, default=4)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--guidance_high", type=float, default=1.0)
    p.add_argument("--guidance_low", type=float, default=1.0)
    p.add_argument("--attn_mode", default="sage_attn2")
    p.add_argument("--offload", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    serve(_parse_args())
