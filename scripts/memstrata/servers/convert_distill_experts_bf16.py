"""Pre-cast the LightX2V 4-step distilled Wan2.2-I2V-A14B experts from on-disk F32 to bf16.

The released distilled safetensors are F32 (~57GB each); the native_wan pipeline runs them in bf16
anyway (convert_model_dtype). Persisting bf16 on disk halves the file (~28GB) and roughly halves the
cold-load read from shared storage. Streams tensor-by-tensor (peak RAM ~= one bf16 copy), so it does
not need 100GB+ of memory.

Usage:
  python convert_distill_experts_bf16.py [SRC_DIR] [DST_DIR]
Defaults: SRC = the F32 distilled dir the user provided; DST = a sibling *-bf16 dir with the SAME
filenames (so setup_wan_distill_native_weights.sh can relink the native ckpt dir to it unchanged).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FILES = (
    "wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors",
    "wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors",
)


def convert_one(src: Path, dst: Path) -> None:
    print(f"[bf16] {src.name}: reading + casting ...", flush=True)
    t0 = time.time()
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        md = f.metadata()
        if md:
            meta.update(md)
        n = 0
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype == torch.float32:
                t = t.to(torch.bfloat16)
            tensors[k] = t.contiguous()
            n += 1
            if n % 200 == 0:
                print(f"[bf16]   {src.name}: {n} tensors cast ({time.time()-t0:.0f}s)", flush=True)
    meta.setdefault("format", "pt")  # diffusers single-file loader expects this
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".safetensors.tmp")
    print(f"[bf16] {src.name}: writing {dst} ...", flush=True)
    save_file(tensors, str(tmp), metadata=meta)
    tmp.replace(dst)
    del tensors
    sz = dst.stat().st_size / 1e9
    print(f"[bf16] {src.name}: DONE -> {sz:.1f} GB in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    src_dir = Path(sys.argv[1] if len(sys.argv) > 1
                   else "${PUBLIC_MODELS_ROOT}/lightx2v/Wan2.2-Distill-Models")
    dst_dir = Path(sys.argv[2] if len(sys.argv) > 2
                   else "${PUBLIC_MODELS_ROOT}/lightx2v/Wan2.2-Distill-Models-bf16")
    for name in FILES:
        s = src_dir / name
        d = dst_dir / name
        if not s.is_file():
            print(f"[bf16] ERROR: missing {s}", flush=True)
            return 1
        if d.is_file() and d.stat().st_size > 0:
            print(f"[bf16] {name}: already exists, skipping", flush=True)
            continue
        convert_one(s, d)
    print(f"[bf16] all done. bf16 dir: {dst_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
