# Reproducing MemStrata paper numbers

`pytest` checks MemStrata contracts; it does not reproduce the paper tables. Use the
`paper-reproduction` branch for the frozen Track A Stage-1 implementation. `main` is
the moving production branch.

## Requirements

You need the matching paper-reproduction VMem-Bench checkout, licensed source videos,
model weights under `PUBLIC_MODELS_ROOT`, a pinned VLM judge, and a GPU. The repositories
do not redistribute LSMDC pixels or third-party model weights. Acquisition and layout
instructions are in VMem-Bench's [`docs/DATA.md`](https://github.com/Suchenl/VMem-Bench/blob/paper-reproduction/docs/DATA.md).

## Track A Stage 1

Keep the repositories side by side and run the adapter from VMem-Bench:

```bash
cd ../VMem-Bench
export MEMSTRATA_SRC="../MemStrata-paper/src"
export MEMSTRATA_TRACKA_NAME_SOURCE=mllm
python3 scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py \
  --adapter memstrata \
  --movie-dir assets/trackA/BlenderOpenMovies/big_buck_bunny \
  --budget 16 \
  --limit 2
```

`MEMSTRATA_TRACKA_NAME_SOURCE=mllm` is required for the paper mechanism. Score the
resulting `visual_selections` with VMem-Bench's `visual_coverage` command.

## GPU production

After installing the generator environments and setting `PUBLIC_MODELS_ROOT`, the
turnkey production launcher is:

```bash
bash scripts/memstrata/run_production.sh
```

Use [`MODELS.md`](MODELS.md) for WeDetect-Ref, DINOv3, Qwen, FLUX, Wan/LightX2V,
and optional SAM3 setup. The `recording` backend is only a CPU plumbing check and is
not a paper reproduction.

```bash
PYTHONPATH=src python3 -m memstrata.production.run \
  --backend recording --decompose none --no-flux --no-autoserve --segments 2
```

See [`REPRODUCE.zh.md`](REPRODUCE.zh.md) for the Chinese version of this guide.
