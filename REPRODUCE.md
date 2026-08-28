# Reproducing paper numbers

`pytest` does **not** reproduce paper tables. It checks contracts (bank, naming, generator protocol) without GPUs or LLMs.

## Branches

| Branch | Meaning |
|---|---|
| `main` | Production. May move. |
| `paper-reproduction` | Frozen Track A Stage-1 implementation for paper comparison. Not `main`. |

Use `paper-reproduction` (and tag `paper-reproduction-v1` when that freeze lands) for any number you intend to compare with the paper.

## What you need for tables

- Source videos (not in this repo). How to download / apply / where to put files: [VMem-Bench `docs/DATA.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/DATA.md) (BBB: `bash scripts/prepare_blender.sh` in that repo)
- Gold / prompts: [huggingface.co/datasets/Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench)
- Scoring + adapters: [github.com/Suchenl/VMem-Bench](https://github.com/Suchenl/VMem-Bench)
- Generator / encoder / VLM weights via `PUBLIC_MODELS_ROOT` (Wan, FLUX, Qwen, …)
- GPUs sufficient for the chosen backend

## Track A Stage-1 command

Run the frozen adapter from the matching `paper-reproduction` VMem-Bench
checkout. Keep the two repositories side by side:

```bash
cd ../VMem-Bench
export MEMSTRATA_SRC="../MemStrata/src"
export MEMSTRATA_TRACKA_NAME_SOURCE=mllm
python3 scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py \
  --adapter memstrata \
  --movie-dir assets/trackA/BlenderOpenMovies/big_buck_bunny \
  --budget 16 \
  --limit 2
```

The runner requires the corresponding source video and writes the
`visual_selections`/manifest artifacts under the movie run directory. Score
those artifacts with the VMem-Bench `visual_coverage` command documented in
its `REPRODUCE.md`. Use `--budget 16` for the paper setting; smaller budgets
are smoke or ablation settings.

A GPU production run is `bash scripts/memstrata/run_production.sh` after `PUBLIC_MODELS_ROOT` is set ([`MODELS.md`](MODELS.md)). The recording backend below only checks that the package imports; it does **not** reproduce paper numbers:

```bash
PYTHONPATH=src python3 -m memstrata.production.run \
  --backend recording --decompose none --no-flux --no-autoserve --segments 2
```

## Citation

See [`CITATION.cff`](CITATION.cff).
