# Reproducing paper numbers

`pytest` does **not** reproduce paper tables. It checks contracts (bank, naming, generator protocol) without GPUs or LLMs.

## Branches

| Branch | Meaning |
|---|---|
| `main` | Production. May move. |
| `paper-reproduction` | Track A Stage-1 freeze (internal `VMem-Track-A-MemStrata` @ `51be2914`). Not `main`. |

Use `paper-reproduction` (and tag `paper-reproduction-v1` when that freeze lands) for any number you intend to compare with the paper.

## What you need for tables

- Source videos (Blender Open Movies + LSMDC; **not** in this repo)
- Gold / prompts: [huggingface.co/datasets/Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench)
- Scoring + adapters: [github.com/Suchenl/VMem-Bench](https://github.com/Suchenl/VMem-Bench)
- Generator / encoder / VLM weights via `PUBLIC_MODELS_ROOT` (Wan, FLUX, Qwen, …)
- GPUs sufficient for the chosen backend

CPU smoke in this repo:

```bash
PYTHONPATH=src python3 -m memstrata.production.run \
  --backend recording --decompose none --no-flux --no-autoserve --segments 2
```
