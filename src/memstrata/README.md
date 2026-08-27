# MemStrata (`src/memstrata`)

Paper-aligned package layout (method §3 four-step loop).

```
memstrata/
  bank/          # Asset bank A_n  (schema.py)
  steps/         # intent → compose → generate/ → decompose → curate
  pipeline.py    # MemStrata orchestrator (+ run_ledger)
  adapters/      # Track A bench JSON (bench.py) — does not import vmem_bench
  encoders/      # face / place / ssl (+ RoleRoutedEmbedding)
  llm/           # MLLM intent + production AngleClassifier (crop → spatial/state)
  lib/           # paths, weights, dedup, media
  extras/        # optional (shot_boundary, …) — not on hot path
  tests/
```

Public API: `from memstrata import MemStrata, MediaTaskGenerator, …`

Production run / generator smoke (from `this repository`; logic in
`memstrata.production.run`, launched by `scripts/memstrata/run_production.sh`):

```bash
# no-GPU backend smoke
PYTHONPATH=src python3 -m memstrata.production.run --backend recording --decompose none --chunks 2
# full screenplay-driven closed loop (GPU)
PYTHONPATH=src python3 -m memstrata.production.run --backend helios_distilled_i2v --flux --force-recompose
```

Design docs (unified under the subproject `docs/`): [`philosophy.md`](../../docs/method/philosophy.md) (highest-level charter; Chinese),
[`design.md`](../../docs/method/design.md) (implementation spec; Chinese), [`generator_wiring.md`](../../docs/method/generator_wiring.md) (Chinese),
[`mllm_roles.md`](../../docs/method/mllm_roles.md) (MLLM role catalog / sampling-parameter spec; Chinese).
