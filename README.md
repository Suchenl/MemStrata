# MemStrata

Stratified memory for **causal long video generation**. A real run needs a **GPU**, generator / encoder / VLM weights, and ffmpeg. Paper tables additionally need [VMem-Bench](https://github.com/Suchenl/VMem-Bench).

> Chinese documentation: [`README.zh.md`](README.zh.md).

## Getting started

Clone the two repos **next to each other**:

```bash
git clone https://github.com/Suchenl/MemStrata.git
git clone https://github.com/Suchenl/VMem-Bench.git
cd MemStrata
python3 -m pip install -e ".[dev]"
export PUBLIC_MODELS_ROOT="$HOME/public_models"   # see MODELS.md
python3 scripts/memstrata/doctor.py
```

Default production path (documented, not hard-wired): FLUX.2 Klein 9B-KV keyframes → Wan2.2-I2V-A14B LightX2V 4-step.

```bash
# downloads: MODELS.md
bash scripts/memstrata/run_production.sh
python3 -m memstrata.production.run --list-backends
```

Weights: [`MODELS.md`](MODELS.md). Paper numbers: [`REPRODUCE.md`](REPRODUCE.md) on branch `paper-reproduction`. Gold: [huggingface.co/datasets/Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench).

`memstrata` and `vmem_bench` never import each other. Evaluation adapters live only in the VMem-Bench repo.

Unit tests and an optional **install check** (`bash scripts/memstrata/cpu_demo.sh`) run without weights; that path is not the method.

## How it works

MemStrata treats a long video's recurring entities (`character` / `prop` / `location`) as a **structured, generation-conditionable memory bank** \(\mathcal{M}_n\). At each step it composes a **minimal-yet-sufficient** visual context for the entities named in the current intent, so the downstream generator reliably reproduces the same identity. The bank is not a "warehouse of frames" but **one conditionable identity dossier per entity**.

The authoritative design spec is [`src/memstrata/docs/design_philosophy.md`](src/memstrata/docs/design_philosophy.md) (six bank-quality axioms + the WHO-before-WHERE admission rule).

### Four-stage causal memory loop (paper §4)

\[(q_n,\tilde p_n,\mathcal{C}_n)=\mathcal{R}(g_n,\mathcal{M}_n)\ \to\ x_n=G_\theta(\tilde p_n,\Phi(\mathcal{C}_n))\ \to\ \mathcal{O}_n=\mathcal{A}(x_n,q_n)\ \to\ \mathcal{M}_{n+1}=\mathcal{W}(\mathcal{M}_n,\mathcal{O}_n)\]

| Paper stage | Skill (`skills/…`) | One line |
|---|---|---|
| Intent Interpretation | `intent_understanding` | external intent \(g_n\) → typed request \(q_n\) (which stored ids + angle/state prefs); FAST name match (0 models), SLOW only when needed |
| (read-path deref) | `composition` + `memory_retrieval` | deterministically deref the ids in \(q_n\) into context \(\mathcal{C}_n\) (LexMax: explicit → state → view → latest) |
| Visual Generation | `generation_routing` + `steps/generate` | reference-conditioned backend generates \(x_n\); the method does **not** alter \(G_\theta\) |
| Evidence Acquisition | `decomposition` + `crop_acquisition` | \(\mathcal{O}_n=\mathcal{O}^{\mathrm{req}}\cup\mathcal{O}^{\mathrm{disc}}\): requested-entity evidence + type-limited self-discovery |
| Stratified Update | `memory_update` | identity re-id → novelty → conflict eviction → per-type budget → cohesion self-audit → export memory snapshot JSON |

### Three cores (fast/slow thinking throughout)

Guiding principle: **combine fast and slow thinking**. Anything solvable deterministically / by name match takes the **fast path** (0 model calls, matters for generation latency); only genuine semantic judgements escalate to **slow** (MLLM).

1. **Intent Interpretation** — parse \(g_n\) into a typed request \(q_n\). FAST name/alias match by default; SLOW MLLM only when a name is missing or ambiguous (and never invents ids outside the bank). Crucially, *how* evidence will be acquired is planned already here.
2. **Evidence Acquisition** — four executable paths:
   - **A. named, matched, no new state** — *fast*: reuse the entity's stored reference images.
   - **B. named, matched, new state** — *slow*: MLLM decides which old states are worth referencing (e.g. a young face for an aged character); after generation, crop the **new state** and store it.
   - **C. named, unmatched, but described** — *slow*: treat as a provisional new entity; after generation, deterministically detect candidates, then VLM-match which crops fit the new entity's name+description, and store the chosen ones.
   - **D. self-discovered** — *slow*: entities that appear but were not in the prompt; type-limited discovery (`VlmEntityDecomposer`) → drop those already covered by A/B/C → re-id and store. Naming is **never** fabricated by the proposer.
3. **Memory Update** — fold each `Observation` into the stratified `AssetBank`: identity anchoring/re-id (χ) → novelty dedup within a compatible stratum → state-conflict eviction (kept as history) → per-type budget \(B_\tau\) attribute-diversity selection → per-chunk cohesion self-audit. Follows **WHO-before-WHERE**: evidence must first prove "this is the entity, and it is clearly visible" before it can compete on angle/state diversity.

### Output: a portable memory package

The write path's deliverable is a human-readable `membank/` package. It contains
`memory.json` (schema `memstrata-memory-1.0`; entity → state → visual memory),
the co-rooted `visual/` image library, and the concatenated
`long_video.mp4` that all timestamps are measured against. The package is
separate from per-segment pipeline logs and can be copied as one unit.

```
<run_dir>/
├── pipeline/                 # internal per-segment records
├── review/                   # review video and review views
└── membank/                  # portable deliverable
    ├── long_video.mp4        # assembled film; timestamp anchor
    ├── memory.json           # snapshot, refreshed after the run
    └── visual/
        ├── characters/<asset_id>/states/<state>/*.png
        ├── props/<asset_id>/states/<state>/*.png
        └── locations/<asset_id>/states/<state>/*.png
```

See [`README.zh.md`](README.zh.md) for the full `memory.json` schema, the write-side quality-gate plan, and the VLM call budget (Chinese).

## Self-check

```bash
python3 -m pytest -q
PYTHONPATH=src python3 -m memstrata.production.run --backend recording --decompose none --no-flux --no-autoserve --segments 2
PYTHONPATH=src python3 -m memstrata.production.run --backend oracle --decompose none --no-flux --no-autoserve --segments 2
```

A real GPU closed loop (Wan / FLUX / Qwen) needs your own weights — see [`MODELS.md`](MODELS.md) and [`docs/operations/models_and_environments.md`](docs/operations/models_and_environments.md). Paper-table numbers are pinned to git branch `paper-reproduction`, not to an arbitrary `main`.

## Citation

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

See [`CITATION.cff`](CITATION.cff). Code is Apache-2.0.
