# memstrata.skills.memory_retrieval

MemStrata's **memory recall** skill — "given the current prompt, which remembered
material do we bring back?" It is deliberately separate from `skills.composition`
(deterministic dereference of already-chosen ids) and from `skills.intent_understanding`
(which *consumes* this skill's name recall). It owns two independent retrieval
mechanisms:

## 1. Name/alias retrieval — `name_match.py` (real read-path anchor)

Model-free deterministic recall over stored assets: match the prompt against each
asset's stored **name and aliases** (English word-boundary + longest-first CJK) and
return the asset ids referred to. This is a **first-class production capability** — the
FAST read path in `skills.intent_understanding` calls it to resolve intent with zero
model calls. It is *not* an ablation.

```python
from memstrata.skills.memory_retrieval import name_match, match_cache, name_hits
ids = name_match(prompt, candidate_assets)          # -> [asset_id, ...]
```

## 2. Frame/segment similarity retrieval — `retrievers.py`

Similarity search over historical segment/frame memory. Some variants are genuine
retrieval mechanisms MemStrata can adopt (text→segment / text→frame / DINO keyframe
diversity); others are deliberately weak **controls** for ablation tables. Kept here so
the method can plug any of them into its recall path (today it uses name match; these
are available to wire in next).

```python
class Retriever(Protocol):
    name: str
    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]: ...
```

`RetrievedRef.source_seconds` is the temporal identity consumed by the causal bench
adapter and maps directly to bench `RetrievedItem.source_seconds`.

### Variants (ablations)

- `seg_uniform_ablation`: text→segment retrieval, then uniform frame sampling from
  selected segments. `RETR_UNIFORM_FPS` supports the planned `{0.5,1,2}` sweep.
- `seg_dinokey_ablation`: text→segment retrieval, then DINOv3 diversity keyframes from
  dense segment samples.
- `seg_framererank_ablation`: coarse text→segment retrieval followed by text→frame
  rerank inside candidate segments. The causal runner can fuse the coarse and fine
  ranked lists with RRF (`k=60`).
- `frame_text_ablation`: direct text→frame retrieval over all sampled historical frames.

### Controls

`recency_ctrl`, `bm25_desc_ctrl`, and `random_ctrl`. Controls only, never headline
baselines.

## Import weight

`name_match` is light (only depends on the asset bank), so the package exports it
eagerly. `retrievers` pulls the heavy encoder substrate (embedding models), so it is
exported **lazily** — importing the name-match read path never forces those deps.

## Encoders (retrievers only)

Default providers are `hash` so the pipeline can be smoke-tested without model loads.
Production providers are configured by retrieval space:

- text→segment: `RETR_TEXT_PROVIDER=qwen3_embedding`
- text→frame: `RETR_FRAME_PROVIDER=siglip2` (the implementation uses the
  same SigLIP2 instance's `embed_text()` for the query and `embed_image()` for
  frames; do not compare Qwen text vectors directly to SigLIP image vectors)
- keyframe diversity: `RETR_KEYFRAME_PROVIDER=dinov3`
- model roots: `PUBLIC_MODELS_ROOT=/path/to/local/snapshots`

`Qwen3-Embedding` also supports a server backend through
`MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT` (OpenAI-compatible `/embeddings`).

## Description dependency

`bm25_desc_ctrl` and `description_only` experiments benefit from asset/rep descriptions.
The write path is responsible for populating those descriptions; this skill skips empty
descriptions and falls back deterministically rather than inventing text.
