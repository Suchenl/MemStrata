# decomposition

Role-aware Asset Decomposition — the paper **Decompose** step (Step 3), formalized as a
reusable skill. `memstrata.steps.decompose` is now a thin re-export shim.

## What it does

Given the entities the intent already **named** (targeted grounding, *not* open-vocab
discovery), for each entity:

1. Isolate a crop from the *generated* chunk via the pluggable `Cropper` protocol.
2. Type-route the encoding (`RoleRoutedEmbedding`: face / location / general by `AssetType`).
3. Optionally resolve spatial/state angles (`memstrata.mllm` angle classifier).
4. Emit one `Observation` (`[image + angle]`) per entity, ready for the curate/commit-to-bank step.

It deliberately does **not** write to the bank — admission, diversity strata, redundancy,
and lifecycle are the curate step's job (`memstrata.steps.curate.MemoryUpdater`).

## Cropper backends (sibling skills)

The `Cropper` protocol (`crop(chunk_video, entity, *, chunk_id) -> str | None`) is provided by:

| skill | method | crop quality | cost |
|---|---|---|---|
| `crop_acquisition` | S5-derived SAM3 + GroundingDINO propose → DINOv3 identity gate → novelty selection | clean masked entity crop (production default) | persistent server, 1 GPU |
| `entity_grounding` | single-VLM (Qwen) tight-box grounding | tight box, no mask | reuses the running Qwen |

**Design note (identity vs novelty):** the identity gate only confirms *"is this our
entity"* (don't bank a background stranger); candidate **selection maximizes novelty** so we
record content the bank does **not** already have; near-duplicate suppression is curate's
`redundancy_threshold`. The three concerns are separated on purpose.

## API

```python
from memstrata.skills.decomposition import RoleAwareDecomposer, NamedEntity, Cropper

decomposer = RoleAwareDecomposer(embedder=..., cropper=<a Cropper>)
observations = decomposer.decompose(chunk_id=i, named_entities=[...], chunk_video=path)
```

`observations_from_packet_dicts(...)` maps bench Track-A `ObservationPacket` rows into
`Observation`s (no discovery), for evaluation runs that bypass live cropping.
