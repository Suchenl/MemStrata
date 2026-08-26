# memory_update

记忆更新 / Memory Update — the paper **Stratified Update** step (Step 4), i.e. the **落库**
logic, formalized as a reusable skill. `memstrata.steps.curate` is now a thin re-export shim.

## What it does

Stratified-updates the `Observation`s produced by the `decomposition` skill into the
stratified `AssetBank`, building a reusable per-entity visual library:

| stage | behavior |
|---|---|
| identity merge | name-anchored: explicit `entity_id` first, else same `name`+`kind`; alias 留痕 |
| admission gates | WHO-before-WHERE: ① dark / low-info reject ② embedding-cohesion floor ③ identity-visible → anchor eligibility |
| diversity strata | `(spatial, state, shot, lighting, pose)` bucket + farthest-point `select_attribute_diverse` under `max_reps_per_asset` — the library "prune by attributes" logic |
| novelty | `redundancy_threshold` discards near-duplicates → the bank records **new** content, not old (pairs with the decompose novelty selection) |
| lifecycle | state events deprecate reps with traceable `deprecated_by` (留痕) + `replaced_by` relation chains; reversible, never silently deleted |
| self-audit | per-asset cohesion sweep (medoid / subcluster) retroactively isolates mixed-identity intruders |
| budget | `max_total_representations` evicts weakest live reps, each asset keeps ≥1 |

## Entry points

- `MemoryUpdater` — the curator class (back-compat aliases: `AssetCurator`, `InverseIngester`).
- `curate_observations(observations, *, chunk_id, state_events?, relations?)` — production path.
- `ingest_packet(packet)` — bench Track-A `ObservationPacket` (authoritative ids, no VLM).
- `ingest_observation(EntityObservation, chunk_id)` — single legacy production observation.
- `audit_cohesion(...)` — standalone identity-cohesion self-audit sweep.
- `export_memory_snapshot(bank, out_dir, ...)` — human-readable memory export mirroring the gt entity schema.

## Notes

- Zero `vmem_bench` import; shared crop-QA / dedup logic is mirrored under
  `memstrata.lib` (`crop_quality`, `dedup`).
- With the offline `HashEmbedding` the cohesion gate self-disables (non-semantic); a
  production run wires a real encoder + a calibrated floor.
