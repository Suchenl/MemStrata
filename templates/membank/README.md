# Memory bank template (`memstrata-memory-1.0`)

This is the canonical on-disk shape of a **live, append-only memory bank** — the
exact layout produced by `memstrata.skills.memory_update.snapshot.export_memory_snapshot`.
Copy this tree as a starting point / reference. The `.png` and `.mp4` here are
empty placeholders; a real run fills them with cropped visual evidence and the
grown film.

```
membank/
├── memory.json                 # deterministic clean CURRENT view (schema below)
├── long_video.mp4              # the grown film; every realized segment is APPENDED here
└── visual/                     # on-disk visual evidence, one crop per representation
    └── <kind_plural>/          # characters/ | props/ | locations/
        └── <asset_id>/
            └── states/
                └── <state>/
                    └── <representation_id><ext>
```

## The one rule: the bank is append-only. Nothing is deleted.

State changes, redundant crops, and superseded evidence are handled by **adding
records and flipping status flags**, never by erasing data on disk.

- **`long_video.mp4`** — the single grown film. Each new realized segment is
  **appended**; it is never truncated or deleted. Every `sec`, `first_seen_sec`,
  and `appearances[].sec` in `memory.json` is a timestamp **on this film's
  timeline**. The `video` header records its relative path and `duration_sec`.
- **`visual/.../<state>/<rep_id>.png`** — every representation's crop is a real
  file on disk and **stays on disk** for audit / reversal, even after it is
  deprecated. Path layout is deterministic:
  `visual/<kind_plural>/<asset_id>/states/<state>/<representation_id><ext>`.
- **`memory.json`** — the clean *current view*, rewritten atomically (temp file +
  `os.replace`) after each segment. It is a projection, not the source of truth.

## What "state changed" actually does

A new appearance state (e.g. a character goes `default` → `soaked`) is a **new
entry under that entity's `states{}`**, with its own `description`,
`first_seen_sec`, `appearances`, and `images`. The **prior state stays** — see
`char_elias` in `memory.json`, which keeps both `default` and `soaked`. Memory
accumulates the full state history of an entity; it does not overwrite the old
state. The identity of the entity (`char_elias`) is stable across states.

## What "deprecation" actually means (this is not deletion)

At the **representation** level, `AssetRepresentation.deprecated: bool` (with
`deprecated_by`) is a **status flag** for a crop that is redundant, an intruder,
or superseded by a better view of the same state. At the **asset** level,
`lifecycle ∈ {rejected, deprecated, failed}` marks an unusable asset.

The clean `memory.json` view simply **filters these out** (deprecated reps and
non-usable assets are omitted) so downstream conditioning only sees current,
trustworthy evidence — but the underlying `AssetBank` record and the `.png` file
**remain**, so any deprecation is reversible and auditable. Deprecate = "stop
surfacing this in the current view", **not** "remove it from the bank".

## Field reference for `memory.json`

Top level:

| key | meaning |
|---|---|
| `schema` | always `memstrata-memory-1.0` |
| `movie_id`, `fps` | provenance header, recorded verbatim |
| `updated_sec` | max known representation second, or explicit override |
| `video.path` | POSIX path (relative to this dir) of the grown film |
| `video.duration_sec` | current film length; grows as segments are appended |
| `entities` | `{asset_id: entity}` — usable assets only, id-sorted |

Per entity:

| key | meaning |
|---|---|
| `name`, `kind` | display name and `character` / `prop` / `location` |
| `description` | the asset descriptor `d` |
| `aliases` | alternate surface names (for read-side name matching) |
| `lifecycle` | asset status `∈ {candidate, reusable, used}` in the clean view |
| `initial_state` | explicit metadata, else earliest-seen state |
| `first_seen_sec` | earliest state `first_seen_sec` for this entity |
| `states` | `{state_label: state}`, ordered by first-seen then segment |

Per state:

| key | meaning |
|---|---|
| `description` | free-form condition/appearance note for this state |
| `first_seen_sec` | earliest appearance second (on the `long_video` timeline) |
| `appearances` | `[{sec, segment}, ...]` sorted by time, every sighting |
| `images` | POSIX crop paths under `visual/`, sorted |
