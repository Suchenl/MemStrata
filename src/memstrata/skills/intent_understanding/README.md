# memstrata.skills.intent_understanding

The paper's **Intent Interpretation** (Stage 3) — the read-path front-end. Split out of
`skills.composition` so Interpret (Stage 3) and Compose (Stage 4) are distinct skills.

## What it does

```
g_n (external intent) + A_n (asset space)
      │  IntentInterpreter.interpret()            ← interpreter.py
      ▼
q_n (CompositionRequest: named refs + angle/state prefs + budget)
```

Dereferencing `q_n` into a Composed Context `C_n` is the separate `skills.composition`
(`compose()`) stage.

- **FAST (default, 0 model calls)** — resolves references by **name/alias match**
  (`skills.memory_retrieval.name_match`); when the prompt has no resolvable name it falls
  back to **description overlap**, then to **recency**, so matching-based composition still
  holds in description-only regimes.
- **SLOW (opt-in)** — `MllmIntentResolver` reasons over the bank listing via the shared
  MLLM transport, but never invents ids outside `A_n`.
- **PLAN (opt-in, `mode="plan"` / `MEMSTRATA_INTENT_MODE=plan`)** — one bounded call returns a
  typed `IntentPlanV1` (`plan.py`): references **plus the required appearance state**, entities
  that must **not** appear, and the **generation route**. A bare id list cannot express those,
  which is what Track B's `persist_state` / `state_change` (the changed look must persist across
  beats) and `deprecation_avoidance` / `false_friend` (a destroyed prop, a look-alike) need.

  Identity stays name-authoritative: the plan may only *name* entities, and names are resolved
  locally through the same matcher the FAST path uses, so a plan can never widen `A_n`.
  Unresolvable names are reported (`plan_unresolved_names`), never invented. Any unusable plan
  falls back to FAST, so enabling PLAN cannot produce an empty read path.

  Two fields carry state the beat itself cannot re-derive later:

  | field | scope | effect |
  |---|---|---|
  | `count_required` | this beat | prefers a stored crop showing that many instances; below 2 it is read as "no constraint", because a planner answers 1 for every single entity |
  | `retired` | every **later** beat | the record is deprecated after this beat composes, so `compose`'s usability gate keeps it out without the planner having to remember |

  `retired` is applied *after* composition on purpose: the beat that destroys a prop is normally
  the beat that shows it. A ban and a reference for the same entity is a planner contradiction;
  the ban wins, except on that retirement overlap, and a surviving `count_required` blocks the
  retirement outright (one of three floats smashing must not delete the other two).

  This is the small, forward-compatible half of
  `docs/method/unified_video_memory_pipeline_DESIGN.md` — the plan *contract* without the
  instance-cache migration, so that migration only has to change the execution side.

## Boundaries

- Name/alias matching is a **memory-retrieval** mechanism and lives in
  `skills.memory_retrieval.name_match`; this skill *consumes* it (does not own it).
- No dereference / representation selection here — that is `skills.composition.compose`.

## Backward compatibility

`skills/composition/intent.py` and `steps/intent.py` are thin re-export shims pointing
here, so existing `from memstrata.skills.composition.intent import ...` /
`from memstrata.steps.intent import ...` call sites keep working unchanged.

## Read-path knobs (for skills/optimization → `context_not_read`)

| knob | effect |
|---|---|
| `mode` | `fast` (model-free) vs `slow` (MLLM resolver) vs `plan` (typed `IntentPlanV1`) |
| `plan_producer` | plan-mode producer; auto-built from the shared MLLM transport when unset |
| `recency_cap` | K assets kept on recency / description fallback |
| `disable_name_anchor` | ablation: name anchoring off → recency proxy |
