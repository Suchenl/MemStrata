# memstrata.skills.composition

The paper's **Compose** step — the *read* side of stratified memory. Moved out of `steps/`
(mirroring `decomposition` / `memory_update`) so it is a reusable capability with its own
`registry.toml` the agent-in-the-loop optimizer can point at.

## What it does

```
g_n (external intent) + A_n (asset space)
      │  IntentInterpreter.interpret()            ← intent.py
      ▼
q_n (CompositionRequest: named refs + angle/state prefs + budget)
      │  compose()                                ← compose.py  (O(1), 0 model calls)
      ▼
C_n (ComposedContext: chosen representation ids per asset, functions, exclusions)
```

- **`intent.py`** — `IntentInterpreter`: FAST (default) resolves references by **name/alias match**
  (model-free); when the prompt has no resolvable name it falls back to **description overlap**
  (not recency), so matching-based composition still holds in description-only regimes. SLOW mode
  optionally calls an MLLM resolver (`MllmIntentResolver`) but never invents ids outside `A_n`.
- **`compose.py`** — `compose()`: dereferences `q_n` against the bank with stratified rep selection
  (spatial/state angle → per-purpose quality → recency), bounded relation expansion
  (`PART_OF`/`LOCATED_IN`), minimal-sufficient budget trimming, and deprecated-rep exclusion.

## Backward compatibility

`steps/intent.py` and `steps/compose.py` are now thin re-export shims, so existing
`from memstrata.steps.intent import ...` / `from memstrata.steps.compose import ...` call sites
(and the pipeline / adapters / tests) keep working unchanged.

## Read-path knobs (for skills/optimization → `context_not_read`)

| knob | effect |
|---|---|
| `mode` | `fast` (model-free) vs `slow` (MLLM resolver) |
| `recency_cap` | K assets kept on recency / description fallback |
| `max_reps_per_asset` | reps a named asset contributes to `C_n` |
| `context_rep_budget` | hard cap on total reps in `C_n` |
| `relation_hops` | bounded structural expansion depth |
| `disable_name_anchor` | ablation: name anchoring off → recency proxy |
