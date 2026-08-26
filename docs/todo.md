# MemStrata — TODO / Known Issues

## [BUG] Forbidden/deprecation control leaks: deprecated entities get re-rendered (Track-B avoidance)

**Discovered:** 2026-07-29, from Track-B end2end scoring (story `0029_ink_wash_painter`).
**Symptom:** on the "indirect reference" avoidance family MemStrata scores only **0.672** avoidance-OK
(vs up to 0.929 for baselines); corpus deprecation-avoidance is 0.891. I.e. a destroyed/deceased entity
is sometimes re-rendered after it should be gone.

### Concrete instance
- Entity **E5 = "pine-soot inkstone boat"** (a prop). Destroyed at GT `seg_076`
  ("…the pine-soot inkstone boat shatters…"). Later GT `seg_114` prompt merely writes its NAME
  ("Ayue writes the four characters for pine-soot inkstone boat … then erases them") → the boat must
  **not** appear. VLM judged MemStrata **DREW E5** (violation); memflow/iamflow/longlive all omitted it.
- This same boat is the col-2 "long-gap recall" hero case (MemStrata correctly recalls it when required),
  so the failure is the *dual* of the strength.

### Root cause (from this run's own logs, `outputs/evaluation/trackB/memstrata/0029_ink_wash_painter/name_anchored/prod_20260729/`)
The memory PLAN is largely correct; the leak is at the plan→generation hand-off. Evidence:

1. **State manager DID deprecate the boat.** In `bank.json`, asset
   `pine-soot inkstone boat gunwales` has `status: deprecated`.
2. **forbid mechanism DOES fire** on 9 segments; several correctly list the boat, e.g.
   `pipeline/segment_061` and `segment_070` forbid `['boat','pine-soot inkstone boat gunwales', …]`,
   `segment_091` (=GT seg_092) forbids `['boat']`.
3. **At `segment_113` (=GT seg_114) the plan did NOT select the boat**:
   `selected_assets = [brush-washing pool, Young Ayue, goat-hair doubi brush]` (no boat);
   `intent_forbidden_asset_ids = []`. The boat appears only in `touched_asset_ids` (name resolved from
   the prompt text). The generation prompt contains the literal noun
   *"…for pine-soot inkstone boat…"* → the text-conditioned video generator paints the noun even with
   **no** boat reference image injected.

### Three distinct sub-bugs to fix
- **(a) Prompt-noun leak (dominant).** The SUT prompt names the deprecated entity verbatim; the
  text-conditioned generator (Wan/VACE) renders named nouns regardless of the memory plan.
  → **Fix:** deprecation-aware prompt rewriting — scrub/negate deprecated-entity nouns from the
  generation prompt (or add them to the generator's negative prompt) before i2v_composed.
- **(b) forbid not propagated to the generator.** Even when the plan sets a forbid (seg_092), it is not
  turned into effective negative conditioning, so the noun still renders.
  → **Fix:** thread `intent_forbidden_asset_ids` into the generator negative prompt / conditioning.
- **(c) Alias fragmentation.** The boat exists as two bank assets — `pine-soot inkstone boat gunwales`
  (`deprecated`) AND `boat-shaped inkstone` (`reusable`, NOT deprecated), plus a generic `boat` token.
  asset-id-keyed forbid intermittently misses (seg_114 forbid was empty).
  → **Fix:** unify lifecycle status across aliased/co-referent representations of one entity; when an
  entity is deprecated, propagate the flag to all its aliases.

### Acceptance / how to verify a fix
- Re-score story 0029 (and the corpus) Track-B; target: indirect-reference avoidance-OK ↑ (from 0.672)
  and deprecation-avoidance ↑ (from 0.891) **without** dropping long-gap recall (must stay ~0.881).
- Spot-check `pipeline/segment_113`: boat noun absent from the generation prompt / present in the
  negative prompt; boat not in the rendered frame.

### Paper note
In the submission we state this plainly as a limitation (MemStrata has not yet realized the
forbidden/deprecation capability), see `fig:trackb-money` caption and §6.3; the detailed mechanism above
is kept here as the engineering record, not in the paper.
