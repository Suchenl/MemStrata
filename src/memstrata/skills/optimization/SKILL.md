# Skill: Production Self-Optimization (agent-in-the-loop)

**Use this when you are running a long MemStrata video production** (`memstrata.production.run`)
and want to watch quality *while it runs*, decide whether the current skill implementations are
misbehaving, and fix them in place — rather than discovering at the end that 200 chunks drifted.

This skill is **for the agent, not the pipeline**. The pipeline keeps generating; you (the agent)
periodically stop to look, diagnose, and act. `monitor.py` is deterministic evidence; the
`continue | tune | rollback-and-rerun | abort` decision is yours.

---

## The loop

Every **N segments** (default N=3–5; smaller early when things are shaky):

1. **Observe.** Run the monitor on the live run dir:

```bash
python -m memstrata.skills.optimization.monitor --run-dir <RUN_DIR> --window 5
```

   It reads `progress.json` (written each chunk), `bank.json`, and `review/` and prints a
   verdict (`healthy | watch | investigate | abort_or_fix`) plus any flagged signals with the
   exact skill + `registry.toml` knob to touch. It also drops
   `<RUN_DIR>/optimization/checkpoint_NNNN.json`.

2. **Look with your own eyes.** The monitor is signals, not ground truth. Open the review view:
   - `<RUN_DIR>/review/INDEX.md` — per-chunk prompt · memory read-in crops · composed keyframe ·
     generated segment · banked observations · bank counts.
   - `<RUN_DIR>/review/long_video.mp4` — the stitched film so far.
   - `<RUN_DIR>/review/keyframes/seg_NNN.png` and `.../observations/seg_NNN__*.png` — is identity
     holding? are banked crops actually the named entity, or garbage/background?

3. **Diagnose → map symptom to skill.** Use the flagged signal's `skill`/`knob`, or the table
   below. The rule: **one hypothesis, one knob, one skill at a time.** Do not shotgun-edit.

4. **Decide:**
   - `healthy`/`watch` → **continue**, next checkpoint later.
   - single `bad` you understand → **tune** the one knob in that skill's `registry.toml` /
     implementation, then continue (new chunks pick up the change; running services may need a
     restart — the monitor's `high_skip`/`keyframe_missing` hints tell you when a service is dead).
   - drift/identity already ruined for several chunks → **rollback-and-rerun**: re-launch from a
     clean point (raise `--chunks` cap or start a fresh timestamped run) after the fix. Segments
     are content-addressed; a fresh run is cheap to compare.
   - two+ `bad` signals or an unhealthy service you can't fix live → **abort**, fix offline,
     relaunch.

5. **Record.** Note the change + why in the run's `optimization/` dir (a one-liner is fine) so the
   next checkpoint knows what you already tried. Never silently re-tune the same knob twice.

---

## Symptom → skill → knob (canonical map)

`monitor.py:SKILL_KNOBS` is the machine-readable source; this is the human view.

| Flagged signal | Meaning | Skill | Knob to consider |
|---|---|---|---|
| `no_new_observations` | decompose yields ~0 crops/chunk | `crop_acquisition` / `entity_grounding` / `decomposition` | lower `identity_threshold` (~0.45→0.30); box confidence |
| `memory_stagnant` | obs produced but bank not growing | `memory_update` / `embedding_deduplication` | raise `dedup_threshold` (admit new views) |
| `memory_explosion` | one asset +1 rep every chunk | `embedding_deduplication` / `memory_update` | lower `dedup_threshold`; per-asset rep cap |
| `context_not_read` | bank non-empty but 0 refs read in | Compose (`steps/compose.py`) / bank selection | name/alias match + identifier deref |
| `ar_drift_risk` | long unbroken `continue_ar` chain | `generation_routing` | recompose sooner / `--force-recompose` |
| `router_infeasible` | picked mode keeps falling back | `generation_routing` | tighten rule feasibility layer |
| `keyframe_missing` | no composed keyframe | `layout_anchor_processing` / flux backend | R3/R4 health; flux server log |
| `high_skip` | chunks skipped after retries | generate backend / services | OOM/resolution; restart dead server |

---

## Guardrails

- **Don't optimize noise.** One weak segment is not a trend. Prefer `--window` ≥ 3 and require a
  signal to repeat before acting (the earlier lesson: don't overfit a single sample).
- **Deterministic first.** Every knob above is a threshold / routing rule, not a model swap.
  Exhaust cheap deterministic fixes before touching models, prompts, or re-serving.
- **Keep the mainline stable.** The validated generator path (FLUX keyframe → Wan2.2-I2V-A14B
  4-step distilled via LightX2V, 480×832) is the baseline; tune memory/routing skills around it,
  don't rewrite the backend mid-run.
- **Evidence in-run, decisions durable.** Signals + checkpoints live under `<RUN_DIR>/optimization/`;
  a durable knob change goes into the skill's `registry.toml` / code with a one-line reason.
