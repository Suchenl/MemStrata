# memstrata.skills.optimization

Agent-in-the-loop self-optimization for long production runs. Not a pipeline stage — a
tool the agent uses to watch a run, diagnose which skill is misbehaving, and fix it in place.

## Files

- `SKILL.md` — the **rule**: the observe → look → diagnose → decide (`continue | tune |
  rollback-and-rerun | abort`) protocol, run every N segments, plus the symptom→skill→knob table.
- `monitor.py` — the **script**: deterministic, stdlib-only. Reads a run dir and prints a verdict
  + flagged signals, each mapped to the skill + `registry.toml` knob to touch. Writes
  `<RUN_DIR>/optimization/checkpoint_NNNN.json`.
- `registry.toml` — skill descriptor + the machine-readable knob lookup.

## Quick use

```bash
# during a live run (progress.json is written each chunk):
python3 -m memstrata.skills.optimization.monitor --run-dir \
  this repository/production/outputs/<story>/<system>/<ts> --window 5

# full JSON (for programmatic use):
python3 -m memstrata.skills.optimization.monitor --run-dir <RUN_DIR> --json
```

## What it reads (no new plumbing needed)

| source | provides |
|---|---|
| `progress.json` (live) / `summary.json` (end) | per-chunk rows: route mode, obs, refs, keyframe, bank counts |
| `review/INDEX.md`, `review/long_video.mp4`, `review/{keyframes,observations,context}/` | human eyeball check |
| `bank.json` | final stratified memory bank |

## Design notes

- **Signals, not verdicts-by-fiat.** The script never edits anything; it produces evidence so the
  agent can decide. This keeps the deterministic core separate from model/agent judgement.
- **One knob at a time.** Every mapped fix is a threshold or routing rule in an existing skill's
  `registry.toml` — cheap, reversible, and attributable. Model swaps are a last resort.
- **Don't overfit one segment.** Use `--window ≥ 3` and require a signal to repeat before acting.
