"""Production self-optimization monitor (agent-in-the-loop).

Reads a *live* or finished MemStrata production run and turns its raw artifacts
(``progress.json`` mid-run / ``summary.json`` at end, ``bank.json``, ``review/``) into
a compact, decision-oriented **checkpoint report**: a handful of health signals, each
flagged healthy / warn / bad, and — when bad — the exact skill + ``registry.toml`` knob
an agent should consider tuning.

This is deliberately deterministic and dependency-light (stdlib only). It does NOT
change anything on its own; it produces evidence so the agent following ``SKILL.md`` can
decide ``continue | tune | rollback-and-rerun | abort``. Never raises into a caller: a
missing/partial run yields a report with ``status="no_data"``.

CLI::

    python -m memstrata.skills.optimization.monitor --run-dir <RUN_DIR> [--window N] [--json]

``--window N`` restricts the signals to the last N segments (default: all so far), which is
what you want when checking "the last few segments" during a long rollout.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Symptom -> (skill, registry knob, one-line fix hint). Kept in sync with skills/*/registry.toml
# and surfaced in optimization/registry.toml so the agent has a single lookup table.
SKILL_KNOBS: dict[str, dict[str, str]] = {
    "no_new_observations": {
        "skill": "crop_acquisition / entity_grounding / decomposition",
        "knob": "crop_acquisition: identity_threshold (lower ~0.45->0.30); entity_grounding: box_conf",
        "hint": "decompose is producing ~0 crops: identity gate too strict for cross-view generated frames, "
                "or grounding is missing the named entity. Loosen identity_threshold / box confidence.",
    },
    "memory_stagnant": {
        "skill": "memory_update / embedding_deduplication",
        "knob": "embedding_deduplication: dedup_threshold (raise); memory_update: admission/novelty gate",
        "hint": "obs are produced but the bank is not growing: curation/dedup is rejecting everything as "
                "redundant. Raise dedup_threshold so genuinely-new views get admitted.",
    },
    "memory_explosion": {
        "skill": "embedding_deduplication / memory_update",
        "knob": "embedding_deduplication: dedup_threshold (lower); memory_update: per-asset rep cap",
        "hint": "one asset accrues a rep every segment with no dedup: near-duplicate crops are all admitted. "
                "Lower dedup_threshold / add a per-asset representation cap.",
    },
    "context_not_read": {
        "skill": "compose (steps/compose.py) / bank selection",
        "knob": "compose: reference selection (name/alias match + identifier deref)",
        "hint": "bank is non-empty but segments read 0 memory refs: the Compose read path is not resolving "
                "referenced_entities to representations. Check name/alias match + identifier dereference.",
    },
    "ar_drift_risk": {
        "skill": "generation_routing",
        "knob": "generation_routing: prefer recompose_keyframe sooner; or run with --force-recompose",
        "hint": "long unbroken continue_ar chain: Helios AR drift accumulates (blur, subject loss). Route to "
                "recompose_keyframe every few beats, or pass --force-recompose.",
    },
    "router_infeasible": {
        "skill": "generation_routing",
        "knob": "generation_routing: rule feasibility layer (restrict modes before the MLLM picks)",
        "hint": "router keeps picking a mode that then falls back: the deterministic feasibility layer is "
                "letting infeasible modes through (e.g. continue_ar with no prior segment).",
    },
    "keyframe_missing": {
        "skill": "layout_anchor_processing / flux backend",
        "knob": "layout_anchor_processing: R3 layout / R4 crop2image; flux backend health",
        "hint": "segments have no composed keyframe: R3/R4 layout or the FLUX I2I fusion failed. Check the MLLM "
                "endpoint and the flux server log.",
    },
    "high_skip": {
        "skill": "generate backend / services",
        "knob": "video backend config (OOM/resolution) or crashed service",
        "hint": "segments are being skipped after retries: a backend/service is unhealthy (OOM, dead server). "
                "Check generator_logs and server logs before continuing.",
    },
}


def _load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (segment rows, meta). Prefer live progress.json, fall back to summary.json."""
    for name in ("progress.json", "summary.json"):
        p = run_dir / name
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                return list(d.get("segments", [])), d
            except Exception:
                continue
    return [], {}


def _signal(name: str, status: str, value: Any, detail: str) -> dict[str, Any]:
    s = {"signal": name, "status": status, "value": value, "detail": detail}
    if status == "bad" and name in SKILL_KNOBS:
        s.update({"skill": SKILL_KNOBS[name]["skill"], "knob": SKILL_KNOBS[name]["knob"],
                  "hint": SKILL_KNOBS[name]["hint"]})
    return s


def analyze(run_dir: Path, window: int = 0) -> dict[str, Any]:
    rows, meta = _load_rows(run_dir)
    if not rows:
        return {"status": "no_data", "run_dir": str(run_dir),
                "detail": "no progress.json/summary.json with segments yet"}
    win = rows[-window:] if window and window > 0 else rows
    n = len(win)

    obs = [len(r.get("new_observations") or []) for r in win]
    refs = [len(r.get("composed_refs") or []) for r in win]
    modes = Counter(str(r.get("used_mode") or r.get("route_mode") or "?") for r in win)
    fallbacks = sum(1 for r in win if r.get("used_mode") and r.get("route_mode")
                    and r["used_mode"] != r["route_mode"])
    kf_missing = sum(1 for r in win if not (r.get("keyframe") or r.get("fused")))
    # bank growth across the window: reps of the last segment vs the first
    first_reps = win[0].get("bank_representations") or {}
    last_reps = win[-1].get("bank_representations") or {}
    growth = sum(last_reps.values()) - sum(first_reps.values())
    max_asset = max(last_reps.items(), key=lambda kv: kv[1], default=("", 0))

    obs_mean = sum(obs) / n if n else 0.0
    refs_mean = sum(refs) / n if n else 0.0
    ar_run = _longest_run([str(r.get("used_mode") or "") for r in win], "continue_ar")
    bank_nonempty = bool(last_reps)

    sigs: list[dict[str, Any]] = []
    # 1. decompose producing observations?
    sigs.append(_signal("no_new_observations",
                        "bad" if obs_mean < 0.25 else ("warn" if obs_mean < 0.75 else "ok"),
                        round(obs_mean, 2), f"mean new observations/segment over last {n}"))
    # 2. memory growing vs producing obs
    if sum(obs) > 0 and growth <= 0:
        sigs.append(_signal("memory_stagnant", "bad", growth,
                            f"{sum(obs)} obs produced but bank grew by {growth}"))
    else:
        sigs.append(_signal("memory_stagnant", "ok", growth, f"bank grew by {growth} reps"))
    # 3. explosion: one asset +1 (or more) essentially every segment
    if max_asset[1] and n >= 4 and max_asset[1] >= n:
        sigs.append(_signal("memory_explosion", "warn", {max_asset[0]: max_asset[1]},
                            f"asset {max_asset[0]} has {max_asset[1]} reps over {n} segments — check dedup"))
    # 4. compose reading memory?
    if bank_nonempty and refs_mean < 0.25:
        sigs.append(_signal("context_not_read", "bad", round(refs_mean, 2),
                            "bank non-empty but ~0 memory refs read into segments"))
    else:
        sigs.append(_signal("context_not_read", "ok", round(refs_mean, 2),
                            f"mean memory refs read/segment over last {n}"))
    # 5. AR drift
    if ar_run >= 4:
        sigs.append(_signal("ar_drift_risk", "bad", ar_run,
                            f"{ar_run} consecutive continue_ar segments"))
    elif ar_run >= 3:
        sigs.append(_signal("ar_drift_risk", "warn", ar_run,
                            f"{ar_run} consecutive continue_ar segments"))
    # 6. router feasibility
    if n and fallbacks / n > 0.3:
        sigs.append(_signal("router_infeasible", "bad", f"{fallbacks}/{n}",
                            "router mode often overridden by fallback"))
    # 7. keyframe missing
    if kf_missing:
        sigs.append(_signal("keyframe_missing", "bad" if kf_missing > n // 2 else "warn",
                            f"{kf_missing}/{n}", "segments without a composed keyframe"))
    # 8. skips (rows are only appended on success; infer from meta.total vs highest segment_id gap)
    done = meta.get("done") or len(rows)
    max_cid = max((r.get("segment_id", -1) for r in rows), default=-1)
    skipped = (max_cid + 1) - len(rows) if max_cid >= 0 else 0
    if skipped > 0:
        sigs.append(_signal("high_skip", "bad" if skipped > 1 else "warn", skipped,
                            f"{skipped} segment(s) skipped after retries"))

    bad = [s for s in sigs if s["status"] == "bad"]
    warn = [s for s in sigs if s["status"] == "warn"]
    verdict = "abort_or_fix" if len(bad) >= 2 else ("investigate" if bad else
              ("watch" if warn else "healthy"))
    return {
        "status": "ok", "run_dir": str(run_dir), "backend": meta.get("backend"),
        "flux": meta.get("flux"), "done": done, "total": meta.get("total"),
        "window": n, "route_mix": dict(modes), "bank": dict(last_reps),
        "verdict": verdict, "signals": sigs,
        "flagged": [s["signal"] for s in bad], "warnings": [s["signal"] for s in warn],
    }


def _longest_run(seq: list[str], token: str) -> int:
    best = cur = 0
    for x in seq:
        cur = cur + 1 if x == token else 0
        best = max(best, cur)
    return best


def _digest(rep: dict[str, Any]) -> str:
    if rep.get("status") != "ok":
        return f"[opt] {rep.get('status')}: {rep.get('detail','')}"
    head = (f"[opt] {rep['done']}/{rep.get('total','?')} done · verdict={rep['verdict'].upper()} · "
            f"route={rep['route_mix']} · bank={rep['bank']}")
    lines = [head]
    for s in rep["signals"]:
        if s["status"] == "ok":
            continue
        mark = "BAD " if s["status"] == "bad" else "warn"
        lines.append(f"  [{mark}] {s['signal']}={s['value']} — {s['detail']}")
        if s.get("knob"):
            lines.append(f"         -> skill: {s['skill']}\n         -> knob: {s['knob']}")
    if len(lines) == 1:
        lines.append("  all signals healthy.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="MemStrata production self-optimization monitor")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--window", type=int, default=0, help="only last N segments (0 = all so far)")
    ap.add_argument("--json", action="store_true", help="print full JSON report")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rep = analyze(run_dir, window=args.window)
    if rep.get("status") == "ok":
        out = run_dir / "optimization"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"checkpoint_{rep['done']:04d}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2) if args.json else _digest(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
