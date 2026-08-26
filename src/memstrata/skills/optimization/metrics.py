"""Objective end-to-end metrics for a MemStrata production run (screenplay as ground truth).

This is the *self-contained* objective scorer for the generator-in-the-loop production runs
(``memstrata.production.run``). It mirrors the **read-path** axis of the public benchmark spec
``docs/benchmark/scoring_v2.md`` (visual-coverage: precision / recall[continuity] / f1 /
redundancy / efficiency) but judges the system's per-segment **context selection** against the
*screenplay's own* ``referenced_entities`` ground truth — so it needs **no gold movie and no
VLM judge**, only ``progress.json``/``summary.json`` + the run's ``screenplay.json``.

Why this is fair and objective (not a VLM opinion):
- The screenplay deterministically states, per shot, which entities are ``referenced`` (the
  prompt names them → the memory read-path must recall them) and which are ``forbidden``
  (``operation=avoid/deprecate`` → must NOT be reused). That is a frozen text GT, exactly the
  role ``gold/segment_annotations.json`` plays for the bench.
- ``continuity`` entities = referenced this segment AND named in some earlier segment (seen before →
  recallable). First appearances are excluded from recall, per scoring_v2 §1.

Per-segment metrics (all in ``[0,1]`` unless noted):
- ``recall``      [headline]  — continuity entities the read-path actually SELECTED / |continuity|.
- ``precision``               — selected assets that are actually referenced this segment / |selected|.
- ``f1``          [headline]  — harmonic mean of the two.
- ``avoidance_ok``            — 1 - (selected ∩ forbidden)/|forbidden| : did it avoid deprecated evidence.
- ``budget``      [descr.]    — |selected| (context size; report, don't score).
- ``redundancy_sim`` [opt]    — DINOv3 per-entity self-similarity of the selected crops (needs torch+weights;
                                 ``null`` otherwise). 1.0 = near-duplicate views, lower = diverse.
- ``memory_growth``           — total representations added across the window.

CLI::

    python -m memstrata.skills.optimization.metrics --run-dir <RUN_DIR> [--screenplay <PATH>] [--json]

``--screenplay`` defaults to ``<run-dir>/screenplay.json`` (run.py always writes it). Writes
``<run-dir>/metrics.json``. Deterministic; never raises into a caller (missing/partial run →
``{"status": "no_data"}``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # pinned by scoring_v2 §4.4


def _load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for name in ("progress.json", "summary.json"):
        p = run_dir / name
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                return list(d.get("segments", [])), d
            except Exception:
                continue
    return [], {}


def _shot_gt(screenplay_path: Path) -> list[dict[str, Any]]:
    """Per-segment GT from the screenplay: referenced + forbidden entity ids, scene start."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # src/
        from memstrata.adapters.screenplay import iter_shots, load_screenplay  # noqa: PLC0415
        sp = load_screenplay(screenplay_path)
        out = []
        for s in iter_shots(sp):
            out.append({"segment_id": s.segment_id, "referenced": list(s.referenced_entities),
                        "forbidden": list(s.forbidden_ids), "scene_start": s.is_scene_start})
        return out
    except Exception:
        return []


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _redundancy_sim(rows: list[dict[str, Any]]) -> float | None:
    """DINOv3 per-entity CLS self-similarity of selected crops, pair-weighted over the run.

    Returns ``None`` if torch / the pinned DINOv3 weights are unavailable (does not block).
    """
    try:
        import itertools  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415
    except Exception:
        return None
    # gather per-(segment,asset) crop paths
    groups: list[list[str]] = []
    for r in rows:
        by_asset: dict[str, list[str]] = {}
        for ref in (r.get("composed_refs") or []):
            pth = ref.get("path")
            if pth and Path(pth).is_file():
                by_asset.setdefault(ref.get("asset_id", "?"), []).append(pth)
        groups.extend([v for v in by_asset.values() if len(v) >= 2])
    if not groups:
        return None
    try:
        proc = AutoImageProcessor.from_pretrained(_DINO_ID)
        model = AutoModel.from_pretrained(_DINO_ID).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
    except Exception:
        return None

    cache: dict[str, Any] = {}

    def emb(p: str):
        if p in cache:
            return cache[p]
        with torch.no_grad():
            im = Image.open(p).convert("RGB")
            inp = {k: v.to(dev) for k, v in proc(images=im, return_tensors="pt").items()}
            cls = model(**inp).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)[0]
        cache[p] = cls
        return cls

    num = den = 0.0
    for g in groups:
        embs = [emb(p) for p in g]
        pairs = list(itertools.combinations(range(len(embs)), 2))
        msim = sum(float(embs[i] @ embs[j]) for i, j in pairs) / len(pairs)
        num += msim * len(pairs)
        den += len(pairs)
    return round(num / den, 4) if den else None


def score_run(run_dir: Path, screenplay_path: Path | None = None, *, window: int = 0,
              with_redundancy: bool = False) -> dict[str, Any]:
    rows, meta = _load_rows(run_dir)
    if not rows:
        return {"status": "no_data", "run_dir": str(run_dir)}
    sp_path = screenplay_path or (run_dir / "screenplay.json")
    gt = {g["segment_id"]: g for g in _shot_gt(sp_path)}
    if not gt:
        return {"status": "no_screenplay", "run_dir": str(run_dir), "screenplay": str(sp_path)}

    win = rows[-window:] if window and window > 0 else rows
    seen: set[str] = set()
    # rebuild the "seen before" set over the FULL prefix so continuity is correct even in a window
    prefix_seen: dict[int, set[str]] = {}
    acc: set[str] = set()
    for cid in sorted(gt):
        prefix_seen[cid] = set(acc)
        acc |= set(gt[cid]["referenced"])

    per: list[dict[str, Any]] = []
    for r in win:
        cid = r.get("segment_id")
        g = gt.get(cid)
        if g is None:
            continue
        ref = set(g["referenced"])
        forb = set(g["forbidden"])
        selected = set(r.get("selected_assets") or [])
        seen_before = prefix_seen.get(cid, set())
        continuity = ref & seen_before
        # recall over continuity; precision over selected; avoidance over forbidden
        recall = (len(continuity & selected) / len(continuity)) if continuity else None
        precision = (len(selected & ref) / len(selected)) if selected else None
        f1 = _f1(precision, recall) if (precision is not None and recall is not None) else None
        avoid_ok = (1.0 - len(selected & forb) / len(forb)) if forb else None
        per.append({
            "segment_id": cid, "referenced": sorted(ref), "continuity": sorted(continuity),
            "selected": sorted(selected), "forbidden": sorted(forb),
            "recall": recall, "precision": precision, "f1": f1, "avoidance_ok": avoid_ok,
            "budget": len(selected), "new_obs": len(r.get("new_observations") or []),
        })

    def _mean(key: str) -> float | None:
        vals = [p[key] for p in per if p.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    reps_first = (win[0].get("bank_representations") or {})
    reps_last = (win[-1].get("bank_representations") or {})
    growth = sum(reps_last.values()) - sum(reps_first.values())

    summary = {
        "recall": _mean("recall"), "precision": _mean("precision"), "f1": _mean("f1"),
        "avoidance_ok": _mean("avoidance_ok"), "budget": _mean("budget"),
        "memory_growth": growth, "segments_scored": len(per),
        "total_representations": sum(reps_last.values()),
        "bank": dict(reps_last),
    }
    if with_redundancy:
        summary["redundancy_sim"] = _redundancy_sim(win)
    return {"status": "ok", "run_dir": str(run_dir), "screenplay": str(sp_path),
            "backend": meta.get("backend"), "flux": meta.get("flux"),
            "done": meta.get("done"), "total": meta.get("total"),
            "window": len(per), "summary": summary, "per_segment": per}


def _digest(rep: dict[str, Any]) -> str:
    if rep.get("status") != "ok":
        return f"[metrics] {rep.get('status')}: {rep.get('run_dir','')}"
    s = rep["summary"]
    return (f"[metrics] {rep.get('done','?')}/{rep.get('total','?')} segments · "
            f"recall={s['recall']} precision={s['precision']} f1={s['f1']} "
            f"avoid_ok={s['avoidance_ok']} budget={s['budget']} "
            f"mem_growth={s['memory_growth']} reps={s['total_representations']}"
            + (f" redundancy_sim={s['redundancy_sim']}" if 'redundancy_sim' in s else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="MemStrata production objective e2e metrics")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--screenplay", default=None)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--redundancy", action="store_true", help="compute DINOv3 redundancy_sim (needs torch+weights)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rep = score_run(run_dir, Path(args.screenplay) if args.screenplay else None,
                    window=args.window, with_redundancy=args.redundancy)
    if rep.get("status") == "ok":
        (run_dir / "metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2) if args.json else _digest(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
