"""Human-friendly review view + incremental stitcher for a MemStrata production run.

The pipeline writes content-addressed blobs (``media/objects/sha256/..``) that dedup well but
are awful to browse. This builds a flat, ordered ``review/`` view next to the run and
(re)stitches the per-segment segments into one growing long video, and writes a visual
``INDEX.md`` that inlines each segment's prompt + the memory crops read in + the fused start
frame + what got banked.

Idempotent and best-effort: safe to call per segment (incremental) or once at the end; never
raises into the caller's loop; stitching degrades gracefully if ffmpeg is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _probe_meta(path: Path) -> dict[str, Any]:
    """Return clip duration/fps, or an empty dict when probing is unavailable."""
    try:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            meta = reader.__next__()
        finally:
            reader.close()
        return {"duration": float(meta["duration"]), "fps": float(meta["fps"])}
    except Exception:
        pass
    probe = shutil.which("ffprobe")
    if probe is None:
        return {}
    try:
        output = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration,avg_frame_rate", "-of", "csv=p=0", str(path)],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        rate, _, duration = output.partition(",")
        numerator, _, denominator = rate.partition("/")
        return {"duration": float(duration), "fps": float(numerator) / float(denominator or 1)}
    except Exception:
        return {}


def _segment_durations(review_dir: Path) -> list[float]:
    """Return per-clip durations in segment order, cached by filename and size."""
    cache_path = review_dir / "durations.json"
    try:
        cache = json.loads(cache_path.read_text())
    except Exception:
        cache = {}
    durations: list[float] = []
    dirty = False
    for segment in sorted((review_dir / "segments").glob("seg_*.mp4")):
        key = f"{segment.name}:{segment.stat().st_size}"
        if key not in cache:
            cache[key] = _probe_meta(segment).get("duration", 0.0)
            dirty = True
        durations.append(float(cache[key]))
    if dirty:
        try:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass
    return durations


def _resolve(src: str | Path | None) -> Path | None:
    """Resolve an artifact path (absolute, or relative to cwd) to an existing file.
    Some summary fields are labels (e.g. 'flux_i2i'), not paths -> return None."""
    if not src:
        return None
    p = Path(src)
    for cand in ([p] if p.is_absolute() else [p, Path.cwd() / p]):
        if cand.is_file():
            return cand
    return None


def _copy(src: str | Path | None, dst: Path) -> bool:
    p = _resolve(src)
    if p is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    return True


def _find_obs_crops(run_dir: Path, segment_id: int) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    segment_dir = run_dir / "observations" / f"segment_{segment_id:03d}"
    if not segment_dir.is_dir():
        return out
    for entity_dir in sorted(p for p in segment_dir.iterdir() if p.is_dir()):
        crops = [c for c in sorted(entity_dir.glob("crop_*.png")) if not c.name.endswith("_mask.png")]
        if crops:
            out.append((entity_dir.name, crops[0]))
    return out


def _stitch(review_dir: Path) -> Path | None:
    segs = sorted((review_dir / "segments").glob("seg_*.mp4"))
    if not segs:
        return None
    ff = _ffmpeg()
    if ff is None:
        return None
    listing = review_dir / "_concat.txt"
    listing.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
    out = review_dir / "long_video.mp4"
    cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return out if out.is_file() else None


def _write_index(review_dir: Path, rows: list[dict[str, Any]], *, title: str = "") -> None:
    lines = [
        f"# {title or 'MemStrata production'} — review view",
        "",
        f"Segments: {len(rows)} · stitched: [`long_video.mp4`](long_video.mp4)",
        "",
        "## Overview",
        "",
        "| # | scene | shot | mode | prompt | refs in | reps (after) | obs | segment |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cid = r["segment_id"]
        reps = r.get("bank_representations") or {}
        reps_s = ", ".join(f"{k}:{v}" for k, v in reps.items())
        obs = str(len(r.get("new_observations") or []))
        seg = f"[seg](segments/seg_{cid:03d}.mp4)" if (review_dir / "segments" / f"seg_{cid:03d}.mp4").is_file() else "—"
        prompt = str(r.get("prompt", "")).replace("|", "\\|")
        lines.append(
            f"| {cid} | {r.get('scene_id','')} | {r.get('shot_id','')} | "
            f"{r.get('used_mode') or r.get('route_mode') or ''} | {prompt} | "
            f"{len(r.get('composed_refs') or [])} | {reps_s} | {obs} | {seg} |"
        )

    lines += ["", "---", "", "## Per-segment context", ""]
    for r in rows:
        cid = r["segment_id"]
        mode = str(r.get("used_mode") or r.get("route_mode") or "")
        route = str(r.get("route_mode") or "")
        lines.append(f"### Segment {cid} · scene `{r.get('scene_id','')}` · shot `{r.get('shot_id','')}` · mode `{mode}`"
                     + (f" (router suggested `{route}`)" if route and route != mode else ""))
        lines += ["", f"**Prompt:** {r.get('prompt','')}", ""]
        ctx_dir = review_dir / "context" / f"seg_{cid:03d}"
        ctx_imgs = sorted(ctx_dir.glob("*.png")) if ctx_dir.is_dir() else []
        if ctx_imgs:
            lines += [f"**Memory read in ({len(ctx_imgs)} refs):**", "",
                      " ".join(f'<img src="context/seg_{cid:03d}/{p.name}" height="120" title="{p.stem}">'
                               for p in ctx_imgs), ""]
        else:
            lines += ["**Memory read in:** _(none — fresh scene / bootstrap)_", ""]
        if (review_dir / "keyframes" / f"seg_{cid:03d}.png").is_file():
            lines += ["**Start frame (composed keyframe → generator):**", "",
                      f'<img src="keyframes/seg_{cid:03d}.png" height="200">', ""]
        if (review_dir / "segments" / f"seg_{cid:03d}.mp4").is_file():
            lines += [f"**Generated segment:** [segments/seg_{cid:03d}.mp4](segments/seg_{cid:03d}.mp4)", ""]
        obs_imgs = sorted((review_dir / "observations").glob(f"seg_{cid:03d}__*.png"))
        if obs_imgs:
            lines += [f"**Banked this segment ({len(obs_imgs)} obs):**", "",
                      " ".join(f'<img src="observations/{p.name}" height="120" title="{p.name.split("__",1)[-1]}">'
                               for p in obs_imgs), ""]
        reps = r.get("bank_representations") or {}
        lines += [f"**Bank after:** {', '.join(f'{k}={v}' for k, v in reps.items())}", "", "---", ""]
    (review_dir / "INDEX.md").write_text("\n".join(lines) + "\n")


def organize_segment(run_dir: str | Path, row: dict[str, Any]) -> None:
    """Place one segment's artifacts into review/. Best-effort, never raises."""
    try:
        run_dir = Path(run_dir)
        review = run_dir / "review"
        cid = int(row["segment_id"])
        _copy(row.get("video"), review / "segments" / f"seg_{cid:03d}.mp4")
        for cand in (row.get("fused"), row.get("keyframe")):  # `fused` may be a label, not a path
            if _copy(cand, review / "keyframes" / f"seg_{cid:03d}.png"):
                break
        for ref in (row.get("composed_refs") or []):
            aid = str(ref.get("asset_id", "ref"))
            rid = str(ref.get("representation_id", "")).replace("@", "_at_").replace("/", "_")
            name = f"{aid}__{rid}.png" if rid else f"{aid}.png"
            _copy(ref.get("path"), review / "context" / f"seg_{cid:03d}" / name)
        for entity_id, crop in _find_obs_crops(run_dir, cid):
            _copy(crop, review / "observations" / f"seg_{cid:03d}__{entity_id}.png")
    except Exception:
        pass


def organize_run(run_dir: str | Path, rows: list[dict[str, Any]] | None = None,
                 *, title: str = "") -> dict[str, Any]:
    """(Re)build the whole review/ view + stitch. Reads summary.json if rows not given."""
    run_dir = Path(run_dir)
    review = run_dir / "review"
    review.mkdir(parents=True, exist_ok=True)
    if rows is None:
        summ = run_dir / "summary.json"
        rows = json.loads(summ.read_text()).get("segments", []) if summ.is_file() else []
    for row in rows:
        organize_segment(run_dir, row)
    long_video = _stitch(review)
    _write_index(review, rows, title=title)
    durations = _segment_durations(review)
    return {"review_dir": str(review),
            "segments": len(list((review / "segments").glob("seg_*.mp4"))),
            "long_video": str(long_video) if long_video else None,
            "segment_durations": durations,
            "duration_sec": sum(durations) if durations else None,
            "fps": _probe_meta(long_video).get("fps") if long_video else None}
