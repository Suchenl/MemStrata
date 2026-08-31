"""Screenplay-driven MemStrata production run (the one runner; supersedes the old
scripts/memstrata/run_generator_pipeline.py smoke demo).

Reads a production_screenplay JSON, seeds the stratified bank from ``main_entities``, then
drives the four-step closed loop shot by shot:

    intent → compose → keyframe(R3/R4[/FLUX]) → generate(video backend) → decompose → curate

using ``MemStrata.for_production`` plus the existing steps/skills. Two decompose modes:

  * ``crop_server`` (default) — the real S5 propose/identify/novelty crop server (needs a GPU);
    the memory bank grows from generated video.
  * ``none`` — skip decompose (oracle-empty), for a no-GPU backend smoke; equivalent to the
    old run_generator_pipeline quick check (``--backend recording|oracle|wan_t2v|...``).

Outputs land under ``<outputs-root>/<story_id>/<system>/<timestamp>/`` with a human-readable
``review/`` view. Invoke via ``python -m memstrata.production.run`` (or the bash launcher
``scripts/memstrata/run_production.sh``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# A single failed segment is absorbed (isolated generator hiccup); this many in a row means the
# run is misconfigured and must stop, because every subsequent segment would start from an empty
# bank and the run would "succeed" while measuring nothing.
_MAX_CONSECUTIVE_SEGMENT_FAILURES = 3


def memstrata_root() -> Path:
    # .../src/memstrata/production/run.py -> parents[3] = this repository
    return Path(__file__).resolve().parents[3]


def _crop_acquisition_digest(run_dir: Path) -> dict[str, Any] | None:
    """Roll the per-entity crop-acquisition summary up into the run manifest (C item 3).

    ``ProposeIdentifyCropper`` writes ``observations/crop_acquisition_summary.json`` in its
    own work dir; surfacing an aggregate in the run manifest makes acquisition miss rate and
    crop provenance auditable without opening a nested per-cropper file. Returns ``None`` when
    the run had no crop server (decompose="none") or produced no crop attempts.
    """
    summary_path = run_dir / "observations" / "crop_acquisition_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entities = summary.get("entities") or {}
    attempts = hits = misses = 0
    sources: dict[str, int] = {}
    for row in entities.values():
        attempts += int(row.get("attempts", 0))
        hits += int(row.get("hits", 0))
        misses += int(row.get("misses", 0))
        for src, count in (row.get("sources") or {}).items():
            sources[src] = sources.get(src, 0) + int(count)
    return {
        "config": summary.get("config", {}),
        "entities": len(entities),
        "attempts": attempts,
        "hits": hits,
        "misses": misses,
        "miss_rate": (misses / attempts) if attempts else None,
        "sources": dict(sorted(sources.items())),
    }


def _anchor_membank_to_film(mem: Any, run_dir: Path, info: dict[str, Any]) -> None:
    """Make ``membank/`` a self-contained bank anchored to the grown film.

    The memory schema's central promise is that every ``sec`` is a timestamp on
    ``long_video.mp4``. That needs three things the pipeline cannot know on its own: the
    film's real path, its fps/duration, and where each segment starts on it. The film is
    *copied* (not linked) into the bank so the directory can be moved or shipped on its own,
    and the copy is skipped while the source is unchanged in size.
    """
    membank = run_dir / "membank"
    membank.mkdir(parents=True, exist_ok=True)
    source = info.get("long_video")
    if source and Path(source).is_file():
        src = Path(source)
        dst = membank / "long_video.mp4"
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        mem.long_video_path = str(dst)
    if info.get("fps"):
        mem.fps = float(info["fps"])
    if info.get("duration_sec") is not None:
        mem.long_video_duration_sec = float(info["duration_sec"])
    durations = info.get("segment_durations") or []
    start = 0.0
    starts: dict[int, float] = {}
    for index, duration in enumerate(durations):
        starts[index] = round(start, 3)
        start += float(duration)
    if starts:
        mem.segment_start_sec = starts
    # The snapshot for this segment was written before the film grew, so refresh the header
    # and timestamps now that the timeline is known.
    mem.write_memory_snapshot()


def build_pipeline(
    *,
    screenplay: dict[str, Any],
    backend_name: str,
    run_dir: Path,
    system: str = "memstrata",
    flux: bool = True,
    flux_backend: str = "flux.2-klein-9b-kv-fp8",
    width: int = 832,
    height: int = 480,
    decompose: str = "crop_server",
    crop_acq_device: str = "",
    models_config: Path | None = None,
    embedder_provider: str = "",
    angle_classifier_mode: str = "",
    discovery: bool = False,
    write_naming: str = "perception",
    resume: bool = False,
):
    """Assemble the seeded bank + generator (+keyframe) + MemStrata.for_production.

    Returns ``(mem, generator, composer)``. All heavy logic is imported from ``memstrata.*``.

    The curator/decomposer are built through ``build_curator`` / ``build_decomposer`` with
    the SAME ``MemoryPolicy`` the pipeline gets. Constructing them directly (as this
    function used to) silently discarded every production knob — the attribute classifier
    included — so the "stratified" bank ran with all-unknown angles, the crop-quality gate
    was off, and the bank-wide budget never applied.
    """
    from memstrata.adapters.screenplay import seed_packet
    from memstrata.bank import AssetBank
    from memstrata.encoders import build_image_embedding
    from memstrata.mllm.angle_classifier import build_angle_classifier
    from memstrata.mllm.crop_attributes import build_crop_attribute_classifier
    from memstrata.pipeline import MemStrata, build_curator, build_decomposer
    from memstrata.skills.memory_update import MemoryPolicy
    from memstrata.steps.generate import MediaTaskGenerator
    from memstrata.steps.generate.backends import build_video_backend
    from memstrata.steps.keyframe import KeyframeComposer

    cfg = models_config or (memstrata_root() / "configs")
    run_dir.mkdir(parents=True, exist_ok=True)

    policy = MemoryPolicy.production(discovery=bool(discovery))
    # Angle/attribute classifiers come from one factory pair so MEMSTRATA_ANGLE_CLASSIFIER
    # (and an explicit --angle-classifier) reach BOTH the decomposer and the curator.
    mode = angle_classifier_mode or None
    angle_classifier = build_angle_classifier(mode=mode)
    crop_attr_classifier = build_crop_attribute_classifier(mode=mode)
    # Embedding provider drives every similarity gate (near-duplicate, cohesion,
    # reconciliation). The deterministic hash default keeps the no-GPU smoke runnable but
    # is non-semantic, so those gates stay off until a real encoder is selected.
    emb = build_image_embedding(provider=embedder_provider or "hash")

    # Resuming an interrupted story means reopening its bank, not rebuilding it: the memory IS the
    # persisted artifact, so a run that died at shot 40 can carry its 40 shots of accumulated
    # identity forward instead of paying for them again.
    persisted_bank = run_dir / "bank.json"
    if resume and persisted_bank.is_file():
        bank = AssetBank.load(persisted_bank)
        print(f"[prod] RESUME: reopened bank with {len(bank.assets)} assets from {persisted_bank}",
              flush=True)
    else:
        bank = AssetBank()
    curator = build_curator(
        bank,
        policy=policy,
        embedder=emb,
        angle_classifier=angle_classifier,
        crop_attribute_classifier=crop_attr_classifier,
    )
    # A reopened bank already holds the seeds; re-ingesting them would fold duplicates back in.
    if not (resume and persisted_bank.is_file()):
        curator.ingest_packet(seed_packet(screenplay))

    video_backend = build_video_backend(
        backend_name, output_dir=run_dir / "media", run_id=system, models_config=cfg)
    # The keyframe composer is attached for real runs (flux keyframe or the crop_server path);
    # the lightweight backend smoke (decompose="none", no FLUX) skips it entirely -> no GPU
    # needed. In the default native keyframe mode it composes multi-image on FLUX with no MLLM;
    # only the legacy MEMSTRATA_KEYFRAME_MODE=collage path calls the Qwen MLLM (R3/R4).
    composer = None
    if flux or decompose == "crop_server":
        image_backend = None
        if flux:
            from memstrata.steps.generate.image_backends.factory import build_image_backend
            image_backend = build_image_backend(
                flux_backend, output_dir=run_dir / "keyframes_flux", run_id=system, models_config=cfg)
        composer = KeyframeComposer(image_backend, width=width, height=height,
                                    work_dir=run_dir / "keyframes")
    generator = MediaTaskGenerator(
        video_backend, bank=bank, model_name=backend_name,
        default_controls={"transition": "continue"},
        keyframe_composer=composer, log_dir=run_dir / "generator_logs")

    decomposer = None
    if decompose == "crop_server":
        from memstrata.skills.crop_acquisition.crop_client import (
            ProposeIdentifyCropper,
            ServerConceptDiscoverer,
        )
        cropper = ProposeIdentifyCropper(
            bank=bank, server_dir=run_dir / "crop_acq_server",
            work_dir=run_dir / "observations", device=str(crop_acq_device))
        discoverer = None
        if policy.discovery and write_naming != "mllm":
            # Reuses the cropper's already-running server, so discovery costs no extra
            # model load — only extra proposals per frame.
            #
            # Skipped under mllm naming: the namer already returns the entities that are visible
            # but unnamed by the prompt as UNANCHORED observations, which is the same D_T role,
            # except they arrive with a label and a description. Running the concept proposer on
            # top only added assets no shot can ask for — 4 of 24 in a measured 13-segment run
            # were 'character_disc_c009_0'-style records.
            discoverer = ServerConceptDiscoverer(
                cropper, work_dir=run_dir / "discoveries")
        # Write-side naming. Without it a first appearance can only enter memory through
        # discovery, which never infers names, so every new entity is banked under a synthetic
        # label (character_disc_c000_0) that the name-authoritative read path cannot resolve —
        # measured: 23 anonymous assets over 21 segments and an empty selection despite a full
        # bank. The namer binds the shot's own wording to entities it visually confirms, which is
        # the same capability Track A selects with MEMSTRATA_TRACKA_NAME_SOURCE=mllm.
        entity_namer = None
        if write_naming == "mllm":
            from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer

            entity_namer = VlmEntityDecomposer()
        decomposer = build_decomposer(
            policy=policy, embedder=emb, cropper=cropper,
            angle_classifier=angle_classifier, discoverer=discoverer,
            entity_namer=entity_namer, namer_frame_dir=run_dir / "observations")

    mem = MemStrata.for_production(
        persist_path=run_dir / "bank.json", policy=policy, bank=bank, generator=generator,
        curator=curator, decomposer=decomposer, embedder=emb,
        angle_classifier=angle_classifier, crop_attribute_classifier=crop_attr_classifier,
        run_dir=run_dir / "pipeline", membank_dir=run_dir / "membank",
        movie_id=str(screenplay.get("story_id", "")))
    return mem, generator, composer


def run_production(
    *,
    screenplay_path: str | Path,
    backend_name: str = "wan22_i2v_a14b_lightx2v_4step",
    system: str = "memstrata",
    outputs_root: Path | None = None,
    run_dir: Path | None = None,
    segments: int = 0,
    flux: bool = True,
    flux_backend: str = "flux.2-klein-9b-kv-fp8",
    force_recompose: bool = False,
    use_router_mllm: bool = True,
    decompose: str = "crop_server",
    crop_acq_device: str = "",
    width: int = 832,
    height: int = 480,
    autoserve: bool = True,
    mllm_gpu: str = "0",
    mllm_port: int = 8000,
    stop_services: bool = False,
    bench_mode: bool = True,
    embedder_provider: str = "",
    angle_classifier_mode: str = "",
    discovery: bool = False,
    write_naming: str = "perception",
    resume: bool = False,
) -> dict[str, Any]:
    from memstrata.adapters.screenplay import iter_shots, load_screenplay
    from memstrata.lib.review import organize_run
    from memstrata.production.services import ServiceManager, required_services
    from memstrata.skills.generation_routing import GenerationRouter

    screenplay = load_screenplay(screenplay_path)
    story_id = str(screenplay.get("story_id", Path(screenplay_path).stem))
    shots = iter_shots(screenplay)
    n = segments if segments and segments > 0 else len(shots)

    root = memstrata_root()
    run_dir = run_dir or (
        (outputs_root or root / "production/outputs") / story_id / system
        / _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prod] story={story_id} shots={len(shots)} run={n} decompose={decompose} out={run_dir}", flush=True)
    (run_dir / "screenplay.json").write_text(
        json.dumps(screenplay, ensure_ascii=False, indent=2), encoding="utf-8")

    # Bring up the services this configuration declares (reuse-first; shared-node safe). The
    # video/image/crop servers self-start inside their backends, so this is just the MLLM endpoint.
    svc = ServiceManager(run_dir)
    if autoserve:
        specs = required_services(flux=flux, decompose=decompose, use_router_mllm=use_router_mllm,
                                  mllm_gpu=mllm_gpu, mllm_port=mllm_port)
        if specs and mllm_port != 8000:
            os.environ["MEMSTRATA_CONTEXT_JUDGER_BASE_URL"] = f"http://127.0.0.1:{mllm_port}/v1"
        svc.ensure_all(specs)

    try:
        return _run_loop(
            screenplay=screenplay, story_id=story_id, shots=shots, n=n, run_dir=run_dir,
            backend_name=backend_name, system=system, flux=flux, flux_backend=flux_backend,
            force_recompose=force_recompose, use_router_mllm=use_router_mllm, decompose=decompose,
            crop_acq_device=crop_acq_device, width=width, height=height,
            organize_run=organize_run, GenerationRouter=GenerationRouter, bench_mode=bench_mode,
            embedder_provider=embedder_provider,
            angle_classifier_mode=angle_classifier_mode, discovery=discovery,
            write_naming=write_naming, resume=resume)
    finally:
        if stop_services:
            svc.shutdown_launched()


def _resume_state(run_dir: Path, n: int) -> tuple[dict[int, dict], list[int]]:
    """Recover the closed shots (keyed by shot id) and the list of shots still to produce.

    Keyed by shot id, not counted: a segment the generator could not produce is skipped without
    writing a record, so ``len(records)`` is smaller than the shot the run actually reached. Resuming
    at that count would re-produce a shot that already exists while leaving the skipped one missing
    forever — the story then ends with a duplicate and a hole, and every later shot is misaligned
    against the prompt stream it is scored against. Addressing shots by id instead makes a resume both
    idempotent and hole-filling.
    """

    progress = run_dir / "progress.json"
    if not progress.is_file():
        return {}, list(range(n))
    try:
        data = json.loads(progress.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, list(range(n))
    closed: dict[int, dict] = {}
    for record in data.get("segments") or []:
        if not isinstance(record, dict):
            continue
        try:
            sid = int(record["segment_id"])
        except (KeyError, TypeError, ValueError):
            continue
        # A shot id outside the requested range belongs to a longer run (e.g. a --segments-limited
        # probe reopening a full story) and must not be inherited.
        if 0 <= sid < n:
            closed.setdefault(sid, record)
    return closed, [i for i in range(n) if i not in closed]


def _run_loop(
    *, screenplay, story_id, shots, n, run_dir, backend_name, system, flux, flux_backend,
    force_recompose, use_router_mllm, decompose, crop_acq_device, width, height,
    organize_run, GenerationRouter, bench_mode=False,
    embedder_provider="", angle_classifier_mode="", discovery=False, write_naming="perception",
    resume=False,
) -> dict[str, Any]:
    mem, _generator, composer = build_pipeline(
        screenplay=screenplay, backend_name=backend_name, run_dir=run_dir, system=system,
        flux=flux, flux_backend=flux_backend, width=width, height=height,
        decompose=decompose, crop_acq_device=crop_acq_device,
        embedder_provider=embedder_provider,
        angle_classifier_mode=angle_classifier_mode, discovery=discovery,
        write_naming=write_naming, resume=resume)
    strat0 = mem.stratification()
    print(f"[prod] policy={mem.policy.name} angle_classifier={type(mem.angle_classifier).__name__} "
          f"crop_attr={type(mem.crop_attribute_classifier).__name__} "
          f"embedder_semantic={mem.curator.embedder_is_semantic} "
          f"cohesion_floor={mem.curator.cohesion_floor} discovery={mem.policy.discovery} "
          f"seed_reps={strat0['representations']}", flush=True)

    # No-GPU smoke (decompose="none") also skips MLLM routing so it runs without a Qwen server.
    from memstrata.bank.schema import NON_USABLE, LifecycleStatus

    # MoVE-Bench Track B fairness contract: in bench_mode the SUT must decide entirely from the
    # prose prompt + its own memory, seeing NO screenplay GT. We therefore CLOSE the two GT
    # leakage paths that exist for the (internal, oracle-assisted) production runs:
    #   (1) forbidden-deprecate: writing shot.forbidden_ids into the bank as DEPRECATED hands the
    #       SUT the avoidance answer (this is what inflated OPTIMIZATION_JOURNAL avoid_ok 0.667→1.0);
    #   (2) referenced_entities/onscreen: feeding shot.referenced_entities (the GT present set) to
    #       the router tells the SUT who is on screen.
    # In bench_mode both are suppressed; the router relies only on prompt/transition/scene_return
    # and "onscreen" is the SUT's OWN prior selection (self-derived, not GT). A manifest records
    # that no GT field was consumed so results are auditable.
    if bench_mode:
        print("[prod] BENCH-MODE: GT leakage closed (no forbidden-deprecate, no GT referenced_entities)", flush=True)
    router = GenerationRouter(use_mllm=use_router_mllm and decompose != "none")
    results: list[dict] = []
    prev_entities: list[str] = []
    gt_consumed = {"forbidden_deprecate": 0, "referenced_entities": 0}
    consecutive_failures = 0
    last_error: BaseException | None = None
    closed: dict[int, dict] = {}
    todo = list(range(n))
    if resume:
        closed, todo = _resume_state(run_dir, n)
        results = [closed[sid] for sid in sorted(closed)]
        if closed:
            holes = [i for i in todo if i < max(closed)]
            print(f"[prod] RESUME: {len(closed)}/{n} shots already closed, {len(todo)} to produce"
                  + (f" (filling holes at {holes})" if holes else ""), flush=True)
    for i in todo:
        shot = shots[i]
        # Continuity is per shot, not per append: when filling a hole the previous on-screen set is
        # the one its own predecessor closed with, not whatever the loop happened to touch last.
        # Only in bench_mode, where "onscreen" is the SUT's own prior selection; the oracle-assisted
        # path derives it from the screenplay's GT instead and must keep doing so.
        if bench_mode:
            if i - 1 in closed:
                prev_entities = list(closed[i - 1].get("selected_assets") or [])
            elif i == 0:
                prev_entities = []
        if composer is not None:
            composer.seed = 2026 + i

        # Lifecycle avoidance (deprecated-evidence hard case): the screenplay's
        # ``operation=avoid/deprecate`` marks an entity as gone (deceased / destroyed /
        # discredited). Deprecate it in the bank BEFORE this segment's compose so the read-path's
        # intrinsic NON_USABLE gate (compose.is_usable) drops it even when the prose still names
        # it (e.g. a memorial plaque engraved with the deceased's name). This is a memory-state
        # transition driven by the production authoring directive — a GT signal, so it is SKIPPED
        # under bench_mode (the SUT must learn avoidance from its own memory/prompt, not from GT).
        if not bench_mode:
            for aid in shot.forbidden_ids:
                asset = mem.bank.assets.get(aid)
                if asset is not None and asset.status not in NON_USABLE:
                    mem.bank.update_status(aid, LifecycleStatus.DEPRECATED)
                    gt_consumed["forbidden_deprecate"] += 1
                    print(f"[prod] segment {i}: deprecate {aid} (screenplay operation=avoid)", flush=True)
        # bench_mode: hide the GT present/forbidden set from the router (prose-only decision).
        if bench_mode:
            routed_refs, routed_onscreen = [], (prev_entities if i > 0 else [])
        else:
            routed_refs, routed_onscreen = shot.referenced_entities, (prev_entities if i > 0 else [])
            gt_consumed["referenced_entities"] += len(shot.referenced_entities)
        decision = router.route(
            prompt=shot.prompt, segment_id=i, referenced_entities=routed_refs,
            onscreen_entities=routed_onscreen, has_prev_segment=(i > 0),
            continue_vs_cut=shot.transition, scene_return=shot.is_scene_start,
            prev_summary=(shots[i - 1].scene_id if i > 0 else ""))
        eff = "recompose_keyframe" if force_recompose else decision.mode.value
        print(f"[prod] segment {i} [{shot.scene_id}/{shot.shot_id}] route={decision.mode.value} "
              f"src={decision.source} eff={eff}", flush=True)

        # decompose="none": bypass the (GPU) decomposer with an empty oracle -> generate-only smoke.
        oracle = [] if decompose == "none" else None
        segment, used = None, None
        for attempt in [eff] + (["recompose_keyframe"] if eff != "recompose_keyframe" else []):
            try:
                segment = mem.run_segment(
                    shot.prompt, segment_id=i, oracle_observations=oracle,
                    generation_controls={"gen_mode": attempt, "transition": shot.transition})
                used = attempt
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[prod] segment {i} mode={attempt} FAILED: {exc!r}", flush=True)
        # "onscreen" for the next segment: bench_mode uses the SUT's OWN selection (self-derived,
        # not GT); otherwise the GT referenced set (legacy oracle-assisted behavior).
        if bench_mode:
            prev_entities = list(segment.context.asset_ids) if segment is not None else []
        else:
            prev_entities = shot.referenced_entities
        if segment is None:
            print(f"[prod] segment {i}: SKIPPED", flush=True)
            consecutive_failures += 1
            # Skipping a segment is meant to absorb an isolated generator hiccup. A run that
            # cannot produce several segments in a row is misconfigured, not unlucky (a gated /
            # missing weight, a dead service), and letting it continue is worse than failing:
            # every later segment starts from an empty bank, so the run *completes* while
            # measuring nothing. Stop and surface the last error instead.
            if consecutive_failures >= _MAX_CONSECUTIVE_SEGMENT_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} consecutive segments failed "
                    f"(last: {last_error!r}); aborting instead of running on an empty bank."
                )
            continue
        consecutive_failures = 0

        kf = (segment.generation.task.controls.get("keyframe_record")
              if segment.generation and segment.generation.task else None)
        composed_refs = []
        for aid, rep_ids in (segment.context.representation_ids or {}).items():
            for rid in rep_ids:
                found = mem.bank.find_representation(rid)
                if found is not None:
                    composed_refs.append({"asset_id": aid, "representation_id": rid,
                                          "path": found[1].object_uri})
        reps = {aid: len(a.representations) for aid, a in mem.bank.assets.items()}
        record = {
            "segment_id": i, "scene_id": shot.scene_id, "shot_id": shot.shot_id,
            "prompt": shot.prompt, "route_mode": decision.mode.value, "used_mode": used,
            "route_source": decision.source, "transition": shot.transition,
            "selected_assets": segment.context.asset_ids, "composed_refs": composed_refs,
            # Which read path answered this segment. "name_recovered" means the primary path
            # returned nothing for a name the bank still holds and the deterministic rematch
            # had to save it — a defect signal worth grepping for after a run, not a normal hit.
            "intent_source": segment.context.intent_resolution_source,
            "keyframe": kf.get("keyframe") if kf else None,
            "fused": kf.get("fused") if kf else None,
            "video": segment.generation.video_path if segment.generation else None,
            "bank_assets": sorted(mem.bank.assets.keys()), "bank_representations": reps,
            "new_observations": [o.observation_id for o in segment.observations],
            "touched_asset_ids": list(segment.touched_asset_ids),
        }
        # Both the assembled review video and the per-shot scoring read this list positionally, so it
        # must stay in shot order even when a hole is filled after later shots were already produced.
        closed[i] = record
        results = [closed[sid] for sid in sorted(closed)]
        print(f"[prod] segment {i}: used={used} assets={segment.context.asset_ids} "
              f"obs={len(segment.observations)} reps={reps}", flush=True)
        # Split the read path when it loses ground between intent and compose: an English replay of
        # a 18-segment run had three segments whose intent resolved the right names (Elias, Mara)
        # yet whose context came back empty, which is invisible when only the composed ids are
        # logged. Printed only on a mismatch so a healthy run stays quiet.
        intent_ids = [ref.asset_id for ref in segment.request.references]
        if intent_ids != list(segment.context.asset_ids):
            dropped = [aid for aid in intent_ids if aid not in set(segment.context.asset_ids)]
            print(f"[prod] segment {i}: read-path drop src={segment.context.intent_resolution_source} "
                  f"intent={intent_ids} composed={list(segment.context.asset_ids)} dropped={dropped}",
                  flush=True)
        # Incremental progress dump so the agent-in-the-loop optimizer (skills/optimization)
        # can diagnose a LIVE run before summary.json exists.
        (run_dir / "progress.json").write_text(
            json.dumps({"story_id": story_id, "system": system, "backend": backend_name,
                        # Closed-shot count, not the loop position: filling a hole at shot 13 of a
                        # story with 74 shots closed must not report progress as 14.
                        "flux": flux, "run_dir": str(run_dir), "done": len(results),
                        "total": len(shots), "segments": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        try:
            info = organize_run(run_dir, results, title=story_id)
            print(f"[prod] review: {info['segments']} segs -> {info['long_video']}", flush=True)
            _anchor_membank_to_film(mem, run_dir, info)
        except Exception as exc:  # noqa: BLE001
            print(f"[prod] organize skipped: {exc!r}", flush=True)

    # Audit manifest: for a Track B (bench_mode) run this must show zero GT consumption; any
    # non-zero count means the run leaked GT and its avoidance/recall numbers are oracle-assisted.
    gt_leakage = "none" if (bench_mode and gt_consumed == {"forbidden_deprecate": 0, "referenced_entities": 0}) \
        else ("oracle_assisted" if not bench_mode else "LEAK_DETECTED")
    manifest = {"story_id": story_id, "system": system, "backend": backend_name,
                "bench_mode": bench_mode, "gt_consumed": gt_consumed, "gt_leakage": gt_leakage,
                "run_dir": str(run_dir)}
    crop_acq = _crop_acquisition_digest(run_dir)
    if crop_acq is not None:
        manifest["crop_acquisition"] = crop_acq
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if bench_mode and gt_leakage != "none":
        raise AssertionError(f"bench_mode run consumed GT: {gt_consumed}")

    # Stratification diagnostic: the direct evidence for the stratified-memory claim.
    # Read spatial_known_ratio / state_known_ratio FIRST — at 0 the attribute classifier
    # was not wired and no angle-stratification result may be reported from this run.
    strat = mem.stratification()
    mem.write_stratification_summary(run_dir / "stratification.json")
    summary = {"story_id": story_id, "system": system, "backend": backend_name,
               "flux": flux, "run_dir": str(run_dir), "bench_mode": bench_mode,
               "gt_leakage": gt_leakage, "segments": results,
               "bank_size": len(mem.bank.assets),
               "policy": mem.policy.name,
               "angle_classifier": type(mem.angle_classifier).__name__,
               "crop_attribute_classifier": type(mem.crop_attribute_classifier).__name__,
               "embedder_is_semantic": mem.curator.embedder_is_semantic,
               "cohesion_floor": mem.curator.cohesion_floor,
               "discovery": mem.policy.discovery,
               "stratification": strat,
               "final_representations": {aid: len(a.representations) for aid, a in mem.bank.assets.items()}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[prod] DONE (bench_mode={bench_mode}, gt_leakage={gt_leakage}):",
          json.dumps(summary["final_representations"], ensure_ascii=False), flush=True)
    print(f"[prod] stratification: spatial_known={strat['spatial_known_ratio']} "
          f"state_known={strat['state_known_ratio']} described={strat['described_ratio']} "
          f"buckets={strat['bucket_coverage']} sources={strat['acquisition_source_counts']}", flush=True)
    if strat["representations"] and strat["spatial_known_ratio"] == 0.0:
        print("[prod] WARNING: every representation has spatial_angle=unknown — the memory is "
              "NOT stratified in this run; do not report stratification results from it "
              "(set --angle-classifier vlm / MEMSTRATA_ANGLE_CLASSIFIER=vlm).", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(memstrata_root() / "src"))
    root = memstrata_root()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screenplay", default=str(root / "production/screenplay/products/en/0000_detective_mystery.json"),
                    help="production_screenplay JSON (default: the detective example)")
    ap.add_argument("--backend", default="wan22_i2v_a14b_lightx2v_4step",
                    help="default video backend: wan22_i2v_a14b_lightx2v_4step (Wan2.2-I2V-A14B "
                         "MoE 4-step distilled, LightX2V engine, 480h×832w, flash_attn2, no offload). "
                         "Also: wan22_i2v_a14b_distill4step (native_wan) | recording | oracle | any "
                         "stem of configs/video_gen/<name>.toml")
    ap.add_argument("--list-backends", action="store_true")
    ap.add_argument("--system", default="memstrata", help="system name for the output path")
    ap.add_argument("--segments", type=int, default=0, help="limit shots (0 = whole screenplay)")
    ap.add_argument("--flux", dest="flux", action="store_true", default=True,
                    help="add FLUX I2I keyframe fusion (DEFAULT ON — keyframes/first-frame need it)")
    ap.add_argument("--no-flux", dest="flux", action="store_false",
                    help="disable FLUX keyframe fusion (R3/R4 collage keyframe only; no photoreal)")
    ap.add_argument("--flux-backend", default="flux.2-klein-9b-kv-fp8")
    ap.add_argument("--force-recompose", action="store_true",
                    help="recompose a fresh keyframe every shot (film-quality; avoids AR drift)")
    ap.add_argument("--no-router-mllm", action="store_true", help="rules-only routing")
    ap.add_argument("--decompose", choices=["crop_server", "none"], default="crop_server",
                    help="crop_server = real S5 GPU cropper (memory grows); none = no-GPU backend smoke")
    ap.add_argument("--crop-acq-device", default="", help="GPU index for the S5 crop server")
    ap.add_argument("--write-naming", choices=["perception", "mllm"], default="perception",
                    help="mllm = MemStrata's VlmEntityDecomposer binds this shot's own prompt "
                         "names to entities it confirms in the generated frames, so a first "
                         "appearance is retrievable by name later; perception = generic labels "
                         "only (identity then rests entirely on visual reconciliation)")
    ap.add_argument("--no-autoserve", action="store_true",
                    help="do not bring up required services (Qwen MLLM); assume they are already up")
    ap.add_argument("--mllm-gpu", default="0", help="GPU index for the auto-served Qwen MLLM endpoint")
    ap.add_argument("--mllm-port", type=int, default=8000, help="port for the Qwen MLLM endpoint")
    ap.add_argument("--stop-services", action="store_true",
                    help="stop services this run launched when it finishes (reused ones are left alone)")
    # bench_mode (no GT leakage) is the DEFAULT: every run that could be evaluated must be fair.
    # The opt-out re-enables the forbidden-deprecate + GT-referenced_entities oracle and is ONLY
    # for internal upper-bound diagnostics — it must NEVER back a published/eval number.
    ap.add_argument("--oracle-assisted", dest="bench_mode", action="store_false", default=True,
                    help="DANGER (diagnostics only): re-enable GT leakage (forbidden-deprecate + GT "
                         "referenced_entities to router). NOT valid for any evaluation. Default is bench-mode.")
    # --- write-path configuration (stratification + similarity gates) ---------------
    ap.add_argument("--angle-classifier", default="", choices=["", "null", "heuristic", "vlm"],
                    help="crop angle/attribute classifier. '' = respect "
                         "MEMSTRATA_ANGLE_CLASSIFIER (default null). Use 'vlm' to actually "
                         "populate spatial/state strata + observation descriptions — with "
                         "'null' every rep stays unknown and NO stratification result may "
                         "be reported from the run.")
    ap.add_argument("--embedder", default="hash",
                    help="image embedding provider for the similarity gates "
                         "(hash|dinov3|insightface|vpr). 'hash' is deterministic and "
                         "offline-safe but NON-semantic, so the cohesion gate and the "
                         "cohesion self-audit stay disabled under it.")
    ap.add_argument("--discovery", action="store_true",
                    help="enable type-constrained discovery (O_disc): bank supported-type "
                         "entities the intent did not request, resolved by identity "
                         "reconciliation. Needs the crop server (--decompose crop_server).")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--outputs-root", type=Path, default=root / "production/outputs")
    ap.add_argument("--run-dir", type=Path, default=None, help="override the timestamped output dir")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --run-dir: reopen its persisted bank and "
                         "skip the shots progress.json shows closed. The external memory is the "
                         "point, so a story that died at shot 40 keeps those 40 shots of identity "
                         "instead of regenerating them.")
    args = ap.parse_args(argv)

    if args.list_backends:
        from memstrata.steps.generate.backends import list_video_backend_names
        print("\n".join(list_video_backend_names()))
        return 0

    try:
        run_production(
            screenplay_path=args.screenplay, backend_name=args.backend, system=args.system,
            outputs_root=args.outputs_root, run_dir=args.run_dir, segments=args.segments,
            flux=args.flux, flux_backend=args.flux_backend, force_recompose=args.force_recompose,
            use_router_mllm=not args.no_router_mllm, decompose=args.decompose,
            crop_acq_device=args.crop_acq_device, width=args.width, height=args.height,
            autoserve=not args.no_autoserve, mllm_gpu=args.mllm_gpu, mllm_port=args.mllm_port,
            stop_services=args.stop_services, bench_mode=args.bench_mode,
            embedder_provider=args.embedder, angle_classifier_mode=args.angle_classifier,
            discovery=args.discovery, write_naming=args.write_naming, resume=args.resume)
    except Exception as exc:  # noqa: BLE001 — surface GPU/weight/config errors clearly
        print(json.dumps({"backend": args.backend, "error": repr(exc)}, ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
