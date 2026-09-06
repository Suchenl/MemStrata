"""Canonical production assembly for observing already-realized video segments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from memstrata.bank import AssetBank
from memstrata.encoders import build_image_embedding
from memstrata.mllm.angle_classifier import build_angle_classifier
from memstrata.mllm.crop_attributes import build_crop_attribute_classifier
from memstrata.pipeline import MemStrata, build_curator, build_decomposer
from memstrata.production.profiles import ProductionProfile, resolve_production_profile
from memstrata.skills.memory_update import MemoryPolicy


def build_realized_segment_pipeline(
    *,
    run_dir: str | Path,
    profile: str | ProductionProfile = "production",
    persist_path: str | Path | None = None,
    movie_id: str = "",
    write_naming: str | None = None,
    discovery: bool = False,
    crop_acq_device: str = "",
    identity_threshold: float = 0.25,
    frame_pos: float = 0.8,
    namer_frames: int = 3,
    embedder_provider: str | None = None,
    angle_classifier_mode: str = "",
    read_slow_fallback: bool | None = None,
    read_max_reps_per_asset: int | None = None,
    read_context_rep_budget: int | None = None,
    max_reps_per_asset: int | None = None,
    wedetect_url: str | None = None,
    require_wedetect: bool | None = None,
    resume: bool = False,
    seed_screenplay: dict[str, Any] | None = None,
) -> MemStrata:
    """Build the production read/write implementation shared by formal Track A runs."""
    from memstrata.skills.crop_acquisition.crop_client import (
        ProposeIdentifyCropper,
        ServerConceptDiscoverer,
    )
    from memstrata.skills.crop_acquisition.wedetect_client import (
        RequiredGrounderError,
        WeDetectRefGrounder,
    )

    selected = resolve_production_profile(profile)
    if selected.name == "paper_tracka_202607":
        conflicts = {
            "write_naming": (write_naming, selected.write_naming),
            "embedder_provider": (embedder_provider, selected.embedder_provider),
            "read_slow_fallback": (read_slow_fallback, selected.read_slow_fallback),
            "read_max_reps_per_asset": (
                read_max_reps_per_asset,
                selected.read_max_reps_per_asset,
            ),
            "require_wedetect": (require_wedetect, selected.require_wedetect),
        }
        invalid = [
            f"{name}={actual!r}"
            for name, (actual, expected) in conflicts.items()
            if actual is not None and actual != expected
        ]
        if invalid:
            raise ValueError(
                f"profile {selected.name!r} is immutable; conflicting overrides: "
                + ", ".join(invalid)
            )
    naming = selected.write_naming if write_naming is None else str(write_naming)
    embedding = selected.embedder_provider if embedder_provider is None else str(embedder_provider)
    slow_fallback = (
        selected.read_slow_fallback
        if read_slow_fallback is None
        else bool(read_slow_fallback)
    )
    read_reps = (
        selected.read_max_reps_per_asset
        if read_max_reps_per_asset is None
        else max(1, int(read_max_reps_per_asset))
    )
    strict_wedetect = (
        selected.require_wedetect if require_wedetect is None else bool(require_wedetect)
    )
    configured_url = (
        os.environ.get("MEMSTRATA_WEDETECT_URL", selected.wedetect_url)
        if wedetect_url is None
        else str(wedetect_url)
    ).strip()
    if strict_wedetect:
        grounder = WeDetectRefGrounder(configured_url, strict=True) if configured_url else None
        if grounder is None or not grounder.healthy():
            raise RequiredGrounderError(
                f"profile {selected.name!r} requires a healthy WeDetect-Ref service; "
                f"checked {configured_url or '<unset>'}"
            )

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    persisted = Path(persist_path) if persist_path else root / "bank.json"
    policy = MemoryPolicy.production(discovery=bool(discovery))
    mode = angle_classifier_mode or None
    angle_classifier = build_angle_classifier(mode=mode)
    crop_attribute_classifier = build_crop_attribute_classifier(mode=mode)
    embedder = build_image_embedding(provider=embedding or "hash")
    bank = AssetBank.load(persisted) if resume and persisted.is_file() else AssetBank()
    curator_options: dict[str, Any] = {}
    if max_reps_per_asset is not None:
        curator_options["max_reps_per_asset"] = int(max_reps_per_asset)
    curator = build_curator(
        bank,
        policy=policy,
        embedder=embedder,
        angle_classifier=angle_classifier,
        crop_attribute_classifier=crop_attribute_classifier,
        **curator_options,
    )
    if seed_screenplay is not None and not (resume and persisted.is_file()):
        from memstrata.adapters.screenplay import seed_packet

        curator.ingest_packet(seed_packet(seed_screenplay))

    server_env = {
        "MEMSTRATA_WEDETECT_URL": configured_url,
        "MEMSTRATA_REQUIRE_WEDETECT": "1" if strict_wedetect else "0",
    }
    cropper = ProposeIdentifyCropper(
        bank=bank,
        server_dir=root / "crop_acq_server",
        work_dir=root / "observations",
        device=str(crop_acq_device),
        identity_threshold=float(identity_threshold),
        frame_pos=float(frame_pos),
        server_env=server_env,
        grounding_backend="wedetect_ref",
        require_wedetect=strict_wedetect,
    )
    entity_namer = None
    if naming == "mllm":
        from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer

        entity_namer = VlmEntityDecomposer()
    discoverer = None
    if policy.discovery and entity_namer is None:
        discoverer = ServerConceptDiscoverer(cropper, work_dir=root / "discoveries")
    decomposer = build_decomposer(
        policy=policy,
        embedder=embedder,
        cropper=cropper,
        angle_classifier=angle_classifier,
        discoverer=discoverer,
        entity_namer=entity_namer,
        namer_frames=int(namer_frames),
        namer_frame_dir=root / "observations",
    )
    return MemStrata.for_production(
        persist_path=persisted,
        manifest_path=root / "run_manifest.json",
        production_profile=selected.name,
        production_provenance={
            "profile": selected.name,
            "write_naming": naming,
            "embedder_provider": embedding,
            "read_slow_fallback": slow_fallback,
            "read_max_reps_per_asset": read_reps,
            "read_context_rep_budget": read_context_rep_budget,
            "grounding_backend": "wedetect_ref",
            "require_wedetect": strict_wedetect,
        },
        policy=policy,
        bank=bank,
        curator=curator,
        decomposer=decomposer,
        embedder=embedder,
        angle_classifier=angle_classifier,
        crop_attribute_classifier=crop_attribute_classifier,
        run_dir=root / "pipeline",
        membank_dir=root / "membank",
        movie_id=movie_id,
        slow_on_miss=slow_fallback,
        read_max_reps_per_asset=read_reps,
        read_context_rep_budget=read_context_rep_budget,
    )


__all__ = ["build_realized_segment_pipeline"]
