"""Canonical production assembly for observing already-realized video segments."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from memstrata.bank import AssetBank
from memstrata.encoders import build_image_embedding
from memstrata.mllm.angle_classifier import build_angle_classifier
from memstrata.mllm.crop_attributes import build_crop_attribute_classifier
from memstrata.pipeline import MemStrata, build_curator, build_decomposer
from memstrata.production.profiles import ProductionProfile, resolve_production_profile
from memstrata.skills.composition.policy import CompositionPolicy
from memstrata.skills.memory_update import MemoryPolicy


def _openai_model_ready(base_url: str, model: str, *, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/models", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    return any(
        isinstance(item, dict) and item.get("id") == model
        for item in payload.get("data", [])
    )


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
    mllm_base_url: str | None = None,
    mllm_model: str | None = None,
    require_mllm: bool | None = None,
    resume: bool = False,
    seed_screenplay: dict[str, Any] | None = None,
    location_scene_validity_enabled: bool | None = None,
    location_resolver_shadow_enabled: bool = False,
    location_scene_plate_candidates: bool = False,
    location_scene_evidence_enabled: bool | None = None,
    location_adaptive_enabled: bool | None = None,
    location_storage_cap: int | None = None,
    location_read_max_refs: int | None = None,
    location_extra_budget_share: float = 0.50,
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
            "mllm_model": (mllm_model, selected.mllm_model),
            "require_mllm": (require_mllm, selected.require_mllm),
            "location_adaptive_enabled": (location_adaptive_enabled, False),
            "location_scene_evidence_enabled": (
                location_scene_evidence_enabled,
                False,
            ),
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
    adaptive_location = (
        selected.adaptive_location_memory
        if location_adaptive_enabled is None
        else bool(location_adaptive_enabled)
    )
    scene_validity = (
        adaptive_location
        if location_scene_validity_enabled is None
        else bool(location_scene_validity_enabled)
    )
    scene_evidence = (
        adaptive_location
        if location_scene_evidence_enabled is None
        else bool(location_scene_evidence_enabled)
    )
    read_budget = (
        selected.read_context_rep_budget
        if read_context_rep_budget is None
        else max(1, int(read_context_rep_budget))
    )
    storage_cap = max(
        1,
        int(
            selected.location_storage_cap
            if location_storage_cap is None
            else location_storage_cap
        ),
    )
    read_location_max = max(
        1,
        int(
            selected.location_read_max_refs
            if location_read_max_refs is None
            else location_read_max_refs
        ),
    )
    strict_wedetect = (
        selected.require_wedetect if require_wedetect is None else bool(require_wedetect)
    )
    strict_mllm = selected.require_mllm if require_mllm is None else bool(require_mllm)
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
    configured_mllm_url = (
        os.environ.get("MEMSTRATA_CONTEXT_JUDGER_BASE_URL", selected.mllm_base_url)
        if mllm_base_url is None
        else str(mllm_base_url)
    ).strip()
    configured_mllm_model = (
        os.environ.get("MEMSTRATA_VLM_MODEL", selected.mllm_model)
        if mllm_model is None
        else str(mllm_model)
    ).strip()
    if naming == "mllm" and strict_mllm and not _openai_model_ready(
        configured_mllm_url, configured_mllm_model
    ):
        raise RuntimeError(
            f"profile {selected.name!r} requires MLLM model "
            f"{configured_mllm_model!r} at {configured_mllm_url}"
        )

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    persisted = Path(persist_path) if persist_path else root / "bank.json"
    policy = MemoryPolicy.production(
        discovery=bool(discovery),
        location_scene_validity_enabled=scene_validity,
        location_resolver_shadow_enabled=bool(location_resolver_shadow_enabled),
        location_adaptive_storage_enabled=adaptive_location,
        location_storage_cap=storage_cap,
    )
    composition_policy = CompositionPolicy(
        adaptive_location_enabled=adaptive_location,
        global_rep_budget=read_budget,
        location_read_max_refs=read_location_max,
        location_extra_budget_share=location_extra_budget_share,
    )
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
        "MEMSTRATA_CROP_IDENTITY_BASE_URL": configured_mllm_url,
        "MEMSTRATA_CROP_IDENTITY_MODEL": configured_mllm_model,
    }
    verify_crop_identity = naming == "mllm"
    extra_acquire_kwargs: dict[str, Any] = {}
    if location_scene_plate_candidates:
        extra_acquire_kwargs["location_scene_plate_candidates"] = True
    if scene_evidence:
        extra_acquire_kwargs["location_scene_evidence_enabled"] = True
    cropper = ProposeIdentifyCropper(
        bank=bank,
        server_dir=root / "crop_acq_server",
        work_dir=root / "observations",
        device=str(crop_acq_device),
        identity_threshold=float(identity_threshold),
        identity_verification_required=verify_crop_identity,
        frame_pos=float(frame_pos),
        extra_acquire_kwargs=extra_acquire_kwargs or None,
        server_env=server_env,
        grounding_backend="wedetect_ref",
        require_wedetect=strict_wedetect,
    )
    entity_namer = None
    if naming == "mllm":
        from memstrata.mllm.runner import HttpTransport, MllmRoleRunner
        from memstrata.skills.decomposition.vlm_decomposer import VlmEntityDecomposer

        transport = HttpTransport(configured_mllm_url)
        entity_namer = VlmEntityDecomposer(
            runner=MllmRoleRunner(
                text_transport=transport,
                vision_transport=transport,
                text_model=configured_mllm_model,
                vision_model=configured_mllm_model,
            )
        )
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
            "read_context_rep_budget": read_budget,
            "grounding_backend": "wedetect_ref",
            "require_wedetect": strict_wedetect,
            "mllm_base_url": configured_mllm_url,
            "mllm_model": configured_mllm_model,
            "require_mllm": strict_mllm,
            "crop_identity_verification": {
                "required_for_established_identity": verify_crop_identity,
                "mode": "image_only_query_to_references",
            },
            **(
                {
                    "location_memory_v2": {
                        "scene_validity_enabled": bool(
                            scene_validity
                        ),
                        "resolver_shadow_enabled": bool(
                            location_resolver_shadow_enabled
                        ),
                        "scene_plate_candidates": bool(
                            location_scene_plate_candidates
                        ),
                        "scene_evidence_enabled": scene_evidence,
                        "adaptive_storage_read": adaptive_location,
                        "storage_cap_guardrail": storage_cap,
                        "read_max_refs": read_location_max,
                        "location_extra_budget_share": float(
                            location_extra_budget_share
                        ),
                    }
                }
                if (
                    scene_validity
                    or location_resolver_shadow_enabled
                    or location_scene_plate_candidates
                    or scene_evidence
                    or adaptive_location
                )
                else {}
            ),
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
        read_context_rep_budget=read_budget,
        composition_policy=composition_policy,
    )


__all__ = ["build_realized_segment_pipeline"]
