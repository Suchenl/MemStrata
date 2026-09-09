from __future__ import annotations

import inspect
from dataclasses import asdict, replace
from pathlib import Path

from memstrata.bank import (
    Asset,
    AssetBank,
    AssetRelation,
    AssetRepresentation,
    AssetType,
    LifecycleStatus,
    RelationType,
)
from memstrata.skills.composition.compose import compose
from memstrata.skills.composition.location_router import (
    LocationReadQuery,
    rank_location_candidates,
)
from memstrata.skills.composition.policy import CompositionPolicy
from memstrata.skills.intent_understanding.interpreter import (
    AssetReference,
    CompositionRequest,
)
from memstrata.skills.memory_update.location_coreset import (
    CORESET_KEY,
    LocationCoresetPolicy,
    update_location_coreset,
)
from memstrata.production.profiles import LOCATION_ADAPTIVE_V1, PAPER_TRACKA_202607
from memstrata.production.realized import build_realized_segment_pipeline
from memstrata.steps.generate.materialize import composed_reference_images


def _rep(
    rep_id: str,
    asset_id: str,
    segment: int,
    embedding: list[float] | None,
    *,
    environment: str = "unknown",
    lighting: str = "day",
    quality: float = 0.8,
    eligible: bool | None = True,
    path: str = "",
) -> AssetRepresentation:
    annotations: dict[str, object] = {
        "quality": quality,
        "encoder_route": "test:2d",
        "crop_attributes": {
            "shot_size": "wide",
            "lighting": lighting,
            "description": f"{environment} {lighting} location",
        },
        "location_metadata": {
            "environment": environment,
            "lighting": lighting,
            "scene_scale": "wide",
        },
    }
    if embedding is not None:
        annotations["embedding"] = embedding
    if eligible is not None:
        annotations["scene_reference_eligible"] = eligible
    return AssetRepresentation(
        representation_id=rep_id,
        asset_id=asset_id,
        object_uri=path or f"/tmp/{rep_id}.png",
        origin_segment_id=segment,
        reference_aspects=["scene_reference"],
        annotations=annotations,
    )


def _location(asset_id: str = "loc") -> Asset:
    return Asset(
        asset_id=asset_id,
        kind=AssetType.LOCATION,
        name=asset_id,
        status=LifecycleStatus.REUSABLE,
    )


def _generic_asset(asset_id: str, kind: AssetType) -> Asset:
    aspect = "identity_anchor" if kind is AssetType.CHARACTER else "object_continuity"
    rep = AssetRepresentation(
        representation_id=f"{asset_id}@0",
        asset_id=asset_id,
        object_uri=f"/tmp/{asset_id}.png",
        origin_segment_id=0,
        reference_aspects=[aspect],
    )
    return Asset(asset_id, kind, asset_id, LifecycleStatus.REUSABLE, representations=[rep])


def _build_clustered_location(
    specs: list[tuple[str, list[float], str, str]],
    *,
    cap: int = 12,
) -> Asset:
    asset = _location()
    policy = LocationCoresetPolicy(
        enabled=True,
        storage_cap=cap,
        cluster_join_distance=0.15,
        cluster_merge_distance=0.05,
    )
    for index, (rep_id, vector, environment, lighting) in enumerate(specs):
        update_location_coreset(
            asset,
            _rep(
                rep_id,
                asset.asset_id,
                index,
                vector,
                environment=environment,
                lighting=lighting,
            ),
            policy=policy,
        )
    return asset


def test_location_cluster_assignment_updates_support() -> None:
    asset = _location()
    policy = LocationCoresetPolicy(enabled=True, cluster_join_distance=0.20)
    update_location_coreset(asset, _rep("r0", "loc", 0, [1.0, 0.0]), policy=policy)
    update_location_coreset(asset, _rep("r1", "loc", 1, [0.99, 0.01]), policy=policy)
    state = asset.metadata[CORESET_KEY]
    assert len(state["clusters"]) == 1
    assert state["clusters"][0]["support_count"] == 2
    assert sum(not rep.deprecated for rep in asset.representations) == 1


def test_known_strata_conflict_creates_cluster() -> None:
    asset = _location()
    policy = LocationCoresetPolicy(enabled=True)
    update_location_coreset(
        asset,
        _rep("inside", "loc", 0, [1.0, 0.0], environment="indoor"),
        policy=policy,
    )
    update_location_coreset(
        asset,
        _rep("outside", "loc", 1, [1.0, 0.0], environment="outdoor"),
        policy=policy,
    )
    assert len(asset.metadata[CORESET_KEY]["clusters"]) == 2


def test_redundant_compatible_clusters_merge() -> None:
    asset = _location()
    policy = LocationCoresetPolicy(
        enabled=True,
        cluster_join_distance=0.01,
        cluster_merge_distance=0.30,
    )
    update_location_coreset(asset, _rep("r0", "loc", 0, [1.0, 0.0]), policy=policy)
    update_location_coreset(asset, _rep("r1", "loc", 1, [0.8, 0.6]), policy=policy)
    assert len(asset.metadata[CORESET_KEY]["clusters"]) == 1
    assert any(rep.deprecated_by == "location_cluster_merge" for rep in asset.representations)


def test_scene_ineligible_rep_never_forms_cluster() -> None:
    asset = _location()
    update_location_coreset(
        asset,
        _rep("bad", "loc", 0, [1.0, 0.0], eligible=False),
        policy=LocationCoresetPolicy(enabled=True),
    )
    assert CORESET_KEY not in asset.metadata
    assert asset.representations[0].annotations["scene_reference_eligible"] is False


def test_unknown_metadata_without_embedding_does_not_explode() -> None:
    asset = _location()
    policy = LocationCoresetPolicy(enabled=True)
    for index in range(5):
        update_location_coreset(
            asset,
            _rep(f"r{index}", "loc", index, None, lighting="unknown"),
            policy=policy,
        )
    assert len(asset.metadata[CORESET_KEY]["clusters"]) == 1


def test_cap_merges_then_evicts_and_preserves_canonical() -> None:
    asset = _location()
    policy = LocationCoresetPolicy(
        enabled=True,
        storage_cap=2,
        cluster_join_distance=0.01,
        cluster_merge_distance=0.02,
    )
    for index, vector in enumerate(([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0])):
        update_location_coreset(
            asset,
            _rep(f"r{index}", "loc", index, list(vector), lighting=f"light-{index}"),
            policy=policy,
        )
    state = asset.metadata[CORESET_KEY]
    assert len(state["clusters"]) == 2
    assert state["cap_hit_count"] == 1
    assert state["canonical_cluster_id"] in {
        cluster["cluster_id"] for cluster in state["clusters"]
    }
    assert any(rep.deprecated_by == "location_storage_cap_guardrail" for rep in asset.representations)


def test_cluster_state_round_trips(tmp_path: Path) -> None:
    asset = _build_clustered_location(
        [("r0", [1.0, 0.0], "indoor", "day")]
    )
    bank = AssetBank({asset.asset_id: asset})
    path = tmp_path / "bank.json"
    bank.save(path)
    restored = AssetBank.load(path).get_asset(asset.asset_id)
    assert restored is not None
    assert restored.metadata[CORESET_KEY] == asset.metadata[CORESET_KEY]


def test_explicit_high_confidence_hint_selects_one() -> None:
    asset = _build_clustered_location(
        [
            ("day", [1.0, 0.0], "outdoor", "day"),
            ("night", [0.0, 1.0], "outdoor", "night"),
        ]
    )
    bank = AssetBank({"loc": asset})
    request = CompositionRequest(
        references=[AssetReference(asset_id="loc", function="scene_reference")],
        context_rep_budget=16,
    )
    context = compose(
        bank,
        request,
        as_of_segment_id=10,
        raw_prompt="A wide nighttime exterior view.",
        policy=CompositionPolicy(
            adaptive_location_enabled=True,
            global_rep_budget=16,
            location_read_max_refs=4,
        ),
    )
    assert context.representation_ids["loc"] == ["night"]
    assert context.selection_trace["assets"]["loc"]["hint_source"] == "raw_prompt"


def test_low_margin_multicluster_selects_minimal_diverse_set() -> None:
    asset = _build_clustered_location(
        [
            ("a", [1.0, 0.0], "unknown", "day"),
            ("b", [0.0, 1.0], "unknown", "night"),
            ("c", [-1.0, 0.0], "indoor", "artificial"),
        ]
    )
    context = compose(
        AssetBank({"loc": asset}),
        CompositionRequest(
            references=[AssetReference(asset_id="loc", function="scene_reference")],
            context_rep_budget=16,
        ),
        as_of_segment_id=10,
        raw_prompt="Return to the location.",
        policy=CompositionPolicy(
            adaptive_location_enabled=True,
            location_read_max_refs=4,
            location_min_marginal_gain=0.01,
            location_coverage_stop=0.99,
        ),
    )
    assert 2 <= len(context.representation_ids["loc"]) <= 4


def test_low_diversity_stops_after_one() -> None:
    asset = _location()
    reps = [
        _rep("a", "loc", 0, [1.0, 0.0]),
        _rep("b", "loc", 1, [0.99, 0.01]),
    ]
    asset.representations = reps
    asset.metadata[CORESET_KEY] = {
        "schema_version": 1,
        "canonical_cluster_id": "c0",
        "next_cluster_index": 2,
        "cap_hit_count": 0,
        "clusters": [
            {
                "cluster_id": f"c{index}",
                "representative_rep_id": rep.representation_id,
                "centroid": rep.annotations["embedding"],
                "encoder_route": "test:2d",
                "support_count": 1,
                "quality_ema": 0.8,
                "strata": {},
            }
            for index, rep in enumerate(reps)
        ],
    }
    for index, rep in enumerate(reps):
        rep.annotations["location_cluster_id"] = f"c{index}"
    context = compose(
        AssetBank({"loc": asset}),
        CompositionRequest(
            references=[AssetReference(asset_id="loc")],
            context_rep_budget=16,
        ),
        as_of_segment_id=10,
        raw_prompt="Return to the location.",
        policy=CompositionPolicy(adaptive_location_enabled=True),
    )
    assert len(context.representation_ids["loc"]) == 1
    assert context.selection_trace["assets"]["loc"]["stop_reason"] == "marginal_gain_stop"


def test_router_filters_future_and_ineligible_reps() -> None:
    asset = _location()
    asset.representations = [
        _rep("past", "loc", 1, [1.0, 0.0]),
        _rep("future", "loc", 5, [0.0, 1.0]),
        _rep("bad", "loc", 2, [-1.0, 0.0], eligible=False),
    ]
    ranking = rank_location_candidates(
        asset,
        LocationReadQuery(raw_prompt="", as_of_segment_id=5),
        policy=CompositionPolicy(adaptive_location_enabled=True),
    )
    assert [candidate.rep.representation_id for candidate in ranking.candidates] == ["past"]
    assert ranking.dropped["future"] == 1
    assert ranking.dropped["ineligible"] == 1


def test_adaptive_relation_expansion_filters_future_metadata() -> None:
    loc = _build_clustered_location(
        [("l0", [1.0, 0.0], "outdoor", "day")]
    )
    char = _generic_asset("char", AssetType.CHARACTER)
    loc.relations.append(
        AssetRelation(
            relation_type=RelationType.LOCATED_IN,
            target_asset_id="char",
            attributes={"origin_segment_id": 10},
        )
    )
    context = compose(
        AssetBank({"loc": loc, "char": char}),
        CompositionRequest(
            references=[AssetReference(asset_id="loc")],
            relation_hops=1,
        ),
        as_of_segment_id=5,
        policy=CompositionPolicy(adaptive_location_enabled=True),
    )
    assert context.expanded == []


def test_part_of_location_neighbor_uses_cross_asset_marginal_gain() -> None:
    prison = _build_clustered_location(
        [("prison-wide", [1.0, 0.0], "indoor", "day")]
    )
    prison.asset_id = "prison"
    prison.name = "Prison"
    for rep in prison.representations:
        rep.asset_id = prison.asset_id
    cell = _build_clustered_location(
        [("cell-interior", [0.0, 1.0], "indoor", "artificial")]
    )
    cell.asset_id = "cell"
    cell.name = "Prison Cell"
    for rep in cell.representations:
        rep.asset_id = cell.asset_id
    cell.relations.append(
        AssetRelation(
            relation_type=RelationType.PART_OF,
            target_asset_id="prison",
            attributes={
                "schema": "memstrata.location-read-neighbor.v1",
                "origin_segment_id": 2,
                "read_neighbor": True,
            },
        )
    )

    context = compose(
        AssetBank({"prison": prison, "cell": cell}),
        CompositionRequest(
            references=[AssetReference(asset_id="prison")],
            context_rep_budget=16,
            relation_hops=1,
        ),
        as_of_segment_id=10,
        raw_prompt="Return to the prison.",
        policy=CompositionPolicy(adaptive_location_enabled=True),
    )

    assert context.expanded == ["cell"]
    assert context.representation_ids == {
        "prison": ["prison-wide"],
        "cell": ["cell-interior"],
    }
    allocation = context.selection_trace["allocation"]
    assert [row["source_cluster_id"] for row in allocation] == [
        "location-cluster-0000",
        "location-cluster-0000",
    ]
    assert allocation[1]["phase"] == "global_marginal"
    assert allocation[1]["location_pool_id"] == "prison"
    assert allocation[1]["relation_provenance"]["direction"] == "incoming"


def test_related_location_extra_cannot_displace_explicit_character_or_prop() -> None:
    char = _generic_asset("char", AssetType.CHARACTER)
    prop = _generic_asset("prop", AssetType.PROP)
    location = _build_clustered_location(
        [("location-wide", [1.0, 0.0], "outdoor", "day")]
    )
    location.asset_id = "location"
    for rep in location.representations:
        rep.asset_id = location.asset_id
    neighbor = _build_clustered_location(
        [("location-detail", [0.0, 1.0], "outdoor", "night")]
    )
    neighbor.asset_id = "neighbor"
    for rep in neighbor.representations:
        rep.asset_id = neighbor.asset_id
    neighbor.relations.append(
        AssetRelation(
            RelationType.PART_OF,
            "location",
            {"origin_segment_id": 1, "read_neighbor": True},
        )
    )
    bank = AssetBank(
        {
            "char": char,
            "prop": prop,
            "location": location,
            "neighbor": neighbor,
        }
    )
    refs = [
        AssetReference(asset_id="char"),
        AssetReference(asset_id="prop"),
        AssetReference(asset_id="location"),
    ]

    context = compose(
        bank,
        CompositionRequest(
            references=refs,
            context_rep_budget=3,
            relation_hops=1,
        ),
        as_of_segment_id=10,
        policy=CompositionPolicy(adaptive_location_enabled=True),
    )

    assert context.representation_ids["char"]
    assert context.representation_ids["prop"]
    assert context.representation_ids["location"]
    assert context.representation_ids["neighbor"] == []
    assert sum(map(len, context.representation_ids.values())) == 3


def test_named_assets_reserved_and_budget_is_strict() -> None:
    char = _generic_asset("char", AssetType.CHARACTER)
    prop = _generic_asset("prop", AssetType.PROP)
    loc = _build_clustered_location(
        [
            ("l0", [1.0, 0.0], "outdoor", "day"),
            ("l1", [0.0, 1.0], "outdoor", "night"),
        ]
    )
    bank = AssetBank({"char": char, "prop": prop, "loc": loc})
    refs = [AssetReference(asset_id=asset_id) for asset_id in ("char", "prop", "loc")]
    context = compose(
        bank,
        CompositionRequest(references=refs, context_rep_budget=3),
        as_of_segment_id=10,
        raw_prompt="The character uses the prop at the location.",
        policy=CompositionPolicy(adaptive_location_enabled=True),
    )
    assert {asset_id: len(context.representation_ids[asset_id]) for asset_id in refs_ids(refs)} == {
        "char": 1,
        "prop": 1,
        "loc": 1,
    }
    assert sum(map(len, context.representation_ids.values())) == 3


def refs_ids(refs: list[AssetReference]) -> list[str]:
    return [ref.asset_id for ref in refs]


def test_more_named_assets_than_budget_records_infeasible() -> None:
    assets = {
        asset_id: _generic_asset(asset_id, AssetType.CHARACTER)
        for asset_id in ("a", "b", "c")
    }
    context = compose(
        AssetBank(assets),
        CompositionRequest(
            references=[AssetReference(asset_id=asset_id) for asset_id in assets],
            context_rep_budget=2,
        ),
        as_of_segment_id=10,
        policy=CompositionPolicy(adaptive_location_enabled=True),
    )
    assert sum(map(len, context.representation_ids.values())) == 2
    assert context.selection_trace["budget_infeasible"] is True
    assert len(context.selection_trace["dropped_primary_asset_ids"]) == 1


def test_adaptive_off_is_legacy_equivalent() -> None:
    asset = _generic_asset("char", AssetType.CHARACTER)
    bank = AssetBank({"char": asset})
    request = CompositionRequest(references=[AssetReference(asset_id="char")])
    baseline = compose(bank, request, as_of_segment_id=2)
    explicit_off = compose(
        bank,
        request,
        as_of_segment_id=2,
        raw_prompt="ignored",
        policy=CompositionPolicy(adaptive_location_enabled=False, global_rep_budget=16),
    )
    assert asdict(explicit_off) == asdict(baseline)


def test_materializer_emits_all_explicit_location_reps(tmp_path: Path) -> None:
    paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        path.write_bytes(b"not-decoded-by-materializer")
        paths.append(path)
    asset = _location()
    asset.representations = [
        _rep("a", "loc", 0, [1.0, 0.0], path=str(paths[0])),
        _rep("b", "loc", 1, [0.0, 1.0], path=str(paths[1])),
    ]
    context = compose(
        AssetBank({"loc": asset}),
        CompositionRequest(
            references=[AssetReference(asset_id="loc")],
            context_rep_budget=2,
        ),
        as_of_segment_id=10,
        raw_prompt="Return to the location.",
        policy=CompositionPolicy(
            adaptive_location_enabled=True,
            location_min_marginal_gain=0.01,
            location_coverage_stop=1.0,
        ),
    )
    refs = composed_reference_images(context, AssetBank({"loc": asset}))
    assert [Path(ref["image"]).name for ref in refs] == [
        Path(rep_id).name + ".png" for rep_id in context.representation_ids["loc"]
    ]
    assert [
        (ref["asset_id"], ref["representation_id"], ref["location_cluster_id"])
        for ref in refs
    ] == [
        ("loc", rep_id, f"legacy-{index:04d}")
        for index, rep_id in enumerate(context.representation_ids["loc"])
    ]


def test_materializer_respects_explicit_empty_selection(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(b"not-decoded-by-materializer")
    asset = _location()
    asset.representations = [_rep("a", "loc", 0, [1.0, 0.0], path=str(path))]
    context = compose(
        AssetBank({"loc": asset}),
        CompositionRequest(references=[AssetReference(asset_id="loc")]),
        as_of_segment_id=10,
    )
    context.representation_ids["loc"] = []
    assert composed_reference_images(context, AssetBank({"loc": asset})) == []


def test_public_read_api_has_no_target_media_parameter() -> None:
    assert "target_image" not in inspect.signature(compose).parameters
    assert "target_video" not in inspect.signature(compose).parameters
    assert set(LocationReadQuery.__dataclass_fields__) == {
        "raw_prompt",
        "planner_hints",
        "as_of_segment_id",
    }


def test_adaptive_profile_is_separate_from_frozen_paper_profile() -> None:
    assert PAPER_TRACKA_202607.read_max_reps_per_asset == 1
    assert PAPER_TRACKA_202607.adaptive_location_memory is False
    assert PAPER_TRACKA_202607.read_context_rep_budget is None
    assert LOCATION_ADAPTIVE_V1.adaptive_location_memory is True
    assert LOCATION_ADAPTIVE_V1.location_storage_cap == 12
    assert LOCATION_ADAPTIVE_V1.location_read_max_refs == 4
    assert LOCATION_ADAPTIVE_V1.read_context_rep_budget == 16


def test_realized_builder_wires_explicit_adaptive_profile(tmp_path: Path) -> None:
    profile = replace(
        LOCATION_ADAPTIVE_V1,
        name="location_adaptive_test",
        write_naming="perception",
        embedder_provider="hash",
        require_wedetect=False,
        require_mllm=False,
    )
    mem = build_realized_segment_pipeline(run_dir=tmp_path, profile=profile)
    assert mem.composition_policy.adaptive_location_enabled is True
    assert mem.composition_policy.global_rep_budget == 16
    assert mem.composition_policy.location_read_max_refs == 4
    assert mem.curator.location_coreset_policy.enabled is True
    assert mem.curator.location_coreset_policy.storage_cap == 12
    assert mem.curator.location_resolver_enabled is True
    assert mem.curator.location_resolver_shadow_enabled is True
    assert (
        mem.production_provenance["location_memory_v2"]["resolver_enabled"]
        is True
    )
