"""MLLM role registry — the single source of truth for *which* MLLM roles the
MemStrata production loop plays, and *with what* model / sampling / schema contract.

This module is a **specification only**: it declares roles as immutable data. It does
NOT open sockets, load models, or call any API — wiring each role to a transport
(e.g. the OpenAI-compatible endpoint used by ``memstrata.mllm.planner.MllmPlanner``)
happens later, per role, once we decide which roles to actually implement.

Design context (paper §3 four-step loop). The paper's official stage names are
Intent Interpretation -> Visual Generation -> Evidence Acquisition -> Stratified
Update; the code module names below (intent / compose / generate / decompose /
curate) are the internal engineering vocabulary for the same loop:

    Intent  ->  Compose  ->  Generate  ->  Decompose  ->  Curate           (+ Offline)

    * "Compose"   is the memory **read** path (assemble the Composed Context).
    * "Decompose" is the memory **write** analysis (break a generated segment back
      into candidate entity crops + attributes) feeding "Curate".

Iron rule (design_philosophy axiom: model-free default read): the *matching* read
path (name / alias match + identifier dereference) is deterministic and calls NO
model (see ``memstrata.steps.intent`` FAST mode). MLLM roles appear only on the
*reasoning* compose path and on the *write* (decompose / curate) path. Every role
below therefore records whether it sits on the per-segment hot path.

All decision / classification / mapping roles are pinned to ``temperature=0.0`` with
JSON-schema-constrained output for reproducibility and eval alignment; only
free-text authoring (captions, prompt writing) and the offline meta-optimizer raise
temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Default model bindings (mirror memstrata.mllm.planner defaults).
DEFAULT_MODEL = "Qwen3.5-9B-Instruct"          # per-segment hot-path roles
OPTIMIZER_MODEL = "Qwen3.5-27B-Instruct"        # offline / high-difficulty reasoning


class Step(str, Enum):
    """Position in the §3 production loop."""

    INTENT = "intent"
    COMPOSE = "compose"
    GENERATE = "generate"
    DECOMPOSE = "decompose"
    CURATE = "curate"
    OFFLINE = "offline"


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"  # multimodal: image crop(s)/frame + text


class Path(str, Enum):
    """Which conceptual path the role serves."""

    READ = "read"          # reasoning read (compose selection); default read is model-free
    COMPOSE = "compose"    # composition planning (layout / region / prompt)
    WRITE = "write"        # decompose + curate (memory write)
    OFFLINE = "offline"    # not on the generation loop


class Status(str, Enum):
    IMPLEMENTED = "implemented"      # code exists today
    PARTIAL = "partial"              # partially covered / needs extension
    PLANNED = "planned"              # spec only, not yet built


@dataclass(frozen=True, slots=True)
class Sampling:
    """Decoding contract for a role. ``thinking`` = enable Qwen reasoning mode."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    thinking: bool = False
    response_format: str = "json_schema"  # "json_schema" | "text"


@dataclass(frozen=True, slots=True)
class RoleSpec:
    id: str                       # stable short id, e.g. "R4"
    key: str                      # stable snake_case key, e.g. "crop_region_assigner"
    title: str
    step: Step
    path: Path
    modality: Modality
    purpose: str
    inputs: str
    output: str                   # short description of the structured output
    sampling: Sampling
    status: Status
    hot_path: bool                # runs per generated segment?
    model: str = DEFAULT_MODEL
    impl_ref: str = ""            # existing code location if any
    notes: str = ""
    schema_fields: tuple[str, ...] = field(default_factory=tuple)


# --- decode presets -----------------------------------------------------------
_DECIDE = Sampling(temperature=0.0, top_p=1.0, thinking=True)     # ambiguous decisions/planning
_CLASSIFY = Sampling(temperature=0.0, top_p=1.0, thinking=False)  # simple structured classification
_AUTHOR = Sampling(temperature=0.2, top_p=0.9, thinking=False, response_format="text")  # short prose
_META = Sampling(temperature=0.3, top_p=0.95, thinking=True, max_tokens=4096)  # offline meta-opt


ROLE_REGISTRY: dict[str, RoleSpec] = {
    "intent_parser": RoleSpec(
        id="R1", key="intent_parser", title="Intent Parser / Director",
        step=Step.INTENT, path=Path.READ, modality=Modality.TEXT,
        purpose="Parse the next-segment prompt g_n into structured intent q_n: which "
                "named entities are referenced, continue-vs-cut, scene-return signals, "
                "and required capabilities.",
        inputs="next-segment prompt g_n; bank asset summaries",
        output="structured intent (selected asset ids + mode + angle prefs)",
        sampling=_DECIDE, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/steps/intent.py::IntentInterpreter (SLOW mode via MllmIntentResolver)",
        notes="FAST mode is model-free (name/alias + description match); this MLLM role "
              "is only the SLOW reasoning read. Keep 0-call default.",
        schema_fields=("selected_asset_ids",),
    ),
    "intent_planner": RoleSpec(
        id="R1b", key="intent_planner", title="Intent Planner (IntentPlanV1)",
        step=Step.INTENT, path=Path.READ, modality=Modality.TEXT,
        purpose="Richer read of the beat than an id list: which stored entities must be "
                "visible AND in which appearance state, which must NOT appear (destroyed / "
                "look-alike), and the generation route (t2v | i2v_composed | i2v_continue).",
        inputs="next-segment prompt g_n; bank asset summaries (name/kind/description)",
        output="IntentPlanV1 (references[name,state_required,must_include], forbidden, route)",
        sampling=_DECIDE, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/mllm/planner.py::MllmPlanner.make_intent_plan "
                 "(consumed by IntentInterpreter mode='plan')",
        notes="Opt-in via MEMSTRATA_INTENT_MODE=plan; exactly one call per segment, and any "
              "unusable plan degrades to the model-free FAST path. Names only — identity "
              "resolution stays local, so the plan cannot widen A_n.",
        schema_fields=("route", "references", "forbidden", "reason"),
    ),
    "asset_selector": RoleSpec(
        id="R2", key="asset_selector", title="Asset Retriever / Selector",
        step=Step.COMPOSE, path=Path.READ, modality=Modality.TEXT,
        purpose="From intent + bank listing, select the minimal-sufficient asset set to "
                "condition the generation (Sufficiency vs Parsimony).",
        inputs="user prompt; available asset list (id/name/kind/description)",
        output="selected_asset_ids",
        sampling=_DECIDE, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/mllm/planner.py::MllmPlanner.select_assets",
        notes="Reasoning read path; respects context_rep_budget (efficient composition).",
        schema_fields=("selected_asset_ids",),
    ),
    "generation_router": RoleSpec(
        id="R2b", key="generation_router", title="Generation-Path Router",
        step=Step.GENERATE, path=Path.COMPOSE, modality=Modality.TEXT,
        purpose="Decide HOW to seed the next segment given prompt g_n + intent + what is on "
                "the previous segment's last frame. Picks one of four generation modes so we "
                "spend FLUX only when we must and keep temporal continuity when we can:\n"
                "  continue_ar        Helios continues from the prior video window "
                "(video=[style_anchor + ~73 recent frames], history_sizes=[16,2,1]); "
                "same scene, subjects already on-screen, camera continuous — cheapest.\n"
                "  reanchor_lastframe prev last frame becomes the i2v anchor + new prompt; "
                "same place, new beat, no off-screen entity to introduce (weaker continuity).\n"
                "  recompose_partial  paste a returning entity crop onto the prev last frame "
                "(Crop2Image) then i2v; scene mostly unchanged but must inject a returning asset.\n"
                "  recompose_keyframe full R3->R4->FLUX fresh keyframe from memory crops; "
                "scene cut / new location / time jump / returning asset absent from prev frame / "
                "repositioning / first segment — the memory-injection path.",
        inputs="intent (continue_vs_cut, scene_return, referenced entities); prev-segment "
               "summary + which referenced entities are visible on its last frame; "
               "whether a prior segment exists",
        output="{mode, reason, recompose_asset_ids[], continuity{scene_same, subjects_onscreen}}",
        # Hot-path routing decision. Thinking off for speed (like R3/R4); hard feasibility
        # constraints (no prior segment => must recompose; scene cut => cannot continue_ar) are
        # enforced by the deterministic rule layer around this call, not left to the model.
        sampling=_CLASSIFY, status=Status.PLANNED, hot_path=True,
        impl_ref="memstrata/skills/generation_routing/router.py::GenerationRouter (rules + MLLM)",
        notes="continue_ar MUST feed Helios via video=[anchor + recent frames] (NOT image=<last "
              "frame>): image and video are mutually exclusive in the Helios pipeline, and a "
              "single last frame cannot preserve motion continuity with the prior segment. "
              "history_sizes=[16,2,1] x temporal_scale 4 => a 73-pixel-frame rolling window + "
              "always-on first-frame (style) anchor. Modes map onto MediaTaskGenerator/"
              "HeliosBackend seeding branches.",
        schema_fields=("mode", "reason", "recompose_asset_ids", "continuity"),
    ),
    "layout_planner": RoleSpec(
        id="R3", key="layout_planner", title="Layout / Spatial Planner",
        step=Step.COMPOSE, path=Path.COMPOSE, modality=Modality.TEXT,
        purpose="Plan a normalized bbox layout for the scene (who/what goes where) so "
                "FLUX gets a color-block spatial anchor and beats the composition lottery.",
        inputs="enhanced prompt + selected assets (labels/kinds)",
        output="elements: [{label, box_2d[ymin,xmin,ymax,xmax], shape}]",
        # Structured layout output — no chain-of-thought needed; thinking mode made each
        # call ~1-2min on Qwen3.5-9B. _CLASSIFY (thinking off, temp 0, json_schema) is ~10x faster.
        sampling=_CLASSIFY, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/skills/layout_anchor_processing/planner.py::LayoutPlanner (R3 via MllmRoleRunner)",
        notes="Coordinates normalized [0,1000]; shape in {human,rectangle,ellipse,line}. "
              "Renders color_block for FLUX (line_art for Qwen-Image-Edit). Vendored from "
              "montage.skills.layout_anchor_processing; live-verified against Qwen3.5-9B.",
        schema_fields=("elements",),
    ),
    "crop_region_assigner": RoleSpec(
        id="R4", key="crop_region_assigner", title="Crop -> Region Assigner",
        step=Step.COMPOSE, path=Path.COMPOSE, modality=Modality.VISION,
        purpose="Decide which retrieved real crop pixels go into which layout region, and "
                "which angle/state variant of each asset to use. This placement decision "
                "MUST be made by the MLLM (in-pipeline), never hard-coded by an operator.",
        inputs="layout elements (R3) + candidate crops per selected asset (+ attributes)",
        output="mapping asset_id -> {box_id, representation_id/variant}",
        # Structured crop->region mapping — no chain-of-thought; thinking off for ~10x speedup.
        sampling=_CLASSIFY, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/skills/layout_anchor_processing/crop2image.py::assign_crops_to_regions + composite_crops",
        notes="Feeds the Crop2Image collage: crops pasted into assigned regions over the "
              "color-block anchor, then FLUX.2 Klein I2I fuses into a coherent keyframe. "
              "Placement decided by the (multimodal) MLLM; deterministic fallback if server "
              "down. Live-verified end-to-end on the single Qwen3.5-9B (r4_fused_keyframe.png).",
        schema_fields=("assignments",),
    ),
    "prompt_composer": RoleSpec(
        id="R5", key="prompt_composer", title="Generation-Prompt Composer",
        step=Step.COMPOSE, path=Path.COMPOSE, modality=Modality.TEXT,
        purpose="Compose the FLUX/video positive prompt from intent + chosen crops, kept "
                "consistent with the crops' actual attributes (no contradicting colors / "
                "clothing that would override the injected identity).",
        inputs="intent + selected assets + their attribute packs",
        output="generation prompt string",
        sampling=_AUTHOR, status=Status.PARTIAL, hot_path=True,
        impl_ref="memstrata/lib/prompt_standardizer.py (deterministic today)",
        notes="Today deterministic enrichment (intent.py appends asset cues). MLLM rewrite "
              "is optional/ponytail; must stay grounded in bank facts.",
    ),
    "view_requester": RoleSpec(
        id="R6", key="view_requester", title="View / Angle Requester",
        step=Step.GENERATE, path=Path.COMPOSE, modality=Modality.TEXT,
        purpose="When the plan needs a new viewpoint of an asset, decide which angle to "
                "render for a reference_image task.",
        inputs="intent + asset current representations",
        output="requested spatial_angle",
        sampling=_CLASSIFY, status=Status.PLANNED, hot_path=False,
        schema_fields=("spatial_angle",),
    ),
    "entity_detector": RoleSpec(
        id="R7", key="entity_detector", title="Entity Detector / Grounder",
        step=Step.DECOMPOSE, path=Path.WRITE, modality=Modality.VISION,
        purpose="Targeted grounding: locate ONE named entity in a generated segment frame "
                "and return a TIGHT bbox (+positive point) so decompose can crop a clean, "
                "entity-isolated observation for candidate ingestion.",
        inputs="one generated segment frame + entity name/kind/description",
        output="{usable, bbox_norm[ymin,xmin,ymax,xmax] 0-1000, point_norm[y,x]}",
        # Structured localization — no chain-of-thought; thinking off for speed.
        sampling=_CLASSIFY, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/skills/entity_grounding/grounding_cropper.py::VlmGroundingCropper "
                 "(mirrors memstrata_bench s5 QwenImageGrounder; runs on the unified Qwen3.5-9B)",
        notes="Targeted (not open-vocab), per design_philosophy decompose. Track A gold "
              "ObservationPackets still bypass this; used for the real closed loop. "
              "SAM3 mask refine + identity-consistency audit are follow-ons (still PARTIAL upstream).",
        schema_fields=("usable", "bbox_norm", "point_norm"),
    ),
    "entity_decomposer": RoleSpec(
        id="R7b", key="entity_decomposer", title="Entity Decomposer (frame → typed named entities)",
        step=Step.DECOMPOSE, path=Path.WRITE, modality=Modality.VISION,
        purpose="Decompose a realized-segment frame into the salient typed entities present "
                "(character/prop/location), each with a concise label and a short visual "
                "description, so the write path can ground + curate them. Type-constrained; "
                "naming/labeling comes from PERCEIVING the frame, never from parsing gold or "
                "the prompt text (the prompt is optional disambiguation context only).",
        inputs="one realized-segment frame (+ optional segment prompt as context)",
        output="entities: [{kind, label, description}]",
        # Structured perception listing — no chain-of-thought; thinking off for speed.
        sampling=_CLASSIFY, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/skills/decomposition/vlm_decomposer.py::VlmEntityDecomposer",
        notes="Companion to R7 entity_detector (which grounds ONE already-named entity to a "
              "tight bbox). R7b proposes WHICH entities exist and their labels; cross-segment "
              "identity is still decided downstream by curate (R9). Runs on the unified "
              "Qwen3.5-9B multimodal endpoint.",
        schema_fields=("entities",),
    ),
    "crop_attribute_classifier": RoleSpec(
        id="R8", key="crop_attribute_classifier", title="Crop Attribute / Angle-State Classifier",
        step=Step.DECOMPOSE, path=Path.WRITE, modality=Modality.VISION,
        purpose="Classify an entity crop into spatial_angle / state_angle (+ shot size / "
                "lighting / occlusion pack) to place it in the stratified bank.",
        inputs="entity crop image + kind/name",
        output="attribute pack {spatial_angle, state_angle, confidence, reasoning}",
        sampling=_CLASSIFY, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/mllm/angle_classifier.py + memstrata/mllm/crop_attributes.py",
        schema_fields=("spatial_angle", "state_angle", "confidence", "reasoning"),
    ),
    "ingest_dedup_judge": RoleSpec(
        id="R9", key="ingest_dedup_judge", title="Ingest / Dedup Judge",
        step=Step.CURATE, path=Path.WRITE, modality=Modality.VISION,
        purpose="Decide whether a new observation is an existing asset (merge) or a new one, "
                "and write a concise descriptive caption.",
        inputs="new observation crop + existing candidate assets",
        output="{matched_asset_id | null, caption, reasoning}",
        sampling=_DECIDE, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/mllm/planner.py::MllmPlanner.make_ingest_decision",
        schema_fields=("matched_asset_id", "caption", "reasoning"),
    ),
    "identity_adjudicator": RoleSpec(
        id="R9b", key="identity_adjudicator", title="Identity Adjudicator (VLM-first, χ gray zone)",
        step=Step.CURATE, path=Path.WRITE, modality=Modality.VISION,
        purpose="Same/different verdict for two crops of the same type, consulted ONLY in the "
                "encoder-score χ gray zone around β_τ. The encoder short-circuits confident "
                "matches and confident new identities; strong-blur crops are deferred; the "
                "judge (temperature 0, per-model θ) never abstains, keeping full coverage "
                "without the encoder-threshold brittleness (paper §4.5 / Sec. exp-idgate).",
        inputs="incoming crop + up to k diverse reference crops of the best type-compatible "
               "candidate (all in one call)",
        output="{same: bool, confidence: float, reasoning}",
        # Structured same/different verdict, thinking off + temperature 0 for reproducibility.
        sampling=_CLASSIFY, status=Status.IMPLEMENTED, hot_path=True,
        impl_ref="memstrata/mllm/identity_judge.py::VlmIdentityJudge "
                 "(gated in memory_update/curator.py::MemoryUpdater._reconcile_identity)",
        notes="Specializes R9's merge/new decision with a calibrated gray-zone gate. Default "
              "NullIdentityJudge abstains → decision reduces to the deterministic χ≥β_τ rule.",
        schema_fields=("same", "confidence", "reasoning"),
    ),
    "admission_gate": RoleSpec(
        id="R10", key="admission_gate", title="Admission Gate",
        step=Step.CURATE, path=Path.WRITE, modality=Modality.VISION,
        purpose="Decide whether a crop is good/novel enough to admit into the bank "
                "(quality + novelty gate).",
        inputs="candidate crop + quality/novelty signals",
        output="{admit: bool, reason}",
        sampling=_CLASSIFY, status=Status.PARTIAL, hot_path=True,
        impl_ref="memstrata/lib/crop_quality.py + who-admission gates (tests)",
        schema_fields=("admit", "reason"),
    ),
    "state_manager": RoleSpec(
        id="R11", key="state_manager", title="State-Update / Deprecation Manager",
        step=Step.CURATE, path=Path.WRITE, modality=Modality.TEXT,
        purpose="Update entity state (changed/damaged), deprecate stale evidence, and "
                "maintain avoidance so deprecated representations are not reused.",
        inputs="asset current state + new observation attributes",
        output="state update ops (state_angle, deprecate ids, avoidance)",
        sampling=_DECIDE, status=Status.PLANNED, hot_path=True,
        schema_fields=("state_angle", "deprecate", "avoidance"),
    ),
    "prompt_optimizer": RoleSpec(
        id="R12", key="prompt_optimizer", title="Prompt Optimizer (meta, offline)",
        step=Step.OFFLINE, path=Path.OFFLINE, modality=Modality.TEXT,
        purpose="Analyze evaluation history and rewrite planner prompt templates to improve "
                "future Sufficiency/Parsimony/Fidelity.",
        inputs="planner templates + evaluation history",
        output="optimized templates + analysis",
        sampling=_META, status=Status.IMPLEMENTED, hot_path=False,
        model=OPTIMIZER_MODEL,
        impl_ref="memstrata/mllm/planner.py::PromptOptimizer",
        schema_fields=("optimized_select_assets_template",
                       "optimized_ingest_decision_template", "analysis"),
    ),
    "quality_judge": RoleSpec(
        id="R13", key="quality_judge", title="In-loop Quality / Consistency Judge",
        step=Step.OFFLINE, path=Path.OFFLINE, modality=Modality.VISION,
        purpose="Optional qualitative judgement of a generated segment vs its references for "
                "hard-case analysis.",
        inputs="generated segment + reference crops",
        output="{consistent: bool, issues, reasoning}",
        sampling=_DECIDE, status=Status.PLANNED, hot_path=False,
        notes="Track A uses a deterministic scorer as the authority; this role is only for "
              "qualitative hard cases, not headline metrics.",
        schema_fields=("consistent", "issues", "reasoning"),
    ),
}


# --- helpers ------------------------------------------------------------------
def get_role(key: str) -> RoleSpec:
    return ROLE_REGISTRY[key]


def roles_for_step(step: Step) -> list[RoleSpec]:
    return [r for r in ROLE_REGISTRY.values() if r.step == step]


def hot_path_roles() -> list[RoleSpec]:
    return [r for r in ROLE_REGISTRY.values() if r.hot_path]


def roles_by_status(status: Status) -> list[RoleSpec]:
    return [r for r in ROLE_REGISTRY.values() if r.status == status]


def validate_registry() -> None:
    """Fail fast on duplicate ids/keys (spec integrity)."""
    ids = [r.id for r in ROLE_REGISTRY.values()]
    keys = list(ROLE_REGISTRY.keys())
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate role ids: {ids}")
    for k, r in ROLE_REGISTRY.items():
        if k != r.key:
            raise ValueError(f"registry key {k!r} != RoleSpec.key {r.key!r}")


validate_registry()


__all__ = [
    "DEFAULT_MODEL", "OPTIMIZER_MODEL",
    "Step", "Modality", "Path", "Status", "Sampling", "RoleSpec",
    "ROLE_REGISTRY", "get_role", "roles_for_step", "hot_path_roles",
    "roles_by_status", "validate_registry",
]
