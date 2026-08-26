"""Generation-path router (role R2b) — rules + MLLM decide HOW to seed the next segment.

Four generation modes (see ``memstrata.mllm.roles``'s ``generation_router``)::

  continue_ar         Helios continues from the prior video window
                      (video=[style_anchor + ~73 recent frames], history_sizes=[16,2,1]).
                      Cheapest; best temporal continuity. Feasible only when a prior
                      segment exists, the scene is unchanged (no cut) and every referenced
                      entity is already on the previous segment's last frame.
  reanchor_lastframe  prev segment's last frame becomes the i2v anchor (image=) + new prompt.
                      Same place, new action beat, no off-screen entity to introduce. Weaker
                      continuity than continue_ar (a single frame cannot carry motion state).
  recompose_partial   paste a returning entity crop onto the prev last frame (Crop2Image),
                      then i2v. Scene mostly unchanged but must inject a returning asset.
  recompose_keyframe  full R3->R4->FLUX fresh keyframe from memory crops, then Helios i2v.
                      Scene cut / new location / time jump / returning asset absent from the
                      prev frame / repositioning / first segment. The memory-injection path.

Design: a deterministic RULE layer first restricts the *feasible* modes (hard physical
constraints — you cannot continue a video that does not exist, cannot AR-continue across a
scene cut). The MLLM (R2b, thinking off) then picks among the feasible set with a reason.
The MLLM's answer is re-validated against feasibility and safely falls back. This keeps hard
constraints deterministic while letting the model make the genuinely ambiguous judgement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GenMode(str, Enum):
    CONTINUE_AR = "continue_ar"
    REANCHOR_LASTFRAME = "reanchor_lastframe"
    RECOMPOSE_PARTIAL = "recompose_partial"
    RECOMPOSE_KEYFRAME = "recompose_keyframe"


ALL_MODES = tuple(m.value for m in GenMode)

ROUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": list(ALL_MODES),
                 "description": "The chosen generation mode for this segment."},
        "reason": {"type": "string", "description": "One concise sentence justifying the choice."},
        "recompose_asset_ids": {
            "type": "array", "items": {"type": "string"},
            "description": "For recompose_* modes: which named assets to (re)inject from memory.",
        },
        "continuity": {
            "type": "object",
            "properties": {
                "scene_same": {"type": "boolean"},
                "subjects_onscreen": {"type": "boolean"},
            },
            "required": ["scene_same", "subjects_onscreen"],
        },
    },
    "required": ["mode", "reason", "recompose_asset_ids", "continuity"],
}

_PROMPT = """You are the generation-path router for a causal long-video system. Decide HOW to \
seed the NEXT video segment, choosing exactly one mode from the FEASIBLE set below.

Next-segment prompt:
"{prompt}"

Situation:
- Previous segment exists: {has_prev}
- Previous segment summary: {prev_summary}
- Entities referenced by the new prompt: {referenced}
- Referenced entities already visible on the previous segment's LAST frame: {onscreen}
- Referenced entities that must RETURN (referenced but NOT on the last frame): {returning}
- Intent signals: continue_vs_cut={continue_vs_cut}, scene_return={scene_return}

FEASIBLE modes (choose one; other modes are ruled out by hard constraints):
{feasible_desc}

Guidance:
- Prefer continue_ar when it is feasible: it preserves motion/temporal continuity most cheaply.
- Use recompose_keyframe when the scene changes or a referenced entity is not on the last frame.
- Use recompose_partial to inject ONE returning entity into an otherwise-continuing scene.
- Use reanchor_lastframe for a new action beat in the same place with no entity to introduce.
Set recompose_asset_ids to the assets to inject for recompose_* modes (else empty)."""

_MODE_DESC = {
    GenMode.CONTINUE_AR: "continue_ar: Helios continues from the prior video window (best continuity, cheapest).",
    GenMode.REANCHOR_LASTFRAME: "reanchor_lastframe: prev last frame as anchor + new prompt (weaker continuity).",
    GenMode.RECOMPOSE_PARTIAL: "recompose_partial: paste a returning crop onto prev last frame, then i2v.",
    GenMode.RECOMPOSE_KEYFRAME: "recompose_keyframe: fresh FLUX keyframe from memory crops (scene cut / new entity).",
}


@dataclass
class RouteDecision:
    mode: GenMode
    reason: str
    recompose_asset_ids: list[str] = field(default_factory=list)
    scene_same: bool = False
    subjects_onscreen: bool = False
    feasible: tuple[str, ...] = ()
    source: str = "rules"  # "rules" (forced) | "mllm" | "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value, "reason": self.reason,
            "recompose_asset_ids": list(self.recompose_asset_ids),
            "continuity": {"scene_same": self.scene_same, "subjects_onscreen": self.subjects_onscreen},
            "feasible": list(self.feasible), "source": self.source,
        }


def _feasible_modes(
    *, has_prev: bool, scene_cut: bool, returning: list[str], onscreen: list[str], referenced: list[str],
) -> list[GenMode]:
    """Hard physical constraints that restrict which modes are even possible."""
    if not has_prev:
        # First segment / nothing to continue from: only a fresh keyframe makes sense.
        return [GenMode.RECOMPOSE_KEYFRAME]
    if scene_cut:
        # A cut breaks the rolling window; you must build a new anchor.
        return [GenMode.RECOMPOSE_KEYFRAME]
    modes: list[GenMode] = []
    # Same scene, no cut:
    if not returning and referenced and all(e in onscreen for e in referenced):
        modes.append(GenMode.CONTINUE_AR)          # everyone on-screen => can AR-continue
    if returning:
        modes.append(GenMode.RECOMPOSE_PARTIAL)     # inject the returning entity
    modes.append(GenMode.REANCHOR_LASTFRAME)        # always an option in-scene
    modes.append(GenMode.RECOMPOSE_KEYFRAME)        # always a safe fallback
    # de-dup preserving order
    seen: set[GenMode] = set()
    return [m for m in modes if not (m in seen or seen.add(m))]


class GenerationRouter:
    """Rules + R2b MLLM decision for the per-segment generation mode."""

    ROLE_KEY = "generation_router"

    def __init__(self, runner: Any | None = None, *, use_mllm: bool = True) -> None:
        self._runner = runner
        self.use_mllm = use_mllm

    def _runner_lazy(self) -> Any | None:
        if self._runner is None and self.use_mllm:
            try:
                from memstrata.mllm.runner import MllmRoleRunner
                self._runner = MllmRoleRunner()
            except Exception as exc:  # noqa: BLE001
                logger.warning("router: MLLM runner unavailable (%r); using rule default", exc)
                self.use_mllm = False
        return self._runner

    def route(
        self,
        *,
        prompt: str,
        segment_id: int,
        referenced_entities: list[str],
        onscreen_entities: list[str],
        has_prev_segment: bool,
        continue_vs_cut: str = "continue",
        scene_return: bool = False,
        prev_summary: str = "",
    ) -> RouteDecision:
        scene_cut = (continue_vs_cut or "continue").lower() == "cut"
        returning = [e for e in referenced_entities if e not in set(onscreen_entities)]
        feasible = _feasible_modes(
            has_prev=has_prev_segment, scene_cut=scene_cut,
            returning=returning, onscreen=onscreen_entities, referenced=referenced_entities,
        )
        feasible_vals = tuple(m.value for m in feasible)

        # Forced (single feasible mode) — no need to spend an MLLM call.
        if len(feasible) == 1:
            m = feasible[0]
            why = ("first segment / no prior video to continue" if not has_prev_segment
                   else "scene cut — rolling window broken, rebuild anchor" if scene_cut
                   else "only feasible mode")
            return RouteDecision(mode=m, reason=why, recompose_asset_ids=list(referenced_entities)
                                 if m != GenMode.CONTINUE_AR else [],
                                 scene_same=not scene_cut, subjects_onscreen=not returning,
                                 feasible=feasible_vals, source="rules")

        runner = self._runner_lazy()
        if runner is not None:
            try:
                instruction = _PROMPT.format(
                    prompt=prompt, has_prev=has_prev_segment, prev_summary=prev_summary or "(none)",
                    referenced=referenced_entities or "(none)", onscreen=onscreen_entities or "(none)",
                    returning=returning or "(none)", continue_vs_cut=continue_vs_cut,
                    scene_return=scene_return,
                    feasible_desc="\n".join(f"- {_MODE_DESC[m]}" for m in feasible),
                )
                out = runner.run(self.ROLE_KEY, instruction=instruction, schema=ROUTER_SCHEMA)
                mode = GenMode(str(out.get("mode", "")))
                if mode in feasible:  # re-validate against feasibility
                    cont = out.get("continuity") or {}
                    return RouteDecision(
                        mode=mode, reason=str(out.get("reason", ""))[:200],
                        recompose_asset_ids=[str(a) for a in (out.get("recompose_asset_ids") or [])],
                        scene_same=bool(cont.get("scene_same", not scene_cut)),
                        subjects_onscreen=bool(cont.get("subjects_onscreen", not returning)),
                        feasible=feasible_vals, source="mllm",
                    )
                logger.warning("router: MLLM chose infeasible mode %s; falling back", out.get("mode"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("router: MLLM call failed (%r); falling back to rule default", exc)

        # Rule fallback: prefer the cheapest feasible continuity mode.
        preferred = next((m for m in (GenMode.CONTINUE_AR, GenMode.RECOMPOSE_PARTIAL,
                                      GenMode.REANCHOR_LASTFRAME, GenMode.RECOMPOSE_KEYFRAME)
                          if m in feasible), feasible[0])
        return RouteDecision(
            mode=preferred, reason="rule default (cheapest feasible continuity mode)",
            recompose_asset_ids=returning if preferred in (GenMode.RECOMPOSE_PARTIAL,
                                                           GenMode.RECOMPOSE_KEYFRAME) else [],
            scene_same=not scene_cut, subjects_onscreen=not returning,
            feasible=feasible_vals, source="fallback",
        )


__all__ = ["GenMode", "GenerationRouter", "RouteDecision", "ROUTER_SCHEMA", "ALL_MODES"]
