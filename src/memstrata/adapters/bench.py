"""Bench Track A adapter: frozen JSON contracts ↔ MemStrata Steps 1 and 4.

Does not import vmem_bench. PromptPacket / ObservationPacket / ComposedContextRecord
shapes are parsed and emitted as plain dicts (schemas_and_contracts.md §3).
"""

from __future__ import annotations

import time
from typing import Any

from memstrata.steps.compose import compose
from memstrata.steps.curate import MemoryUpdater
from memstrata.steps.intent import INTENT_MODE_FAST, IntentInterpreter, MllmIntentResolver
from memstrata.bank import AssetBank
from memstrata.mllm.planner import MllmPlanner

SCHEMA_VERSION = "2.0.0"
REQUIRED = "required"


class BenchReplayAdapter:
    """Track A harness surface: handle_prompt (Step 1) + handle_observation (Step 4)."""

    def __init__(
        self,
        asset_space: AssetBank,
        ingester: MemoryUpdater,
        composer: Any = None,
        planner: MllmPlanner | None = None,
        intent_mode: str = INTENT_MODE_FAST,
        disable_name_anchor: bool = False,
    ) -> None:
        _ = composer  # composition is model-free via compose(); kept for call-site compat
        self.asset_space = asset_space
        self.ingester = ingester
        resolver = MllmIntentResolver(planner) if planner is not None else None
        self.interpreter = IntentInterpreter(
            asset_space,
            resolver=resolver,
            mode=intent_mode,
            disable_name_anchor=disable_name_anchor,
        )

    def handle_observation(self, observation_packet: dict) -> None:
        self.ingester.ingest_packet(observation_packet)

    def handle_prompt(self, prompt_packet: dict) -> dict:
        start = time.perf_counter()
        segment_id = int(prompt_packet["segment_id"])
        prompt = str(prompt_packet["prompt"])

        request, model_calls = self.interpreter.interpret(prompt, segment_id=segment_id)
        context = compose(self.asset_space, request)

        selected_output: list[dict[str, Any]] = []
        per_asset: list[dict[str, str]] = []
        for asset_id in context.asset_ids:
            selected_output.append({
                "asset_id": asset_id,
                "representation_ids": list(context.representation_ids.get(asset_id, [])),
                "function": context.functions.get(asset_id, "identity_anchor"),
                "strength": REQUIRED,
            })
            per_asset.append({
                "asset_ref": asset_id,
                "requirement": context.requirements.get(asset_id, "continuity"),
            })

        return {
            "schema_version": SCHEMA_VERSION,
            "segment_id": segment_id,
            "selected": selected_output,
            "instruction": {
                "per_asset": per_asset,
                "exclusions": list(context.exclusions),
            },
            "memory_keys": list(context.asset_ids),
            "timing_ms": (time.perf_counter() - start) * 1000.0,
            "model_calls": model_calls,
            "intent_mode_requested": request.requested_mode,
            "intent_mode_used": request.used_mode,
            "intent_fallback_reason": request.fallback_reason,
            "enhanced_prompt": context.enhanced_prompt,
        }
