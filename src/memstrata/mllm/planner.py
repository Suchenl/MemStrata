"""MLLM-based Planner and Prompt Optimizer for MemStrata."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from memstrata.lib.paths import memstrata_root

# Default configurations
DEFAULT_INFERENCE_MODEL = "Qwen3.5-9B-Instruct"
DEFAULT_OPTIMIZER_MODEL = "Qwen3.5-27B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


class MllmPlanner:
    """MLLM-based planner for active asset selection, deduplication, and persistence decisions."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        prompts_path: Path | str | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("MEMSTRATA_CONTEXT_JUDGER_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or DEFAULT_INFERENCE_MODEL
        
        # Prompts persistence path (always inside MemStrata project)
        if prompts_path is not None:
            self.prompts_path = Path(prompts_path)
        else:
            self.prompts_path = memstrata_root() / "planner_prompts.json"

        # Default prompt templates
        self.select_assets_template = (
            "You are an expert director and asset manager. Given a user prompt describing the next video segment "
            "to generate, and a list of available assets in our memory bank, select the most relevant and necessary "
            "assets that must be conditioned/composed into the generation context to ensure visual continuity and "
            "instruction fidelity.\n\n"
            "User Prompt for Next Segment:\n{user_prompt}\n\n"
            "Available Assets in Memory Bank:\n{assets_list}\n\n"
            "Select only the assets that are truly needed. Do not select redundant or irrelevant assets.\n"
            "Return a JSON object with a list of selected asset IDs under the field 'selected_asset_ids'."
        )

        self.ingest_decision_template = (
            "You are an expert asset archivist. We have a new visual observation of an entity (character, location, "
            "or prop) from a generated video segment, and a list of existing candidate assets in our memory bank.\n"
            "Decide if this new observation represents the exact same entity as one of the existing candidates "
            "(to merge them), or if it is a completely new entity (to create a new asset). Also, generate a concise, "
            "highly descriptive caption for this entity.\n\n"
            "New Observation:\n- Kind: {obs_kind}\n- Image Path: {obs_image_path}\n\n"
            "Existing Candidate Assets:\n{candidates_list}\n\n"
            "Return a JSON object with the following fields:\n"
            "1. 'matched_asset_id': The ID of the matching candidate asset, or null if it is a new asset.\n"
            "2. 'caption': A concise, descriptive caption focusing on key visual features (appearance, colors, "
            "style, textures) of the entity in the new observation.\n"
            "3. 'reasoning': A brief explanation of your decision."
        )

        self.load_prompts()

    def load_prompts(self) -> None:
        """Load optimized prompts from disk if available."""
        if self.prompts_path.is_file():
            try:
                with self.prompts_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.select_assets_template = data.get("select_assets_template", self.select_assets_template)
                    self.ingest_decision_template = data.get("ingest_decision_template", self.ingest_decision_template)
            except Exception:
                pass

    def save_prompts(self) -> None:
        """Save current prompts to disk."""
        try:
            self.prompts_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "select_assets_template": self.select_assets_template,
                "ingest_decision_template": self.ingest_decision_template,
            }
            with self.prompts_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _call_api(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request to the OpenAI-compatible API."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "planner_response",
                    "schema": schema,
                    "strict": True,
                },
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            raise RuntimeError(f"Planner VLM API call failed: {e}") from e

    def select_assets(self, user_prompt: str, assets: list[dict[str, Any]]) -> list[str]:
        """Select relevant assets for composition using MLLM."""
        assets_list = ""
        for a in assets:
            assets_list += f"- ID: {a['id']}, Name: {a['name']}, Kind: {a['kind']}, Description: {a.get('description', '')}\n"

        prompt = self.select_assets_template.format(
            user_prompt=user_prompt,
            assets_list=assets_list,
        )

        messages = [{"role": "user", "content": prompt}]
        schema = {
            "type": "object",
            "properties": {
                "selected_asset_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["selected_asset_ids"],
            "additionalProperties": False,
        }

        try:
            result = self._call_api(messages, schema)
            return list(result.get("selected_asset_ids", []))
        except Exception:
            return []

    def make_intent_plan(self, user_prompt: str, assets: list[dict[str, Any]]) -> dict[str, Any]:
        """One bounded call producing an IntentPlanV1-shaped dict (see skills.intent_understanding.plan).

        Richer than ``select_assets``: besides *which* entities the beat needs, it returns the
        required appearance state, entities that must NOT appear, and the generation route.
        Returns ``{}`` on any failure so the read path can fall back to the model-free FAST path.
        """
        # Imported here (not at module scope) so the MLLM transport layer never has an
        # import-time dependency on the skills layer that consumes it.
        from memstrata.skills.intent_understanding.plan import PLAN_INSTRUCTION, PLAN_JSON_SCHEMA

        assets_list = "".join(
            f"- Name: {a.get('name', '')}, Kind: {a.get('kind', '')}, "
            f"Description: {a.get('description', '')}\n"
            for a in assets
        )
        messages = [
            {
                "role": "user",
                "content": PLAN_INSTRUCTION.format(
                    user_prompt=user_prompt, assets_list=assets_list
                ),
            }
        ]
        try:
            return dict(self._call_api(messages, PLAN_JSON_SCHEMA) or {})
        except Exception:
            return {}

    def make_ingest_decision(self, obs: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide if a new observation matches any candidate asset using MLLM."""
        candidates_list = ""
        for c in candidates:
            candidates_list += f"- ID: {c['id']}, Name: {c['name']}, Kind: {c['kind']}, Description: {c.get('description', '')}\n"

        prompt = self.ingest_decision_template.format(
            obs_kind=obs["kind"],
            obs_image_path=obs["image_path"],
            candidates_list=candidates_list if candidates_list else "None (Memory bank is empty for this kind)",
        )

        messages = [{"role": "user", "content": prompt}]
        schema = {
            "type": "object",
            "properties": {
                "matched_asset_id": {"type": ["string", "null"]},
                "caption": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["matched_asset_id", "caption", "reasoning"],
            "additionalProperties": False,
        }

        try:
            return self._call_api(messages, schema)
        except Exception:
            # Fallback on API failure
            return {
                "matched_asset_id": None,
                "caption": f"A newly observed {obs['kind']}",
                "reasoning": "Fallback due to API error",
            }


class PromptOptimizer:
    """Self-updating optimizer that reads evaluation history and updates planner prompt templates."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("MEMSTRATA_CONTEXT_JUDGER_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or DEFAULT_OPTIMIZER_MODEL

    def _call_api(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request to the OpenAI-compatible API."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Slightly higher temperature for creative prompt engineering
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "optimizer_response",
                    "schema": schema,
                    "strict": True,
                },
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            raise RuntimeError(f"Prompt Optimizer API call failed: {e}") from e

    def optimize_prompts(self, planner: MllmPlanner, evaluation_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze evaluation logs and optimize prompt templates in-place."""
        meta_prompt = (
            "You are a meta-prompt optimizer. Your task is to analyze the evaluation history of an asset "
            "composition system and optimize the prompt templates used by our MLLM planner to improve future performance.\n\n"
            "Current Select Assets Prompt Template:\n{current_select_assets_template}\n\n"
            "Current Ingest Decision Prompt Template:\n{current_ingest_decision_template}\n\n"
            "Evaluation History & Feedback (including metrics like Sufficiency, Parsimony, Compactness, Fidelity, "
            "Avoidance, and any errors/failures):\n{evaluation_history}\n\n"
            "Analyze the failures and successes. Identify if the planner is selecting too many/too few assets, "
            "or making incorrect deduplication decisions. Rewrite the prompt templates to address these issues. "
            "Ensure the output format remains JSON with the exact same fields.\n\n"
            "Return a JSON object with the newly optimized prompt templates and your analysis."
        ).format(
            current_select_assets_template=planner.select_assets_template,
            current_ingest_decision_template=planner.ingest_decision_template,
            evaluation_history=json.dumps(evaluation_history, ensure_ascii=False, indent=2),
        )

        messages = [{"role": "user", "content": meta_prompt}]
        schema = {
            "type": "object",
            "properties": {
                "optimized_select_assets_template": {"type": "string"},
                "optimized_ingest_decision_template": {"type": "string"},
                "analysis": {"type": "string"},
            },
            "required": [
                "optimized_select_assets_template",
                "optimized_ingest_decision_template",
                "analysis",
            ],
            "additionalProperties": False,
        }

        try:
            result = self._call_api(messages, schema)
            
            # Update planner templates in-place
            planner.select_assets_template = result["optimized_select_assets_template"]
            planner.ingest_decision_template = result["optimized_ingest_decision_template"]
            
            # Save the newly optimized prompts to disk
            planner.save_prompts()
            
            return {
                "success": True,
                "analysis": result["analysis"],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
