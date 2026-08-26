"""Unified Prompt Standardization and Refinement Engine.

This module centralizes all prompt standardization, translation, and refinement
logic for different media generation backends (FLUX, LTX-2.3, Wan/VACE). It converts
highly structured screenplay screenplay prompts into optimal model-specific inputs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_screenplay_prompt(prompt: str) -> dict[str, str]:
    """Parse a structured screenplay format prompt into a dictionary of fields.

    Expected format:
    Narrative goal: ...
    Visible actions: ...
    Dialogue or narration: ...
    Active characters or subjects: ...
    Scene state: ...
    Continuity requirements: ...
    """
    lines = prompt.split("\n")
    data = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip()
    return data


def standardize_prompt(prompt: str, target_model: str, quality_preset: str | None = None) -> str:
    """Standardize and refine any raw input prompt for a specific target generator model.

    Supported target models:
        - "flux_klein": Optimized for FLUX.2 image generation (JSON or NLP structure, no negative prompt).
        - "ltx23": Optimized for LTX-2.3 video generation (dense cinematic caption).
        - "native_vace": Optimized for Wan/VACE video generation (highly descriptive, visual-focused).
        - "longcat": Optimized for LongCat-Video (dense cinematic caption with identity continuity).
    """
    target = target_model.lower()

    # 1. Check if input is a structured screenplay format
    is_screenplay = any(key in prompt for key in ["Narrative goal", "Visible actions", "Scene state"])

    if is_screenplay:
        data = parse_screenplay_prompt(prompt)

        def clean_field(s: str) -> str:
            return s.strip().rstrip(".").rstrip(",")

        narrative_goal = clean_field(data.get("Narrative goal", ""))
        actions = clean_field(data.get("Visible actions", ""))
        dialogue = clean_field(data.get("Dialogue or narration", ""))
        characters = clean_field(data.get("Active characters or subjects", ""))
        scene_state = data.get("Scene state", "")
        continuity = clean_field(data.get("Continuity requirements", ""))
        start_state = clean_field(data.get("Start state", ""))
        end_state = clean_field(data.get("End state", ""))

        # Clean narrative goal prefixes
        goal_cleaned = narrative_goal
        for prefix in ["A video of ", "A video showing ", "A shot of ", "Video of ", "A photo of ", "Photo of "]:
            if goal_cleaned.lower().startswith(prefix.lower()):
                goal_cleaned = goal_cleaned[len(prefix):]

        # Parse Scene State for details
        loc, lighting, style_desc = "", "", ""
        if scene_state:
            kv_pairs = [kv.split("=") for kv in scene_state.split(";") if "=" in kv]
            kv_dict = {k.strip(): v.strip() for k, v in kv_pairs}
            loc = clean_field(kv_dict.get("location") or kv_dict.get("place", ""))
            lighting = clean_field(kv_dict.get("lighting") or kv_dict.get("light", ""))
            style_desc = clean_field(kv_dict.get("style") or kv_dict.get("genre", ""))

        if "flux" in target:
            # FLUX.2 follows structured JSON prompts best; keep each control axis explicit.
            structured_prompt: dict[str, Any] = {
                "scene": goal_cleaned,
                "subjects": [],
                "action": actions,
                "context": {},
                "style": style_desc or "cinematic film still",
            }
            if characters:
                structured_prompt["subjects"].append({"description": characters})
            if loc:
                structured_prompt["context"]["location"] = loc
            if lighting:
                structured_prompt["context"]["lighting"] = lighting
            refined_prompt = json.dumps(structured_prompt, ensure_ascii=False)
            return preprocess_de_ai_prompt(refined_prompt, quality_preset)

        elif "ltx" in target or "longcat" in target:
            # LTX-2.3 / LongCat-Video: Dense, continuous physical description
            parts = []
            setting_parts = []
            if loc:
                setting_parts.append(f"in a {loc}")
            if lighting:
                setting_parts.append(f"under {lighting} lighting")
            if style_desc:
                setting_parts.append(f"in a {style_desc} style")

            setting_desc = f"The scene is set {' '.join(setting_parts)}. " if setting_parts else ""
            char_desc = f"Featuring {characters}. " if characters else ""

            if goal_cleaned:
                parts.append(f"A high-quality cinematic video of {goal_cleaned}. {char_desc}{setting_desc}")
            else:
                parts.append(f"A high-quality cinematic video. {char_desc}{setting_desc}")

            if actions:
                parts.append(f"The actions unfold: {actions}.")
            if "longcat" in target and (start_state or end_state):
                trajectory = []
                if start_state:
                    trajectory.append(f"begin from {start_state}")
                if end_state:
                    trajectory.append(f"progress to {end_state}")
                parts.append(
                    "Segment trajectory: "
                    + " and ".join(trajectory)
                    + ". If this segment is video continuation, continue forward from the conditioning tail frames; do not reset to a previous wide establishing shot, do not repeat the same opening composition, and do not restart the action from an earlier state."
                )
            if continuity:
                parts.append(f"Maintaining strict continuity: {continuity}.")
            if dialogue:
                parts.append(f"The audio or narration features: {dialogue}.")

            return " ".join(parts).strip()

        elif "vace" in target or "wan" in target:
            # Native VACE/Wan: Highly imaginative, descriptive, but no continuity lists (since VACE's text encoder
            # is highly visual but doesn't interpret structured metadata list syntax well)
            parts = []
            setting_parts = []
            if loc:
                setting_parts.append(f"in a {loc}")
            if lighting:
                setting_parts.append(f"with {lighting} lighting")
            if style_desc:
                setting_parts.append(f"styled as {style_desc}")
            setting_str = " " + ", ".join(setting_parts) if setting_parts else ""

            char_str = f" {characters}" if characters else " a subject"
            action_str = f" performing: {actions}" if actions else ""

            if goal_cleaned:
                parts.append(f"A cohesive cinematic video showing {goal_cleaned} as{char_str}{action_str}{setting_str}.")
            else:
                parts.append(f"A beautiful cinematic video showing{char_str}{action_str}{setting_str}.")

            # Keep VACE highly visual, discard pure textual dialogue metadata unless it contains visual cues
            return " ".join(parts).strip()

        else:
            # General fallback
            return prompt

    # 2. For non-screenplay prompts (raw descriptions/NLP):
    if "flux" in target:
        try:
            json.loads(prompt)
            return preprocess_de_ai_prompt(prompt, quality_preset)
        except (json.JSONDecodeError, TypeError):
            structured_prompt = {
                "scene": prompt,
                "subjects": [],
                "style": "cinematic film still",
            }
            return preprocess_de_ai_prompt(json.dumps(structured_prompt, ensure_ascii=False), quality_preset)
    elif "ltx" in target or "longcat" in target:
        # Ensure LTX-2.3 / LongCat prompts are wrapped cleanly
        if not any(p in prompt.lower() for p in ["video of", "video showing", "cinematic video"]):
            return f"A high-quality cinematic video of {prompt}"
        return prompt
    elif "vace" in target or "wan" in target:
        # Ensure VACE has strong photographic style descriptors
        if not any(p in prompt.lower() for p in ["video", "motion"]):
            return f"A beautiful cinematic video of {prompt}"
        return prompt

    return prompt


def preprocess_de_ai_prompt(prompt: str, quality_preset: str | None) -> str:
    """Clean up CG-like buzzwords and inject photographic film anchors to shatter AI plastic bias."""
    dirty_words = ["photorealistic", "hyperrealistic", "highly detailed", "8k", "flawless skin", "studio lighting"]
    anchors = "raw photo, un-retouched, shot on 35mm film, Fujifilm Superia, natural skin texture, visible pores, slight film grain"

    def clean_text(text: str) -> str:
        for word in dirty_words:
            text = text.replace(f", {word}", "").replace(word, "")
            text = text.replace(word, "")
        return text

    # Detect and process structured JSON prompts
    try:
        data = json.loads(prompt)

        def process_node(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: process_node(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [process_node(item) for item in node]
            elif isinstance(node, str):
                return clean_text(node)
            return node

        processed_data = process_node(data)

        if quality_preset == "raw_film":
            if isinstance(processed_data, dict):
                if "style" in processed_data and isinstance(processed_data["style"], str):
                    if processed_data["style"].strip():
                        processed_data["style"] += f", {anchors}"
                    else:
                        processed_data["style"] = anchors
                elif "scene" in processed_data and isinstance(processed_data["scene"], str):
                    processed_data["scene"] += f", {anchors}"
                else:
                    processed_data["style"] = anchors

        return json.dumps(processed_data, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        # Fallback to standard natural language prompt processing
        if quality_preset != "raw_film":
            return prompt

        prompt = clean_text(prompt)
        prompt += f", {anchors}"
        return prompt
