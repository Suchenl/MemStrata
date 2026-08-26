"""Layout Anchor Planning (role R3).

Adapted from ``src/montage/skills/layout_anchor_processing/planner.py``: the layout
schema + prompt are kept, but the model call is rewired from Montage's
``ModelRegistry`` to the MemStrata ``MllmRoleRunner`` so this carries no ``montage``
dependency and honours the ``layout_planner`` (R3) sampling/schema contract declared
in ``memstrata.mllm.roles``.

Pipeline role: turn a screenplay/enhanced-prompt into a normalized bbox layout, which
``ColorBlockProcessor`` renders into a color-block anchor for FLUX.2 Klein (the seed for
the Crop2Image keyframe). R4 then decides which real crop fills which region.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from memstrata.mllm.roles import get_role
from memstrata.mllm.runner import MllmRoleRunner

logger = logging.getLogger(__name__)

# JSON Schema to enforce structured output from the MLLM.
LAYOUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "description": "List of visual layout elements representing the composition plan.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The name or semantic label of the visual entity (e.g., speaker, podium, reporter, wall, floor, background).",
                    },
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "2D bounding box in normalized coordinates [ymin, xmin, ymax, xmax] in the range [0, 1000].",
                    },
                    "shape": {
                        "type": "string",
                        "enum": ["rectangle", "ellipse", "line", "human"],
                        "description": "The geometric shape to render for this element. Use 'human' for speakers/actors/persons.",
                    },
                },
                "required": ["label", "box_2d", "shape"],
            },
        }
    },
    "required": ["elements"],
}

LAYOUT_PLANNER_PROMPT_TEMPLATE = """You are a professional cinematic layout planner and storyboard director.
Based on the following screenplay/description, please output a 2D bounding box coordinate plan in normalized JSON format [0, 1000] for all major visual entities (e.g., characters, props, background elements, wall, floor, sky).

Screenplay / Description:
"{screenplay}"

Guidelines:
1. Use normalized coordinates in the range [0, 1000].
2. The coordinate format for 'box_2d' must be [ymin, xmin, ymax, xmax].
3. Assign an appropriate shape to each element:
   - "rectangle": standard rectangular objects, background blocks, walls, floors.
   - "ellipse": round or spherical objects.
   - "line": thin lines or linear objects.
   - "human": use this shape for any person, actor, speaker, or character.
4. Ensure the layout matches the spatial relationships described in the screenplay (e.g., "on the left", "on the right", "in the background").
5. Do not overlap key foreground entities unless they are physically interacting.
"""

ROLE_KEY = "layout_planner"  # R3


class LayoutPlanner:
    """Query the R3 MLLM role for a structured layout plan from a screenplay."""

    def __init__(self, runner: MllmRoleRunner | None = None) -> None:
        self.runner = runner or MllmRoleRunner()
        # Fail fast if the registry drifts away from this planner's contract.
        role = get_role(ROLE_KEY)
        if "elements" not in role.schema_fields:
            raise ValueError(f"role {ROLE_KEY} must declare 'elements' in schema_fields")

    def plan_layout(self, screenplay: str) -> List[Dict[str, Any]]:
        """Plan the spatial layout for a screenplay. Returns a list of element dicts."""
        prompt = LAYOUT_PLANNER_PROMPT_TEMPLATE.format(screenplay=screenplay)
        logger.info("Querying R3 layout_planner for layout plan...")
        result = self.runner.run(ROLE_KEY, instruction=prompt, schema=LAYOUT_SCHEMA)
        elements = list(result.get("elements", []))
        logger.info("R3 produced %d layout elements.", len(elements))
        return elements
