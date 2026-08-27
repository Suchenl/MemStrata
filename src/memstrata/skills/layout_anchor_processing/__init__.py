"""Layout Anchor Processing (vendored from upstream skills.layout_anchor_processing).

Planning (R3 MLLM role) + parsing/scaling/rendering of structured layout anchors into
color-block (FLUX.2 Klein) or line-art (Qwen-Image-Edit) composition blueprints. The
planner is rewired to MemStrata's MllmRoleRunner; processors are verbatim copies with
no external dependency beyond PIL.
"""

from __future__ import annotations

from memstrata.skills.layout_anchor_processing.base import (
    DEFAULT_COLORS,
    BaseLayoutProcessor,
    LayoutElement,
    get_deterministic_color,
)
from memstrata.skills.layout_anchor_processing.color_block_processor import ColorBlockProcessor
from memstrata.skills.layout_anchor_processing.line_art_processor import LineArtProcessor
from memstrata.skills.layout_anchor_processing.crop2image import (
    R4_ROLE,
    R4_SCHEMA,
    CropRef,
    assign_crops_to_regions,
    composite_crops,
    crop2image_canvas,
)
from memstrata.skills.layout_anchor_processing.planner import (
    LAYOUT_PLANNER_PROMPT_TEMPLATE,
    LAYOUT_SCHEMA,
    ROLE_KEY,
    LayoutPlanner,
)

__all__ = [
    "DEFAULT_COLORS",
    "BaseLayoutProcessor",
    "LayoutElement",
    "get_deterministic_color",
    "ColorBlockProcessor",
    "LineArtProcessor",
    "LayoutPlanner",
    "LAYOUT_SCHEMA",
    "LAYOUT_PLANNER_PROMPT_TEMPLATE",
    "ROLE_KEY",
    "CropRef",
    "R4_ROLE",
    "R4_SCHEMA",
    "assign_crops_to_regions",
    "composite_crops",
    "crop2image_canvas",
]
