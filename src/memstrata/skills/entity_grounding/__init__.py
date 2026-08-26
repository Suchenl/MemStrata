"""Entity grounding skill (production R7).

VLM-grounded crop acquisition for the decompose write-path: locate a named entity in a
generated frame and return a tight crop. Mirrors vmem_bench S5 QwenImageGrounder,
reusing the unified Qwen3.5-9B via MllmRoleRunner (zero-import of vmem_bench).
"""

from memstrata.skills.entity_grounding.grounding_cropper import (
    GROUNDING_SCHEMA,
    VlmGroundingCropper,
    canonicalize_box_point,
)

__all__ = ["VlmGroundingCropper", "canonicalize_box_point", "GROUNDING_SCHEMA"]
