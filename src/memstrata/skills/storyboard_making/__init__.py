"""Storyboard Making Skill.

One FLUX pass paints all keyframes of a shot sequence in a shared latent, then the sheet is
sliced back into equally sized panels. Handles multi-style prompt assembly, shot-scope and
inter-panel timing declarations, video-aligned dimension calculation, and white-gutter slicing.
"""

from __future__ import annotations

from .storyboard_maker import SHOT_SCOPE_ACROSS, SHOT_SCOPE_WITHIN, StoryboardMaker

__all__ = ["StoryboardMaker", "SHOT_SCOPE_WITHIN", "SHOT_SCOPE_ACROSS"]
