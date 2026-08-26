"""Color block layout anchor processor.

Renders layout elements as solid color blocks on a dark background — the optimal
control signal for FLUX.2 Klein. Vendored verbatim from
``src/montage/skills/layout_anchor_processing/color_block_processor.py``.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from PIL import Image, ImageDraw

from memstrata.skills.layout_anchor_processing.base import DEFAULT_COLORS, BaseLayoutProcessor, LayoutElement

logger = logging.getLogger(__name__)


class ColorBlockProcessor(BaseLayoutProcessor):
    """Processor that renders layout elements as solid color blocks for FLUX.2 Klein."""

    def render_anchor(
        self,
        elements: List[LayoutElement],
        width: int | None = None,
        height: int | None = None,
        normalized_range: Tuple[float, float] | None = (0, 1000),
        background_color: Tuple[int, int, int] | None = None
    ) -> Image.Image:
        """Render the layout elements onto a PIL image canvas as solid color blocks."""
        w = width or self.default_width
        h = height or self.default_height
        bg = background_color or DEFAULT_COLORS["background"]

        # Create a new blank canvas with a dark background
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)

        # Render elements sequentially
        for elem in elements:
            xmin, ymin, xmax, ymax = self.scale_coordinates(
                elem.box_2d, w, h, normalized_range=normalized_range
            )

            # Skip degenerate boxes
            if xmin >= xmax or ymin >= ymax:
                logger.warning(f"Skipping degenerate layout box for element '{elem.label}': {(xmin, ymin, xmax, ymax)}")
                continue

            self._draw_element(draw, xmin, ymin, xmax, ymax, elem)

        return img

    def _draw_element(
        self,
        draw: ImageDraw.ImageDraw,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        elem: LayoutElement
    ) -> None:
        """Internal helper to draw solid shapes."""
        color = elem.fill_color
        shape = elem.shape
        w = xmax - xmin
        h = ymax - ymin

        if shape == "rectangle":
            draw.rectangle((xmin, ymin, xmax, ymax), fill=color)
        elif shape == "ellipse":
            draw.ellipse((xmin, ymin, xmax, ymax), fill=color)
        elif shape == "line":
            draw.line((xmin, ymin, xmax, ymax), fill=color, width=max(2, int(min(w, h) * 0.1)))
        elif shape in {"ellipse_and_rect", "human", "person", "actor", "speaker"}:
            # Draw a beautifully scaled, simplified solid human figure
            # 1. Head (ellipse, top 20% of height)
            head_w = int(w * 0.45)
            head_h = min(head_w, int(h * 0.2))
            head_xmin = xmin + (w - head_w) // 2
            head_ymin = ymin
            head_xmax = head_xmin + head_w
            head_ymax = head_ymin + head_h
            draw.ellipse((head_xmin, head_ymin, head_xmax, head_ymax), fill=color)

            # 2. Torso (rectangle, middle 40% of height)
            torso_w = int(w * 0.6)
            torso_h = int(h * 0.4)
            torso_xmin = xmin + (w - torso_w) // 2
            torso_ymin = head_ymax
            torso_xmax = torso_xmin + torso_w
            torso_ymax = torso_ymin + torso_h
            draw.rectangle((torso_xmin, torso_ymin, torso_xmax, torso_ymax), fill=color)

            # 3. Legs (lines, bottom 40% of height)
            leg_width = max(2, int(w * 0.1))
            # Left leg
            draw.line(
                (torso_xmin + leg_width, torso_ymax, xmin + int(w * 0.2), ymax),
                fill=(22, 24, 30),  # Darker color for legs/pants to match sketch prior
                width=leg_width
            )
            # Right leg
            draw.line(
                (torso_xmax - leg_width, torso_ymax, xmin + int(w * 0.8), ymax),
                fill=(22, 24, 30),
                width=leg_width
            )
        else:
            # Fallback to rectangle for unknown shapes
            logger.warning(f"Unknown shape '{shape}' for element '{elem.label}'. Falling back to rectangle.")
            draw.rectangle((xmin, ymin, xmax, ymax), fill=color)
