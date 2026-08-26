"""Base classes and utilities for layout anchor processing.

Vendored verbatim from ``src/montage/skills/layout_anchor_processing/base.py``
(only the package location changed) so MemStrata carries no ``montage`` import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

# Default color palette for common visual entities to maintain visual consistency
DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "background": (38, 38, 42),
    "wall": (55, 57, 62),
    "floor": (72, 64, 57),
    "speaker": (188, 143, 110),
    "actor": (188, 143, 110),
    "human": (188, 143, 110),
    "person": (188, 143, 110),
    "podium": (112, 70, 42),
    "table": (112, 70, 42),
    "desk": (112, 70, 42),
    "chair": (78, 49, 31),
    "reporter": (125, 100, 82),
    "crowd": (125, 100, 82),
    "audience": (125, 100, 82),
    "sky": (135, 206, 235),
    "ground": (34, 139, 34),
    "tree": (46, 139, 87),
    "car": (220, 20, 60),
    "building": (105, 105, 105),
}


def get_deterministic_color(label: str) -> Tuple[int, int, int]:
    """Get a deterministic RGB color for a given label.

    If the label is in DEFAULT_COLORS, returns the pre-defined color.
    Otherwise, generates a nice, usable color based on the hash of the label.
    """
    label_key = label.lower().strip()
    if label_key in DEFAULT_COLORS:
        return DEFAULT_COLORS[label_key]

    # Deterministic hash-based color generation
    h = abs(hash(label_key))
    r = 50 + (h % 150)
    g = 50 + ((h >> 8) % 150)
    b = 50 + ((h >> 16) % 150)
    return (r, g, b)


class LayoutElement:
    """Represents a single visual element in the layout anchor plan."""

    def __init__(
        self,
        label: str,
        box_2d: List[Union[int, float]],  # [ymin, xmin, ymax, xmax]
        fill_color: Union[List[int], Tuple[int, int, int], None] = None,
        shape: str = "rectangle",
        **kwargs: Any
    ) -> None:
        self.label = label
        self.box_2d = box_2d
        self.shape = shape.lower().strip()
        self.extra = kwargs

        # Assign color (either provided, default, or deterministic)
        if fill_color is not None:
            self.fill_color = tuple(fill_color)
        else:
            self.fill_color = get_deterministic_color(label)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutElement":
        """Create a LayoutElement from a dictionary."""
        if "label" not in data or "box_2d" not in data:
            raise ValueError("Layout element dictionary must contain 'label' and 'box_2d' keys.")
        return cls(
            label=data["label"],
            box_2d=data["box_2d"],
            fill_color=data.get("fill_color"),
            shape=data.get("shape", "rectangle"),
            **{k: v for k, v in data.items() if k not in {"label", "box_2d", "fill_color", "shape"}}
        )


class BaseLayoutProcessor:
    """Base processor for parsing and scaling structured layout anchors."""

    def __init__(self, default_width: int = 832, default_height: int = 480) -> None:
        self.default_width = default_width
        self.default_height = default_height

    def parse_layout(self, layout_input: Union[str, List[Dict[str, Any]], Path]) -> List[LayoutElement]:
        """Parse layout elements from a JSON string, list of dicts, or a file path."""
        if isinstance(layout_input, Path) or (isinstance(layout_input, str) and layout_input.endswith(".json")):
            path = Path(layout_input)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(layout_input, str):
            data = json.loads(layout_input)
        else:
            data = layout_input

        if not isinstance(data, list):
            raise ValueError("Layout JSON input must be a list of elements.")

        return [LayoutElement.from_dict(item) for item in data]

    def scale_coordinates(
        self,
        box_2d: List[Union[int, float]],
        target_w: int,
        target_h: int,
        normalized_range: Tuple[float, float] | None = (0, 1000)
    ) -> Tuple[int, int, int, int]:
        """Scale coordinates from a normalized range (e.g. [0, 1000]) to target pixel dimensions.

        Returns:
            Tuple of absolute coordinates: (xmin, ymin, xmax, ymax)
        """
        ymin, xmin, ymax, xmax = box_2d

        if normalized_range is not None:
            norm_min, norm_max = normalized_range
            norm_span = norm_max - norm_min

            ymin = ((ymin - norm_min) / norm_span) * target_h
            xmin = ((xmin - norm_min) / norm_span) * target_w
            ymax = ((ymax - norm_min) / norm_span) * target_h
            xmax = ((xmax - norm_min) / norm_span) * target_w

        # Ensure coordinates are within image boundaries and rounded to integers
        xmin_abs = max(0, min(int(round(xmin)), target_w))
        ymin_abs = max(0, min(int(round(ymin)), target_h))
        xmax_abs = max(0, min(int(round(xmax)), target_w))
        ymax_abs = max(0, min(int(round(ymax)), target_h))

        return xmin_abs, ymin_abs, xmax_abs, ymax_abs
