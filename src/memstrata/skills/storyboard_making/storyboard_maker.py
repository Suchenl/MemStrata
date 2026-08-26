"""Storyboard Making Skill Module.

One FLUX.2-klein pass paints every keyframe of a shot sequence in a single latent, so
self-attention shares the character's face, wardrobe and lighting across all panels — an
identity lock that independently generated keyframes cannot give. This module owns the two
halves of that trick:

1. Prompt assembly (``format_prompt``): style template + **shot scope** (are these panels
   keyframes of ONE shot, or separate shots?) + **explicit inter-panel timing** + mandatory
   grid geometry + the de-AI film anchors.
2. Panel extraction (``slice_storyboard``): find the white gutters, cut, **trim the leftover
   white border**, then force every panel to one identical resolution.

Both halves are needed. Asking the model for equal-size panels is necessary but never
sufficient: FLUX's gutters drift by a few pixels, so the panels must also be normalised on
our side before they can serve as video-model conditioning frames.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

#: ``format_prompt(shot_scope=...)`` values.
SHOT_SCOPE_WITHIN = "within_shot"
SHOT_SCOPE_ACROSS = "across_shots"

_SHOT_SCOPE_TEXT = {
    SHOT_SCOPE_WITHIN: (
        "Shot scope (mandatory): these {panel_count} panels are successive KEYFRAMES OF ONE "
        "SINGLE CONTINUOUS SHOT, not separate shots. Camera position, lens, focal length, "
        "framing, shot size, background and lighting are IDENTICAL in every panel; only the "
        "subject's pose and action advance in time. Never cut to another angle, another "
        "location or another lighting setup."
    ),
    SHOT_SCOPE_ACROSS: (
        "Shot scope (mandatory): these {panel_count} panels are {panel_count} SEPARATE SHOTS "
        "of the same scene, as a film would cut between them. Camera angle, framing and shot "
        "size SHOULD differ from panel to panel, but the character's identity and face, the "
        "wardrobe, the props and the location must stay exactly the same in all panels."
    ),
}

_GEOMETRY_TEXT = (
    "Grid geometry (mandatory): exactly {cols} columns by {rows} rows. Every panel has an "
    "identical width, height and aspect ratio. Panels are separated by narrow pure-white "
    "gutters of even thickness, and the whole sheet is surrounded by a pure-white margin. No "
    "content may cross or overlap a gutter. No panel numbers, no captions, no page numbers and "
    "no frame borders other than the white gutters."
)


class StoryboardMaker:
    """Storyboard Maker class implementing the storyboard creation skill."""

    def __init__(self, styles_path: Path | None = None) -> None:
        if styles_path is None:
            styles_path = Path(__file__).parent / "styles.json"
        
        self.styles_path = styles_path
        self.styles = self._load_styles()

    def _load_styles(self) -> Dict[str, Any]:
        """Load styles from styles.json."""
        try:
            with open(self.styles_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load styles from {self.styles_path}: {e}")
            return {}

    def get_available_styles(self) -> List[Dict[str, str]]:
        """Get a list of available styles with their metadata."""
        return [
            {
                "key": key,
                "name_zh": val.get("name_zh", ""),
                "name_en": val.get("name_en", ""),
                "description": val.get("description", "")
            }
            for key, val in self.styles.items()
        ]

    def calculate_dimensions(
        self,
        video_w: int,
        video_h: int,
        cols: int,
        rows: int,
        gutter: int = 16,
        scale_factor: float = 0.75
    ) -> Tuple[int, int]:
        """Calculate optimal storyboard dimensions based on target video dimensions.
        
        Formula:
            W_board = S * W_video * C + (C - 1) * G
            H_board = S * H_video * R + (R - 1) * G
        """
        w_board = int(scale_factor * video_w * cols + (cols - 1) * gutter)
        h_board = int(scale_factor * video_h * rows + (rows - 1) * gutter)
        
        # Ensure dimensions are multiples of 8 or 16 for stable diffusion models
        w_board = (w_board // 16) * 16
        h_board = (h_board // 16) * 16
        
        return w_board, h_board

    def format_prompt(
        self,
        style_key: str,
        panel_descriptions: List[str],
        cols: int,
        rows: int,
        *,
        shot_scope: str = SHOT_SCOPE_WITHIN,
        panel_times_sec: Sequence[float] | None = None,
        raw_film: bool = True,
    ) -> Tuple[str, str]:
        """Format the prompt and negative prompt for a given style and panel descriptions.

        Parameters
        ----------
        shot_scope:
            ``SHOT_SCOPE_WITHIN`` (panels are keyframes inside one continuous shot) or
            ``SHOT_SCOPE_ACROSS`` (panels are separate shots of the same scene). This changes
            what the model is allowed to vary between panels — camera continuity versus a cut —
            so it must be stated, not inferred.
        panel_times_sec:
            Each panel's timestamp in seconds on the shot's own timeline. Turned into explicit
            per-panel deltas so the amount of change between neighbours matches the real time
            gap instead of being arbitrary.
        raw_film:
            Apply the de-AI treatment (strip "photorealistic"/"8k"-class buzzwords, append the
            35mm film anchors). Shared with the video prompt path so both use one wording.
        """
        style = self.styles.get(style_key)
        if not style:
            raise ValueError(f"Style '{style_key}' not found in styles.json")
        if shot_scope not in _SHOT_SCOPE_TEXT:
            raise ValueError(
                f"shot_scope must be one of {sorted(_SHOT_SCOPE_TEXT)}, got {shot_scope!r}"
            )

        panel_count = len(panel_descriptions)
        expected_count = cols * rows
        if panel_count != expected_count:
            logger.warning(
                f"Panel description count ({panel_count}) does not match grid layout "
                f"{cols}x{rows} ({expected_count} panels expected). Adjusting description list."
            )
            if panel_count < expected_count:
                panel_descriptions = panel_descriptions + [""] * (expected_count - panel_count)
            else:
                panel_descriptions = panel_descriptions[:expected_count]

        offsets = self._panel_offsets(panel_times_sec, expected_count)

        # Format individual panel descriptions, each carrying its own time delta.
        formatted_panels = []
        for i, desc in enumerate(panel_descriptions):
            stamp = ""
            if offsets is not None:
                stamp = (
                    " (first keyframe, t=0s)" if i == 0
                    else f" (+{offsets[i]:.1f}s after Panel {i})"
                )
            formatted_panels.append(f"Panel {i + 1}{stamp}: {desc}")

        panels_text = "\n".join(formatted_panels)

        # Format the main prompt template
        prompt = style["prompt_template"].format(
            panel_count=expected_count,
            cols=cols,
            rows=rows,
            panels_description=panels_text
        )

        sections = [
            prompt,
            _SHOT_SCOPE_TEXT[shot_scope].format(panel_count=expected_count),
            _GEOMETRY_TEXT.format(cols=cols, rows=rows),
        ]
        if offsets is not None and panel_times_sec:
            span = float(panel_times_sec[-1]) - float(panel_times_sec[0])
            gaps = ", ".join(f"{g:.1f}s" for g in offsets[1:])
            sections.append(
                f"Timing (mandatory): the panels are {gaps} apart and span {span:.1f}s in "
                "total. Scale the visible change between neighbouring panels to these gaps — a "
                "short gap means only a slight advance of the same pose, a long gap means a "
                "clearly later moment of the action."
            )
        if raw_film:
            sections = self._apply_film_anchors(sections)
        prompt = "\n\n".join(sections)

        negative_prompt = style.get("negative_prompt", "")
        return prompt, negative_prompt

    @staticmethod
    def _panel_offsets(
        panel_times_sec: Sequence[float] | None, expected_count: int
    ) -> List[float] | None:
        """Per-panel gap to its predecessor, or None when no timeline was supplied."""
        if not panel_times_sec:
            return None
        if len(panel_times_sec) != expected_count:
            logger.warning(
                "panel_times_sec has %d entries but the grid holds %d panels; timing omitted",
                len(panel_times_sec), expected_count,
            )
            return None
        times = [float(t) for t in panel_times_sec]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(f"panel_times_sec must be non-decreasing, got {times}")
        return [0.0] + [b - a for a, b in zip(times, times[1:])]

    @staticmethod
    def _apply_film_anchors(sections: List[str]) -> List[str]:
        """Drop the AI-render buzzwords and add a film-look section.

        Wording is shared with the video prompt path so the two never drift; the anchors form
        their own trailing section instead of being glued onto whatever sentence ends the prompt.
        """
        try:
            from memstrata.lib.prompt_standardizer import FILM_ANCHORS, strip_ai_buzzwords
        except Exception as exc:  # noqa: BLE001 - the skill stays usable standalone
            logger.warning("de-AI wording unavailable (%s); prompt left as-is", exc)
            return sections
        cleaned = [strip_ai_buzzwords(section) for section in sections]
        cleaned.append(f"Film look (mandatory): {FILM_ANCHORS}.")
        return cleaned

    def slice_storyboard(
        self,
        image_path: Union[str, Path],
        cols: int,
        rows: int,
        gutter: int = 16,
        out_dir: Union[str, Path] | None = None,
        adaptive_detection: bool = True,
        trim_white: bool = True,
        target_size: Tuple[int, int] | None = None,
    ) -> List[Path]:
        """Slice a storyboard image into individual panels.

        Gutters are located by adaptive white detection with a mathematical grid fallback. Two
        steps then make the panels usable as video conditioning frames:

        ``trim_white``
            Shave the residual white margin off each panel. The sheet is asked for an outer
            white margin and even gutters, but the cut lands a few pixels inside the white, so
            without this a keyframe carries white bars that the video model would animate.
        ``target_size``
            Final ``(w, h)`` for every panel, e.g. the video's ``(832, 480)``. When omitted the
            panels are still forced to one common size, because a downstream multi-keyframe call
            requires all references to share a resolution.
        """
        image_path = Path(image_path)
        if out_dir is None:
            out_dir = image_path.parent / "sliced_panels"
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path)
        W, H = img.size

        # Calculate exact panel size (float division for precision)
        panel_w = (W - (cols - 1) * gutter) / cols
        panel_h = (H - (rows - 1) * gutter) / rows

        x_gutters = []
        y_gutters = []

        if adaptive_detection:
            try:
                # Convert to grayscale for brightness analysis
                img_l = img.convert("L")
                
                # Fast column/row average calculation using PIL's resize projection trick
                col_img = img_l.resize((W, 1))
                row_img = img_l.resize((1, H))
                
                col_averages = [col_img.getpixel((x, 0)) for x in range(W)]
                row_averages = [row_img.getpixel((0, y)) for y in range(H)]
                
                # Search window size (10% of expected panel width/height)
                w_col = int((W / cols) * 0.1)
                w_row = int((H / rows) * 0.1)
                
                # 1. Detect vertical gutters (columns)
                for j in range(1, cols):
                    # Expected mathematical center of the j-th gutter
                    x_exp = int(round(j * panel_w + (j - 0.5) * gutter))
                    x_at, how = _gutter_center(col_averages, x_exp, w_col, W)
                    x_gutters.append(x_at)
                    logger.info(f"Adaptive Gutter: vertical gutter at x={x_at} ({how})")

                # 2. Detect horizontal gutters (rows)
                for i in range(1, rows):
                    # Expected mathematical center of the i-th gutter
                    y_exp = int(round(i * panel_h + (i - 0.5) * gutter))
                    y_at, how = _gutter_center(row_averages, y_exp, w_row, H)
                    y_gutters.append(y_at)
                    logger.info(f"Adaptive Gutter: horizontal gutter at y={y_at} ({how})")
            except Exception as e:
                logger.warning(f"Adaptive gutter detection failed: {e}. Falling back to mathematical grid.")
                adaptive_detection = False

        # If adaptive detection is disabled or failed, use mathematical grid fallback
        if not adaptive_detection:
            x_gutters = [int(round(j * panel_w + (j - 0.5) * gutter)) for j in range(1, cols)]
            y_gutters = [int(round(i * panel_h + (i - 0.5) * gutter)) for i in range(1, rows)]

        # Define split boundaries (including image edges)
        x_splits = [0] + x_gutters + [W]
        y_splits = [0] + y_gutters + [H]

        panels: List[Image.Image] = []
        panel_idx = 1

        # With trimming on, cut at the gutter centre and let the white trim find the exact panel
        # edge; pulling back half a gutter first would shave a pixel row off the content, since a
        # gutter of even width has no integer centre. Without trimming the old inset is kept.
        shave = 0 if trim_white else gutter // 2

        for r in range(rows):
            for c in range(cols):
                # Calculate coordinates with gutter offsets
                x_start = x_splits[c] + (shave if c > 0 else 0)
                x_end = x_splits[c+1] - (shave if c < cols - 1 else 0)
                y_start = y_splits[r] + (shave if r > 0 else 0)
                y_end = y_splits[r+1] - (shave if r < rows - 1 else 0)

                # Boundary safety checks
                x_start = max(0, min(x_start, W))
                x_end = max(0, min(x_end, W))
                y_start = max(0, min(y_start, H))
                y_end = max(0, min(y_end, H))

                # Crop, shave the leftover white, save
                panel_img = img.crop((x_start, y_start, x_end, y_end))
                if trim_white:
                    panel_img = _trim_white_border(panel_img)
                panels.append(panel_img)
                logger.info(
                    f"Sliced Panel {panel_idx}: box=({x_start}, {y_start}, {x_end}, {y_end}), "
                    f"size={panel_img.size}"
                )
                panel_idx += 1

        panels = _normalize_panel_sizes(panels, target_size=target_size)

        sliced_paths = []
        for index, panel_img in enumerate(panels, start=1):
            panel_path = out_dir / f"panel_{index:02d}.png"
            panel_img.save(panel_path)
            sliced_paths.append(panel_path)
            logger.info(f"Panel {index} -> {panel_path} size={panel_img.size}")

        return sliced_paths


def _gutter_center(
    profile: List[int], expected: int, half_window: int, limit: int, *, white: int = 200
) -> Tuple[int, str]:
    """Centre of the white gutter run nearest ``expected`` in a brightness profile.

    Taking the run's centre rather than its brightest line matters: a gutter is a plateau of
    equally white lines, so "brightest" resolves to whichever edge is scanned first and biases
    every cut into the neighbouring panel by half a gutter.
    """
    start = max(0, expected - half_window)
    end = min(limit - 1, expected + half_window)
    window = profile[start : end + 1]
    if not window or max(window) - min(window) < 15:
        return expected, "flat, math fallback"  # uniform window: no gutter to lock onto
    if max(window) < white:
        return expected, "no white run, math fallback"

    runs: List[Tuple[int, int]] = []
    run_start: int | None = None
    for index in range(start, end + 2):
        is_white = index <= end and profile[index] >= white
        if is_white and run_start is None:
            run_start = index
        elif not is_white and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None
    best = min(runs, key=lambda run: abs((run[0] + run[1]) // 2 - expected))
    return (best[0] + best[1]) // 2, "detected"


def _trim_white_border(
    image: Image.Image, *, threshold: int = 242, white_frac: float = 0.97, max_frac: float = 0.2
) -> Image.Image:
    """Shave near-white edge lines off a panel.

    Walks in from each side while that border line is almost entirely near-white, so the white
    gutter/margin left by the cut disappears while genuine bright content (a sky, an overexposed
    window) is kept: a real image line is never ~97% white across its full length. Refuses to
    eat more than ``max_frac`` of either dimension, so a legitimately white-heavy panel cannot
    collapse.
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    def row_is_white(y: int) -> bool:
        hits = sum(1 for x in range(width) if pixels[x, y] >= threshold)
        return hits >= white_frac * width

    def col_is_white(x: int) -> bool:
        hits = sum(1 for y in range(height) if pixels[x, y] >= threshold)
        return hits >= white_frac * height

    # A panel with no non-white content has no border to find; eating into it would only
    # disguise the real problem (an empty or blown-out generation).
    if all(row_is_white(y) for y in range(height)):
        logger.warning("panel is entirely near-white; leaving it untrimmed")
        return image

    max_x, max_y = int(width * max_frac), int(height * max_frac)
    left = 0
    while left < max_x and col_is_white(left):
        left += 1
    right = width
    while right > width - max_x and col_is_white(right - 1):
        right -= 1
    top = 0
    while top < max_y and row_is_white(top):
        top += 1
    bottom = height
    while bottom > height - max_y and row_is_white(bottom - 1):
        bottom -= 1

    if right - left < 8 or bottom - top < 8:
        logger.warning("white trim would empty the panel; keeping it untouched")
        return image
    if (left, top, right, bottom) == (0, 0, width, height):
        return image
    logger.info("Trimmed white border: (%d, %d, %d, %d) from %dx%d", left, top, right, bottom,
                width, height)
    return image.crop((left, top, right, bottom))


def _normalize_panel_sizes(
    panels: List[Image.Image], *, target_size: Tuple[int, int] | None = None
) -> List[Image.Image]:
    """Force every panel to one identical resolution.

    A multi-keyframe video call rejects (or silently distorts) references of differing sizes, and
    trimming leaves panels a few pixels apart. Each panel is centre-cropped to the target aspect
    ratio first and only then resized, so nothing is stretched.
    """
    if not panels:
        return panels
    if target_size is None:
        width = min(p.size[0] for p in panels)
        height = min(p.size[1] for p in panels)
        target_size = (width, height)
    tw, th = target_size
    target_ratio = tw / th

    out: List[Image.Image] = []
    for panel in panels:
        pw, ph = panel.size
        if abs(pw / ph - target_ratio) > 1e-3:
            if pw / ph > target_ratio:  # too wide -> trim the sides
                crop_w = int(round(ph * target_ratio))
                offset = (pw - crop_w) // 2
                panel = panel.crop((offset, 0, offset + crop_w, ph))
            else:  # too tall -> trim top/bottom
                crop_h = int(round(pw / target_ratio))
                offset = (ph - crop_h) // 2
                panel = panel.crop((0, offset, pw, offset + crop_h))
        if panel.size != (tw, th):
            panel = panel.resize((tw, th), Image.LANCZOS)
        out.append(panel)
    sizes = {p.size for p in out}
    if len(sizes) != 1:  # defensive: the contract downstream depends on this
        raise AssertionError(f"panel normalisation failed, got sizes {sizes}")
    return out
