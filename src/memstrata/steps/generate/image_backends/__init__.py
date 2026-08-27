"""Image generation / keyframe backends (keyframe-first stage 1).

Vendored self-contained from upstream project ``models/image_generation`` (no ``montage`` imports).
The keyframe-first main pipeline uses these to synthesise Layout Anchor keyframes (FLUX),
which the video backends then expand into video (Wan2.2-I2V-A14B + SVI / Morphic LoRA).
"""

from .base import ImageGenerationModel, apply_photographic_grain, preprocess_de_ai_prompt
from .factory import build_image_backend, list_image_backend_names
from .flux_klein_backend import FluxKleinImageBackend

__all__ = [
    "ImageGenerationModel",
    "FluxKleinImageBackend",
    "build_image_backend",
    "list_image_backend_names",
    "preprocess_de_ai_prompt",
    "apply_photographic_grain",
]
