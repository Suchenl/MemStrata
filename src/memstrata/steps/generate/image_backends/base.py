"""Image generation model protocol and common utilities.

Vendored from upstream project ``the upstream source tree/models/image_generation/base.py`` (import paths
rewritten to stay self-contained under ``this repository`` — no ``montage`` imports).
"""

from typing import Any, Protocol

from memstrata.lib.prompt_standardizer import preprocess_de_ai_prompt as _preprocess_de_ai_prompt
from memstrata.steps.generate.schemas import GenerationArtifact, MediaGenerationTask


class ImageGenerationModel(Protocol):
    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        """Generate or edit an image according to an approved task."""


def preprocess_de_ai_prompt(prompt: str, quality_preset: str | None) -> str:
    """Clean up CG-like buzzwords and inject photographic film anchors to shatter AI plastic bias."""
    return _preprocess_de_ai_prompt(prompt, quality_preset)


def apply_photographic_grain(image: Any, noise_strength: float = 0.015) -> Any:
    """Apply a very subtle film grain and mild sharpening to shatter the VAE CG gloss."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageFilter  # noqa: PLC0415

    # 1. Subtle sharpening to break down VAE-smoothed edges
    image_sharp = image.filter(ImageFilter.SHARPEN)

    # 2. Add Gaussian noise to simulate analog photographic emulsion grain
    img_array = np.array(image_sharp, dtype=np.float32)
    noise = np.random.normal(0, noise_strength * 255, img_array.shape)
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img_array)
