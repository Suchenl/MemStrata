from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from memstrata.lib.weights import hf_cache_dir
from memstrata.encoders.base import EmbeddingModel, Vector, l2_normalize, _load_preprocess_config


class DinoV3Embedding:
    """Optional DINOv3 backend (latest self-supervised ViT features)."""

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        *,
        device: str | None = None,
        pooling: str = "cls",
    ) -> None:
        if pooling not in {"cls", "mean", "mean_patch"}:
            raise ValueError("pooling must be one of 'cls', 'mean', 'mean_patch'")
        self.model_id = model_id
        self.name = f"dinov3:{model_id}"
        self.pooling = pooling
        self._device = device
        self._model = None
        self._processor = None
        self._num_register_tokens = 0
        self.dim = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DinoV3Embedding requires the 'vision' extra (torch, transformers, Pillow). "
                "Use HashEmbedding for offline/testing runs."
            ) from exc
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = hf_cache_dir()
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        except ImportError:
            self._processor = None
            self._manual_cfg = _load_preprocess_config(self.model_id)
        self._model = (
            AutoModel.from_pretrained(self.model_id, cache_dir=cache_dir).to(self._device).eval()
        )
        self.dim = int(self._model.config.hidden_size)
        self._num_register_tokens = int(getattr(self._model.config, "num_register_tokens", 0) or 0)

    def embed_image(self, image: str | Path) -> Vector:
        self._ensure_loaded()
        from memstrata.lib.media import load_crop_rgb_for_model

        torch = self._torch
        pil = load_crop_rgb_for_model(image)
        if self._processor is not None:
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        else:
            inputs = {"pixel_values": self._manual_pixels(pil).to(self._device)}
        with torch.no_grad():
            outputs = self._model(**inputs)
        tokens = outputs.last_hidden_state[0]
        if self.pooling == "cls":
            pooled = tokens[0]
        elif self.pooling == "mean_patch":
            start = 1 + self._num_register_tokens
            pooled = tokens[start:].mean(dim=0)
        else:
            pooled = tokens.mean(dim=0)
        return l2_normalize(pooled.float().cpu().tolist())

    def _manual_pixels(self, pil: object):
        import numpy as np
        from PIL import Image

        torch = self._torch
        cfg = self._manual_cfg
        height, width = cfg["height"], cfg["width"]
        resized = pil.resize((width, height), Image.BILINEAR)
        arr = np.asarray(resized, dtype="float32") * cfg["rescale"]
        arr = (arr - np.asarray(cfg["mean"], dtype="float32")) / np.asarray(cfg["std"], dtype="float32")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor
