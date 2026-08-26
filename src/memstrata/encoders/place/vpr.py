from __future__ import annotations

from pathlib import Path

from memstrata.lib.weights import configure_torch_hub, weights_root
from memstrata.encoders.base import EmbeddingModel, Vector, l2_normalize


class VprEmbedding:
    """Location/place backend (Visual Place Recognition) for the ``location`` role."""

    def __init__(
        self,
        method: str = "megaloc",
        *,
        backbone: str | None = None,
        descriptors_dimension: int | None = None,
        weights: str | None = None,
        image_size: int = 322,
        device: str | None = None,
        repo: str = "submodules/_5_perception/scene_embedding/VPR-methods-evaluation",
    ) -> None:
        self.method = method
        self.backbone = backbone
        self.descriptors_dimension = descriptors_dimension
        self.weights = weights
        self.image_size = image_size
        self.repo = repo
        self.name = f"vpr:{method}"
        self.dim = descriptors_dimension or 0
        self._device = device
        self._model = None
        self._torch = None

    _HUB = {
        "cosplace": ("gmberton/cosplace", "get_trained_model"),
        "eigenplaces": ("gmberton/eigenplaces", "get_trained_model"),
        "megaloc": ("gmberton/MegaLoc", "get_trained_model"),
        "salad": ("serizba/salad", "dinov2_salad"),
        "cricavpr": ("Lu-Feng/CricaVPR", "trained_model"),
    }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "VprEmbedding requires the 'vision' extra (torch). Use a stub for offline runs."
            ) from exc
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        configure_torch_hub()

        model = self._load_megaloc_local(torch) if self.method == "megaloc" else None
        if model is not None:
            pass
        elif self.method in self._HUB:
            repo, entry = self._HUB[self.method]
            kwargs: dict = {}
            if self.backbone:
                kwargs["backbone"] = self.backbone
            if self.descriptors_dimension:
                kwargs["fc_output_dim"] = self.descriptors_dimension
            model = torch.hub.load(repo, entry, trust_repo=True, **kwargs)
        else:
            import sys

            repo_path = Path(self.repo).resolve()
            if repo_path.is_dir() and str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            try:
                from vpr_models import get_model
            except ImportError as exc:
                raise RuntimeError(
                    f"Could not import VPR-methods-evaluation from {self.repo!r}. Install its "
                    "extra deps / submodules, or use a torch.hub method "
                    f"({', '.join(sorted(self._HUB))})."
                ) from exc
            model = get_model(self.method, self.backbone, self.descriptors_dimension)

        if self.weights and Path(self.weights).is_file():
            state = torch.load(self.weights, map_location="cpu")
            model.load_state_dict(state.get("model_state_dict", state), strict=False)
        self._model = model.to(self._device).eval()

    def _load_megaloc_local(self, torch):
        import sys

        repo_dir = weights_root() / "torch_hub" / "gmberton_MegaLoc_main"
        weights_file = Path(self.weights) if self.weights else (
            weights_root() / "location_embedding" / "MegaLoc" / "model.safetensors"
        )
        if not weights_file.is_file():
            base_dir = weights_root() / "hub" / "models--gberton--MegaLoc" / "snapshots"
            candidates = list(base_dir.glob("*/model.safetensors")) if base_dir.is_dir() else []
            if candidates:
                weights_file = candidates[0]

        if not (repo_dir.is_dir() and weights_file.is_file()):
            return None
        try:
            from safetensors.torch import load_file

            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            from megaloc_model import MegaLoc

            model = MegaLoc()
            model.load_state_dict(load_file(str(weights_file)))
            self.weights = None
            return model
        except Exception:
            return None

    def embed_image(self, image: str | Path) -> Vector:
        self._ensure_loaded()
        import numpy as np
        from PIL import Image
        from memstrata.lib.media import load_crop_rgb_for_model

        torch = self._torch
        pil = load_crop_rgb_for_model(image).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        arr = np.asarray(pil, dtype="float32") / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
        std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        with torch.no_grad():
            descriptor = self._model(tensor)
        vector = descriptor.squeeze(0).float().cpu().tolist()
        self.dim = len(vector)
        return l2_normalize(vector)
