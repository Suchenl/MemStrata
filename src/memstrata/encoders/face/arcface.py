from __future__ import annotations

from pathlib import Path

from memstrata.lib.weights import weights_root
from memstrata.encoders.base import EmbeddingModel, Vector, l2_normalize


class FaceNotFound(ValueError):
    """Raised when an ArcFace embedding is requested for an image with no detectable face."""


class ArcFaceEmbedding:
    """Face-identity backend (ArcFace via InsightFace) for the ``face`` / character role."""

    def __init__(
        self,
        pack: str = "buffalo_l",
        *,
        root: str | None = None,
        det_size: int = 640,
    ) -> None:
        self.pack = pack
        self.root = root
        self.det_size = det_size
        self.name = f"arcface:{pack}"
        self.dim = 512
        self._app = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "ArcFaceEmbedding requires the 'insightface' + 'onnxruntime' packages. "
                "Install them (pip install insightface onnxruntime) or use a stub for offline runs."
            ) from exc
        root = self.root or str(weights_root() / "human_face_embedding")
        app = FaceAnalysis(name=self.pack, root=root)
        app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        self._app = app

    def embed_image(self, image: str | Path) -> Vector:
        self._ensure_loaded()
        import numpy as np
        from memstrata.lib.media import load_crop_rgb_for_model

        pil = load_crop_rgb_for_model(image)
        bgr = np.asarray(pil)[:, :, ::-1]
        faces = self._app.get(np.ascontiguousarray(bgr))
        if not faces:
            raise FaceNotFound(f"no face detected in {image}")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return l2_normalize([float(v) for v in face.normed_embedding])
