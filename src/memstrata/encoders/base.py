"""Common embedding interfaces, fallbacks, and routing protocols for MemStrata."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class EmbeddingModel(Protocol):
    """A frozen image encoder mapping an image to a unit-norm vector."""

    name: str
    dim: int

    def embed_image(self, image: str | Path) -> Vector: ...


@runtime_checkable
class TextEmbeddingModel(Protocol):
    """A frozen text encoder mapping text to a unit-norm vector."""

    name: str
    dim: int

    def embed_text(self, text: str) -> Vector: ...


@runtime_checkable
class CrossModalEmbeddingModel(EmbeddingModel, TextEmbeddingModel, Protocol):
    """Shared text/image embedding space for text→frame retrieval."""


def l2_normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must share the same dimensionality")
    return sum(a * b for a, b in zip(left, right))


def cosine_distance(left: Vector, right: Vector) -> float:
    return 1.0 - cosine_similarity(left, right)


def _load_preprocess_config(model_id: str) -> dict:
    import json

    defaults = {
        "height": 224,
        "width": 224,
        "rescale": 1.0 / 255.0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    config_path = Path(model_id) / "preprocessor_config.json"
    if not config_path.is_file():
        return defaults
    data = json.loads(config_path.read_text())
    size = data.get("size") or {}
    return {
        "height": int(size.get("height", 224)),
        "width": int(size.get("width", 224)),
        "rescale": float(data.get("rescale_factor", defaults["rescale"])),
        "mean": list(data.get("image_mean", defaults["mean"])),
        "std": list(data.get("image_std", defaults["std"])),
    }


_EMBEDDING_CACHE: dict[tuple[str, str | None, str | None], EmbeddingModel] = {}


def _public_model_root() -> Path:
    root = os.environ.get("PUBLIC_MODELS_ROOT", "").strip()
    if not root:
        raise FileNotFoundError(
            "PUBLIC_MODELS_ROOT is not set; heavy embedding providers resolve local "
            "weights under this root. Set PUBLIC_MODELS_ROOT or pass weights=..."
        )
    return Path(root)


def _resolve_local_model(
    *,
    provider: str,
    model: str | None,
    weights: str | None,
    default_rel: str,
    env_var: str,
) -> str:
    explicit = weights or os.environ.get(env_var)
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
        raise FileNotFoundError(
            f"{provider} weights not found: {path}. Set {env_var} or pass weights=... "
            "to an existing local snapshot."
        )
    if model and Path(model).expanduser().exists():
        return str(Path(model).expanduser())
    rel = model or default_rel
    cand = _public_model_root() / rel
    if cand.exists():
        return str(cand)
    if os.environ.get("MEMSTRATA_ALLOW_HF_DOWNLOAD") == "1":
        return rel
    raise FileNotFoundError(
        f"{provider} weights not found under PUBLIC_MODELS_ROOT: {cand}. "
        "Set the provider-specific *_WEIGHTS env var, pass weights=..., or set "
        f"MEMSTRATA_ALLOW_HF_DOWNLOAD=1 to let transformers resolve {rel!r}."
    )


def _as_vector(value: Any) -> Vector:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [float(v) for v in value]


class OpenAITextEmbedding:
    """OpenAI-compatible embedding endpoint client for server-backed text encoders."""

    def __init__(self, *, endpoint: str, model: str, api_key: str = "") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.name = f"openai-embedding:{model}"
        self.dim = 0

    def _embed_many(self, texts: list[str]) -> list[Vector]:
        url = self.endpoint
        if not url.endswith("/embeddings"):
            url = f"{url}/embeddings"
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=float(os.environ.get("MEMSTRATA_EMBED_TIMEOUT", "120"))) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = sorted(data.get("data", []), key=lambda row: int(row.get("index", 0)))
        vectors = [l2_normalize([float(v) for v in row["embedding"]]) for row in rows]
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors

    def embed_text(self, text: str) -> Vector:
        return self._embed_many([text])[0]


class Qwen3TextEmbedding:
    """Qwen3-Embedding text encoder with lazy heavy imports."""

    INSTRUCT = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:"
    )

    def __init__(self, *, model_id: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.name = f"qwen3-embedding@{self.device}"
        self.dim = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)

    def _encode(self, texts: list[str]) -> list[Vector]:
        torch = self._torch
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
            emb = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
        return [_as_vector(row) for row in emb]

    def embed_text(self, text: str) -> Vector:
        return self._encode([text])[0]

    def embed_query(self, text: str) -> Vector:
        return self.embed_text(self.INSTRUCT + text)

    def embed_doc(self, text: str) -> Vector:
        return self.embed_text(text)


class SigLIP2Embedding:
    """SigLIP2 shared text/image encoder for text→frame retrieval."""

    def __init__(self, *, model_id: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.name = f"siglip2@{self.device}"
        cfg = getattr(self.model, "config", None)
        self.dim = int(
            getattr(getattr(cfg, "text_config", None), "hidden_size", 0)
            or getattr(cfg, "projection_dim", 0)
            or 0
        )

    def embed_text(self, text: str) -> Vector:
        torch = self._torch
        inputs = self.processor(
            text=[text], return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        with torch.no_grad():
            feat = self.model.get_text_features(**inputs)
            feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        return _as_vector(feat)

    def embed_image(self, image: str | Path) -> Vector:
        import torch
        from PIL import Image

        try:
            img = Image.open(image).convert("RGB")
        except Exception:  # noqa: BLE001 - missing smoke-test images become stable handles.
            return HashEmbedding(dim=self.dim or 64).embed_image(image)
        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
            feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        return _as_vector(feat)


def _construct_image_embedding(
    backend: str, model: str | None, weights: str | None
) -> EmbeddingModel:
    if backend == "siglip2":
        model_id = _resolve_local_model(
            provider="siglip2",
            model=model,
            weights=weights,
            default_rel="google/siglip2-base-patch16-512",
            env_var="MEMSTRATA_SIGLIP2_WEIGHTS",
        )
        return SigLIP2Embedding(model_id=model_id)
    if backend in {"dinov3", "dinov2"}:
        from memstrata.encoders.ssl.dinov3 import DinoV3Embedding
        model_id = model or "facebook/dinov3-vitb16-pretrain-lvd1689m"
        if weights and Path(weights).exists():
            model_id = weights
        return DinoV3Embedding(model_id=str(model_id))
    if backend == "insightface":
        from memstrata.encoders.face.arcface import ArcFaceEmbedding
        return ArcFaceEmbedding(
            pack=model or "buffalo_l",
            root=weights if weights else None,
        )
    if backend == "vpr":
        from memstrata.encoders.place.vpr import VprEmbedding
        return VprEmbedding(
            method=model or "megaloc",
            weights=weights if weights else None,
        )
    return HashEmbedding()


def _construct_text_embedding(
    backend: str, model: str | None, weights: str | None
) -> TextEmbeddingModel:
    if backend in {"qwen3", "qwen3_embedding", "qwen3-embedding"}:
        endpoint = os.environ.get("MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT", "").strip()
        if endpoint:
            return OpenAITextEmbedding(
                endpoint=endpoint,
                model=model or os.environ.get("MEMSTRATA_QWEN3_EMBEDDING_MODEL", "Qwen3-Embedding-4B"),
                api_key=os.environ.get("MEMSTRATA_QWEN3_EMBEDDING_API_KEY", ""),
            )
        model_id = _resolve_local_model(
            provider="qwen3_embedding",
            model=model,
            weights=weights,
            default_rel="Qwen/Qwen3-Embedding-4B",
            env_var="MEMSTRATA_QWEN3_EMBEDDING_WEIGHTS",
        )
        return Qwen3TextEmbedding(model_id=model_id)
    if backend == "siglip2":
        model_id = _resolve_local_model(
            provider="siglip2",
            model=model,
            weights=weights,
            default_rel="google/siglip2-base-patch16-512",
            env_var="MEMSTRATA_SIGLIP2_WEIGHTS",
        )
        return SigLIP2Embedding(model_id=model_id)
    return HashEmbedding()


def build_image_embedding(
    *,
    provider: str = "hash",
    model: str | None = None,
    weights: str | None = None,
    use_cache: bool = True,
    **_: object,
) -> EmbeddingModel:
    """Construct (and by default process-cache) an image-embedding backend.

    Heavy backends (dinov3 / insightface / vpr) load weights once per
    ``(provider, model, weights)`` key so repeatedly constructing pipelines does
    not reload the same model. Pass ``use_cache=False`` to force a fresh build.
    """

    backend = (provider or "hash").lower()
    key = (backend, model, weights)
    if use_cache:
        cached = _EMBEDDING_CACHE.get(key)
        if cached is not None:
            return cached
    instance = _construct_image_embedding(backend, model, weights)
    if use_cache:
        _EMBEDDING_CACHE[key] = instance
    return instance


def build_text_embedding(
    *,
    provider: str = "hash",
    model: str | None = None,
    weights: str | None = None,
    use_cache: bool = True,
    **_: object,
) -> TextEmbeddingModel:
    """Construct (and by default process-cache) a text-embedding backend."""

    backend = (provider or "hash").lower()
    key = (f"text:{backend}", model, weights)
    if use_cache:
        cached = _EMBEDDING_CACHE.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
    instance = _construct_text_embedding(backend, model, weights)
    if use_cache:
        _EMBEDDING_CACHE[key] = instance  # type: ignore[assignment]
    return instance


class HashEmbedding:
    """Deterministic, dependency-free fallback embedding."""

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = "hash-fallback"

    def embed_image(self, image: str | Path) -> Vector:
        # An empty/blank crop_path (e.g. video-free "text-gold" where entities carry
        # references but no pixels) must NOT be read: ``Path("")`` resolves to ``.``
        # (the cwd, a directory) and ``read_bytes()`` would raise IsADirectoryError.
        # Fall back to hashing the string handle so each distinct reference stays
        # deterministic and distinguishable.
        text = str(image)
        path = Path(text) if text else None
        seed = path.read_bytes() if (path is not None and path.is_file()) else text.encode("utf-8")
        components: Vector = []
        counter = 0
        while len(components) < self.dim:
            digest = sha256(seed + counter.to_bytes(4, "big")).digest()
            for index in range(0, len(digest), 2):
                if len(components) >= self.dim:
                    break
                raw = int.from_bytes(digest[index : index + 2], "big")
                components.append((raw / 65535.0) * 2.0 - 1.0)
            counter += 1
        return l2_normalize(components)

    def embed_text(self, text: str) -> Vector:
        return self.embed_image(f"text:{text}")

    def embed_query(self, text: str) -> Vector:
        return self.embed_text(text)

    def embed_doc(self, text: str) -> Vector:
        return self.embed_text(text)


class RoleRoutedEmbedding:
    """Routes each crop to the embedding backend appropriate for the asset's *kind*."""

    name = "role-routed"

    def __init__(
        self,
        *,
        general: EmbeddingModel,
        face: EmbeddingModel | None = None,
        location: EmbeddingModel | None = None,
        face_kinds: tuple[str, ...] = ("character",),
        location_kinds: tuple[str, ...] = ("location",),
    ) -> None:
        self.general = general
        self.face = face
        self.location = location
        self.face_kinds = {str(k) for k in face_kinds}
        self.location_kinds = {str(k) for k in location_kinds}
        self.dim = getattr(general, "dim", 0)

    def route(self, kind: str | None) -> tuple[str, EmbeddingModel]:
        val = getattr(kind, "value", kind)
        key = str(val) if val is not None else ""
        if self.face is not None and key in self.face_kinds:
            return "face", self.face
        if self.location is not None and key in self.location_kinds:
            return "location", self.location
        return "general", self.general

    def route_name(self, kind: str | None) -> str:
        return self.route(kind)[0]

    def embed_with_route(self, image: str | Path, kind: str | None) -> tuple[Vector, str]:
        route, model = self.route(kind)
        if route == "general":
            return model.embed_image(image), model.name
        try:
            return model.embed_image(image), model.name
        except Exception:  # noqa: BLE001
            return self.general.embed_image(image), self.general.name

    def embed_for_kind(self, image: str | Path, kind: str | None) -> Vector:
        return self.embed_with_route(image, kind)[0]

    def embed_image(self, image: str | Path) -> Vector:
        return self.general.embed_image(image)


def build_role_routed_embedding_from_env() -> RoleRoutedEmbedding:
    """Build opt-in production routes while keeping the offline default deterministic."""
    def _build(prefix: str, default: str) -> EmbeddingModel:
        return build_image_embedding(
            provider=os.environ.get(f"{prefix}_PROVIDER", default),
            model=os.environ.get(f"{prefix}_MODEL"),
            weights=os.environ.get(f"{prefix}_WEIGHTS"),
        )

    general = _build("MEMSTRATA_GENERAL_EMBEDDER", "hash")
    face_provider = os.environ.get("MEMSTRATA_FACE_EMBEDDER_PROVIDER", "").strip()
    place_provider = os.environ.get("MEMSTRATA_PLACE_EMBEDDER_PROVIDER", "").strip()
    return RoleRoutedEmbedding(
        general=general,
        face=_build("MEMSTRATA_FACE_EMBEDDER", face_provider) if face_provider else None,
        location=_build("MEMSTRATA_PLACE_EMBEDDER", place_provider) if place_provider else None,
    )
