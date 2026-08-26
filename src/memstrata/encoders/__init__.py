"""Type-routed image encoders for Step 3 (face / place / general SSL)."""

from memstrata.encoders.base import (
    EmbeddingModel,
    HashEmbedding,
    RoleRoutedEmbedding,
    TextEmbeddingModel,
    Vector,
    build_image_embedding,
    build_role_routed_embedding_from_env,
    build_text_embedding,
    cosine_distance,
    cosine_similarity,
    l2_normalize,
)

__all__ = [
    "EmbeddingModel",
    "HashEmbedding",
    "RoleRoutedEmbedding",
    "TextEmbeddingModel",
    "Vector",
    "build_image_embedding",
    "build_role_routed_embedding_from_env",
    "build_text_embedding",
    "cosine_distance",
    "cosine_similarity",
    "l2_normalize",
]
