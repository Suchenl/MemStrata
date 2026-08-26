# Embedding Deduplication Skill

This skill provides model-free deduplication over embedding vectors, including identity matching and quality-seeded non-redundant selection.

## Features

- **Identity Matching**: Matches a query vector against a list of candidate vectors using cosine similarity with a configurable threshold.
- **Non-Redundant Selection**: Performs greedy farthest-point selection seeded by a quality score to select the most diverse and high-quality subset of vectors.

## Usage

```python
from memstrata.skills.embedding_deduplication import match_to_existing, select_non_redundant

# Match query vector to existing candidates
best_id, score = match_to_existing(query_vector, candidates=[("id1", vec1), ("id2", vec2)])

# Select non-redundant vectors
selected_indices = select_non_redundant(
    vectors=[vec1, vec2, vec3, vec4],
    max_keep=2,
    min_distance=0.15,
    quality=[0.9, 0.8, 0.95, 0.7]
)
```
