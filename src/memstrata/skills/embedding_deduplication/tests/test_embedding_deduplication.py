"""Tests for model-free embedding deduplication."""

from unittest import TestCase

from memstrata.skills.embedding_deduplication import (
    match_to_existing,
    select_non_redundant,
)


class EmbeddingDeduplicationTest(TestCase):
    def test_match_to_existing_finds_closest_above_threshold(self) -> None:
        # Simple unit vectors
        candidates = [
            ("id1", [1.0, 0.0, 0.0]),
            ("id2", [0.0, 1.0, 0.0]),
        ]
        # Query is very close to id1
        query = [0.99, 0.1, 0.0]
        best_id, score = match_to_existing(query, candidates, threshold=0.8)
        self.assertEqual(best_id, "id1")
        self.assertGreater(score, 0.9)

        # Query is close to id1 but below threshold
        best_id_low, score_low = match_to_existing(query, candidates, threshold=0.999)
        self.assertIsNone(best_id_low)

    def test_select_non_redundant_keeps_only_new_information(self) -> None:
        # Three vectors: v1 and v2 are identical, v3 is orthogonal
        vectors = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        # Seed quality: v1 (index 0) has quality 1.0, v2 (index 1) has 0.5, v3 (index 2) has 0.9.
        # Farthest-point selection should keep index 0 (best quality) and index 2 (orthogonal),
        # but discard index 1 (duplicate of index 0).
        keep = select_non_redundant(vectors, max_keep=5, min_distance=0.15, quality=[1.0, 0.5, 0.9])
        self.assertEqual(keep, [0, 2])
