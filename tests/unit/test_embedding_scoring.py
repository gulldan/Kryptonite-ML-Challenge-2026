from __future__ import annotations

import math

import pytest

from kryptonite.models import (
    average_normalized_embeddings,
    cosine_score_matrix,
    cosine_score_pairs,
    l2_normalize_embeddings,
    rank_cosine_scores,
)


def test_l2_normalize_embeddings_rejects_zero_norm_rows() -> None:
    with pytest.raises(ValueError, match="zero-norm embedding"):
        l2_normalize_embeddings([[1.0, 0.0], [0.0, 0.0]])


def test_average_normalized_embeddings_renormalizes_the_mean() -> None:
    averaged = average_normalized_embeddings([[2.0, 0.0], [0.0, 3.0]])

    expected = 1.0 / math.sqrt(2.0)
    assert averaged.tolist() == pytest.approx([expected, expected])


def test_cosine_score_pairs_normalizes_unnormalized_inputs() -> None:
    scores = cosine_score_pairs(
        [[3.0, 0.0], [0.0, 5.0]],
        [[9.0, 0.0], [0.0, -2.0]],
        normalize=True,
    )

    assert scores.tolist() == pytest.approx([1.0, -1.0])


def test_cosine_score_matrix_and_ranking_return_sorted_top_matches() -> None:
    score_matrix = cosine_score_matrix(
        [[1.0, 0.0]],
        [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]],
        normalize=True,
    )
    indices, scores = rank_cosine_scores(score_matrix, top_k=2)

    assert score_matrix.shape == (1, 3)
    assert score_matrix[0].tolist() == pytest.approx([1.0, 0.70710678, -1.0])
    assert indices.tolist() == [[0, 1]]
    assert scores[0].tolist() == pytest.approx([1.0, 0.70710678])
