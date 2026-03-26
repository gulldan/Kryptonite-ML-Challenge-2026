"""Embedding normalization and cosine scoring helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_EMBEDDING_NORM_EPSILON = 1e-12


def ensure_embedding_matrix(
    values: Any,
    *,
    field_name: str,
) -> np.ndarray:
    """Coerce a single embedding or a batch of embeddings into a finite 2D matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        if matrix.size == 0:
            raise ValueError(f"{field_name} must not be empty.")
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim != 2:
        raise ValueError(
            f"{field_name} must be a 1D embedding vector or a 2D embedding matrix; "
            f"got shape {tuple(matrix.shape)}."
        )

    if matrix.shape[0] == 0:
        raise ValueError(f"{field_name} must contain at least one embedding row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{field_name} must contain embeddings with at least one dimension.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{field_name} must contain only finite numeric values.")
    return matrix


def l2_normalize_embeddings(
    values: Any,
    *,
    field_name: str = "embeddings",
    epsilon: float = DEFAULT_EMBEDDING_NORM_EPSILON,
) -> np.ndarray:
    """Apply row-wise L2 normalization to one or more embeddings."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    matrix = ensure_embedding_matrix(values, field_name=field_name)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = np.where(norms[:, 0] <= epsilon)[0]
    if zero_rows.size:
        row_index = int(zero_rows[0]) + 1
        raise ValueError(f"{field_name} contains a zero-norm embedding at row {row_index}.")
    return matrix / norms


def average_normalized_embeddings(
    values: Any,
    *,
    field_name: str = "embeddings",
) -> np.ndarray:
    """Normalize each embedding row, mean-pool them, then renormalize the result."""

    normalized = l2_normalize_embeddings(values, field_name=field_name)
    pooled = normalized.mean(axis=0, keepdims=True)
    return l2_normalize_embeddings(
        pooled,
        field_name=f"{field_name}_pooled",
    )[0]


def cosine_score_pairs(
    left_embeddings: Any,
    right_embeddings: Any,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Compute one cosine score per aligned embedding pair."""

    left = ensure_embedding_matrix(left_embeddings, field_name="left_embeddings")
    right = ensure_embedding_matrix(right_embeddings, field_name="right_embeddings")
    if left.shape != right.shape:
        raise ValueError(
            "left_embeddings and right_embeddings must have the same shape for pairwise scoring; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )

    if normalize:
        left = l2_normalize_embeddings(left, field_name="left_embeddings")
        right = l2_normalize_embeddings(right, field_name="right_embeddings")

    return np.clip(np.sum(left * right, axis=1), -1.0, 1.0)


def cosine_score_matrix(
    query_embeddings: Any,
    reference_embeddings: Any,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Compute a cosine similarity matrix for query-to-reference scoring."""

    queries = ensure_embedding_matrix(query_embeddings, field_name="query_embeddings")
    references = ensure_embedding_matrix(reference_embeddings, field_name="reference_embeddings")
    if queries.shape[1] != references.shape[1]:
        raise ValueError(
            "query_embeddings and reference_embeddings must have the same embedding dimension; "
            f"got {queries.shape[1]} and {references.shape[1]}."
        )

    if normalize:
        queries = l2_normalize_embeddings(queries, field_name="query_embeddings")
        references = l2_normalize_embeddings(references, field_name="reference_embeddings")

    return np.clip(queries @ references.T, -1.0, 1.0)


def rank_cosine_scores(
    score_matrix: Any,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-query top-k column indices and scores from a 2D score matrix."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"score_matrix must be 2D; got shape {tuple(matrix.shape)}.")
    if matrix.shape[0] == 0:
        raise ValueError("score_matrix must contain at least one query row.")
    if matrix.shape[1] == 0:
        raise ValueError("score_matrix must contain at least one reference column.")
    if not np.isfinite(matrix).all():
        raise ValueError("score_matrix must contain only finite numeric values.")

    effective_top_k = min(top_k, matrix.shape[1])
    candidate_indices = np.argpartition(matrix, kth=-effective_top_k, axis=1)[:, -effective_top_k:]
    candidate_scores = matrix[np.arange(matrix.shape[0])[:, None], candidate_indices]
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    sorted_indices = candidate_indices[np.arange(matrix.shape[0])[:, None], order]
    sorted_scores = candidate_scores[np.arange(matrix.shape[0])[:, None], order]
    return sorted_indices, sorted_scores


__all__ = [
    "DEFAULT_EMBEDDING_NORM_EPSILON",
    "average_normalized_embeddings",
    "cosine_score_matrix",
    "cosine_score_pairs",
    "ensure_embedding_matrix",
    "l2_normalize_embeddings",
    "rank_cosine_scores",
]
