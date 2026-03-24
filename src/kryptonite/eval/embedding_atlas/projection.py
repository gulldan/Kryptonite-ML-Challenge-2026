"""Projection and neighbor-lookup helpers for embedding atlases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ProjectionMethod


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    coordinates: np.ndarray
    explained_variance_ratio_2d: float


def project_embeddings(
    embeddings: np.ndarray,
    *,
    method: ProjectionMethod,
) -> ProjectionResult:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if method == "cosine_pca":
        matrix = _l2_normalize_rows(matrix)

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2:
        padding = np.zeros((matrix.shape[0], 2), dtype=np.float64)
        padding[:, 0] = centered[:, 0] if centered.shape[1] else 0.0
        return ProjectionResult(coordinates=padding, explained_variance_ratio_2d=1.0)

    coordinates = centered @ vh[:2].T
    explained = singular_values * singular_values
    total_explained = float(explained.sum())
    explained_ratio = float(explained[:2].sum() / total_explained) if total_explained > 0.0 else 0.0
    return ProjectionResult(
        coordinates=coordinates.astype(np.float64, copy=False),
        explained_variance_ratio_2d=round(explained_ratio, 6),
    )


def compute_cosine_neighbors(
    embeddings: np.ndarray,
    *,
    top_k: int,
    block_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float64))
    row_count = matrix.shape[0]
    effective_k = min(top_k, max(0, row_count - 1))
    if effective_k == 0:
        return (
            np.empty((row_count, 0), dtype=np.int64),
            np.empty((row_count, 0), dtype=np.float64),
        )

    neighbor_indices = np.empty((row_count, effective_k), dtype=np.int64)
    neighbor_distances = np.empty((row_count, effective_k), dtype=np.float64)
    for start in range(0, row_count, block_size):
        end = min(start + block_size, row_count)
        similarities = matrix[start:end] @ matrix.T
        diagonal_rows = np.arange(end - start)
        similarities[diagonal_rows, diagonal_rows + start] = -np.inf

        candidate_indices = np.argpartition(similarities, kth=-effective_k, axis=1)[
            :, -effective_k:
        ]
        candidate_scores = similarities[np.arange(end - start)[:, None], candidate_indices]
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        sorted_indices = candidate_indices[np.arange(end - start)[:, None], order]
        sorted_scores = candidate_scores[np.arange(end - start)[:, None], order]

        neighbor_indices[start:end] = sorted_indices
        neighbor_distances[start:end] = 1.0 - sorted_scores
    return neighbor_indices, neighbor_distances


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe_norms


__all__ = ["ProjectionResult", "compute_cosine_neighbors", "project_embeddings"]
