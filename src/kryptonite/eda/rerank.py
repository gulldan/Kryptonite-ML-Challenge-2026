"""Hubness-aware reranking helpers for retrieval EDA."""

from __future__ import annotations

import numpy as np


def reciprocal_local_rerank(
    *,
    query_indices: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    gallery_topk: np.ndarray,
    top_k: int,
    reciprocal_top: int = 20,
    reciprocal_bonus: float = 0.03,
    density_z: np.ndarray | None = None,
    density_penalty: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    out_i = np.empty((indices.shape[0], top_k), dtype=np.int64)
    out_s = np.empty((indices.shape[0], top_k), dtype=np.float32)
    for row, query_index in enumerate(query_indices):
        candidates = indices[row]
        reciprocal = np.asarray(
            [
                query_index in gallery_topk[int(candidate), :reciprocal_top]
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        adjusted = scores[row] + reciprocal_bonus * reciprocal
        if density_z is not None:
            adjusted = adjusted - density_penalty * density_z[candidates]
        order = np.argsort(adjusted)[::-1][:top_k]
        out_i[row] = candidates[order]
        out_s[row] = scores[row, order]
    return out_i, out_s


def density_zscore(gallery_scores: np.ndarray, *, top_n: int = 20) -> np.ndarray:
    density = gallery_scores[:, :top_n].mean(axis=1)
    return (density - density.mean()) / max(float(density.std()), 1e-6)


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(values.astype(np.float64))
    n = sorted_values.size
    total = float(sorted_values.sum())
    if total == 0.0:
        return 0.0
    return float((2 * np.arange(1, n + 1) @ sorted_values) / (n * total) - (n + 1) / n)
