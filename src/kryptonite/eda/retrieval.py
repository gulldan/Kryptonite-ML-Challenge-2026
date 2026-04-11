"""Local retrieval-baseline analysis from precomputed embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    summary: dict[str, Any]
    per_query: pl.DataFrame
    worst_queries: pl.DataFrame
    confused_speaker_pairs: pl.DataFrame


def evaluate_retrieval_embeddings(
    embeddings: np.ndarray,
    labels: list[str],
    filepaths: list[str],
    *,
    top_k: int = 10,
    normalize: bool = True,
) -> RetrievalEvaluation:
    """Evaluate same-pool speaker retrieval with self-match exclusion."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2D matrix.")
    if len(labels) != matrix.shape[0]:
        raise ValueError("labels length must match embedding rows.")
    if len(filepaths) != matrix.shape[0]:
        raise ValueError("filepaths length must match embedding rows.")
    if matrix.shape[0] < 2:
        raise ValueError("At least two embeddings are required for retrieval evaluation.")

    effective_k = min(top_k, matrix.shape[0] - 1)
    scores = cosine_score_matrix(matrix, matrix, normalize=normalize)
    np.fill_diagonal(scores, -1_000_000_000.0)
    indices, top_scores = rank_cosine_scores(scores, top_k=effective_k)
    label_array = np.asarray(labels, dtype=object)
    ranked_labels = label_array[indices]
    hits = ranked_labels == label_array[:, None]
    precision = hits.mean(axis=1)
    top1_correct = hits[:, 0]
    first_correct_rank = _first_correct_rank(hits)

    per_query = pl.DataFrame(
        {
            "query_index": np.arange(matrix.shape[0], dtype=np.int64),
            "filepath": filepaths,
            "speaker_id": labels,
            f"precision_at_{effective_k}": precision,
            "top1_correct": top1_correct,
            "first_correct_rank": first_correct_rank,
            "top_indices": [row.tolist() for row in indices],
            "top_scores": [row.tolist() for row in top_scores],
            "top_speaker_ids": [row.tolist() for row in ranked_labels],
        }
    )
    confused_pairs = _build_confused_pairs(labels=label_array, indices=indices, hits=hits)
    same_cosines, different_cosines = _sample_cosine_distributions(scores, label_array)
    summary = {
        "query_count": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "top_k": int(effective_k),
        f"precision_at_{effective_k}": round(float(np.mean(precision)), 8),
        "top1_accuracy": round(float(np.mean(top1_correct)), 8),
        "mean_first_correct_rank": (
            round(float(np.mean(first_correct_rank[first_correct_rank > 0])), 6)
            if np.any(first_correct_rank > 0)
            else None
        ),
        "same_speaker_median_cos": _median_or_none(same_cosines),
        "different_speaker_95p_cos": _quantile_or_none(different_cosines, 0.95),
    }
    if (
        summary["same_speaker_median_cos"] is not None
        and summary["different_speaker_95p_cos"] is not None
    ):
        summary["margin_same_median_minus_diff95"] = round(
            float(summary["same_speaker_median_cos"]) - float(summary["different_speaker_95p_cos"]),
            8,
        )
    return RetrievalEvaluation(
        summary=summary,
        per_query=per_query,
        worst_queries=per_query.sort(
            [f"precision_at_{effective_k}", "first_correct_rank"],
            descending=[False, True],
        ).head(100),
        confused_speaker_pairs=confused_pairs,
    )


def _first_correct_rank(hits: np.ndarray) -> np.ndarray:
    ranks = np.zeros(hits.shape[0], dtype=np.int64)
    for row_index, row in enumerate(hits):
        positions = np.flatnonzero(row)
        if positions.size:
            ranks[row_index] = int(positions[0]) + 1
    return ranks


def _l2_normalize(values: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = np.where(norms[:, 0] <= epsilon)[0]
    if zero_rows.size:
        raise ValueError(f"embeddings contains a zero-norm row at index {int(zero_rows[0])}.")
    return values / norms


def cosine_score_matrix(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    queries = np.asarray(query_embeddings, dtype=np.float64)
    references = np.asarray(reference_embeddings, dtype=np.float64)
    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError("query_embeddings and reference_embeddings must be 2D matrices.")
    if queries.shape[1] != references.shape[1]:
        raise ValueError("query_embeddings and reference_embeddings dimensions must match.")
    if normalize:
        queries = _l2_normalize(queries)
        references = _l2_normalize(references)
    return np.clip(queries @ references.T, -1.0, 1.0)


def rank_cosine_scores(score_matrix: np.ndarray, *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("score_matrix must be a non-empty 2D matrix.")
    effective_top_k = min(top_k, matrix.shape[1])
    candidate_indices = np.argpartition(matrix, kth=-effective_top_k, axis=1)[:, -effective_top_k:]
    candidate_scores = matrix[np.arange(matrix.shape[0])[:, None], candidate_indices]
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    sorted_indices = candidate_indices[np.arange(matrix.shape[0])[:, None], order]
    sorted_scores = candidate_scores[np.arange(matrix.shape[0])[:, None], order]
    return sorted_indices, sorted_scores


def _build_confused_pairs(
    *,
    labels: np.ndarray,
    indices: np.ndarray,
    hits: np.ndarray,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_index in range(indices.shape[0]):
        query_speaker = str(labels[query_index])
        for neighbour_index, is_hit in zip(indices[query_index], hits[query_index], strict=True):
            if bool(is_hit):
                continue
            rows.append(
                {
                    "query_speaker_id": query_speaker,
                    "neighbour_speaker_id": str(labels[int(neighbour_index)]),
                    "count": 1,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "query_speaker_id": pl.Utf8,
                "neighbour_speaker_id": pl.Utf8,
                "count": pl.Int64,
            }
        )
    return (
        pl.DataFrame(rows)
        .group_by(["query_speaker_id", "neighbour_speaker_id"])
        .agg(pl.col("count").sum())
        .sort("count", descending=True)
    )


def _sample_cosine_distributions(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_pairs: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    upper = np.triu_indices(scores.shape[0], k=1)
    same_mask = labels[upper[0]] == labels[upper[1]]
    pair_scores = scores[upper]
    if pair_scores.size > max_pairs:
        rng = np.random.default_rng(2026)
        sample = rng.choice(pair_scores.size, size=max_pairs, replace=False)
        pair_scores = pair_scores[sample]
        same_mask = same_mask[sample]
    return pair_scores[same_mask], pair_scores[~same_mask]


def _median_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else round(float(np.median(values)), 8)


def _quantile_or_none(values: np.ndarray, quantile: float) -> float | None:
    return None if values.size == 0 else round(float(np.quantile(values, quantile)), 8)
