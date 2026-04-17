"""Backbone-neighbor fusion helpers for transductive retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RankScoreFusionConfig:
    """Configuration for row-wise top-k fusion across two backbone graphs."""

    experiment_id: str
    left_name: str = "eres2netv2"
    right_name: str = "campp"
    left_weight: float = 0.75
    right_weight: float = 0.25
    source_top_k: int = 200
    output_top_k: int = 100
    rank_weight: float = 1.0
    score_z_weight: float = 0.15
    min_score_z: float = -3.0
    max_score_z: float = 3.0


def fuse_topk_rank_score(
    *,
    left_indices: np.ndarray,
    left_scores: np.ndarray,
    right_indices: np.ndarray,
    right_scores: np.ndarray,
    config: RankScoreFusionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse two sorted top-k graphs with rank and row-wise robust score features."""

    if left_indices.shape != left_scores.shape:
        raise ValueError("left_indices and left_scores must have the same shape")
    if right_indices.shape != right_scores.shape:
        raise ValueError("right_indices and right_scores must have the same shape")
    if left_indices.shape != right_indices.shape:
        raise ValueError("left and right top-k arrays must have the same shape")
    if config.source_top_k > left_indices.shape[1]:
        raise ValueError("source_top_k is larger than the cached top-k arrays")
    if config.output_top_k > 2 * config.source_top_k:
        raise ValueError("output_top_k cannot exceed the union of both source top-k sets")

    started = time.perf_counter()
    row_count = left_indices.shape[0]
    fused_indices = np.empty((row_count, config.output_top_k), dtype=np.int64)
    fused_scores = np.empty((row_count, config.output_top_k), dtype=np.float32)
    overlap_counts = np.zeros(row_count, dtype=np.int32)
    candidate_counts = np.zeros(row_count, dtype=np.int32)

    for row in range(row_count):
        row_scores: dict[int, float] = {}
        left_candidates = left_indices[row, : config.source_top_k]
        right_candidates = right_indices[row, : config.source_top_k]
        overlap_counts[row] = int(
            len(set(left_candidates.tolist()).intersection(right_candidates.tolist()))
        )
        _add_source_scores(
            row_scores=row_scores,
            row_index=row,
            candidates=left_candidates,
            scores=left_scores[row, : config.source_top_k],
            source_weight=config.left_weight,
            rank_weight=config.rank_weight,
            score_z_weight=config.score_z_weight,
            min_score_z=config.min_score_z,
            max_score_z=config.max_score_z,
        )
        _add_source_scores(
            row_scores=row_scores,
            row_index=row,
            candidates=right_candidates,
            scores=right_scores[row, : config.source_top_k],
            source_weight=config.right_weight,
            rank_weight=config.rank_weight,
            score_z_weight=config.score_z_weight,
            min_score_z=config.min_score_z,
            max_score_z=config.max_score_z,
        )
        if len(row_scores) < config.output_top_k:
            raise ValueError(
                f"row {row} has only {len(row_scores)} fused candidates; need {config.output_top_k}"
            )
        candidate_counts[row] = len(row_scores)
        best = sorted(row_scores.items(), key=lambda item: (-item[1], item[0]))[
            : config.output_top_k
        ]
        fused_indices[row] = np.asarray([candidate for candidate, _ in best], dtype=np.int64)
        fused_scores[row] = np.asarray([score for _, score in best], dtype=np.float32)

    elapsed_s = time.perf_counter() - started
    meta: dict[str, Any] = {
        "fusion_experiment_id": config.experiment_id,
        "fusion_left_name": config.left_name,
        "fusion_right_name": config.right_name,
        "fusion_left_weight": config.left_weight,
        "fusion_right_weight": config.right_weight,
        "fusion_source_top_k": config.source_top_k,
        "fusion_output_top_k": config.output_top_k,
        "fusion_rank_weight": config.rank_weight,
        "fusion_score_z_weight": config.score_z_weight,
        "fusion_elapsed_s": round(elapsed_s, 6),
        "fusion_candidate_count_p50": float(np.quantile(candidate_counts, 0.50)),
        "fusion_candidate_count_p95": float(np.quantile(candidate_counts, 0.95)),
        "fusion_source_overlap_p50": float(np.quantile(overlap_counts, 0.50)),
        "fusion_source_overlap_p95": float(np.quantile(overlap_counts, 0.95)),
    }
    return fused_indices, fused_scores, meta


def _add_source_scores(
    *,
    row_scores: dict[int, float],
    row_index: int,
    candidates: np.ndarray,
    scores: np.ndarray,
    source_weight: float,
    rank_weight: float,
    score_z_weight: float,
    min_score_z: float,
    max_score_z: float,
) -> None:
    score_z = _robust_z(scores, min_value=min_score_z, max_value=max_score_z)
    denominator = max(len(candidates) - 1, 1)
    for rank, candidate in enumerate(candidates.tolist()):
        candidate_int = int(candidate)
        if candidate_int == row_index:
            continue
        rank_score = 1.0 - (rank / denominator)
        source_score = (rank_weight * rank_score) + (score_z_weight * float(score_z[rank]))
        combined = source_weight * max(source_score, 0.0)
        row_scores[candidate_int] = row_scores.get(candidate_int, 0.0) + combined


def _robust_z(scores: np.ndarray, *, min_value: float, max_value: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)
    return np.clip((values - median) / scale, min_value, max_value).astype(np.float32)
