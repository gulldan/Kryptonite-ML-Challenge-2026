"""Class-aware retrieval utilities for transductive public inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ClassFirstConfig:
    top_k: int = 10
    min_class_candidates: int = 6
    class_fallback_k: int = 3
    max_class_candidates: int = 600
    embedding_weight: float = 0.50
    class_overlap_weight: float = 0.35
    same_top1_bonus: float = 0.15
    fallback_rank_bonus: float = 0.03
    bucket_backfill: bool = True


@dataclass(frozen=True, slots=True)
class ClassAdjustedScoreConfig:
    class_overlap_top_k: int = 3
    class_overlap_weight: float = 0.08
    same_top1_bonus: float = 0.02
    same_query_topk_bonus: float = 0.01


def class_first_rerank(
    *,
    embeddings: np.ndarray,
    top_class_indices: np.ndarray,
    top_class_probs: np.ndarray,
    fallback_indices: np.ndarray,
    fallback_scores: np.ndarray,
    config: ClassFirstConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Prefer candidates assigned to the same classifier speaker bucket."""

    _validate_rerank_inputs(
        embeddings=embeddings,
        top_class_indices=top_class_indices,
        top_class_probs=top_class_probs,
        fallback_indices=fallback_indices,
        fallback_scores=fallback_scores,
        config=config,
    )
    buckets = _build_class_buckets(top_class_indices[:, 0], top_class_probs[:, 0])
    class_counts = np.asarray([len(bucket) for bucket in buckets], dtype=np.int64)
    out_i = np.empty((embeddings.shape[0], config.top_k), dtype=np.int64)
    out_s = np.empty((embeddings.shape[0], config.top_k), dtype=np.float32)
    class_used = np.zeros(embeddings.shape[0], dtype=bool)
    class_candidate_counts = np.zeros(embeddings.shape[0], dtype=np.int32)

    for row in range(embeddings.shape[0]):
        candidates = _class_candidates_for_row(
            row=row,
            embeddings=embeddings,
            top_class_indices=top_class_indices,
            fallback_row=fallback_indices[row],
            buckets=buckets,
            config=config,
        )
        class_candidate_counts[row] = len(candidates)
        selected: list[int] = []
        selected_scores: list[float] = []
        if len(candidates) >= config.min_class_candidates:
            ranked_candidates, ranked_scores = _rank_class_candidates(
                row=row,
                candidates=candidates,
                embeddings=embeddings,
                top_class_indices=top_class_indices,
                top_class_probs=top_class_probs,
                fallback_row=fallback_indices[row],
                config=config,
            )
            for candidate, score in zip(ranked_candidates, ranked_scores, strict=True):
                selected.append(int(candidate))
                selected_scores.append(float(score))
                if len(selected) == config.top_k:
                    break
            class_used[row] = len(selected) > 0

        seen = set(selected)
        if len(selected) < config.top_k:
            for candidate, score in zip(fallback_indices[row], fallback_scores[row], strict=True):
                candidate_int = int(candidate)
                if candidate_int == row or candidate_int in seen:
                    continue
                selected.append(candidate_int)
                selected_scores.append(float(score))
                seen.add(candidate_int)
                if len(selected) == config.top_k:
                    break
        out_i[row] = np.asarray(selected, dtype=np.int64)
        out_s[row] = np.asarray(selected_scores, dtype=np.float32)

    non_empty_counts = class_counts[class_counts > 0]
    return (
        out_i,
        out_s,
        {
            "class_used_share": float(class_used.mean()),
            "class_candidate_count_p50": float(np.quantile(class_candidate_counts, 0.50)),
            "class_candidate_count_p95": float(np.quantile(class_candidate_counts, 0.95)),
            "top1_class_count": int(len(non_empty_counts)),
            "top1_class_size_p50": float(np.quantile(non_empty_counts, 0.50)),
            "top1_class_size_p95": float(np.quantile(non_empty_counts, 0.95)),
            "top1_class_size_p99": float(np.quantile(non_empty_counts, 0.99)),
            "top1_class_size_max": int(non_empty_counts.max()),
        },
    )


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def class_adjusted_topk(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    top_class_indices: np.ndarray,
    top_class_probs: np.ndarray,
    config: ClassAdjustedScoreConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Re-rank a fixed top-k cache with soft classifier posterior evidence."""

    if indices.shape != scores.shape:
        raise ValueError("indices and scores must have the same shape")
    if top_class_indices.shape != top_class_probs.shape:
        raise ValueError("top_class_indices and top_class_probs must have the same shape")
    if indices.shape[0] != top_class_indices.shape[0]:
        raise ValueError("top-k cache and class cache must have the same row count")
    if config.class_overlap_top_k > top_class_indices.shape[1]:
        raise ValueError("class_overlap_top_k is larger than the class cache width")

    overlap = topk_class_overlap(
        indices=indices,
        top_class_indices=top_class_indices,
        top_class_probs=top_class_probs,
        class_top_k=config.class_overlap_top_k,
    )
    candidate_top1 = top_class_indices[indices, 0]
    query_top = top_class_indices[:, : config.class_overlap_top_k]
    same_top1 = candidate_top1 == top_class_indices[:, 0][:, None]
    same_query_topk = np.zeros(indices.shape, dtype=bool)
    for rank in range(config.class_overlap_top_k):
        same_query_topk |= candidate_top1 == query_top[:, rank][:, None]

    adjusted = (
        scores
        + config.class_overlap_weight * overlap
        + config.same_top1_bonus * same_top1.astype(np.float32)
        + config.same_query_topk_bonus * same_query_topk.astype(np.float32)
    ).astype(np.float32, copy=False)
    order = np.argsort(adjusted, axis=1)[:, ::-1]
    sorted_indices = np.take_along_axis(indices, order, axis=1)
    sorted_scores = np.take_along_axis(adjusted, order, axis=1)
    moved = sorted_indices[:, 0] != indices[:, 0]
    return (
        sorted_indices,
        sorted_scores,
        {
            "class_overlap_top_k": config.class_overlap_top_k,
            "class_overlap_weight": config.class_overlap_weight,
            "same_top1_bonus": config.same_top1_bonus,
            "same_query_topk_bonus": config.same_query_topk_bonus,
            "class_overlap_mean": float(overlap.mean()),
            "class_overlap_p95": float(np.quantile(overlap, 0.95)),
            "same_top1_edge_share": float(same_top1.mean()),
            "same_query_topk_edge_share": float(same_query_topk.mean()),
            "adjusted_top1_changed_share": float(moved.mean()),
        },
    )


def topk_class_overlap(
    *,
    indices: np.ndarray,
    top_class_indices: np.ndarray,
    top_class_probs: np.ndarray,
    class_top_k: int,
) -> np.ndarray:
    """Return sparse posterior dot-product for each row and top-k candidate."""

    if class_top_k <= 0:
        raise ValueError("class_top_k must be positive")
    overlap = np.zeros(indices.shape, dtype=np.float32)
    for query_rank in range(class_top_k):
        query_classes = top_class_indices[:, query_rank][:, None]
        query_probs = top_class_probs[:, query_rank][:, None]
        for candidate_rank in range(class_top_k):
            candidate_classes = top_class_indices[indices, candidate_rank]
            candidate_probs = top_class_probs[indices, candidate_rank]
            overlap += (candidate_classes == query_classes) * candidate_probs * query_probs
    return overlap


def _validate_rerank_inputs(
    *,
    embeddings: np.ndarray,
    top_class_indices: np.ndarray,
    top_class_probs: np.ndarray,
    fallback_indices: np.ndarray,
    fallback_scores: np.ndarray,
    config: ClassFirstConfig,
) -> None:
    if top_class_indices.shape != top_class_probs.shape:
        raise ValueError("top_class_indices and top_class_probs must have the same shape")
    if fallback_indices.shape != fallback_scores.shape:
        raise ValueError("fallback_indices and fallback_scores must have the same shape")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if top_class_indices.shape[0] != embeddings.shape[0]:
        raise ValueError("class top-k row count must match embeddings")
    if fallback_indices.shape[0] != embeddings.shape[0]:
        raise ValueError("fallback top-k row count must match embeddings")
    if top_class_indices.shape[1] < config.class_fallback_k:
        raise ValueError("class top-k width is smaller than class_fallback_k")
    if fallback_indices.shape[1] < config.top_k:
        raise ValueError("fallback top-k width is smaller than output top-k")


def _class_candidates_for_row(
    *,
    row: int,
    embeddings: np.ndarray,
    top_class_indices: np.ndarray,
    fallback_row: np.ndarray,
    buckets: list[np.ndarray],
    config: ClassFirstConfig,
) -> np.ndarray:
    allowed_classes = {
        int(class_id) for class_id in top_class_indices[row, : config.class_fallback_k].tolist()
    }
    seen = {row}
    selected: list[int] = []

    for candidate in fallback_row:
        candidate_int = int(candidate)
        if candidate_int in seen:
            continue
        if int(top_class_indices[candidate_int, 0]) not in allowed_classes:
            continue
        selected.append(candidate_int)
        seen.add(candidate_int)
    if len(selected) >= config.min_class_candidates:
        return np.asarray(selected, dtype=np.int64)
    if not config.bucket_backfill:
        return np.asarray(selected, dtype=np.int64)

    for class_rank in range(config.class_fallback_k):
        class_id = int(top_class_indices[row, class_rank])
        if class_id >= len(buckets):
            continue
        for candidate in buckets[class_id][: config.max_class_candidates]:
            candidate_int = int(candidate)
            if candidate_int in seen:
                continue
            selected.append(candidate_int)
            seen.add(candidate_int)
        if len(selected) >= config.max_class_candidates:
            break
    if not selected:
        return np.empty(0, dtype=np.int64)
    candidates = np.asarray(selected, dtype=np.int64)
    if len(candidates) <= config.max_class_candidates:
        return candidates
    sims = embeddings[candidates] @ embeddings[row]
    keep = np.argpartition(-sims, config.max_class_candidates - 1)[: config.max_class_candidates]
    return candidates[keep]


def _rank_class_candidates(
    *,
    row: int,
    candidates: np.ndarray,
    embeddings: np.ndarray,
    top_class_indices: np.ndarray,
    top_class_probs: np.ndarray,
    fallback_row: np.ndarray,
    config: ClassFirstConfig,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = embeddings[candidates] @ embeddings[row]
    class_overlap = _sparse_top_class_overlap(
        query_indices=top_class_indices[row],
        query_probs=top_class_probs[row],
        candidate_indices=top_class_indices[candidates],
        candidate_probs=top_class_probs[candidates],
    )
    same_top1 = (top_class_indices[candidates, 0] == top_class_indices[row, 0]).astype(np.float32)
    fallback_bonus = _fallback_rank_bonus(candidates, fallback_row, config.fallback_rank_bonus)
    scores = (
        config.embedding_weight * cosine
        + config.class_overlap_weight * class_overlap
        + config.same_top1_bonus * same_top1
        + fallback_bonus
    )
    order = np.argsort(-scores, kind="stable")
    return candidates[order], scores[order].astype(np.float32, copy=False)


def _sparse_top_class_overlap(
    *,
    query_indices: np.ndarray,
    query_probs: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_probs: np.ndarray,
) -> np.ndarray:
    matches = candidate_indices[:, :, None] == query_indices[None, None, :]
    weighted = candidate_probs[:, :, None] * query_probs[None, None, :]
    return np.sum(matches * weighted, axis=(1, 2), dtype=np.float32)


def _fallback_rank_bonus(
    candidates: np.ndarray,
    fallback_row: np.ndarray,
    weight: float,
) -> np.ndarray:
    if weight <= 0.0:
        return np.zeros(len(candidates), dtype=np.float32)
    ranks = {int(candidate): rank for rank, candidate in enumerate(fallback_row.tolist())}
    bonuses = np.zeros(len(candidates), dtype=np.float32)
    for position, candidate in enumerate(candidates.tolist()):
        rank = ranks.get(int(candidate))
        if rank is not None:
            bonuses[position] = weight / float(rank + 1)
    return bonuses


def _build_class_buckets(
    top1_class_indices: np.ndarray, top1_class_probs: np.ndarray
) -> list[np.ndarray]:
    class_count = int(top1_class_indices.max()) + 1
    buckets: list[list[int]] = [[] for _ in range(class_count)]
    for row in np.argsort(-top1_class_probs, kind="stable").tolist():
        class_id = int(top1_class_indices[row])
        buckets[int(class_id)].append(row)
    return [np.asarray(bucket, dtype=np.int64) for bucket in buckets]
