"""Transductive graph/community postprocessing for retrieval submissions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.community_graph import (
    cluster_edge_weights,
    mutual_components,
    mutual_mask,
    shared_neighbor_mask,
    split_oversized_clusters,
    weighted_label_propagation,
)
from kryptonite.eda.community_io import (
    evaluate_labelled_topk,
    write_cluster_assignments,
    write_submission,
)
from kryptonite.eda.community_types import (
    ClusterFirstConfig,
    CommunityConfig,
    LabelPropagationConfig,
)
from kryptonite.eda.leaderboard_alignment import public_lb_for, public_status_for
from kryptonite.eda.rerank import density_zscore, gini
from kryptonite.eda.submission import validate_submission


def exact_topk(
    embeddings: np.ndarray,
    *,
    top_k: int,
    batch_size: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact cosine top-k neighbors, excluding self matches."""

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    matrix = torch.from_numpy(np.asarray(embeddings, dtype=np.float32).copy()).to(device)
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    indices = np.empty((matrix.shape[0], top_k), dtype=np.int64)
    scores = np.empty((matrix.shape[0], top_k), dtype=np.float32)
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        sims = matrix[start:end] @ matrix.T
        sims[
            torch.arange(end - start, device=device), torch.arange(start, end, device=device)
        ] = -torch.inf
        values, top_indices = torch.topk(sims, k=top_k, dim=1)
        indices[start:end] = top_indices.cpu().numpy()
        scores[start:end] = values.cpu().numpy()
    return indices, scores


def community_rerank(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    config: CommunityConfig,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Prefer same mutual-kNN component candidates and fall back to local-scaled ranking."""

    if indices.shape != scores.shape:
        raise ValueError("indices and scores must have the same shape")
    if indices.shape[1] < max(config.edge_top, config.reciprocal_top, config.rank_top, top_k):
        raise ValueError("top-k cache is smaller than the requested community config")

    rank_indices, rank_scores, rank_meta = adjusted_ranking(
        indices=indices,
        scores=scores,
        rank_top=config.rank_top,
        reciprocal_top=config.reciprocal_top,
        reciprocal_bonus=config.reciprocal_bonus,
        density_penalty=config.density_penalty,
    )
    components, component_sizes, graph_meta = mutual_components(
        indices=indices,
        scores=scores,
        edge_top=config.edge_top,
        reciprocal_top=config.reciprocal_top,
        score_quantile=config.edge_score_quantile,
        min_score=config.edge_min_score,
        shared_top=config.shared_top,
        shared_min_count=config.shared_min_count,
    )
    out_i = np.empty((indices.shape[0], top_k), dtype=np.int64)
    out_s = np.empty((indices.shape[0], top_k), dtype=np.float32)
    component_used = np.zeros(indices.shape[0], dtype=bool)
    same_component_counts = np.zeros(indices.shape[0], dtype=np.int16)
    for row in range(indices.shape[0]):
        own_component = components[row]
        own_component_size = int(component_sizes[own_component])
        fallback = rank_indices[row]
        fallback_scores = rank_scores[row]
        selected: list[int] = []
        selected_scores: list[float] = []
        if config.component_min_size <= own_component_size <= config.component_max_size:
            in_component = components[fallback] == own_component
            component_positions = np.flatnonzero(in_component)
            same_component_counts[row] = len(component_positions)
            if len(component_positions) >= config.component_min_candidates:
                for position in component_positions:
                    candidate = int(fallback[position])
                    if candidate == row:
                        continue
                    selected.append(candidate)
                    selected_scores.append(float(fallback_scores[position]))
                    if len(selected) == top_k:
                        break
                component_used[row] = len(selected) > 0
        seen = set(selected)
        if len(selected) < top_k:
            for candidate, score in zip(fallback, fallback_scores, strict=True):
                candidate_int = int(candidate)
                if candidate_int == row or candidate_int in seen:
                    continue
                selected.append(candidate_int)
                selected_scores.append(float(score))
                seen.add(candidate_int)
                if len(selected) == top_k:
                    break
        out_i[row] = np.asarray(selected, dtype=np.int64)
        out_s[row] = np.asarray(selected_scores, dtype=np.float32)
    meta = {
        **rank_meta,
        **graph_meta,
        "component_min_size": config.component_min_size,
        "component_max_size": config.component_max_size,
        "component_min_candidates": config.component_min_candidates,
        "component_used_share": float(component_used.mean()),
        "same_component_candidates_p50": float(np.quantile(same_component_counts, 0.50)),
        "same_component_candidates_p95": float(np.quantile(same_component_counts, 0.95)),
    }
    return out_i, out_s, meta


def cluster_first_rerank(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    config: ClusterFirstConfig,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Cluster the full test graph first, then prefer within-cluster retrieval."""

    if indices.shape != scores.shape:
        raise ValueError("indices and scores must have the same shape")
    if indices.shape[1] < max(config.edge_top, config.reciprocal_top, config.rank_top, top_k):
        raise ValueError("top-k cache is smaller than the requested cluster-first config")

    rank_indices, rank_scores, rank_meta = adjusted_ranking(
        indices=indices,
        scores=scores,
        rank_top=config.rank_top,
        reciprocal_top=config.reciprocal_top,
        reciprocal_bonus=config.reciprocal_bonus,
        density_penalty=config.density_penalty,
    )
    edge_mask, edge_weights, edge_meta = cluster_edge_weights(
        indices=indices,
        scores=scores,
        config=config,
    )
    labels, propagation_meta = weighted_label_propagation(
        indices=indices[:, : config.edge_top],
        scores=edge_weights,
        edge_mask=edge_mask,
        iterations=config.iterations,
        self_weight=config.self_weight,
        label_size_penalty=config.label_size_penalty,
    )
    split_meta: dict[str, Any] = {}
    if config.split_oversized:
        labels, split_meta = split_oversized_clusters(
            labels=labels,
            indices=indices,
            edge_mask=edge_mask,
            cluster_max_size=config.cluster_max_size,
            split_edge_top=config.split_edge_top,
        )
    cluster_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    out_i = np.empty((indices.shape[0], top_k), dtype=np.int64)
    out_s = np.empty((indices.shape[0], top_k), dtype=np.float32)
    cluster_used = np.zeros(indices.shape[0], dtype=bool)
    same_cluster_counts = np.zeros(indices.shape[0], dtype=np.int16)
    for row in range(indices.shape[0]):
        own_label = labels[row]
        own_cluster_size = int(cluster_sizes[own_label])
        fallback = rank_indices[row]
        fallback_scores = rank_scores[row]
        selected: list[int] = []
        selected_scores: list[float] = []
        if config.cluster_min_size <= own_cluster_size <= config.cluster_max_size:
            same_cluster = labels[fallback] == own_label
            cluster_positions = np.flatnonzero(same_cluster)
            same_cluster_counts[row] = len(cluster_positions)
            if len(cluster_positions) >= config.cluster_min_candidates:
                for position in cluster_positions:
                    candidate = int(fallback[position])
                    if candidate == row:
                        continue
                    selected.append(candidate)
                    selected_scores.append(float(fallback_scores[position]))
                    if len(selected) == top_k:
                        break
                cluster_used[row] = len(selected) > 0
        seen = set(selected)
        if len(selected) < top_k:
            for candidate, score in zip(fallback, fallback_scores, strict=True):
                candidate_int = int(candidate)
                if candidate_int == row or candidate_int in seen:
                    continue
                selected.append(candidate_int)
                selected_scores.append(float(score))
                seen.add(candidate_int)
                if len(selected) == top_k:
                    break
        out_i[row] = np.asarray(selected, dtype=np.int64)
        out_s[row] = np.asarray(selected_scores, dtype=np.float32)
    meta = {
        **rank_meta,
        **edge_meta,
        **propagation_meta,
        **split_meta,
        "iterations": config.iterations,
        "cluster_min_size": config.cluster_min_size,
        "cluster_max_size": config.cluster_max_size,
        "cluster_min_candidates": config.cluster_min_candidates,
        "cluster_used_share": float(cluster_used.mean()),
        "same_cluster_candidates_p50": float(np.quantile(same_cluster_counts, 0.50)),
        "same_cluster_candidates_p95": float(np.quantile(same_cluster_counts, 0.95)),
        "cluster_count": int(len(cluster_sizes)),
        "cluster_size_p50": float(np.quantile(cluster_sizes, 0.50)),
        "cluster_size_p95": float(np.quantile(cluster_sizes, 0.95)),
        "cluster_size_p99": float(np.quantile(cluster_sizes, 0.99)),
        "cluster_size_max": int(cluster_sizes.max()),
    }
    return out_i, out_s, labels, meta


def label_propagation_rerank(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    config: LabelPropagationConfig,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Weighted label propagation on a mutual-kNN graph followed by within-label ranking."""

    if indices.shape != scores.shape:
        raise ValueError("indices and scores must have the same shape")
    rank_top = max(config.rank_top, top_k)
    if indices.shape[1] < max(config.edge_top, config.reciprocal_top, rank_top, top_k):
        raise ValueError("top-k cache is smaller than the requested label-propagation config")

    rank_indices, rank_scores, rank_meta = adjusted_ranking(
        indices=indices,
        scores=scores,
        rank_top=rank_top,
        reciprocal_top=config.reciprocal_top,
        reciprocal_bonus=config.reciprocal_bonus,
        density_penalty=config.density_penalty,
    )
    edge_mask = mutual_mask(indices, edge_top=config.edge_top, reciprocal_top=config.reciprocal_top)
    if config.shared_min_count > 0:
        edge_mask &= shared_neighbor_mask(
            indices=indices,
            edge_top=config.edge_top,
            shared_top=config.shared_top,
            shared_min_count=config.shared_min_count,
            base_mask=edge_mask,
        )
    labels, label_meta = weighted_label_propagation(
        indices=indices[:, : config.edge_top],
        scores=scores[:, : config.edge_top],
        edge_mask=edge_mask,
        iterations=config.iterations,
    )
    label_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    out_i = np.empty((indices.shape[0], top_k), dtype=np.int64)
    out_s = np.empty((indices.shape[0], top_k), dtype=np.float32)
    label_used = np.zeros(indices.shape[0], dtype=bool)
    same_label_counts = np.zeros(indices.shape[0], dtype=np.int16)
    for row in range(indices.shape[0]):
        own_label = labels[row]
        own_label_size = int(label_sizes[own_label])
        fallback = rank_indices[row]
        fallback_scores = rank_scores[row]
        selected: list[int] = []
        selected_scores: list[float] = []
        if config.label_min_size <= own_label_size <= config.label_max_size:
            same_label = labels[fallback] == own_label
            label_positions = np.flatnonzero(same_label)
            same_label_counts[row] = len(label_positions)
            if len(label_positions) >= config.label_min_candidates:
                for position in label_positions:
                    candidate = int(fallback[position])
                    if candidate == row:
                        continue
                    selected.append(candidate)
                    selected_scores.append(float(fallback_scores[position]))
                    if len(selected) == top_k:
                        break
                label_used[row] = len(selected) > 0
        seen = set(selected)
        if len(selected) < top_k:
            for candidate, score in zip(fallback, fallback_scores, strict=True):
                candidate_int = int(candidate)
                if candidate_int == row or candidate_int in seen:
                    continue
                selected.append(candidate_int)
                selected_scores.append(float(score))
                seen.add(candidate_int)
                if len(selected) == top_k:
                    break
        out_i[row] = np.asarray(selected, dtype=np.int64)
        out_s[row] = np.asarray(selected_scores, dtype=np.float32)
    meta = {
        **rank_meta,
        **label_meta,
        "edge_top": config.edge_top,
        "iterations": config.iterations,
        "shared_top": config.shared_top,
        "shared_min_count": config.shared_min_count,
        "label_min_size": config.label_min_size,
        "label_max_size": config.label_max_size,
        "label_min_candidates": config.label_min_candidates,
        "label_used_share": float(label_used.mean()),
        "same_label_candidates_p50": float(np.quantile(same_label_counts, 0.50)),
        "same_label_candidates_p95": float(np.quantile(same_label_counts, 0.95)),
        "label_count": int(len(label_sizes)),
        "label_size_p50": float(np.quantile(label_sizes, 0.50)),
        "label_size_p95": float(np.quantile(label_sizes, 0.95)),
        "label_size_p99": float(np.quantile(label_sizes, 0.99)),
        "label_size_max": int(label_sizes.max()),
    }
    return out_i, out_s, meta


def adjusted_ranking(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    rank_top: int,
    reciprocal_top: int,
    reciprocal_bonus: float,
    density_penalty: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """B8-style reciprocal plus local-density adjusted ranking."""

    rank_indices = indices[:, :rank_top]
    rank_scores = scores[:, :rank_top]
    density_z = density_zscore(scores, top_n=20)
    reciprocal = mutual_mask(indices, edge_top=rank_top, reciprocal_top=reciprocal_top)
    adjusted = rank_scores + reciprocal_bonus * reciprocal
    adjusted = adjusted - density_penalty * density_z[rank_indices]
    order = np.argsort(adjusted, axis=1)[:, ::-1]
    sorted_indices = np.take_along_axis(rank_indices, order, axis=1)
    sorted_scores = np.take_along_axis(rank_scores, order, axis=1)
    return (
        sorted_indices,
        sorted_scores,
        {
            "rank_top": rank_top,
            "reciprocal_top": reciprocal_top,
            "reciprocal_bonus": reciprocal_bonus,
            "density_penalty": density_penalty,
            "reciprocal_share_rank": float(reciprocal.mean()),
        },
    )


def run_public_community_package(
    *,
    embeddings_path: Path,
    manifest_csv: Path,
    template_csv: Path,
    output_dir: Path,
    configs: list[CommunityConfig | LabelPropagationConfig | ClusterFirstConfig],
    top_cache_k: int = 100,
    search_batch_size: int = 2048,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(embeddings_path)
    manifest = pl.read_csv(manifest_csv)
    started = time.perf_counter()
    indices, scores = exact_topk(
        embeddings, top_k=top_cache_k, batch_size=search_batch_size, device="cuda"
    )
    search_s = time.perf_counter() - started
    rows = []
    for config in configs:
        started = time.perf_counter()
        if isinstance(config, LabelPropagationConfig):
            top_indices, top_scores, meta = label_propagation_rerank(
                indices=indices, scores=scores, config=config, top_k=10
            )
        elif isinstance(config, ClusterFirstConfig):
            top_indices, top_scores, labels, meta = cluster_first_rerank(
                indices=indices, scores=scores, config=config, top_k=10
            )
            write_cluster_assignments(
                manifest=manifest,
                labels=labels,
                output_csv=output_dir / f"clusters_{config.experiment_id}.csv",
            )
        else:
            top_indices, top_scores, meta = community_rerank(
                indices=indices, scores=scores, config=config, top_k=10
            )
        rerank_s = time.perf_counter() - started
        submission_path = output_dir / f"submission_{config.experiment_id}.csv"
        write_submission(manifest=manifest, top_indices=top_indices, output_csv=submission_path)
        validation = validate_submission(template_csv=template_csv, submission_csv=submission_path)
        validation_path = output_dir / f"submission_{config.experiment_id}_validation.json"
        validation_path.write_text(
            __import__("json").dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        indegree = np.bincount(top_indices.ravel(), minlength=manifest.height)
        rows.append(
            {
                "experiment_id": config.experiment_id,
                "public_lb": public_lb_for(config.experiment_id),
                "public_submission_status": public_status_for(config.experiment_id),
                "validator_passed": bool(validation["passed"]),
                "search_s": round(search_s, 6),
                "rerank_s": round(rerank_s, 6),
                "top1_score_mean": float(top_scores[:, 0].mean()),
                "top10_mean_score_mean": float(top_scores.mean()),
                "indegree_gini_10": gini(indegree),
                "indegree_max_10": int(indegree.max()),
                **meta,
            }
        )
    pl.DataFrame(rows).write_csv(output_dir / "public_community_runs_summary.csv")


__all__ = [
    "ClusterFirstConfig",
    "CommunityConfig",
    "LabelPropagationConfig",
    "adjusted_ranking",
    "cluster_edge_weights",
    "cluster_first_rerank",
    "community_rerank",
    "evaluate_labelled_topk",
    "exact_topk",
    "label_propagation_rerank",
    "mutual_components",
    "mutual_mask",
    "run_public_community_package",
    "shared_neighbor_mask",
    "split_oversized_clusters",
    "weighted_label_propagation",
    "write_cluster_assignments",
    "write_submission",
]
