"""Transductive graph/community postprocessing for retrieval submissions."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.leaderboard_alignment import public_lb_for, public_status_for
from kryptonite.eda.rerank import density_zscore, gini
from kryptonite.eda.submission import validate_submission


@dataclass(frozen=True, slots=True)
class CommunityConfig:
    experiment_id: str
    edge_top: int = 20
    reciprocal_top: int = 20
    rank_top: int = 100
    component_min_size: int = 11
    component_max_size: int = 120
    component_min_candidates: int = 6
    reciprocal_bonus: float = 0.03
    density_penalty: float = 0.02
    edge_score_quantile: float | None = None
    edge_min_score: float | None = None
    shared_top: int = 20
    shared_min_count: int = 0


@dataclass(frozen=True, slots=True)
class LabelPropagationConfig:
    experiment_id: str
    edge_top: int = 10
    reciprocal_top: int = 20
    rank_top: int = 100
    iterations: int = 5
    label_min_size: int = 5
    label_max_size: int = 120
    label_min_candidates: int = 3
    shared_top: int = 20
    shared_min_count: int = 0
    reciprocal_bonus: float = 0.03
    density_penalty: float = 0.02


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
    if indices.shape[1] < max(config.edge_top, config.reciprocal_top, config.rank_top, top_k):
        raise ValueError("top-k cache is smaller than the requested label-propagation config")

    rank_indices, rank_scores, rank_meta = adjusted_ranking(
        indices=indices,
        scores=scores,
        rank_top=config.rank_top,
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


def weighted_label_propagation(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    edge_mask: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic weighted label propagation over a sparse neighbor graph."""

    labels = np.arange(indices.shape[0], dtype=np.int64)
    edge_counts = edge_mask.sum(axis=1)
    for iteration in range(iterations):
        changed = 0
        for row in range(indices.shape[0]):
            if edge_counts[row] == 0:
                continue
            label_scores: dict[int, float] = {}
            for candidate, weight in zip(
                indices[row, edge_mask[row]], scores[row, edge_mask[row]], strict=True
            ):
                label = int(labels[int(candidate)])
                label_scores[label] = label_scores.get(label, 0.0) + float(weight)
            best_label = max(label_scores.items(), key=lambda item: (item[1], -item[0]))[0]
            if labels[row] != best_label:
                labels[row] = best_label
                changed += 1
        if changed == 0:
            return _compact_labels(labels), {
                "label_propagation_iterations_run": iteration + 1,
                "label_propagation_last_changed": changed,
                "label_propagation_edge_share": float(edge_mask.mean()),
            }
    return _compact_labels(labels), {
        "label_propagation_iterations_run": iterations,
        "label_propagation_last_changed": changed,
        "label_propagation_edge_share": float(edge_mask.mean()),
    }


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    _, dense_labels = np.unique(labels, return_inverse=True)
    return dense_labels.astype(np.int64)


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


def mutual_components(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    edge_top: int,
    reciprocal_top: int,
    score_quantile: float | None = None,
    min_score: float | None = None,
    shared_top: int = 20,
    shared_min_count: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build connected components from mutual-kNN edges."""

    n_rows = indices.shape[0]
    mask = mutual_mask(indices, edge_top=edge_top, reciprocal_top=reciprocal_top)
    threshold = min_score
    if score_quantile is not None:
        threshold = float(np.quantile(scores[:, :edge_top], score_quantile))
    if threshold is not None:
        mask &= scores[:, :edge_top] >= threshold
    if shared_min_count > 0:
        mask &= shared_neighbor_mask(
            indices=indices,
            edge_top=edge_top,
            shared_top=shared_top,
            shared_min_count=shared_min_count,
            base_mask=mask,
        )
    src = np.repeat(np.arange(n_rows, dtype=np.int64), edge_top)[mask.ravel()]
    dst = indices[:, :edge_top].ravel()[mask.ravel()]
    dsu = _DisjointSet(n_rows)
    for left, right in zip(src.tolist(), dst.tolist(), strict=True):
        dsu.union(left, right)
    components = dsu.labels()
    component_sizes = np.bincount(components, minlength=int(components.max()) + 1)
    nontrivial = component_sizes[component_sizes > 1]
    meta = {
        "edge_top": edge_top,
        "mutual_edge_count": int(len(src)),
        "mutual_edge_share": float(mask.mean()),
        "component_count": int(len(component_sizes)),
        "nontrivial_component_count": int(len(nontrivial)),
        "component_size_p50": float(np.quantile(component_sizes, 0.50)),
        "component_size_p95": float(np.quantile(component_sizes, 0.95)),
        "component_size_p99": float(np.quantile(component_sizes, 0.99)),
        "component_size_max": int(component_sizes.max()),
        "edge_score_threshold": threshold,
        "shared_top": shared_top,
        "shared_min_count": shared_min_count,
    }
    return components, component_sizes, meta


def shared_neighbor_mask(
    *,
    indices: np.ndarray,
    edge_top: int,
    shared_top: int,
    shared_min_count: int,
    base_mask: np.ndarray,
) -> np.ndarray:
    """Return mask for candidate pairs with enough shared top-neighborhood."""

    shared_top = min(shared_top, indices.shape[1])
    neighbor_sets = [set(row) for row in indices[:, :shared_top].tolist()]
    out = np.zeros((indices.shape[0], edge_top), dtype=bool)
    for row in range(indices.shape[0]):
        row_set = neighbor_sets[row]
        for position in np.flatnonzero(base_mask[row]):
            candidate = int(indices[row, position])
            if len(row_set.intersection(neighbor_sets[candidate])) >= shared_min_count:
                out[row, position] = True
    return out


def mutual_mask(indices: np.ndarray, *, edge_top: int, reciprocal_top: int) -> np.ndarray:
    """Return a boolean mask for edges whose reverse appears in reciprocal_top."""

    n_rows = indices.shape[0]
    rows = np.arange(n_rows, dtype=np.int64)
    reverse_keys = (indices[:, :reciprocal_top].astype(np.int64).ravel() * n_rows) + np.repeat(
        rows, reciprocal_top
    )
    forward_keys = (np.repeat(rows, edge_top) * n_rows) + indices[:, :edge_top].astype(
        np.int64
    ).ravel()
    return np.isin(forward_keys, reverse_keys, assume_unique=False).reshape(n_rows, edge_top)


def evaluate_labelled_topk(
    *,
    experiment_id: str,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    manifest: pl.DataFrame,
    query_only: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    labels = manifest["speaker_id"].cast(pl.Utf8).to_list()
    paths = manifest["filepath"].cast(pl.Utf8).to_list()
    if query_only and "is_query" in manifest.columns:
        query_indices = manifest.filter(pl.col("is_query"))["gallery_index"].to_numpy()
    elif "gallery_index" in manifest.columns:
        query_indices = manifest["gallery_index"].to_numpy()
    else:
        query_indices = np.arange(manifest.height)
    rows = []
    for query_index in query_indices:
        neighbours = top_indices[int(query_index)]
        scores = top_scores[int(query_index)]
        query_label = labels[int(query_index)]
        hits = [labels[int(index)] == query_label for index in neighbours]
        rows.append(
            {
                "experiment_id": experiment_id,
                "query_idx": int(query_index),
                "query_path": paths[int(query_index)],
                "speaker_id": query_label,
                "p10": float(np.mean(hits)),
                "n_correct_top10": int(sum(hits)),
                "top1_correct": bool(hits[0]),
                "top1_score": float(scores[0]),
                "top10_mean_score": float(np.mean(scores)),
            }
        )
    frame = pl.DataFrame(rows)
    summary = {
        "experiment_id": experiment_id,
        "query_count": frame.height,
        "gallery_count": manifest.height,
        "p10": float(frame["p10"].mean()),
        "top1_accuracy": float(frame["top1_correct"].mean()),
        "mean_top10_score": float(frame["top10_mean_score"].mean()),
    }
    return frame, summary


def write_submission(
    *,
    manifest: pl.DataFrame,
    top_indices: np.ndarray,
    output_csv: Path,
) -> None:
    paths = manifest["filepath"].cast(pl.Utf8).to_list()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filepath", "neighbours"])
        for filepath, neighbours in zip(paths, top_indices, strict=True):
            writer.writerow([filepath, ",".join(str(int(index)) for index in neighbours)])


def run_public_community_package(
    *,
    embeddings_path: Path,
    manifest_csv: Path,
    template_csv: Path,
    output_dir: Path,
    configs: list[CommunityConfig | LabelPropagationConfig],
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


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        if parent != value:
            self.parent[value] = self.find(parent)
        return int(self.parent[value])

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(index) for index in range(len(self.parent))], dtype=np.int64)
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int64)
