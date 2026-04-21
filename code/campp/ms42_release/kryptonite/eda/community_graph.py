"""Graph primitives used by transductive retrieval reranking."""

from __future__ import annotations

from typing import Any

import numpy as np

from kryptonite.eda.rerank import density_zscore


def cluster_edge_weights(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build weighted mutual-kNN edges for cluster-first inference."""

    edge_scores = scores[:, : config.edge_top]
    edge_mask = mutual_mask(
        indices,
        edge_top=config.edge_top,
        reciprocal_top=config.reciprocal_top,
    )
    threshold = config.edge_min_score
    if config.edge_score_quantile is not None:
        threshold = float(np.quantile(edge_scores, config.edge_score_quantile))
    if threshold is not None:
        edge_mask &= edge_scores >= threshold
    shared_counts = shared_neighbor_counts(
        indices=indices,
        edge_top=config.edge_top,
        shared_top=config.shared_top,
        base_mask=edge_mask,
    )
    if config.shared_min_count > 0:
        edge_mask &= shared_counts >= config.shared_min_count
    density_z = density_zscore(scores, top_n=min(20, scores.shape[1]))
    rank_bonus = config.rank_weight / (1.0 + np.arange(config.edge_top, dtype=np.float32))
    edge_weights = (
        edge_scores
        + rank_bonus[None, :]
        + config.shared_weight * (shared_counts.astype(np.float32) / max(config.shared_top, 1))
        - config.density_penalty * density_z[indices[:, : config.edge_top]]
    )
    edge_weights = np.where(edge_mask, edge_weights, 0.0).astype(np.float32, copy=False)
    nonzero_counts = edge_mask.sum(axis=1)
    active_shared = shared_counts[edge_mask]
    shared_mean = float(active_shared.mean()) if active_shared.size else 0.0
    return (
        edge_mask,
        edge_weights,
        {
            "edge_top": config.edge_top,
            "edge_score_threshold": threshold,
            "cluster_edge_count": int(edge_mask.sum()),
            "cluster_edge_share": float(edge_mask.mean()),
            "cluster_edge_degree_p50": float(np.quantile(nonzero_counts, 0.50)),
            "cluster_edge_degree_p95": float(np.quantile(nonzero_counts, 0.95)),
            "shared_top": config.shared_top,
            "shared_min_count": config.shared_min_count,
            "shared_count_mean": shared_mean,
            "shared_weight": config.shared_weight,
            "cluster_rank_weight": config.rank_weight,
        },
    )


def split_oversized_clusters(
    *,
    labels: np.ndarray,
    indices: np.ndarray,
    edge_mask: np.ndarray,
    cluster_max_size: int,
    split_edge_top: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Split oversized labels using stricter induced mutual edges."""

    label_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    oversized = np.flatnonzero(label_sizes > cluster_max_size)
    if len(oversized) == 0:
        return labels, {
            "split_oversized_cluster_count": 0,
            "split_oversized_rows": 0,
            "split_edge_top": split_edge_top,
        }

    next_label = int(labels.max()) + 1
    out = labels.copy()
    edge_top = min(split_edge_top, edge_mask.shape[1])
    split_rows = 0
    for label in oversized.tolist():
        members = np.flatnonzero(labels == label)
        split_rows += int(len(members))
        local_index = {int(member): position for position, member in enumerate(members.tolist())}
        dsu = _DisjointSet(len(members))
        for member in members.tolist():
            member_position = local_index[int(member)]
            for candidate in indices[int(member), :edge_top][edge_mask[int(member), :edge_top]]:
                candidate_position = local_index.get(int(candidate))
                if candidate_position is not None:
                    dsu.union(member_position, candidate_position)
        local_labels = dsu.labels()
        local_sizes = np.bincount(local_labels, minlength=int(local_labels.max()) + 1)
        if len(local_sizes) == 1:
            continue
        first = True
        for local_label in range(len(local_sizes)):
            local_members = members[local_labels == local_label]
            if first:
                out[local_members] = label
                first = False
            else:
                out[local_members] = next_label
                next_label += 1
    return _compact_labels(out), {
        "split_oversized_cluster_count": int(len(oversized)),
        "split_oversized_rows": split_rows,
        "split_edge_top": split_edge_top,
    }


def weighted_label_propagation(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    edge_mask: np.ndarray,
    iterations: int,
    self_weight: float = 0.0,
    label_size_penalty: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic weighted label propagation over a sparse neighbor graph."""

    labels = np.arange(indices.shape[0], dtype=np.int64)
    edge_counts = edge_mask.sum(axis=1)
    for iteration in range(iterations):
        changed = 0
        label_counts = np.bincount(labels, minlength=indices.shape[0])
        for row in range(indices.shape[0]):
            if edge_counts[row] == 0:
                continue
            label_scores: dict[int, float] = {}
            if self_weight > 0.0:
                current_label = int(labels[row])
                label_scores[current_label] = float(self_weight)
            for candidate, weight in zip(
                indices[row, edge_mask[row]], scores[row, edge_mask[row]], strict=True
            ):
                label = int(labels[int(candidate)])
                label_scores[label] = label_scores.get(label, 0.0) + float(weight)
            if label_size_penalty > 0.0:
                best_label = max(
                    label_scores.items(),
                    key=lambda item: (
                        item[1] / float(label_counts[item[0]]) ** label_size_penalty,
                        -item[0],
                    ),
                )[0]
            else:
                best_label = max(label_scores.items(), key=lambda item: (item[1], -item[0]))[0]
            if labels[row] != best_label:
                labels[row] = best_label
                changed += 1
        if changed == 0:
            return _compact_labels(labels), {
                "label_propagation_iterations_run": iteration + 1,
                "label_propagation_last_changed": changed,
                "label_propagation_edge_share": float(edge_mask.mean()),
                "label_propagation_self_weight": self_weight,
                "label_propagation_label_size_penalty": label_size_penalty,
            }
    return _compact_labels(labels), {
        "label_propagation_iterations_run": iterations,
        "label_propagation_last_changed": changed,
        "label_propagation_edge_share": float(edge_mask.mean()),
        "label_propagation_self_weight": self_weight,
        "label_propagation_label_size_penalty": label_size_penalty,
    }


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    _, dense_labels = np.unique(labels, return_inverse=True)
    return dense_labels.astype(np.int64)


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

    return base_mask & (
        shared_neighbor_counts(
            indices=indices,
            edge_top=edge_top,
            shared_top=shared_top,
            base_mask=base_mask,
        )
        >= shared_min_count
    )


def shared_neighbor_counts(
    *,
    indices: np.ndarray,
    edge_top: int,
    shared_top: int,
    base_mask: np.ndarray,
) -> np.ndarray:
    """Count shared top-neighbors for candidate pairs selected by base_mask."""

    shared_top = min(shared_top, indices.shape[1])
    neighbor_sets = [set(row) for row in indices[:, :shared_top].tolist()]
    out = np.zeros((indices.shape[0], edge_top), dtype=np.int16)
    for row in range(indices.shape[0]):
        row_set = neighbor_sets[row]
        for position in np.flatnonzero(base_mask[row]):
            candidate = int(indices[row, position])
            out[row, position] = len(row_set.intersection(neighbor_sets[candidate]))
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
