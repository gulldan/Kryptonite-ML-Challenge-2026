from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

_SELECTOR_COLUMNS = [
    "row_index",
    "component_id",
    "component_size",
    "pseudo_spk",
    "anchor_index",
    "top1_score",
    "top1_cosine",
    "top1_margin",
    "path",
    "dur",
    "start",
    "stop",
    "orig_filepath",
    "manifest_split",
    "duration_bucket",
    "profile_bucket",
    "prior_distance",
]
_EPSILON = 1e-6


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def _selector_meta_df(bundle: dict[str, object]) -> pd.DataFrame:
    meta = bundle.get("selector_meta_df")
    if not isinstance(meta, pd.DataFrame):
        raise TypeError("validation bundle must contain selector_meta_df as a DataFrame")
    return meta.drop(columns=["spk", "oracle_spk"], errors="ignore").copy()


def _bundle_arrays(
    bundle: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topk_idx = np.asarray(bundle["topk_idx"], dtype=np.int64)
    topk_sim = np.asarray(bundle["topk_sim"], dtype=np.float32)
    top1_margin = np.asarray(bundle["top1_margin"], dtype=np.float32)
    if topk_idx.shape != topk_sim.shape:
        raise ValueError("topk_idx and topk_sim must have the same shape")
    if top1_margin.shape != (topk_idx.shape[0],):
        raise ValueError("top1_margin must be aligned with topk_idx rows")
    return topk_idx, topk_sim, top1_margin


def _connected_components_for_edges(
    num_nodes: int,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if num_nodes <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if edges.size == 0:
        return np.arange(num_nodes, dtype=np.int64), np.ones(num_nodes, dtype=np.int64)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    component_count, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    counts = np.bincount(labels, minlength=component_count).astype(np.int64, copy=False)
    return labels.astype(np.int64, copy=False), counts


def _weighted_prior_distance(
    features: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    centered = (features - mu[None, :]) / scale[None, :]
    return np.sum(weight[None, :] * np.square(centered), axis=1, dtype=np.float32)


def _append_stage(
    stage_counts: list[dict[str, int]],
    stage: str,
    row_count: int,
    cluster_count: int = 0,
) -> None:
    stage_counts.append(
        {
            "stage": str(stage),
            "row_count": int(row_count),
            "cluster_count": int(cluster_count),
        }
    )


def _quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float32), float(q)))


def _build_mutual_margin_edges(
    *,
    topk_idx: np.ndarray,
    top1_margin: np.ndarray,
    min_margin: float | None,
) -> np.ndarray:
    num_rows = topk_idx.shape[0]
    neighbor_sets = [set(row.tolist()) for row in topk_idx]
    required_margin = 0.0 if min_margin is None else float(min_margin)
    edges: list[tuple[int, int]] = []
    for src_index in range(num_rows):
        if float(top1_margin[src_index]) < required_margin:
            continue
        for dst_index in topk_idx[src_index]:
            dst = int(dst_index)
            if src_index == dst:
                continue
            if src_index not in neighbor_sets[dst]:
                continue
            if float(top1_margin[dst]) < required_margin:
                continue
            left, right = sorted((src_index, dst))
            edges.append((left, right))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.asarray(edges, dtype=np.int64), axis=0)


def _component_frame(
    *,
    selector_meta: pd.DataFrame,
    topk_idx: np.ndarray,
    topk_sim: np.ndarray,
    top1_margin: np.ndarray,
    row_indices: np.ndarray,
    component_id: int,
    prefix: str,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    component_rows = selector_meta.iloc[row_indices].copy().reset_index(drop=True)
    component_rows.insert(0, "row_index", row_indices.astype(np.int64, copy=False))
    component_rows["component_id"] = int(component_id)
    component_rows["component_size"] = int(len(row_indices))
    component_rows["pseudo_spk"] = f"{prefix}_{component_id}"
    component_rows["anchor_index"] = topk_idx[row_indices, 0].astype(np.int64, copy=False)
    component_rows["top1_score"] = topk_sim[row_indices, 0].astype(np.float32, copy=False)
    component_rows["top1_cosine"] = component_rows["top1_score"].to_numpy(copy=False)
    component_rows["top1_margin"] = top1_margin[row_indices].astype(np.float32, copy=False)
    if extra_columns:
        for name, value in extra_columns.items():
            if isinstance(value, np.ndarray):
                component_rows[name] = value[row_indices]
            else:
                component_rows[name] = value
    extra_tail = [column for column in component_rows.columns if column not in _SELECTOR_COLUMNS]
    return component_rows.loc[:, _SELECTOR_COLUMNS + extra_tail]


def _limit_rows_per_cluster(
    accepted_df: pd.DataFrame,
    *,
    max_rows_per_cluster: int,
) -> pd.DataFrame:
    if accepted_df.empty:
        return accepted_df.copy()
    sort_by: list[str] = []
    ascending: list[bool] = []
    for name, is_ascending in (
        ("top1_score", False),
        ("top1_margin", False),
        ("prior_distance", True),
        ("inbound_degree", True),
        ("row_index", True),
    ):
        if name in accepted_df.columns:
            sort_by.append(name)
            ascending.append(is_ascending)
    ranked = accepted_df.sort_values(
        by=sort_by,
        ascending=ascending,
        kind="mergesort",
    ).copy()
    ranked["_cluster_rank"] = ranked.groupby("component_id", sort=False).cumcount() + 1
    ranked = ranked.loc[ranked["_cluster_rank"] <= int(max_rows_per_cluster)].copy()
    return ranked.drop(columns=["_cluster_rank"]).reset_index(drop=True)


@dataclass(slots=True)
class PseudoLabelSelectionConfig:
    min_cluster_size: int
    max_cluster_size: int | None = None
    min_top1_score: float | None = None
    min_top1_margin: float | None = None
    max_indegree_quantile: float | None = None
    indegree_top_k: int | None = None
    max_rows_per_cluster: int | None = None

    def __post_init__(self) -> None:
        if self.min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be positive")
        if self.max_cluster_size is not None:
            if self.max_cluster_size < self.min_cluster_size:
                raise ValueError("max_cluster_size must be >= min_cluster_size")
        if self.min_top1_score is not None and not (0.0 <= self.min_top1_score <= 1.0):
            raise ValueError("min_top1_score must satisfy 0 <= value <= 1")
        if self.min_top1_margin is not None and self.min_top1_margin < 0.0:
            raise ValueError("min_top1_margin must be non-negative")
        if self.max_indegree_quantile is not None and not (0.0 < self.max_indegree_quantile <= 1.0):
            raise ValueError("max_indegree_quantile must satisfy 0 < value <= 1")
        if self.indegree_top_k is not None and self.indegree_top_k <= 0:
            raise ValueError("indegree_top_k must be positive when provided")
        if self.max_rows_per_cluster is not None and self.max_rows_per_cluster <= 0:
            raise ValueError("max_rows_per_cluster must be positive when provided")


def select_pseudo_label_rows(
    *,
    bundle: dict[str, object],
    prior: dict[str, object],
    selection: PseudoLabelSelectionConfig,
    feature_names: list[str],
    label_prefix: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selector_meta = _selector_meta_df(bundle)
    if selector_meta.empty:
        summary = {
            "selection_stage_counts": [{"stage": "initial", "row_count": 0, "cluster_count": 0}],
            "selection_thresholds": {},
            "final_row_count": 0,
            "final_cluster_count": 0,
        }
        return pd.DataFrame(columns=_SELECTOR_COLUMNS), summary

    _require_columns(selector_meta, feature_names, "selector_meta_df")
    topk_idx, topk_sim, top1_margin = _bundle_arrays(bundle)
    if selector_meta.shape[0] != topk_idx.shape[0]:
        raise ValueError("selector_meta_df must align with topk arrays")

    allowed_nodes = np.ones(topk_idx.shape[0], dtype=bool)
    stage_counts: list[dict[str, int]] = []
    thresholds: dict[str, float] = {}
    _append_stage(stage_counts, "initial", int(allowed_nodes.sum()))

    top1_score = topk_sim[:, 0].astype(np.float32, copy=False)
    if selection.min_top1_score is not None:
        allowed_nodes &= top1_score >= float(selection.min_top1_score)
        thresholds["top1_score_min"] = float(selection.min_top1_score)
        _append_stage(stage_counts, "top1_score", int(allowed_nodes.sum()))

    if selection.min_top1_margin is not None:
        allowed_nodes &= top1_margin >= float(selection.min_top1_margin)
        thresholds["top1_margin_min"] = float(selection.min_top1_margin)
        _append_stage(stage_counts, "top1_margin", int(allowed_nodes.sum()))

    effective_top_k = topk_idx.shape[1]
    if selection.indegree_top_k is not None:
        effective_top_k = min(int(selection.indegree_top_k), topk_idx.shape[1])
    inbound_degree = np.bincount(
        topk_idx[:, :effective_top_k].reshape(-1),
        minlength=topk_idx.shape[0],
    ).astype(np.int64, copy=False)
    if selection.max_indegree_quantile is not None:
        indegree_threshold = _quantile(inbound_degree, selection.max_indegree_quantile)
        allowed_nodes &= inbound_degree <= indegree_threshold
        thresholds[f"indegree_at_{effective_top_k}_max"] = indegree_threshold
        _append_stage(stage_counts, f"indegree_at_{effective_top_k}", int(allowed_nodes.sum()))

    base_edges = _build_mutual_margin_edges(
        topk_idx=topk_idx,
        top1_margin=top1_margin,
        min_margin=selection.min_top1_margin,
    )
    if base_edges.size == 0:
        summary = {
            "selection_stage_counts": stage_counts,
            "selection_thresholds": thresholds,
            "final_row_count": 0,
            "final_cluster_count": 0,
        }
        return pd.DataFrame(
            columns=_SELECTOR_COLUMNS + ["inbound_degree", "diversity_score"]
        ), summary

    pruned_edges = base_edges[allowed_nodes[base_edges[:, 0]] & allowed_nodes[base_edges[:, 1]]]
    labels, counts = _connected_components_for_edges(len(selector_meta), pruned_edges)

    mu = np.asarray(prior["mu"], dtype=np.float32)
    scale = np.clip(np.asarray(prior["scale"], dtype=np.float32), _EPSILON, None)
    weight = np.asarray(prior["weight"], dtype=np.float32)

    frames: list[pd.DataFrame] = []
    next_component_id = 0
    for raw_component_id, component_size in enumerate(counts.tolist()):
        if component_size < selection.min_cluster_size:
            continue
        if selection.max_cluster_size is not None and component_size > selection.max_cluster_size:
            continue
        component_rows = np.flatnonzero(labels == raw_component_id)
        if component_rows.size == 0:
            continue
        x_comp = (
            selector_meta.iloc[component_rows]
            .loc[:, feature_names]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )
        centroid = np.median(x_comp, axis=0).astype(np.float32, copy=False)
        prior_distance = float(
            _weighted_prior_distance(
                centroid.reshape(1, -1),
                mu=mu,
                scale=scale,
                weight=weight,
            )[0]
        )
        feature_std = np.std(x_comp, axis=0, ddof=0).astype(np.float32, copy=False)
        diversity_score = float(np.sum(weight * (feature_std / scale), dtype=np.float32))
        if prior_distance > float(prior["prior_distance_threshold"]):
            continue
        if diversity_score < float(prior["diversity_floor"]):
            continue
        frames.append(
            _component_frame(
                selector_meta=selector_meta,
                topk_idx=topk_idx,
                topk_sim=topk_sim,
                top1_margin=top1_margin,
                row_indices=component_rows,
                component_id=next_component_id,
                prefix=label_prefix,
                extra_columns={
                    "inbound_degree": inbound_degree,
                    "prior_distance": prior_distance,
                    "diversity_score": diversity_score,
                },
            )
        )
        next_component_id += 1

    accepted_df = (
        pd.concat(frames, axis=0, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_SELECTOR_COLUMNS + ["inbound_degree", "diversity_score"])
    )
    _append_stage(
        stage_counts,
        "prior_distance",
        int(len(accepted_df)),
        int(accepted_df["component_id"].nunique()) if not accepted_df.empty else 0,
    )

    if selection.max_rows_per_cluster is not None:
        accepted_df = _limit_rows_per_cluster(
            accepted_df,
            max_rows_per_cluster=selection.max_rows_per_cluster,
        )
        thresholds["max_rows_per_cluster"] = float(selection.max_rows_per_cluster)
        _append_stage(
            stage_counts,
            "cluster_cap",
            int(len(accepted_df)),
            int(accepted_df["component_id"].nunique()) if not accepted_df.empty else 0,
        )

    accepted_df = accepted_df.sort_values(
        by=["component_id", "row_index"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary = {
        "selection_stage_counts": stage_counts,
        "selection_thresholds": thresholds,
        "final_row_count": int(len(accepted_df)),
        "final_cluster_count": int(accepted_df["component_id"].nunique())
        if not accepted_df.empty
        else 0,
        "prior_distance_max": float(prior["prior_distance_threshold"]),
        "diversity_floor_min": float(prior["diversity_floor"]),
        "indegree_top_k": int(effective_top_k),
    }
    if "duration_bucket" in accepted_df.columns:
        summary["final_duration_bucket_count"] = int(
            accepted_df["duration_bucket"].astype(str).nunique()
        )
    if "profile_bucket" in accepted_df.columns:
        summary["final_profile_bucket_count"] = int(
            accepted_df["profile_bucket"].astype(str).nunique()
        )
    return accepted_df, summary
