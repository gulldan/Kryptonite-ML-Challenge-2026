"""Stability helpers for multi-teacher soft pseudo-label probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kryptonite.eda.community import ClusterFirstConfig, cluster_first_rerank

FuseCallback = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]


def stability_scores(
    *,
    args: Any,
    teachers: list[Any],
    main_indices: np.ndarray,
    main_labels: np.ndarray,
    fuse_teachers: FuseCallback,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure co-membership stability under configured teacher-dropout variants."""

    if not args.stability_drop_teacher:
        return np.ones(main_labels.shape[0], dtype=np.float32), {"stability_variant_count": 0}
    variants: list[np.ndarray] = []
    for name in args.stability_drop_teacher:
        kept = [teacher for teacher in teachers if teacher.spec.name != name]
        if len(kept) == len(teachers):
            raise ValueError(f"Unknown --stability-drop-teacher={name!r}.")
        indices, scores, _, _ = fuse_teachers(
            teachers=kept,
            output_top_k=args.top_cache_k,
            score_z_weight=args.score_z_weight,
            raw_score_weight=args.raw_score_weight,
            rank_weight=args.rank_weight,
            reciprocal_bonus=args.reciprocal_bonus,
            agreement_bonus=args.agreement_bonus,
            shared_weight=args.shared_weight,
            hubness_penalty=args.hubness_penalty,
        )
        _, _, labels, _ = cluster_first_rerank(
            indices=indices,
            scores=scores,
            config=ClusterFirstConfig(
                experiment_id=f"{args.experiment_id}_drop_{name}",
                edge_top=args.cluster_edge_top,
                reciprocal_top=args.cluster_reciprocal_top,
                rank_top=args.cluster_rank_top,
                iterations=args.cluster_iterations,
                cluster_min_size=args.cluster_min_size,
                cluster_max_size=args.cluster_max_size,
                cluster_min_candidates=args.cluster_min_candidates,
                shared_top=args.cluster_shared_top,
                shared_min_count=args.cluster_shared_min_count,
                reciprocal_bonus=args.cluster_reciprocal_bonus,
                density_penalty=args.cluster_density_penalty,
                edge_score_quantile=args.cluster_edge_score_quantile,
                edge_min_score=args.cluster_edge_min_score,
                shared_weight=args.cluster_shared_weight,
                rank_weight=args.cluster_rank_weight,
                self_weight=args.cluster_self_weight,
                label_size_penalty=args.cluster_label_size_penalty,
                split_oversized=not args.no_split_oversized,
                split_edge_top=args.split_edge_top,
            ),
            top_k=10,
        )
        variants.append(
            _co_membership_stability(main_indices, main_labels, labels, args.stability_top)
        )
    stacked = np.stack(variants, axis=0)
    stability = stacked.mean(axis=0).astype(np.float32, copy=False)
    return stability, {
        "stability_variant_count": len(variants),
        "stability_drop_teacher": list(args.stability_drop_teacher),
        "stability_mean": float(stability.mean()),
        "stability_p10": float(np.quantile(stability, 0.10)),
        "stability_p50": float(np.quantile(stability, 0.50)),
    }


def _co_membership_stability(
    main_indices: np.ndarray,
    main_labels: np.ndarray,
    variant_labels: np.ndarray,
    top_k: int,
) -> np.ndarray:
    top_k = min(top_k, main_indices.shape[1])
    out = np.empty(main_labels.shape[0], dtype=np.float32)
    for row in range(main_labels.shape[0]):
        candidates = main_indices[row, :top_k]
        main_same = main_labels[candidates] == main_labels[row]
        variant_same = variant_labels[candidates] == variant_labels[row]
        union = np.logical_or(main_same, variant_same).sum()
        out[row] = (
            1.0 if union == 0 else float(np.logical_and(main_same, variant_same).sum() / union)
        )
    return out
