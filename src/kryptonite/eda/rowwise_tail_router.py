"""Query-conditioned routing across retrieval tail submissions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kryptonite.eda.community import (
    LabelPropagationConfig,
    adjusted_ranking,
    mutual_mask,
    shared_neighbor_mask,
    weighted_label_propagation,
)


@dataclass(frozen=True, slots=True)
class RowwiseTailRouterConfig:
    """Heuristic gates for query-conditioned tail selection."""

    default_policy: str = "classaware_c4"
    full_policy: str = "full_c4"
    exact_policy: str = "exact"
    weak_policy: str = "weak_c4"
    reciprocal_policy: str = "reciprocal_only"
    soup_policy: str | None = "soup_c4"
    classaware_policy: str | None = "classaware_c4"
    low_margin: float = 0.015
    high_margin: float = 0.045
    low_reciprocal_support: int = 1
    high_reciprocal_support: int = 3
    suspicious_label_max_size: int = 95
    extreme_label_max_size: int = 140
    min_same_label_candidates: int = 3
    strong_same_label_candidates: int = 6
    min_consensus_overlap: int = 5
    strong_consensus_overlap: int = 7
    low_class_entropy: float = 0.55
    high_class_entropy: float = 0.85
    short_duration_s: float = 2.0
    soup_consensus_margin: float = 0.02


@dataclass(frozen=True, slots=True)
class GraphTailDiagnostics:
    """Per-row features used by the row-wise router."""

    margin_top1_top10: np.ndarray
    reciprocal_support: np.ndarray
    labels: np.ndarray
    label_sizes: np.ndarray
    row_label_size: np.ndarray
    same_label_candidates: np.ndarray
    label_usable: np.ndarray
    label_confidence: np.ndarray


def graph_tail_diagnostics(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    config: LabelPropagationConfig,
    top_k: int = 10,
) -> GraphTailDiagnostics:
    """Compute per-query graph confidence features for the C4-style tail."""

    if indices.shape != scores.shape:
        raise ValueError("indices and scores must have the same shape")
    if indices.ndim != 2:
        raise ValueError("indices and scores must be 2D arrays")
    required_top = max(config.edge_top, config.reciprocal_top, config.rank_top, top_k)
    if indices.shape[1] < required_top:
        raise ValueError(
            f"top-k cache width {indices.shape[1]} is smaller than required {required_top}"
        )

    rank_indices, _, _ = adjusted_ranking(
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
    labels, _ = weighted_label_propagation(
        indices=indices[:, : config.edge_top],
        scores=scores[:, : config.edge_top],
        edge_mask=edge_mask,
        iterations=config.iterations,
    )
    label_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    row_label_size = label_sizes[labels].astype(np.int32, copy=False)
    same_label_candidates = np.asarray(
        [
            int(np.count_nonzero(labels[rank_indices[row]] == labels[row]))
            for row in range(indices.shape[0])
        ],
        dtype=np.int16,
    )
    label_usable = (
        (row_label_size >= config.label_min_size)
        & (row_label_size <= config.label_max_size)
        & (same_label_candidates >= config.label_min_candidates)
    )
    max_possible_same = np.maximum(np.minimum(row_label_size - 1, config.rank_top), 1)
    label_confidence = same_label_candidates.astype(np.float32) / max_possible_same.astype(
        np.float32
    )
    reciprocal_support = mutual_mask(
        indices, edge_top=top_k, reciprocal_top=config.reciprocal_top
    ).sum(axis=1)
    margin_column = min(top_k - 1, scores.shape[1] - 1)
    margin = scores[:, 0] - scores[:, margin_column]
    return GraphTailDiagnostics(
        margin_top1_top10=margin.astype(np.float32, copy=False),
        reciprocal_support=reciprocal_support.astype(np.int16, copy=False),
        labels=labels,
        label_sizes=label_sizes,
        row_label_size=row_label_size,
        same_label_candidates=same_label_candidates,
        label_usable=label_usable,
        label_confidence=label_confidence.astype(np.float32, copy=False),
    )


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return normalized row-wise entropy for truncated class probabilities."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    values = np.asarray(probabilities, dtype=np.float32)
    clipped = np.clip(values, 0.0, 1.0)
    row_sum = clipped.sum(axis=1, keepdims=True)
    normalized = clipped / np.maximum(row_sum, 1e-12)
    entropy = -(normalized * np.log(np.maximum(normalized, 1e-12))).sum(axis=1)
    return (entropy / np.log(max(values.shape[1], 2))).astype(np.float32, copy=False)


def policy_consensus_scores(candidates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Score each policy by how much its row top-k agrees with all candidate policies."""

    _validate_candidate_shapes(candidates)
    names = list(candidates)
    row_count, top_k = candidates[names[0]].shape
    scores = {name: np.zeros(row_count, dtype=np.float32) for name in names}
    for row in range(row_count):
        votes: dict[int, int] = {}
        for name in names:
            for candidate in candidates[name][row].tolist():
                candidate_int = int(candidate)
                votes[candidate_int] = votes.get(candidate_int, 0) + 1
        for name in names:
            scores[name][row] = sum(votes[int(value)] for value in candidates[name][row]) / (
                top_k * len(names)
            )
    return scores


def policy_overlap_counts(
    *,
    candidates: dict[str, np.ndarray],
    reference_policy: str,
) -> dict[str, np.ndarray]:
    """Return row-wise top-k set overlap against one reference policy."""

    _validate_candidate_shapes(candidates)
    if reference_policy not in candidates:
        raise KeyError(f"Unknown reference policy: {reference_policy}")
    row_count = candidates[reference_policy].shape[0]
    reference = candidates[reference_policy]
    overlaps: dict[str, np.ndarray] = {}
    for name, values in candidates.items():
        counts = np.empty(row_count, dtype=np.int16)
        for row in range(row_count):
            counts[row] = len(set(reference[row].tolist()) & set(values[row].tolist()))
        overlaps[name] = counts
    return overlaps


def route_tail_policies(
    *,
    candidates: dict[str, np.ndarray],
    diagnostics: GraphTailDiagnostics,
    config: RowwiseTailRouterConfig,
    class_entropy: np.ndarray | None = None,
    durations_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Select one retrieval policy per query and return routed neighbours."""

    _validate_candidate_shapes(candidates)
    _validate_policy_names(candidates, config)
    row_count, top_k = candidates[config.default_policy].shape
    _validate_feature_length(row_count, diagnostics)
    if class_entropy is not None and class_entropy.shape[0] != row_count:
        raise ValueError("class_entropy row count must match candidates")
    if durations_s is not None and durations_s.shape[0] != row_count:
        raise ValueError("durations_s row count must match candidates")

    consensus = policy_consensus_scores(candidates)
    full_overlaps = policy_overlap_counts(
        candidates=candidates, reference_policy=config.full_policy
    )
    selected_names: list[str] = []
    reason_names: list[str] = []
    selected = np.empty((row_count, top_k), dtype=np.int64)
    default_entropy = np.full(row_count, np.nan, dtype=np.float32)
    entropy = class_entropy if class_entropy is not None else default_entropy
    default_durations = np.full(row_count, np.nan, dtype=np.float32)
    durations = durations_s if durations_s is not None else default_durations

    for row in range(row_count):
        policy, reason = _select_policy_for_row(
            row=row,
            candidates=candidates,
            diagnostics=diagnostics,
            config=config,
            consensus=consensus,
            full_overlaps=full_overlaps,
            class_entropy=entropy,
            durations_s=durations,
        )
        selected_names.append(policy)
        reason_names.append(reason)
        selected[row] = candidates[policy][row]
    details = {
        "selected_policy": np.asarray(selected_names, dtype=object),
        "selected_reason": np.asarray(reason_names, dtype=object),
        **{f"consensus_{name}": values for name, values in consensus.items()},
        **{f"overlap_full_{name}": values for name, values in full_overlaps.items()},
    }
    return selected, np.asarray(selected_names, dtype=object), details


def _select_policy_for_row(
    *,
    row: int,
    candidates: dict[str, np.ndarray],
    diagnostics: GraphTailDiagnostics,
    config: RowwiseTailRouterConfig,
    consensus: dict[str, np.ndarray],
    full_overlaps: dict[str, np.ndarray],
    class_entropy: np.ndarray,
    durations_s: np.ndarray,
) -> tuple[str, str]:
    margin = float(diagnostics.margin_top1_top10[row])
    reciprocal_support = int(diagnostics.reciprocal_support[row])
    label_size = int(diagnostics.row_label_size[row])
    same_label = int(diagnostics.same_label_candidates[row])
    label_usable = bool(diagnostics.label_usable[row])
    entropy = float(class_entropy[row])
    duration = float(durations_s[row])

    suspicious = (
        (not label_usable)
        or margin < config.low_margin
        or reciprocal_support <= config.low_reciprocal_support
        or same_label < config.min_same_label_candidates
        or label_size > config.suspicious_label_max_size
    )
    extreme = (
        label_size > config.extreme_label_max_size
        or reciprocal_support == 0
        or (
            not np.isnan(duration)
            and duration < config.short_duration_s
            and margin < config.high_margin
        )
    )
    strong = (
        label_usable
        and margin >= config.high_margin
        and reciprocal_support >= config.high_reciprocal_support
        and same_label >= config.strong_same_label_candidates
        and label_size <= config.suspicious_label_max_size
    )

    if config.classaware_policy is not None and config.classaware_policy in candidates:
        class_overlap = int(full_overlaps[config.classaware_policy][row])
        if (
            not np.isnan(entropy)
            and entropy <= config.low_class_entropy
            and class_overlap >= config.min_consensus_overlap
            and not extreme
        ):
            return config.classaware_policy, "low_entropy_classaware_agreement"
        if (
            strong
            and (np.isnan(entropy) or entropy <= config.high_class_entropy)
            and class_overlap >= config.strong_consensus_overlap
        ):
            return config.classaware_policy, "strong_graph_classaware_agreement"

    if config.soup_policy is not None and config.soup_policy in candidates and not suspicious:
        soup_advantage = consensus[config.soup_policy][row] - consensus[config.full_policy][row]
        if soup_advantage >= config.soup_consensus_margin:
            return config.soup_policy, "soup_consensus_advantage"

    if suspicious:
        if extreme and config.exact_policy in candidates:
            return config.exact_policy, "extreme_low_graph_confidence_exact"
        if (
            config.weak_policy in candidates
            and full_overlaps[config.weak_policy][row] >= config.min_consensus_overlap
        ):
            return config.weak_policy, "weak_c4_low_graph_confidence"
        if config.reciprocal_policy in candidates:
            return config.reciprocal_policy, "reciprocal_low_graph_confidence"
        if config.exact_policy in candidates:
            return config.exact_policy, "exact_low_graph_confidence"

    if strong:
        return config.full_policy, "strong_graph_full_c4"
    return config.default_policy, "default_policy"


def _validate_candidate_shapes(candidates: dict[str, np.ndarray]) -> None:
    if not candidates:
        raise ValueError("At least one candidate policy is required")
    shapes = {name: values.shape for name, values in candidates.items()}
    first_shape = next(iter(shapes.values()))
    for name, shape in shapes.items():
        if shape != first_shape:
            raise ValueError(f"Candidate {name} has shape {shape}, expected {first_shape}")
        if len(shape) != 2:
            raise ValueError(f"Candidate {name} must be a 2D neighbour matrix")


def _validate_policy_names(
    candidates: dict[str, np.ndarray], config: RowwiseTailRouterConfig
) -> None:
    required = [
        config.default_policy,
        config.full_policy,
        config.exact_policy,
        config.weak_policy,
        config.reciprocal_policy,
    ]
    missing = [name for name in required if name not in candidates]
    if missing:
        raise KeyError(f"Missing required candidate policies: {missing}")


def _validate_feature_length(row_count: int, diagnostics: GraphTailDiagnostics) -> None:
    fields: dict[str, np.ndarray] = {
        "margin_top1_top10": diagnostics.margin_top1_top10,
        "reciprocal_support": diagnostics.reciprocal_support,
        "labels": diagnostics.labels,
        "row_label_size": diagnostics.row_label_size,
        "same_label_candidates": diagnostics.same_label_candidates,
        "label_usable": diagnostics.label_usable,
        "label_confidence": diagnostics.label_confidence,
    }
    for name, values in fields.items():
        if values.shape[0] != row_count:
            raise ValueError(f"{name} row count {values.shape[0]} != {row_count}")
