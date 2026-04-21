"""Configuration dataclasses for community-style retrieval reranking."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class ClusterFirstConfig:
    experiment_id: str
    edge_top: int = 50
    reciprocal_top: int = 100
    rank_top: int = 300
    iterations: int = 8
    cluster_min_size: int = 11
    cluster_max_size: int = 220
    cluster_min_candidates: int = 6
    shared_top: int = 50
    shared_min_count: int = 2
    reciprocal_bonus: float = 0.03
    density_penalty: float = 0.02
    edge_score_quantile: float | None = None
    edge_min_score: float | None = None
    shared_weight: float = 0.04
    rank_weight: float = 0.02
    self_weight: float = 0.0
    label_size_penalty: float = 0.0
    split_oversized: bool = True
    split_edge_top: int = 12
