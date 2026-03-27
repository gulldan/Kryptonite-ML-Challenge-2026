"""Lightweight trainable score-normalization helpers for offline verification experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .score_normalization import (
    build_score_normalization_context,
    compute_identifier_cohort_statistics,
    resolve_trial_score_records,
)

DEFAULT_TAS_NORM_TOP_K = 100
DEFAULT_TAS_NORM_STD_EPSILON = 1e-6
DEFAULT_TAS_NORM_LEARNING_RATE = 0.1
DEFAULT_TAS_NORM_MAX_STEPS = 300
DEFAULT_TAS_NORM_L2_REGULARIZATION = 1e-3
DEFAULT_TAS_NORM_EARLY_STOPPING_PATIENCE = 25
DEFAULT_TAS_NORM_MIN_RELATIVE_LOSS_IMPROVEMENT = 1e-4

TAS_NORM_MODEL_JSON_NAME = "tas_norm_model.json"
VERIFICATION_TAS_NORM_SCORES_JSONL_NAME = "verification_scores_tas_norm.jsonl"

TAS_NORM_FEATURE_NAMES = (
    "raw_score",
    "as_norm_score",
    "cohort_mean",
    "cohort_std",
    "cohort_mean_gap",
    "cohort_std_gap",
)


@dataclass(frozen=True, slots=True)
class TasNormTrainingConfig:
    learning_rate: float = DEFAULT_TAS_NORM_LEARNING_RATE
    max_steps: int = DEFAULT_TAS_NORM_MAX_STEPS
    l2_regularization: float = DEFAULT_TAS_NORM_L2_REGULARIZATION
    early_stopping_patience: int = DEFAULT_TAS_NORM_EARLY_STOPPING_PATIENCE
    min_relative_loss_improvement: float = DEFAULT_TAS_NORM_MIN_RELATIVE_LOSS_IMPROVEMENT
    balance_classes: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.min_relative_loss_improvement < 0.0:
            raise ValueError("min_relative_loss_improvement must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TasNormFeatureBatch:
    feature_names: tuple[str, ...]
    feature_matrix: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, Any]]
    raw_scores: np.ndarray
    as_norm_scores: np.ndarray
    cohort_size: int
    embedding_dim: int
    top_k: int
    effective_top_k: int
    floored_std_count: int
    excluded_same_speaker_count: int

    @property
    def trial_count(self) -> int:
        return int(self.feature_matrix.shape[0])

    def slice(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        resolved = np.asarray(indices, dtype=np.int64)
        return (
            self.feature_matrix[resolved],
            self.labels[resolved],
            [self.rows[int(i)] for i in resolved],
        )


@dataclass(frozen=True, slots=True)
class TasNormTrainingSummary:
    steps_completed: int
    converged: bool
    initial_loss: float
    final_loss: float
    best_loss: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TasNormModel:
    feature_names: tuple[str, ...]
    feature_offsets: tuple[float, ...]
    feature_scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    training: TasNormTrainingSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "feature_offsets": list(self.feature_offsets),
            "feature_scales": list(self.feature_scales),
            "weights": list(self.weights),
            "bias": self.bias,
            "training": self.training.to_dict(),
        }


def prepare_tas_norm_feature_batch(
    score_rows: list[dict[str, Any]],
    *,
    embeddings_path: Path | str,
    metadata_path: Path | str,
    cohort_bank_root: Path | str,
    top_k: int = DEFAULT_TAS_NORM_TOP_K,
    std_epsilon: float = DEFAULT_TAS_NORM_STD_EPSILON,
    exclude_matching_speakers: bool = True,
    embeddings_key: str = "embeddings",
    ids_key: str | None = "point_ids",
    point_id_field: str = "atlas_point_id",
) -> TasNormFeatureBatch:
    context = build_score_normalization_context(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        cohort_bank_root=cohort_bank_root,
        embeddings_key=embeddings_key,
        ids_key=ids_key,
        point_id_field=point_id_field,
    )
    trial_records = resolve_trial_score_records(score_rows)

    unique_identifiers: list[str] = []
    seen_identifiers: set[str] = set()
    for record in trial_records:
        for identifier in (record.left_identifier, record.right_identifier):
            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)
            unique_identifiers.append(identifier)

    identifier_to_stats, stats_summary = compute_identifier_cohort_statistics(
        context,
        identifiers=tuple(unique_identifiers),
        top_k=top_k,
        std_epsilon=std_epsilon,
        exclude_matching_speakers=exclude_matching_speakers,
    )

    rows: list[dict[str, Any]] = []
    feature_rows: list[list[float]] = []
    raw_scores: list[float] = []
    as_norm_scores: list[float] = []
    labels: list[int] = []

    for raw_row, record in zip(score_rows, trial_records, strict=True):
        left_stats = identifier_to_stats[record.left_identifier]
        right_stats = identifier_to_stats[record.right_identifier]
        as_norm_score = 0.5 * (
            ((record.raw_score - left_stats.mean) / left_stats.std)
            + ((record.raw_score - right_stats.mean) / right_stats.std)
        )
        cohort_mean = 0.5 * (left_stats.mean + right_stats.mean)
        cohort_std = 0.5 * (left_stats.std + right_stats.std)
        cohort_mean_gap = abs(left_stats.mean - right_stats.mean)
        cohort_std_gap = abs(left_stats.std - right_stats.std)

        feature_rows.append(
            [
                record.raw_score,
                as_norm_score,
                cohort_mean,
                cohort_std,
                cohort_mean_gap,
                cohort_std_gap,
            ]
        )
        raw_scores.append(record.raw_score)
        as_norm_scores.append(as_norm_score)
        labels.append(record.label)
        rows.append(
            {
                **raw_row,
                "raw_score": round(record.raw_score, 8),
                "as_norm_score": round(as_norm_score, 8),
                "left_cohort_mean": round(left_stats.mean, 8),
                "right_cohort_mean": round(right_stats.mean, 8),
                "left_cohort_std": round(left_stats.std, 8),
                "right_cohort_std": round(right_stats.std, 8),
                "cohort_mean": round(cohort_mean, 8),
                "cohort_std": round(cohort_std, 8),
                "cohort_mean_gap": round(cohort_mean_gap, 8),
                "cohort_std_gap": round(cohort_std_gap, 8),
            }
        )

    excluded_same_speaker_count = sum(
        stats.excluded_same_speaker_count for stats in identifier_to_stats.values()
    )
    return TasNormFeatureBatch(
        feature_names=TAS_NORM_FEATURE_NAMES,
        feature_matrix=np.asarray(feature_rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.float64),
        rows=rows,
        raw_scores=np.asarray(raw_scores, dtype=np.float64),
        as_norm_scores=np.asarray(as_norm_scores, dtype=np.float64),
        cohort_size=context.cohort_size,
        embedding_dim=context.embedding_dim,
        top_k=top_k,
        effective_top_k=stats_summary.effective_top_k,
        floored_std_count=stats_summary.floored_std_count,
        excluded_same_speaker_count=excluded_same_speaker_count,
    )


def fit_tas_norm_model(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    training_config: TasNormTrainingConfig | None = None,
    feature_names: tuple[str, ...] = TAS_NORM_FEATURE_NAMES,
) -> TasNormModel:
    config = training_config or TasNormTrainingConfig()
    features = np.asarray(feature_matrix, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional.")
    if target.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if features.shape[0] != target.shape[0]:
        raise ValueError("feature_matrix and labels must contain the same number of rows.")
    if features.shape[0] == 0:
        raise ValueError("feature_matrix must not be empty.")
    if len(feature_names) != features.shape[1]:
        raise ValueError("feature_names must match the feature_matrix column count.")
    unique_labels = {int(value) for value in np.unique(target)}
    if unique_labels != {0, 1}:
        raise ValueError("TAS-norm model fitting requires both positive and negative labels.")

    offsets = features.mean(axis=0)
    scales = features.std(axis=0, ddof=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    standardized = (features - offsets) / scales

    weights = np.zeros((standardized.shape[1],), dtype=np.float64)
    positive_rate = float(target.mean())
    bias = 0.0 if not 0.0 < positive_rate < 1.0 else math.log(positive_rate / (1.0 - positive_rate))
    sample_weights = _build_sample_weights(target, balance_classes=config.balance_classes)

    initial_loss = _weighted_logistic_loss(
        logits=standardized @ weights + bias,
        labels=target,
        sample_weights=sample_weights,
        l2_regularization=config.l2_regularization,
        weights=weights,
    )
    best_loss = initial_loss
    best_weights = weights.copy()
    best_bias = bias
    steps_completed = 0
    stagnant_steps = 0
    converged = False

    for step in range(1, config.max_steps + 1):
        logits = standardized @ weights + bias
        probabilities = _sigmoid(logits)
        errors = (probabilities - target) * sample_weights
        grad_weights = (standardized.T @ errors) / float(target.shape[0])
        grad_weights += config.l2_regularization * weights
        grad_bias = float(errors.mean())

        weights = weights - config.learning_rate * grad_weights
        bias = bias - config.learning_rate * grad_bias
        loss = _weighted_logistic_loss(
            logits=standardized @ weights + bias,
            labels=target,
            sample_weights=sample_weights,
            l2_regularization=config.l2_regularization,
            weights=weights,
        )
        steps_completed = step

        relative_improvement = (best_loss - loss) / max(best_loss, 1e-12)
        if relative_improvement > config.min_relative_loss_improvement:
            best_loss = loss
            best_weights = weights.copy()
            best_bias = bias
            stagnant_steps = 0
        else:
            stagnant_steps += 1
            if stagnant_steps >= config.early_stopping_patience:
                converged = True
                break

    final_loss = _weighted_logistic_loss(
        logits=standardized @ best_weights + best_bias,
        labels=target,
        sample_weights=sample_weights,
        l2_regularization=config.l2_regularization,
        weights=best_weights,
    )
    training_summary = TasNormTrainingSummary(
        steps_completed=steps_completed,
        converged=converged or steps_completed < config.max_steps,
        initial_loss=round(float(initial_loss), 8),
        final_loss=round(float(final_loss), 8),
        best_loss=round(float(best_loss), 8),
    )
    return TasNormModel(
        feature_names=feature_names,
        feature_offsets=tuple(float(value) for value in offsets),
        feature_scales=tuple(float(value) for value in scales),
        weights=tuple(float(value) for value in best_weights),
        bias=round(float(best_bias), 8),
        training=training_summary,
    )


def apply_tas_norm_model(
    batch: TasNormFeatureBatch,
    model: TasNormModel,
    *,
    indices: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if batch.feature_names != model.feature_names:
        raise ValueError("TasNormModel feature_names do not match the prepared feature batch.")

    if indices is None:
        resolved_indices = np.arange(batch.trial_count, dtype=np.int64)
    else:
        resolved_indices = np.asarray(indices, dtype=np.int64)
    features = batch.feature_matrix[resolved_indices]
    probabilities = predict_tas_norm_probabilities(features, model)

    score_rows: list[dict[str, Any]] = []
    for row, probability in zip(
        [batch.rows[int(index)] for index in resolved_indices],
        probabilities,
        strict=True,
    ):
        score_rows.append(
            {
                **row,
                "score": round(float(probability), 8),
                "score_normalization": "tas-norm",
            }
        )
    return score_rows


def predict_tas_norm_probabilities(feature_matrix: np.ndarray, model: TasNormModel) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional.")
    if features.shape[1] != len(model.feature_names):
        raise ValueError("feature_matrix column count does not match TasNormModel.feature_names.")

    offsets = np.asarray(model.feature_offsets, dtype=np.float64)
    scales = np.asarray(model.feature_scales, dtype=np.float64)
    weights = np.asarray(model.weights, dtype=np.float64)
    standardized = (features - offsets) / scales
    logits = standardized @ weights + float(model.bias)
    return _sigmoid(logits)


def _build_sample_weights(labels: np.ndarray, *, balance_classes: bool) -> np.ndarray:
    if not balance_classes:
        return np.ones_like(labels, dtype=np.float64)
    positive_count = float(labels.sum())
    negative_count = float(labels.shape[0] - positive_count)
    positive_weight = (
        1.0 if positive_count == 0.0 else 0.5 * float(labels.shape[0]) / positive_count
    )
    negative_weight = (
        1.0 if negative_count == 0.0 else 0.5 * float(labels.shape[0]) / negative_count
    )
    return np.where(labels > 0.5, positive_weight, negative_weight)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _weighted_logistic_loss(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    l2_regularization: float,
    weights: np.ndarray,
) -> float:
    probabilities = _sigmoid(logits)
    probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    losses = -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
    weighted = losses * sample_weights
    return float(weighted.mean() + 0.5 * l2_regularization * float(np.dot(weights, weights)))


__all__ = [
    "DEFAULT_TAS_NORM_EARLY_STOPPING_PATIENCE",
    "DEFAULT_TAS_NORM_L2_REGULARIZATION",
    "DEFAULT_TAS_NORM_LEARNING_RATE",
    "DEFAULT_TAS_NORM_MAX_STEPS",
    "DEFAULT_TAS_NORM_MIN_RELATIVE_LOSS_IMPROVEMENT",
    "DEFAULT_TAS_NORM_STD_EPSILON",
    "DEFAULT_TAS_NORM_TOP_K",
    "TAS_NORM_FEATURE_NAMES",
    "TAS_NORM_MODEL_JSON_NAME",
    "TasNormFeatureBatch",
    "TasNormModel",
    "TasNormTrainingConfig",
    "TasNormTrainingSummary",
    "VERIFICATION_TAS_NORM_SCORES_JSONL_NAME",
    "apply_tas_norm_model",
    "fit_tas_norm_model",
    "predict_tas_norm_probabilities",
    "prepare_tas_norm_feature_batch",
]
