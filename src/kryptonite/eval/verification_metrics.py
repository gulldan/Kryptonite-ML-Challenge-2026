"""Verification metrics and operating-point computations for score files."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .verification_data import load_verification_score_rows


@dataclass(frozen=True, slots=True)
class VerificationMetricsSummary:
    trial_count: int
    positive_count: int
    negative_count: int
    eer: float
    eer_threshold: float
    min_dcf: float
    min_dcf_threshold: float
    p_target: float
    c_miss: float
    c_fa: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationOperatingPoint:
    threshold: float
    true_accept_count: int
    true_reject_count: int
    false_accept_count: int
    false_reject_count: int
    true_accept_rate: float
    true_reject_rate: float
    false_accept_rate: float
    false_reject_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": (None if not math.isfinite(self.threshold) else self.threshold),
            "true_accept_count": self.true_accept_count,
            "true_reject_count": self.true_reject_count,
            "false_accept_count": self.false_accept_count,
            "false_reject_count": self.false_reject_count,
            "true_accept_rate": self.true_accept_rate,
            "true_reject_rate": self.true_reject_rate,
            "false_accept_rate": self.false_accept_rate,
            "false_reject_rate": self.false_reject_rate,
        }


def normalize_verification_score_rows(
    score_rows: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    labels: list[int] = []
    scores: list[float] = []
    for index, row in enumerate(score_rows, start=1):
        try:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid verification score row at index {index}: {row!r}") from exc
    normalized_labels, normalized_scores = normalize_verification_score_arrays(labels, scores)
    return [
        {"label": int(label), "score": float(score)}
        for label, score in zip(normalized_labels, normalized_scores, strict=True)
    ]


def normalize_verification_score_arrays(
    labels: Any,
    scores: Any,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_labels = np.asarray(labels, dtype=np.int8)
    normalized_scores = np.asarray(scores, dtype=np.float64)
    if normalized_labels.ndim != 1 or normalized_scores.ndim != 1:
        raise ValueError("Verification labels and scores must be 1D arrays.")
    if normalized_labels.shape[0] != normalized_scores.shape[0]:
        raise ValueError(
            "Verification labels and scores must have the same length; "
            f"got {normalized_labels.shape[0]} and {normalized_scores.shape[0]}."
        )
    if normalized_labels.shape[0] == 0:
        raise ValueError("Verification metrics require at least one score.")
    if np.any((normalized_labels != 0) & (normalized_labels != 1)):
        raise ValueError("Verification labels must be 0 or 1.")
    if not np.isfinite(normalized_scores).all():
        raise ValueError("Verification scores must be finite.")
    return normalized_labels, normalized_scores


def build_verification_operating_points(
    score_rows: list[dict[str, Any]] | list[dict[str, float | int]],
) -> list[VerificationOperatingPoint]:
    labels, scores = _labels_and_scores_from_rows(score_rows)
    return build_verification_operating_points_from_arrays(labels=labels, scores=scores)


def build_verification_operating_points_from_arrays(
    *,
    labels: Any,
    scores: Any,
    max_points: int | None = None,
) -> list[VerificationOperatingPoint]:
    (
        positive_count,
        negative_count,
        thresholds,
        accepted_positives,
        accepted_negatives,
    ) = _build_operating_point_arrays(labels=labels, scores=scores)
    selected_indices = np.arange(thresholds.shape[0], dtype=np.int64)
    if max_points is not None and thresholds.shape[0] + 1 > max_points:
        selected_indices = _downsample_operating_point_indices(
            point_count=thresholds.shape[0],
            max_points=max_points - 1,
        )

    points = [
        VerificationOperatingPoint(
            threshold=float("inf"),
            true_accept_count=0,
            true_reject_count=negative_count,
            false_accept_count=0,
            false_reject_count=positive_count,
            true_accept_rate=0.0,
            true_reject_rate=1.0,
            false_accept_rate=0.0,
            false_reject_rate=1.0,
        )
    ]
    for point_index in selected_indices.tolist():
        true_accept_count = int(accepted_positives[point_index])
        false_accept_count = int(accepted_negatives[point_index])
        false_reject_count = positive_count - true_accept_count
        true_reject_count = negative_count - false_accept_count
        points.append(
            VerificationOperatingPoint(
                threshold=float(thresholds[point_index]),
                true_accept_count=true_accept_count,
                true_reject_count=true_reject_count,
                false_accept_count=false_accept_count,
                false_reject_count=false_reject_count,
                true_accept_rate=true_accept_count / positive_count,
                true_reject_rate=true_reject_count / negative_count,
                false_accept_rate=false_accept_count / negative_count,
                false_reject_rate=false_reject_count / positive_count,
            )
        )
    return points


def compute_verification_metrics(
    score_rows: list[dict[str, Any]],
    *,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> VerificationMetricsSummary:
    labels, scores = _labels_and_scores_from_rows(score_rows)
    return compute_verification_metrics_from_arrays(
        labels=labels,
        scores=scores,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )


def compute_verification_metrics_from_arrays(
    *,
    labels: Any,
    scores: Any,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> VerificationMetricsSummary:
    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must be within (0, 1)")
    if c_miss <= 0.0:
        raise ValueError("c_miss must be positive")
    if c_fa <= 0.0:
        raise ValueError("c_fa must be positive")

    (
        positive_count,
        negative_count,
        thresholds,
        accepted_positives,
        accepted_negatives,
    ) = _build_operating_point_arrays(labels=labels, scores=scores)
    false_accept_rates = accepted_negatives / negative_count
    false_reject_rates = (positive_count - accepted_positives) / positive_count
    eer, eer_threshold = _compute_eer_from_arrays(
        thresholds=thresholds,
        false_accept_rates=false_accept_rates,
        false_reject_rates=false_reject_rates,
    )
    min_dcf, min_dcf_threshold = _compute_min_dcf_from_arrays(
        thresholds=thresholds,
        false_accept_rates=false_accept_rates,
        false_reject_rates=false_reject_rates,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )
    return VerificationMetricsSummary(
        trial_count=int(np.asarray(labels).shape[0]),
        positive_count=positive_count,
        negative_count=negative_count,
        eer=round(eer, 6),
        eer_threshold=round(eer_threshold, 6),
        min_dcf=round(min_dcf, 6),
        min_dcf_threshold=round(min_dcf_threshold, 6),
        p_target=round(p_target, 6),
        c_miss=round(c_miss, 6),
        c_fa=round(c_fa, 6),
    )


def _labels_and_scores_from_rows(
    score_rows: list[dict[str, Any]] | list[dict[str, float | int]],
) -> tuple[np.ndarray, np.ndarray]:
    labels = [int(row["label"]) for row in score_rows]
    scores = [float(row["score"]) for row in score_rows]
    return normalize_verification_score_arrays(labels, scores)


def _build_operating_point_arrays(
    *,
    labels: Any,
    scores: Any,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    normalized_labels, normalized_scores = normalize_verification_score_arrays(labels, scores)
    positive_count = int((normalized_labels == 1).sum())
    negative_count = int((normalized_labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Verification metrics require both positive and negative trials.")

    order = np.argsort(normalized_scores, kind="mergesort")[::-1]
    sorted_scores = normalized_scores[order]
    sorted_labels = normalized_labels[order]
    accepted_positives = np.cumsum(sorted_labels == 1, dtype=np.int64)
    accepted_negatives = np.cumsum(sorted_labels == 0, dtype=np.int64)
    threshold_end_indices = np.flatnonzero(np.r_[np.diff(sorted_scores) != 0.0, True]).astype(
        np.int64, copy=False
    )
    thresholds = sorted_scores[threshold_end_indices]
    return (
        positive_count,
        negative_count,
        thresholds,
        accepted_positives[threshold_end_indices],
        accepted_negatives[threshold_end_indices],
    )


def _compute_eer_from_arrays(
    *,
    thresholds: np.ndarray,
    false_accept_rates: np.ndarray,
    false_reject_rates: np.ndarray,
) -> tuple[float, float]:
    best_index = int(np.argmin(np.abs(false_accept_rates - false_reject_rates)))
    best_eer = float((false_accept_rates[best_index] + false_reject_rates[best_index]) / 2.0)
    best_threshold = float(thresholds[best_index])

    deltas = false_accept_rates - false_reject_rates
    zero_match = np.isclose(deltas, 0.0, atol=1e-12)
    if zero_match.any():
        exact_index = int(np.flatnonzero(zero_match)[0])
        return float(false_accept_rates[exact_index]), float(thresholds[exact_index])

    left_deltas = deltas[:-1]
    right_deltas = deltas[1:]
    crossing_indices = np.flatnonzero(left_deltas * right_deltas < 0.0)
    if crossing_indices.size:
        left_index = int(crossing_indices[0])
        right_index = left_index + 1
        ratio = left_deltas[left_index] / (left_deltas[left_index] - right_deltas[left_index])
        eer = false_accept_rates[left_index] + ratio * (
            false_accept_rates[right_index] - false_accept_rates[left_index]
        )
        threshold = thresholds[left_index] + ratio * (
            thresholds[right_index] - thresholds[left_index]
        )
        return float(eer), float(threshold)
    return best_eer, best_threshold


def _compute_min_dcf_from_arrays(
    *,
    thresholds: np.ndarray,
    false_accept_rates: np.ndarray,
    false_reject_rates: np.ndarray,
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> tuple[float, float]:
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    if default_cost <= 0.0:
        raise ValueError("Default detection cost must be positive.")
    normalized_costs = (
        c_miss * false_reject_rates * p_target + c_fa * false_accept_rates * (1.0 - p_target)
    ) / default_cost
    best_index = int(np.argmin(normalized_costs))
    return float(normalized_costs[best_index]), float(thresholds[best_index])


def _downsample_operating_point_indices(
    *,
    point_count: int,
    max_points: int,
) -> np.ndarray:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)
    indices = np.linspace(0, point_count - 1, num=max_points, dtype=np.int64)
    return np.unique(indices)


__all__ = [
    "VerificationMetricsSummary",
    "VerificationOperatingPoint",
    "build_verification_operating_points",
    "build_verification_operating_points_from_arrays",
    "compute_verification_metrics",
    "compute_verification_metrics_from_arrays",
    "load_verification_score_rows",
    "normalize_verification_score_arrays",
    "normalize_verification_score_rows",
]
