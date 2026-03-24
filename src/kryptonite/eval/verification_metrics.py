"""Verification metrics for score files produced by baseline pipelines."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
class _ThresholdMetrics:
    threshold: float
    false_accept_rate: float
    false_reject_rate: float


def load_verification_score_rows(
    path: Path | str,
) -> list[dict[str, Any]]:
    score_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(score_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {score_path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"No score rows found in {score_path}")
    return rows


def compute_verification_metrics(
    score_rows: list[dict[str, Any]],
    *,
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

    normalized_rows = _normalize_score_rows(score_rows)
    positive_count = sum(1 for row in normalized_rows if row["label"] == 1)
    negative_count = sum(1 for row in normalized_rows if row["label"] == 0)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Verification metrics require both positive and negative trials.")

    operating_points = _build_operating_points(normalized_rows)
    eer, eer_threshold = _compute_eer(operating_points)
    min_dcf, min_dcf_threshold = _compute_min_dcf(
        operating_points,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )

    return VerificationMetricsSummary(
        trial_count=len(normalized_rows),
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


def _normalize_score_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    normalized: list[dict[str, float | int]] = []
    for index, row in enumerate(score_rows, start=1):
        try:
            label = int(row["label"])
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid verification score row at index {index}: {row!r}") from exc
        if label not in {0, 1}:
            raise ValueError(f"Verification labels must be 0 or 1, got {label!r} at row {index}.")
        if not math.isfinite(score):
            raise ValueError(f"Verification score must be finite, got {score!r} at row {index}.")
        normalized.append({"label": label, "score": score})
    return normalized


def _build_operating_points(score_rows: list[dict[str, float | int]]) -> list[_ThresholdMetrics]:
    positives = sum(1 for row in score_rows if row["label"] == 1)
    negatives = sum(1 for row in score_rows if row["label"] == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("Verification metrics require both positive and negative labels.")

    sorted_rows = sorted(score_rows, key=lambda row: float(row["score"]), reverse=True)
    points = [
        _ThresholdMetrics(
            threshold=float("inf"),
            false_accept_rate=0.0,
            false_reject_rate=1.0,
        )
    ]

    accepted_positives = 0
    accepted_negatives = 0
    index = 0
    while index < len(sorted_rows):
        threshold = float(sorted_rows[index]["score"])
        while index < len(sorted_rows) and float(sorted_rows[index]["score"]) == threshold:
            if int(sorted_rows[index]["label"]) == 1:
                accepted_positives += 1
            else:
                accepted_negatives += 1
            index += 1
        false_accept_rate = accepted_negatives / negatives
        false_reject_rate = (positives - accepted_positives) / positives
        points.append(
            _ThresholdMetrics(
                threshold=threshold,
                false_accept_rate=false_accept_rate,
                false_reject_rate=false_reject_rate,
            )
        )
    return points


def _compute_eer(points: list[_ThresholdMetrics]) -> tuple[float, float]:
    best_point = min(
        points,
        key=lambda point: abs(point.false_accept_rate - point.false_reject_rate),
    )
    best_eer = (best_point.false_accept_rate + best_point.false_reject_rate) / 2.0
    best_threshold = best_point.threshold

    for left, right in zip(points, points[1:], strict=True):
        left_delta = left.false_accept_rate - left.false_reject_rate
        right_delta = right.false_accept_rate - right.false_reject_rate

        if math.isclose(left_delta, 0.0, abs_tol=1e-12):
            return left.false_accept_rate, left.threshold
        if math.isclose(right_delta, 0.0, abs_tol=1e-12):
            return right.false_accept_rate, right.threshold
        if left_delta * right_delta > 0.0:
            continue

        ratio = left_delta / (left_delta - right_delta)
        eer = left.false_accept_rate + ratio * (
            right.false_accept_rate - left.false_accept_rate
        )
        if math.isfinite(left.threshold) and math.isfinite(right.threshold):
            threshold = left.threshold + ratio * (right.threshold - left.threshold)
        else:
            threshold = right.threshold
        return eer, threshold

    return best_eer, best_threshold


def _compute_min_dcf(
    points: list[_ThresholdMetrics],
    *,
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> tuple[float, float]:
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    if default_cost <= 0.0:
        raise ValueError("Default detection cost must be positive.")

    def normalized_cost(point: _ThresholdMetrics) -> float:
        raw_cost = (
            c_miss * point.false_reject_rate * p_target
            + c_fa * point.false_accept_rate * (1.0 - p_target)
        )
        return raw_cost / default_cost

    best_point = min(points, key=normalized_cost)
    return normalized_cost(best_point), best_point.threshold
