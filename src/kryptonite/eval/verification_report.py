"""Build and write rich verification-evaluation reports."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .verification_data import (
    load_verification_metadata_rows,
    load_verification_trial_rows,
)
from .verification_metrics import (
    VerificationMetricsSummary,
    VerificationOperatingPoint,
    build_verification_operating_points_from_arrays,
    compute_verification_metrics_from_arrays,
    normalize_verification_score_arrays,
)

DEFAULT_SLICE_FIELDS = ("duration_bucket", "gender", "device", "channel", "language")

VERIFICATION_REPORT_JSON_NAME = "verification_eval_report.json"
VERIFICATION_REPORT_MARKDOWN_NAME = "verification_eval_report.md"
VERIFICATION_ROC_CURVE_JSONL_NAME = "verification_roc_curve.jsonl"
VERIFICATION_DET_CURVE_JSONL_NAME = "verification_det_curve.jsonl"
VERIFICATION_CALIBRATION_CURVE_JSONL_NAME = "verification_calibration_curve.jsonl"
VERIFICATION_HISTOGRAM_JSON_NAME = "verification_score_histogram.json"
_NORMAL_DIST = NormalDist()


@dataclass(frozen=True, slots=True)
class VerificationReportInputs:
    scores_path: str
    trials_path: str | None = None
    metadata_path: str | None = None
    raw_scores_path: str | None = None
    score_normalization: str | None = None
    score_normalization_summary_path: str | None = None
    embeddings_path: str | None = None
    cohort_bank_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationScoreStatistics:
    min_score: float
    max_score: float
    mean_score: float
    score_std: float
    mean_positive_score: float | None
    mean_negative_score: float | None
    score_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationHistogramBin:
    lower_bound: float
    upper_bound: float
    count: int
    positive_count: int
    negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationCalibrationSummary:
    coefficient: float
    intercept: float
    brier_score: float
    log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationCalibrationBin:
    lower_probability: float
    upper_probability: float
    mean_predicted_probability: float
    observed_positive_rate: float
    trial_count: int
    positive_count: int
    negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationDetPoint:
    threshold: float
    false_accept_rate: float
    false_reject_rate: float
    det_x: float
    det_y: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": (None if not math.isfinite(self.threshold) else self.threshold),
            "false_accept_rate": self.false_accept_rate,
            "false_reject_rate": self.false_reject_rate,
            "det_x": self.det_x,
            "det_y": self.det_y,
        }


@dataclass(frozen=True, slots=True)
class VerificationEvaluationSummary:
    metrics: VerificationMetricsSummary
    score_statistics: VerificationScoreStatistics
    calibration: VerificationCalibrationSummary
    roc_point_count: int
    det_point_count: int
    histogram_bin_count: int
    calibration_bin_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "score_statistics": self.score_statistics.to_dict(),
            "calibration": self.calibration.to_dict(),
            "roc_point_count": self.roc_point_count,
            "det_point_count": self.det_point_count,
            "histogram_bin_count": self.histogram_bin_count,
            "calibration_bin_count": self.calibration_bin_count,
        }


@dataclass(frozen=True, slots=True)
class VerificationEvaluationReport:
    inputs: VerificationReportInputs
    summary: VerificationEvaluationSummary
    histogram: tuple[VerificationHistogramBin, ...]
    calibration_bins: tuple[VerificationCalibrationBin, ...]
    roc_points: tuple[VerificationOperatingPoint, ...]
    det_points: tuple[VerificationDetPoint, ...]

    def to_dict(self, *, include_curves: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "inputs": self.inputs.to_dict(),
            "summary": self.summary.to_dict(),
            "histogram": [bucket.to_dict() for bucket in self.histogram],
            "calibration_bins": [bucket.to_dict() for bucket in self.calibration_bins],
        }
        if include_curves:
            payload["roc_points"] = [point.to_dict() for point in self.roc_points]
            payload["det_points"] = [point.to_dict() for point in self.det_points]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenVerificationEvaluationReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    roc_curve_path: str
    det_curve_path: str
    calibration_curve_path: str
    histogram_path: str
    summary: VerificationEvaluationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "roc_curve_path": self.roc_curve_path,
            "det_curve_path": self.det_curve_path,
            "calibration_curve_path": self.calibration_curve_path,
            "histogram_path": self.histogram_path,
            "summary": self.summary.to_dict(),
        }


def build_verification_evaluation_report(
    score_rows: list[dict[str, Any]],
    *,
    scores_path: Path | str,
    trials_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    raw_scores_path: Path | str | None = None,
    score_normalization: str | None = None,
    score_normalization_summary_path: Path | str | None = None,
    embeddings_path: Path | str | None = None,
    cohort_bank_path: Path | str | None = None,
    trial_rows: list[dict[str, Any]] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    slice_fields: tuple[str, ...] = DEFAULT_SLICE_FIELDS,
    histogram_bins: int = 20,
    calibration_bins: int = 10,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> VerificationEvaluationReport:
    labels = [int(row["label"]) for row in score_rows]
    scores = [float(row["score"]) for row in score_rows]
    return build_verification_evaluation_report_from_arrays(
        labels=labels,
        scores=scores,
        scores_path=scores_path,
        trials_path=trials_path,
        metadata_path=metadata_path,
        raw_scores_path=raw_scores_path,
        score_normalization=score_normalization,
        score_normalization_summary_path=score_normalization_summary_path,
        embeddings_path=embeddings_path,
        cohort_bank_path=cohort_bank_path,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        slice_fields=slice_fields,
        histogram_bins=histogram_bins,
        calibration_bins=calibration_bins,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )


def build_verification_evaluation_report_from_arrays(
    *,
    labels: Any,
    scores: Any,
    scores_path: Path | str,
    trials_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    raw_scores_path: Path | str | None = None,
    score_normalization: str | None = None,
    score_normalization_summary_path: Path | str | None = None,
    embeddings_path: Path | str | None = None,
    cohort_bank_path: Path | str | None = None,
    trial_rows: list[dict[str, Any]] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    slice_fields: tuple[str, ...] = DEFAULT_SLICE_FIELDS,
    histogram_bins: int = 20,
    calibration_bins: int = 10,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    max_curve_points: int | None = 4096,
    emit_progress: Callable[[str], None] | None = None,
) -> VerificationEvaluationReport:
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive.")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive.")

    normalized_labels, normalized_scores = normalize_verification_score_arrays(labels, scores)
    if emit_progress is not None:
        emit_progress(f"[verification] rows={normalized_labels.shape[0]} phase=metrics")
    metrics = compute_verification_metrics_from_arrays(
        labels=normalized_labels,
        scores=normalized_scores,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )
    if emit_progress is not None:
        emit_progress(f"[verification] rows={normalized_labels.shape[0]} phase=curves")
    roc_points = tuple(
        build_verification_operating_points_from_arrays(
            labels=normalized_labels,
            scores=normalized_scores,
            max_points=max_curve_points,
        )
    )
    det_points = tuple(_build_det_points(roc_points))
    score_statistics = _build_score_statistics_from_arrays(
        labels=normalized_labels,
        scores=normalized_scores,
    )
    histogram = tuple(
        _build_histogram_from_arrays(
            labels=normalized_labels,
            scores=normalized_scores,
            histogram_bins=histogram_bins,
        )
    )
    if emit_progress is not None:
        emit_progress(f"[verification] rows={normalized_labels.shape[0]} phase=calibration")
    calibration_summary, calibration_curve = _build_calibration_curve_from_arrays(
        labels=normalized_labels,
        scores=normalized_scores,
        calibration_bins=calibration_bins,
    )

    summary = VerificationEvaluationSummary(
        metrics=metrics,
        score_statistics=score_statistics,
        calibration=calibration_summary,
        roc_point_count=len(roc_points),
        det_point_count=len(det_points),
        histogram_bin_count=len(histogram),
        calibration_bin_count=len(calibration_curve),
    )
    return VerificationEvaluationReport(
        inputs=VerificationReportInputs(
            scores_path=str(scores_path),
            trials_path=(None if trials_path is None else str(trials_path)),
            metadata_path=(None if metadata_path is None else str(metadata_path)),
            raw_scores_path=(None if raw_scores_path is None else str(raw_scores_path)),
            score_normalization=score_normalization,
            score_normalization_summary_path=(
                None
                if score_normalization_summary_path is None
                else str(score_normalization_summary_path)
            ),
            embeddings_path=(None if embeddings_path is None else str(embeddings_path)),
            cohort_bank_path=(None if cohort_bank_path is None else str(cohort_bank_path)),
        ),
        summary=summary,
        histogram=histogram,
        calibration_bins=tuple(calibration_curve),
        roc_points=roc_points,
        det_points=det_points,
    )


def write_verification_evaluation_report(
    report: VerificationEvaluationReport,
    *,
    output_root: Path | str,
) -> WrittenVerificationEvaluationReport:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    report_json_path = output_path / VERIFICATION_REPORT_JSON_NAME
    report_markdown_path = output_path / VERIFICATION_REPORT_MARKDOWN_NAME
    roc_curve_path = output_path / VERIFICATION_ROC_CURVE_JSONL_NAME
    det_curve_path = output_path / VERIFICATION_DET_CURVE_JSONL_NAME
    calibration_curve_path = output_path / VERIFICATION_CALIBRATION_CURVE_JSONL_NAME
    histogram_path = output_path / VERIFICATION_HISTOGRAM_JSON_NAME

    report_json_path.write_text(
        json.dumps(report.to_dict(include_curves=False), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_verification_evaluation_markdown(report), encoding="utf-8"
    )
    roc_curve_path.write_text(
        "".join(json.dumps(point.to_dict(), sort_keys=True) + "\n" for point in report.roc_points),
        encoding="utf-8",
    )
    det_curve_path.write_text(
        "".join(json.dumps(point.to_dict(), sort_keys=True) + "\n" for point in report.det_points),
        encoding="utf-8",
    )
    calibration_curve_path.write_text(
        "".join(
            json.dumps(point.to_dict(), sort_keys=True) + "\n" for point in report.calibration_bins
        ),
        encoding="utf-8",
    )
    histogram_path.write_text(
        json.dumps([bucket.to_dict() for bucket in report.histogram], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    return WrittenVerificationEvaluationReport(
        output_root=str(output_path),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        roc_curve_path=str(roc_curve_path),
        det_curve_path=str(det_curve_path),
        calibration_curve_path=str(calibration_curve_path),
        histogram_path=str(histogram_path),
        summary=report.summary,
    )


def render_verification_evaluation_markdown(report: VerificationEvaluationReport) -> str:
    metrics = report.summary.metrics
    score_statistics = report.summary.score_statistics
    calibration = report.summary.calibration
    lines = [
        "# Verification Evaluation Report",
        "",
        "## Inputs",
        "",
        f"- Scores: `{report.inputs.scores_path}`",
        f"- Trials: `{report.inputs.trials_path}`",
        f"- Metadata: `{report.inputs.metadata_path}`",
    ]
    if report.inputs.embeddings_path is not None:
        lines.append(f"- Embeddings: `{report.inputs.embeddings_path}`")
    if report.inputs.cohort_bank_path is not None:
        lines.append(f"- Cohort bank: `{report.inputs.cohort_bank_path}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- Trials: `{metrics.trial_count}`",
            f"- Positives: `{metrics.positive_count}`",
            f"- Negatives: `{metrics.negative_count}`",
            f"- EER: `{metrics.eer}` at threshold `{metrics.eer_threshold}`",
            f"- MinDCF: `{metrics.min_dcf}` at threshold `{metrics.min_dcf_threshold}`",
            "",
            "## Score Distribution",
            "",
            f"- Score range: `{score_statistics.min_score}` to `{score_statistics.max_score}`",
            f"- Mean score: `{score_statistics.mean_score}`",
            f"- Score std: `{score_statistics.score_std}`",
            f"- Mean positive score: `{score_statistics.mean_positive_score}`",
            f"- Mean negative score: `{score_statistics.mean_negative_score}`",
            f"- Score gap: `{score_statistics.score_gap}`",
            "",
            "## Calibration",
            "",
            f"- Platt coefficient: `{calibration.coefficient}`",
            f"- Platt intercept: `{calibration.intercept}`",
            f"- Brier score: `{calibration.brier_score}`",
            f"- Log loss: `{calibration.log_loss}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_score_statistics(
    normalized_rows: list[dict[str, float | int]],
) -> VerificationScoreStatistics:
    labels = np.asarray([int(row["label"]) for row in normalized_rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in normalized_rows], dtype=np.float64)
    return _build_score_statistics_from_arrays(labels=labels, scores=scores)


def _build_score_statistics_from_arrays(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
) -> VerificationScoreStatistics:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    mean_positive = _rounded_optional_mean(positive_scores.tolist())
    mean_negative = _rounded_optional_mean(negative_scores.tolist())
    score_gap = (
        None
        if mean_positive is None or mean_negative is None
        else round(mean_positive - mean_negative, 6)
    )
    return VerificationScoreStatistics(
        min_score=round(float(scores.min()), 6),
        max_score=round(float(scores.max()), 6),
        mean_score=round(float(scores.mean()), 6),
        score_std=round(float(scores.std(ddof=0)), 6),
        mean_positive_score=mean_positive,
        mean_negative_score=mean_negative,
        score_gap=score_gap,
    )


def _build_histogram(
    normalized_rows: list[dict[str, float | int]],
    *,
    histogram_bins: int,
) -> list[VerificationHistogramBin]:
    labels = np.asarray([int(row["label"]) for row in normalized_rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in normalized_rows], dtype=np.float64)
    return _build_histogram_from_arrays(
        labels=labels,
        scores=scores,
        histogram_bins=histogram_bins,
    )


def _build_histogram_from_arrays(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    histogram_bins: int,
) -> list[VerificationHistogramBin]:
    if math.isclose(float(scores.min()), float(scores.max()), abs_tol=1e-12):
        positive_count = int((labels == 1).sum())
        negative_count = int(labels.shape[0] - positive_count)
        return [
            VerificationHistogramBin(
                lower_bound=round(float(scores.min()), 6),
                upper_bound=round(float(scores.max()), 6),
                count=int(labels.shape[0]),
                positive_count=positive_count,
                negative_count=negative_count,
            )
        ]

    edges = np.linspace(scores.min(), scores.max(), num=histogram_bins + 1, dtype=np.float64)
    bucket_indices = np.searchsorted(edges, scores, side="right") - 1
    bucket_indices = np.clip(bucket_indices, 0, histogram_bins - 1)
    buckets = np.bincount(bucket_indices, minlength=histogram_bins)
    positive_buckets = np.bincount(bucket_indices[labels == 1], minlength=histogram_bins)
    negative_buckets = np.bincount(bucket_indices[labels == 0], minlength=histogram_bins)
    return [
        VerificationHistogramBin(
            lower_bound=round(float(edges[index]), 6),
            upper_bound=round(float(edges[index + 1]), 6),
            count=int(buckets[index]),
            positive_count=int(positive_buckets[index]),
            negative_count=int(negative_buckets[index]),
        )
        for index in range(histogram_bins)
    ]


def _build_calibration_curve(
    normalized_rows: list[dict[str, float | int]],
    *,
    calibration_bins: int,
) -> tuple[VerificationCalibrationSummary, list[VerificationCalibrationBin]]:
    scores = np.asarray([float(row["score"]) for row in normalized_rows], dtype=np.float64)
    labels = np.asarray([float(row["label"]) for row in normalized_rows], dtype=np.float64)
    return _build_calibration_curve_from_arrays(
        labels=labels,
        scores=scores,
        calibration_bins=calibration_bins,
    )


def _build_calibration_curve_from_arrays(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    calibration_bins: int,
) -> tuple[VerificationCalibrationSummary, list[VerificationCalibrationBin]]:
    coefficient, intercept = _fit_platt_scaler(scores, labels)
    probabilities = _sigmoid((scores * coefficient) + intercept)
    clipped_probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    brier_score = float(np.mean((clipped_probabilities - labels) ** 2))
    log_loss = float(
        -np.mean(
            (labels * np.log(clipped_probabilities))
            + ((1.0 - labels) * np.log(1.0 - clipped_probabilities))
        )
    )

    order = np.argsort(clipped_probabilities)
    ordered_probabilities = clipped_probabilities[order]
    ordered_labels = labels[order]
    boundaries = np.linspace(
        0,
        len(ordered_probabilities),
        num=min(calibration_bins, len(ordered_probabilities)) + 1,
        dtype=int,
    )

    bins: list[VerificationCalibrationBin] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop <= start:
            continue
        probabilities_slice = ordered_probabilities[start:stop]
        labels_slice = ordered_labels[start:stop]
        positive_count = int(labels_slice.sum())
        negative_count = int(labels_slice.shape[0] - positive_count)
        bins.append(
            VerificationCalibrationBin(
                lower_probability=round(float(probabilities_slice.min()), 6),
                upper_probability=round(float(probabilities_slice.max()), 6),
                mean_predicted_probability=round(float(probabilities_slice.mean()), 6),
                observed_positive_rate=round(float(labels_slice.mean()), 6),
                trial_count=int(labels_slice.shape[0]),
                positive_count=positive_count,
                negative_count=negative_count,
            )
        )

    return (
        VerificationCalibrationSummary(
            coefficient=round(float(coefficient), 6),
            intercept=round(float(intercept), 6),
            brier_score=round(brier_score, 6),
            log_loss=round(log_loss, 6),
        ),
        bins,
    )


def _build_det_points(
    operating_points: tuple[VerificationOperatingPoint, ...] | list[VerificationOperatingPoint],
) -> list[VerificationDetPoint]:
    det_points: list[VerificationDetPoint] = []
    for point in operating_points:
        det_points.append(
            VerificationDetPoint(
                threshold=round(float(point.threshold), 6),
                false_accept_rate=round(point.false_accept_rate, 6),
                false_reject_rate=round(point.false_reject_rate, 6),
                det_x=round(_rate_to_probit(point.false_accept_rate), 6),
                det_y=round(_rate_to_probit(point.false_reject_rate), 6),
            )
        )
    return det_points


def _fit_platt_scaler(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([scores, np.ones_like(scores)])
    coefficients = np.zeros(2, dtype=np.float64)
    regularization = 1e-6

    for _ in range(64):
        logits = design @ coefficients
        probabilities = _sigmoid(logits)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        adjusted_response = logits + ((labels - probabilities) / weights)
        xtwx = design.T @ (weights[:, None] * design)
        xtwz = design.T @ (weights * adjusted_response)
        updated = np.linalg.solve(xtwx + (regularization * np.eye(2)), xtwz)
        if float(np.max(np.abs(updated - coefficients))) < 1e-8:
            coefficients = updated
            break
        coefficients = updated
    return float(coefficients[0]), float(coefficients[1])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _rate_to_probit(rate: float) -> float:
    clamped_rate = min(max(rate, 1e-6), 1.0 - 1e-6)
    return float(_NORMAL_DIST.inv_cdf(clamped_rate))


def _rounded_optional_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


__all__ = [
    "DEFAULT_SLICE_FIELDS",
    "VERIFICATION_CALIBRATION_CURVE_JSONL_NAME",
    "VERIFICATION_DET_CURVE_JSONL_NAME",
    "VERIFICATION_HISTOGRAM_JSON_NAME",
    "VERIFICATION_REPORT_JSON_NAME",
    "VERIFICATION_REPORT_MARKDOWN_NAME",
    "VERIFICATION_ROC_CURVE_JSONL_NAME",
    "VerificationCalibrationBin",
    "VerificationCalibrationSummary",
    "VerificationDetPoint",
    "VerificationEvaluationReport",
    "VerificationEvaluationSummary",
    "VerificationHistogramBin",
    "VerificationReportInputs",
    "VerificationScoreStatistics",
    "WrittenVerificationEvaluationReport",
    "build_verification_evaluation_report",
    "build_verification_evaluation_report_from_arrays",
    "load_verification_metadata_rows",
    "load_verification_trial_rows",
    "render_verification_evaluation_markdown",
    "write_verification_evaluation_report",
]
