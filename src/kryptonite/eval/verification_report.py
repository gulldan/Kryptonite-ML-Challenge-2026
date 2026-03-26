"""Build and write rich verification-evaluation reports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .verification_dashboard import (
    VERIFICATION_SLICE_DASHBOARD_HTML_NAME as DASHBOARD_HTML_NAME,
)
from .verification_dashboard import (
    render_verification_slice_dashboard_html,
)
from .verification_data import (
    load_verification_metadata_rows,
    load_verification_trial_rows,
    resolve_trial_side_identifier,
)
from .verification_error_analysis import (
    VERIFICATION_ERROR_ANALYSIS_JSON_NAME as ERROR_ANALYSIS_JSON_NAME,
)
from .verification_error_analysis import (
    VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME as ERROR_ANALYSIS_MARKDOWN_NAME,
)
from .verification_error_analysis import (
    VerificationErrorAnalysisReport,
    build_verification_error_analysis,
    write_verification_error_analysis_report,
)
from .verification_metrics import (
    VerificationMetricsSummary,
    VerificationOperatingPoint,
    build_verification_operating_points,
    compute_verification_metrics,
    normalize_verification_score_rows,
)
from .verification_slices import DEFAULT_SLICE_FIELDS, group_verification_rows_by_slice

VERIFICATION_REPORT_JSON_NAME = "verification_eval_report.json"
VERIFICATION_REPORT_MARKDOWN_NAME = "verification_eval_report.md"
VERIFICATION_ROC_CURVE_JSONL_NAME = "verification_roc_curve.jsonl"
VERIFICATION_DET_CURVE_JSONL_NAME = "verification_det_curve.jsonl"
VERIFICATION_CALIBRATION_CURVE_JSONL_NAME = "verification_calibration_curve.jsonl"
VERIFICATION_HISTOGRAM_JSON_NAME = "verification_score_histogram.json"
VERIFICATION_SLICE_BREAKDOWN_JSONL_NAME = "verification_slice_breakdown.jsonl"
VERIFICATION_SLICE_DASHBOARD_HTML_NAME = DASHBOARD_HTML_NAME
VERIFICATION_ERROR_ANALYSIS_JSON_NAME = ERROR_ANALYSIS_JSON_NAME
VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME = ERROR_ANALYSIS_MARKDOWN_NAME
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
class VerificationSliceSummary:
    field_name: str
    field_value: str
    trial_count: int
    positive_count: int
    negative_count: int
    mean_score: float
    mean_positive_score: float | None
    mean_negative_score: float | None
    score_gap: float | None
    eer: float | None
    eer_threshold: float | None
    min_dcf: float | None
    min_dcf_threshold: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationEvaluationSummary:
    metrics: VerificationMetricsSummary
    score_statistics: VerificationScoreStatistics
    calibration: VerificationCalibrationSummary
    roc_point_count: int
    det_point_count: int
    histogram_bin_count: int
    calibration_bin_count: int
    slice_row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "score_statistics": self.score_statistics.to_dict(),
            "calibration": self.calibration.to_dict(),
            "roc_point_count": self.roc_point_count,
            "det_point_count": self.det_point_count,
            "histogram_bin_count": self.histogram_bin_count,
            "calibration_bin_count": self.calibration_bin_count,
            "slice_row_count": self.slice_row_count,
        }


@dataclass(frozen=True, slots=True)
class VerificationEvaluationReport:
    inputs: VerificationReportInputs
    summary: VerificationEvaluationSummary
    histogram: tuple[VerificationHistogramBin, ...]
    calibration_bins: tuple[VerificationCalibrationBin, ...]
    slice_breakdown: tuple[VerificationSliceSummary, ...]
    roc_points: tuple[VerificationOperatingPoint, ...]
    det_points: tuple[VerificationDetPoint, ...]
    error_analysis: VerificationErrorAnalysisReport | None = None

    def to_dict(self, *, include_curves: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "inputs": self.inputs.to_dict(),
            "summary": self.summary.to_dict(),
            "histogram": [bucket.to_dict() for bucket in self.histogram],
            "calibration_bins": [bucket.to_dict() for bucket in self.calibration_bins],
            "slice_breakdown": [row.to_dict() for row in self.slice_breakdown],
        }
        if include_curves:
            payload["roc_points"] = [point.to_dict() for point in self.roc_points]
            payload["det_points"] = [point.to_dict() for point in self.det_points]
        if self.error_analysis is not None:
            payload["error_analysis"] = self.error_analysis.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class WrittenVerificationEvaluationReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    slice_dashboard_path: str
    roc_curve_path: str
    det_curve_path: str
    calibration_curve_path: str
    histogram_path: str
    slice_breakdown_path: str
    summary: VerificationEvaluationSummary
    error_analysis_json_path: str | None = None
    error_analysis_markdown_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "slice_dashboard_path": self.slice_dashboard_path,
            "roc_curve_path": self.roc_curve_path,
            "det_curve_path": self.det_curve_path,
            "calibration_curve_path": self.calibration_curve_path,
            "histogram_path": self.histogram_path,
            "slice_breakdown_path": self.slice_breakdown_path,
            "summary": self.summary.to_dict(),
            "error_analysis_json_path": self.error_analysis_json_path,
            "error_analysis_markdown_path": self.error_analysis_markdown_path,
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
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive.")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive.")

    normalized_rows = normalize_verification_score_rows(score_rows)
    metrics = compute_verification_metrics(
        score_rows,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )
    roc_points = tuple(build_verification_operating_points(normalized_rows))
    det_points = tuple(_build_det_points(roc_points))
    score_statistics = _build_score_statistics(normalized_rows)
    histogram = tuple(_build_histogram(normalized_rows, histogram_bins=histogram_bins))
    calibration_summary, calibration_curve = _build_calibration_curve(
        normalized_rows,
        calibration_bins=calibration_bins,
    )
    resolved_trial_rows = trial_rows
    if resolved_trial_rows is None and trials_path is not None:
        resolved_trial_rows = load_verification_trial_rows(trials_path)
    resolved_metadata_rows = metadata_rows
    if resolved_metadata_rows is None and metadata_path is not None:
        resolved_metadata_rows = load_verification_metadata_rows(metadata_path)
    slice_breakdown = tuple(
        _build_slice_breakdown(
            raw_score_rows=score_rows,
            normalized_rows=normalized_rows,
            trial_rows=resolved_trial_rows,
            metadata_rows=resolved_metadata_rows,
            slice_fields=slice_fields,
            p_target=p_target,
            c_miss=c_miss,
            c_fa=c_fa,
        )
    )
    error_analysis = None
    if resolved_trial_rows is not None or any(
        resolve_trial_side_identifier(row, "left") or resolve_trial_side_identifier(row, "right")
        for row in score_rows
    ):
        error_analysis = build_verification_error_analysis(
            score_rows,
            decision_threshold=metrics.eer_threshold,
            threshold_source="eer",
            trial_rows=resolved_trial_rows,
            metadata_rows=resolved_metadata_rows,
            slice_fields=slice_fields,
        )

    summary = VerificationEvaluationSummary(
        metrics=metrics,
        score_statistics=score_statistics,
        calibration=calibration_summary,
        roc_point_count=len(roc_points),
        det_point_count=len(det_points),
        histogram_bin_count=len(histogram),
        calibration_bin_count=len(calibration_curve),
        slice_row_count=len(slice_breakdown),
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
        slice_breakdown=slice_breakdown,
        roc_points=roc_points,
        det_points=det_points,
        error_analysis=error_analysis,
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
    slice_dashboard_path = output_path / VERIFICATION_SLICE_DASHBOARD_HTML_NAME
    roc_curve_path = output_path / VERIFICATION_ROC_CURVE_JSONL_NAME
    det_curve_path = output_path / VERIFICATION_DET_CURVE_JSONL_NAME
    calibration_curve_path = output_path / VERIFICATION_CALIBRATION_CURVE_JSONL_NAME
    histogram_path = output_path / VERIFICATION_HISTOGRAM_JSON_NAME
    slice_breakdown_path = output_path / VERIFICATION_SLICE_BREAKDOWN_JSONL_NAME
    error_analysis_json_path = None
    error_analysis_markdown_path = None

    report_json_path.write_text(
        json.dumps(report.to_dict(include_curves=False), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_verification_evaluation_markdown(report), encoding="utf-8"
    )
    slice_dashboard_path.write_text(
        render_verification_slice_dashboard_html(report),
        encoding="utf-8",
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
    slice_breakdown_path.write_text(
        "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in report.slice_breakdown),
        encoding="utf-8",
    )
    if report.error_analysis is not None:
        written_error_analysis = write_verification_error_analysis_report(
            report.error_analysis,
            output_root=output_path,
        )
        error_analysis_json_path = written_error_analysis.report_json_path
        error_analysis_markdown_path = written_error_analysis.report_markdown_path

    return WrittenVerificationEvaluationReport(
        output_root=str(output_path),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        slice_dashboard_path=str(slice_dashboard_path),
        roc_curve_path=str(roc_curve_path),
        det_curve_path=str(det_curve_path),
        calibration_curve_path=str(calibration_curve_path),
        histogram_path=str(histogram_path),
        slice_breakdown_path=str(slice_breakdown_path),
        summary=report.summary,
        error_analysis_json_path=error_analysis_json_path,
        error_analysis_markdown_path=error_analysis_markdown_path,
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
    if report.inputs.raw_scores_path is not None:
        lines.append(f"- Raw scores: `{report.inputs.raw_scores_path}`")
    if report.inputs.score_normalization is not None:
        lines.append(f"- Score normalization: `{report.inputs.score_normalization}`")
    if report.inputs.score_normalization_summary_path is not None:
        lines.append(
            f"- Score-normalization summary: `{report.inputs.score_normalization_summary_path}`"
        )
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
    if report.slice_breakdown:
        lines.extend(["", "## Slice Breakdown", ""])
        for field_name in dict.fromkeys(row.field_name for row in report.slice_breakdown):
            lines.append(f"### `{field_name}`")
            for row in sorted(
                (item for item in report.slice_breakdown if item.field_name == field_name),
                key=lambda item: (-item.trial_count, item.field_value),
            )[:5]:
                lines.append(
                    "- "
                    f"`{row.field_value}`: trials `{row.trial_count}`, "
                    f"EER `{row.eer}`, minDCF `{row.min_dcf}`, gap `{row.score_gap}`"
                )
            lines.append("")
    if report.error_analysis is not None:
        error_summary = report.error_analysis.summary
        lines.extend(
            [
                "",
                "## Error Analysis",
                "",
                f"- Threshold source: `{error_summary.threshold_source}`",
                f"- Decision threshold: `{error_summary.decision_threshold}`",
                f"- False accepts: `{error_summary.false_accept_count}` "
                f"({error_summary.false_accept_rate})",
                f"- False rejects: `{error_summary.false_reject_count}` "
                f"({error_summary.false_reject_rate})",
            ]
        )
        if report.error_analysis.priority_findings:
            lines.extend(["", "### Priority Weak Spots", ""])
            for finding in report.error_analysis.priority_findings[:5]:
                lines.append(
                    "- "
                    f"**{finding.title}**: {finding.evidence} "
                    f"(errors `{finding.error_count}` / trials `{finding.trial_count}`, "
                    f"rate `{finding.error_rate}`)"
                )
    return "\n".join(lines).rstrip() + "\n"


def _build_score_statistics(
    normalized_rows: list[dict[str, float | int]],
) -> VerificationScoreStatistics:
    scores = np.asarray([float(row["score"]) for row in normalized_rows], dtype=np.float64)
    positive_scores = [float(row["score"]) for row in normalized_rows if int(row["label"]) == 1]
    negative_scores = [float(row["score"]) for row in normalized_rows if int(row["label"]) == 0]
    mean_positive = _rounded_optional_mean(positive_scores)
    mean_negative = _rounded_optional_mean(negative_scores)
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
    scores = np.asarray([float(row["score"]) for row in normalized_rows], dtype=np.float64)
    if math.isclose(float(scores.min()), float(scores.max()), abs_tol=1e-12):
        positive_count = sum(1 for row in normalized_rows if int(row["label"]) == 1)
        negative_count = len(normalized_rows) - positive_count
        return [
            VerificationHistogramBin(
                lower_bound=round(float(scores.min()), 6),
                upper_bound=round(float(scores.max()), 6),
                count=len(normalized_rows),
                positive_count=positive_count,
                negative_count=negative_count,
            )
        ]

    edges = np.linspace(scores.min(), scores.max(), num=histogram_bins + 1, dtype=np.float64)
    buckets = [0 for _ in range(histogram_bins)]
    positive_buckets = [0 for _ in range(histogram_bins)]
    negative_buckets = [0 for _ in range(histogram_bins)]
    for row in normalized_rows:
        score = float(row["score"])
        bucket_index = min(int(np.searchsorted(edges, score, side="right")) - 1, histogram_bins - 1)
        bucket_index = max(bucket_index, 0)
        buckets[bucket_index] += 1
        if int(row["label"]) == 1:
            positive_buckets[bucket_index] += 1
        else:
            negative_buckets[bucket_index] += 1
    return [
        VerificationHistogramBin(
            lower_bound=round(float(edges[index]), 6),
            upper_bound=round(float(edges[index + 1]), 6),
            count=buckets[index],
            positive_count=positive_buckets[index],
            negative_count=negative_buckets[index],
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


def _build_slice_breakdown(
    *,
    raw_score_rows: list[dict[str, Any]],
    normalized_rows: list[dict[str, float | int]],
    trial_rows: list[dict[str, Any]] | None,
    metadata_rows: list[dict[str, Any]] | None,
    slice_fields: tuple[str, ...],
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> list[VerificationSliceSummary]:
    grouped_rows = group_verification_rows_by_slice(
        raw_score_rows=raw_score_rows,
        normalized_rows=normalized_rows,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        slice_fields=slice_fields,
    )
    if not grouped_rows:
        return []

    slice_rows: list[VerificationSliceSummary] = []
    for (field_name, field_value), rows in sorted(
        grouped_rows.items(),
        key=lambda item: (item[0][0], -len(item[1]), item[0][1]),
    ):
        positive_scores = [float(row["score"]) for row in rows if int(row["label"]) == 1]
        negative_scores = [float(row["score"]) for row in rows if int(row["label"]) == 0]
        mean_positive = _rounded_optional_mean(positive_scores)
        mean_negative = _rounded_optional_mean(negative_scores)
        score_gap = (
            None
            if mean_positive is None or mean_negative is None
            else round(mean_positive - mean_negative, 6)
        )
        metrics: VerificationMetricsSummary | None = None
        if positive_scores and negative_scores:
            metrics = compute_verification_metrics(
                list(rows),
                p_target=p_target,
                c_miss=c_miss,
                c_fa=c_fa,
            )
        slice_rows.append(
            VerificationSliceSummary(
                field_name=field_name,
                field_value=field_value,
                trial_count=len(rows),
                positive_count=len(positive_scores),
                negative_count=len(negative_scores),
                mean_score=round(
                    sum(float(row["score"]) for row in rows) / float(len(rows)),
                    6,
                ),
                mean_positive_score=mean_positive,
                mean_negative_score=mean_negative,
                score_gap=score_gap,
                eer=(None if metrics is None else metrics.eer),
                eer_threshold=(None if metrics is None else metrics.eer_threshold),
                min_dcf=(None if metrics is None else metrics.min_dcf),
                min_dcf_threshold=(None if metrics is None else metrics.min_dcf_threshold),
            )
        )
    return slice_rows


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
    "VERIFICATION_ERROR_ANALYSIS_JSON_NAME",
    "VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME",
    "VERIFICATION_HISTOGRAM_JSON_NAME",
    "VERIFICATION_REPORT_JSON_NAME",
    "VERIFICATION_REPORT_MARKDOWN_NAME",
    "VERIFICATION_ROC_CURVE_JSONL_NAME",
    "VERIFICATION_SLICE_DASHBOARD_HTML_NAME",
    "VERIFICATION_SLICE_BREAKDOWN_JSONL_NAME",
    "VerificationCalibrationBin",
    "VerificationCalibrationSummary",
    "VerificationDetPoint",
    "VerificationEvaluationReport",
    "VerificationEvaluationSummary",
    "VerificationHistogramBin",
    "VerificationReportInputs",
    "VerificationScoreStatistics",
    "VerificationSliceSummary",
    "WrittenVerificationEvaluationReport",
    "build_verification_evaluation_report",
    "load_verification_metadata_rows",
    "load_verification_trial_rows",
    "render_verification_evaluation_markdown",
    "write_verification_evaluation_report",
]
