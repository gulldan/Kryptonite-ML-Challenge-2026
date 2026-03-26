"""Named threshold calibration profiles for verification score bundles."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .verification_metrics import (
    VerificationMetricsSummary,
    VerificationOperatingPoint,
    build_verification_operating_points,
    compute_verification_metrics,
    normalize_verification_score_rows,
)
from .verification_slices import group_verification_rows_by_slice

VERIFICATION_THRESHOLD_CALIBRATION_JSON_NAME = "verification_threshold_calibration.json"
VERIFICATION_THRESHOLD_CALIBRATION_MARKDOWN_NAME = "verification_threshold_calibration.md"


@dataclass(frozen=True, slots=True)
class ThresholdProfileSpec:
    name: str
    selection_method: str
    target_false_accept_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Threshold profile names must not be empty.")
        if self.selection_method not in {"eer", "min_dcf", "target_far"}:
            raise ValueError("selection_method must be one of: eer, min_dcf, target_far.")
        if self.selection_method == "target_far":
            if self.target_false_accept_rate is None:
                raise ValueError("target_false_accept_rate is required for target_far profiles.")
            if not 0.0 <= self.target_false_accept_rate <= 1.0:
                raise ValueError("target_false_accept_rate must be within [0.0, 1.0].")
        elif self.target_false_accept_rate is not None:
            raise ValueError("target_false_accept_rate is only valid for target_far profiles.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationThresholdProfile:
    name: str
    selection_method: str
    threshold: float
    true_accept_count: int
    true_reject_count: int
    false_accept_count: int
    false_reject_count: int
    true_accept_rate: float
    true_reject_rate: float
    false_accept_rate: float
    false_reject_rate: float
    target_false_accept_rate: float | None = None
    p_target: float | None = None
    c_miss: float | None = None
    c_fa: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["threshold"] = None if not math.isfinite(self.threshold) else self.threshold
        return payload


@dataclass(frozen=True, slots=True)
class VerificationSliceThresholdCalibration:
    slice_field: str
    slice_value: str
    trial_count: int
    positive_count: int
    negative_count: int
    profiles: tuple[VerificationThresholdProfile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_field": self.slice_field,
            "slice_value": self.slice_value,
            "trial_count": self.trial_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "profiles": [profile.to_dict() for profile in self.profiles],
        }


@dataclass(frozen=True, slots=True)
class VerificationThresholdCalibrationSummary:
    trial_count: int
    positive_count: int
    negative_count: int
    global_profile_count: int
    slice_group_count: int
    min_slice_trials: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationThresholdCalibrationReport:
    metrics: VerificationMetricsSummary
    profile_specs: tuple[ThresholdProfileSpec, ...]
    summary: VerificationThresholdCalibrationSummary
    global_profiles: tuple[VerificationThresholdProfile, ...]
    slice_profiles: tuple[VerificationSliceThresholdCalibration, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "profile_specs": [spec.to_dict() for spec in self.profile_specs],
            "summary": self.summary.to_dict(),
            "global_profiles": [profile.to_dict() for profile in self.global_profiles],
            "slice_profiles": [profile.to_dict() for profile in self.slice_profiles],
        }


@dataclass(frozen=True, slots=True)
class WrittenVerificationThresholdCalibrationReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    summary: VerificationThresholdCalibrationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "summary": self.summary.to_dict(),
        }


def build_default_threshold_profile_specs(
    *,
    demo_target_far: float = 0.05,
    production_target_far: float = 0.01,
) -> tuple[ThresholdProfileSpec, ...]:
    return (
        ThresholdProfileSpec(name="balanced", selection_method="eer"),
        ThresholdProfileSpec(name="min_dcf", selection_method="min_dcf"),
        ThresholdProfileSpec(
            name="demo",
            selection_method="target_far",
            target_false_accept_rate=demo_target_far,
        ),
        ThresholdProfileSpec(
            name="production",
            selection_method="target_far",
            target_false_accept_rate=production_target_far,
        ),
    )


def build_verification_threshold_calibration_report(
    score_rows: list[dict[str, Any]],
    *,
    raw_score_rows: list[dict[str, Any]] | None = None,
    trial_rows: list[dict[str, Any]] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    profile_specs: tuple[ThresholdProfileSpec, ...] | None = None,
    slice_fields: tuple[str, ...] = (),
    min_slice_trials: int = 25,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> VerificationThresholdCalibrationReport:
    if min_slice_trials <= 0:
        raise ValueError("min_slice_trials must be positive.")

    normalized_rows = normalize_verification_score_rows(score_rows)
    metrics = compute_verification_metrics(
        score_rows,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )
    points = tuple(build_verification_operating_points(normalized_rows))
    resolved_profile_specs = (
        build_default_threshold_profile_specs() if profile_specs is None else profile_specs
    )

    global_profiles = tuple(
        _select_threshold_profile(
            spec,
            points=points,
            p_target=p_target,
            c_miss=c_miss,
            c_fa=c_fa,
        )
        for spec in resolved_profile_specs
    )

    slice_profiles: list[VerificationSliceThresholdCalibration] = []
    resolved_raw_rows = score_rows if raw_score_rows is None else raw_score_rows
    grouped_rows = group_verification_rows_by_slice(
        raw_score_rows=resolved_raw_rows,
        normalized_rows=normalized_rows,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        slice_fields=slice_fields,
    )
    for (slice_field, slice_value), rows in sorted(
        grouped_rows.items(),
        key=lambda item: (item[0][0], -len(item[1]), item[0][1]),
    ):
        if len(rows) < min_slice_trials:
            continue
        positive_count = sum(1 for row in rows if int(row["label"]) == 1)
        negative_count = len(rows) - positive_count
        if positive_count == 0 or negative_count == 0:
            continue
        local_points = tuple(build_verification_operating_points(list(rows)))
        profiles = tuple(
            _select_threshold_profile(
                spec,
                points=local_points,
                p_target=p_target,
                c_miss=c_miss,
                c_fa=c_fa,
            )
            for spec in resolved_profile_specs
        )
        slice_profiles.append(
            VerificationSliceThresholdCalibration(
                slice_field=slice_field,
                slice_value=slice_value,
                trial_count=len(rows),
                positive_count=positive_count,
                negative_count=negative_count,
                profiles=profiles,
            )
        )

    return VerificationThresholdCalibrationReport(
        metrics=metrics,
        profile_specs=resolved_profile_specs,
        summary=VerificationThresholdCalibrationSummary(
            trial_count=metrics.trial_count,
            positive_count=metrics.positive_count,
            negative_count=metrics.negative_count,
            global_profile_count=len(global_profiles),
            slice_group_count=len(slice_profiles),
            min_slice_trials=min_slice_trials,
        ),
        global_profiles=global_profiles,
        slice_profiles=tuple(slice_profiles),
    )


def write_verification_threshold_calibration_report(
    report: VerificationThresholdCalibrationReport,
    *,
    output_root: Path | str,
) -> WrittenVerificationThresholdCalibrationReport:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    report_json_path = output_path / VERIFICATION_THRESHOLD_CALIBRATION_JSON_NAME
    report_markdown_path = output_path / VERIFICATION_THRESHOLD_CALIBRATION_MARKDOWN_NAME
    report_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_verification_threshold_calibration_markdown(report),
        encoding="utf-8",
    )
    return WrittenVerificationThresholdCalibrationReport(
        output_root=str(output_path),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        summary=report.summary,
    )


def render_verification_threshold_calibration_markdown(
    report: VerificationThresholdCalibrationReport,
) -> str:
    metrics = report.metrics
    lines = [
        "# Verification Threshold Calibration",
        "",
        "## Summary",
        "",
        f"- Trials: `{metrics.trial_count}`",
        f"- Positives: `{metrics.positive_count}`",
        f"- Negatives: `{metrics.negative_count}`",
        f"- EER: `{metrics.eer}` at threshold `{_format_threshold(metrics.eer_threshold)}`",
        "- "
        f"MinDCF: `{metrics.min_dcf}` at threshold "
        f"`{_format_threshold(metrics.min_dcf_threshold)}`",
        f"- Slice-aware threshold groups: `{report.summary.slice_group_count}`",
        "",
        "## Global Operating Points",
        "",
    ]
    for profile in report.global_profiles:
        lines.append(
            "- "
            f"`{profile.name}` (`{profile.selection_method}`): threshold "
            f"`{_format_threshold(profile.threshold)}`, FAR `{profile.false_accept_rate}`, "
            f"FRR `{profile.false_reject_rate}`, TAR `{profile.true_accept_rate}`, "
            f"TRR `{profile.true_reject_rate}`{_render_profile_suffix(profile)}"
        )

    lines.extend(["", "## Slice-Aware Thresholds", ""])
    if not report.slice_profiles:
        lines.append(
            "- Not emitted. Pass one or more slice fields plus enough trial volume "
            "to calibrate them."
        )
    else:
        current_field = None
        for slice_profile in report.slice_profiles:
            if slice_profile.slice_field != current_field:
                current_field = slice_profile.slice_field
                lines.extend(["", f"### `{current_field}`", ""])
            lines.append(
                "- "
                f"`{slice_profile.slice_value}`: trials `{slice_profile.trial_count}`, positives "
                f"`{slice_profile.positive_count}`, negatives `{slice_profile.negative_count}`"
            )
            for profile in slice_profile.profiles:
                lines.append(
                    f"  - `{profile.name}`: threshold `{_format_threshold(profile.threshold)}`, "
                    f"FAR `{profile.false_accept_rate}`, FRR `{profile.false_reject_rate}`"
                    f"{_render_profile_suffix(profile)}"
                )
    return "\n".join(lines).rstrip() + "\n"


def _select_threshold_profile(
    spec: ThresholdProfileSpec,
    *,
    points: tuple[VerificationOperatingPoint, ...],
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> VerificationThresholdProfile:
    point: VerificationOperatingPoint
    if spec.selection_method == "eer":
        point = min(
            points,
            key=lambda candidate: abs(candidate.false_accept_rate - candidate.false_reject_rate),
        )
    elif spec.selection_method == "min_dcf":
        point = min(
            points,
            key=lambda candidate: _normalized_detection_cost(
                candidate,
                p_target=p_target,
                c_miss=c_miss,
                c_fa=c_fa,
            ),
        )
    elif spec.selection_method == "target_far":
        assert spec.target_false_accept_rate is not None
        allowed_points = [
            candidate
            for candidate in points
            if candidate.false_accept_rate <= spec.target_false_accept_rate + 1e-12
        ]
        point = min(
            allowed_points,
            key=lambda candidate: (
                candidate.false_reject_rate,
                abs(candidate.false_accept_rate - spec.target_false_accept_rate),
                _threshold_sort_key(candidate.threshold),
            ),
        )
    else:
        raise ValueError(f"Unsupported selection method: {spec.selection_method!r}")

    return VerificationThresholdProfile(
        name=spec.name,
        selection_method=spec.selection_method,
        threshold=round(float(point.threshold), 6),
        true_accept_count=point.true_accept_count,
        true_reject_count=point.true_reject_count,
        false_accept_count=point.false_accept_count,
        false_reject_count=point.false_reject_count,
        true_accept_rate=round(point.true_accept_rate, 6),
        true_reject_rate=round(point.true_reject_rate, 6),
        false_accept_rate=round(point.false_accept_rate, 6),
        false_reject_rate=round(point.false_reject_rate, 6),
        target_false_accept_rate=(
            None
            if spec.target_false_accept_rate is None
            else round(spec.target_false_accept_rate, 6)
        ),
        p_target=(None if spec.selection_method != "min_dcf" else round(p_target, 6)),
        c_miss=(None if spec.selection_method != "min_dcf" else round(c_miss, 6)),
        c_fa=(None if spec.selection_method != "min_dcf" else round(c_fa, 6)),
    )


def _normalized_detection_cost(
    point: VerificationOperatingPoint,
    *,
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> float:
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    if default_cost <= 0.0:
        raise ValueError("Default detection cost must be positive.")
    raw_cost = c_miss * point.false_reject_rate * p_target + c_fa * point.false_accept_rate * (
        1.0 - p_target
    )
    return raw_cost / default_cost


def _threshold_sort_key(threshold: float) -> float:
    if not math.isfinite(threshold):
        return float("inf")
    return -float(threshold)


def _format_threshold(threshold: float) -> str:
    if not math.isfinite(threshold):
        return "inf"
    return f"{threshold:.6f}"


def _render_profile_suffix(profile: VerificationThresholdProfile) -> str:
    if profile.selection_method == "target_far":
        return f", FAR budget `{profile.target_false_accept_rate}`"
    if profile.selection_method == "min_dcf":
        return f", p_target `{profile.p_target}`, c_miss `{profile.c_miss}`, c_fa `{profile.c_fa}`"
    return ""


__all__ = [
    "ThresholdProfileSpec",
    "VERIFICATION_THRESHOLD_CALIBRATION_JSON_NAME",
    "VERIFICATION_THRESHOLD_CALIBRATION_MARKDOWN_NAME",
    "VerificationSliceThresholdCalibration",
    "VerificationThresholdCalibrationReport",
    "VerificationThresholdCalibrationSummary",
    "VerificationThresholdProfile",
    "WrittenVerificationThresholdCalibrationReport",
    "build_default_threshold_profile_specs",
    "build_verification_threshold_calibration_report",
    "render_verification_threshold_calibration_markdown",
    "write_verification_threshold_calibration_report",
]
