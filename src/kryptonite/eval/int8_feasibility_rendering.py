"""Rendering helpers for reproducible INT8 feasibility reports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .int8_feasibility_models import (
    INT8_FEASIBILITY_JSON_NAME,
    INT8_FEASIBILITY_MARKDOWN_NAME,
    Int8FeasibilityArtifactRef,
    Int8FeasibilityReport,
    WrittenInt8FeasibilityReport,
)


def render_int8_feasibility_markdown(report: Int8FeasibilityReport) -> str:
    included_categories = ", ".join(
        f"`{item}`" for item in report.calibration_set.include_categories
    )
    excluded_scenarios = (
        ", ".join(f"`{item}`" for item in report.calibration_set.exclude_scenarios)
        if report.calibration_set.exclude_scenarios
        else "-"
    )
    duration_bucket_summary = (
        f"short=`{report.calibration_set.short_count}`, "
        f"mid=`{report.calibration_set.mid_count}`, "
        f"long=`{report.calibration_set.long_count}`, "
        f"unknown=`{report.calibration_set.unknown_duration_count}`"
    )
    lines = [
        f"# {report.title}",
        "",
        f"- Report id: `{report.report_id}`",
        f"- Candidate: `{report.candidate_label}`",
        f"- Decision: `{report.summary.decision}`",
        f"- Model version: `{report.model_version or 'unknown'}`",
        f"- Structural stub: `{_format_bool(report.structural_stub)}`",
        f"- Export profile: `{report.export_profile or 'unknown'}`",
        f"- Selected calibration inputs: `{report.calibration_set.selected_input_count}`",
        "",
        "## Summary",
        "",
        report.summary_text.strip(),
        "",
        "## Decision Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        status = "pass" if check.passed else "fail"
        lines.append(f"| {check.name} | `{status}` | {check.detail} |")

    lines.extend(
        [
            "",
            "## Calibration Set",
            "",
            f"- Source catalog: `{report.calibration_set.source_catalog_path}`",
            f"- Included categories: {included_categories}",
            f"- Excluded scenarios: {excluded_scenarios}",
            f"- Category counts: {_format_category_counts(report.calibration_set.category_counts)}",
            f"- Duration buckets: {duration_bucket_summary}",
            f"- Total selected duration (s): `{report.calibration_set.total_duration_seconds}`",
            "",
            "| Scenario | Category | Duration (s) | Audio path |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in report.calibration_set.selected_inputs:
        duration = "-" if item.duration_seconds is None else f"{item.duration_seconds:.3f}"
        lines.append(f"| {item.scenario_id} | {item.category} | {duration} | `{item.audio_path}` |")
    if not report.calibration_set.selected_inputs:
        lines.append("| - | - | - | - |")
    if report.calibration_set.warnings:
        lines.extend(["", "Calibration warnings:"])
        lines.extend(f"- {warning}" for warning in report.calibration_set.warnings)

    lines.extend(
        [
            "",
            "## Backend Measurements",
            "",
            "| Backend | EER | minDCF | Latency ms/audio | Peak RSS MiB | Peak CUDA MiB |",
            "| --- | --- | --- | --- | --- | --- |",
            _render_measurement_row(report.fp16),
            _render_measurement_row(report.int8),
            "",
            "## Delta Summary",
            "",
            f"- EER delta (INT8 - FP16): `{_format_float(report.deltas.eer_delta, digits=6)}`",
            (
                "- minDCF delta (INT8 - FP16): "
                f"`{_format_float(report.deltas.min_dcf_delta, digits=6)}`"
            ),
            (
                "- Latency speedup ratio (FP16 / INT8): "
                f"`{_format_float(report.deltas.latency_speedup_ratio, digits=6)}`"
            ),
            (
                "- Process RSS delta MiB (INT8 - FP16): "
                f"`{_format_float(report.deltas.process_rss_delta_mib, digits=6)}`"
            ),
            (
                "- CUDA allocated delta MiB (INT8 - FP16): "
                f"`{_format_float(report.deltas.cuda_allocated_delta_mib, digits=6)}`"
            ),
            "",
            "## Artifact Snapshot",
            "",
            "| Artifact | Required | Exists | Kind | Configured path |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for artifact in report.artifacts:
        lines.append(_render_artifact_row(artifact))

    if report.validation_commands:
        lines.extend(["", "## Validation Commands", ""])
        lines.extend(f"- `{command}`" for command in report.validation_commands)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines) + "\n"


def write_int8_feasibility_report(
    report: Int8FeasibilityReport,
) -> WrittenInt8FeasibilityReport:
    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report_json_path = output_root / INT8_FEASIBILITY_JSON_NAME
    report_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path = output_root / INT8_FEASIBILITY_MARKDOWN_NAME
    report_markdown_path.write_text(
        render_int8_feasibility_markdown(report),
        encoding="utf-8",
    )

    source_config_copy_path = None
    if report.source_config_path is not None:
        source_config_copy = output_root / "sources" / "int8_feasibility_config.toml"
        source_config_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report.source_config_path, source_config_copy)
        source_config_copy_path = str(source_config_copy)

    return WrittenInt8FeasibilityReport(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        source_config_copy_path=source_config_copy_path,
        summary=report.summary,
    )


def _render_measurement_row(report) -> str:
    return (
        "| "
        f"{report.label} | "
        f"{_format_float(report.eer, digits=6)} | "
        f"{_format_float(report.min_dcf, digits=6)} | "
        f"{_format_float(report.mean_ms_per_audio_at_largest_batch, digits=6)} | "
        f"{_format_float(report.peak_process_rss_mib, digits=3)} | "
        f"{_format_float(report.peak_cuda_allocated_mib, digits=3)} |"
    )


def _render_artifact_row(artifact: Int8FeasibilityArtifactRef) -> str:
    exists = "yes" if artifact.exists else "no"
    required = "yes" if artifact.required else "no"
    configured_path = artifact.configured_path or "-"
    return (
        f"| {artifact.label} | `{required}` | `{exists}` | `{artifact.kind}` | "
        f"`{configured_path}` |"
    )


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if value else "false"


def _format_float(value: float | None, *, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_category_counts(category_counts: dict[str, int]) -> str:
    if not category_counts:
        return "-"
    return ", ".join(f"`{key}`={value}" for key, value in sorted(category_counts.items()))


__all__ = [
    "render_int8_feasibility_markdown",
    "write_int8_feasibility_report",
]
