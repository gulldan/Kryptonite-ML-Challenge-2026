"""Manifest-driven loudness normalization comparison reports."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from kryptonite.config import NormalizationConfig, VADConfig
from kryptonite.data import AudioLoadRequest, iter_manifest_audio, load_audio
from kryptonite.deployment import resolve_project_path

_TARGET_TOLERANCE_DB = 0.5


@dataclass(frozen=True, slots=True)
class LoudnessComparisonRecord:
    manifest_path: str
    line_number: int | None
    audio_path: str
    speaker_id: str | None
    utterance_id: str | None
    duration_seconds: float
    source_rms_dbfs: float | None
    output_rms_dbfs: float | None
    target_loudness_dbfs: float
    requested_gain_db: float
    applied_gain_db: float
    gain_clamped: bool
    peak_limited: bool
    applied: bool
    skip_reason: str
    alignment_error: float
    degradation_check_passed: bool

    @property
    def target_error_db(self) -> float | None:
        if self.output_rms_dbfs is None:
            return None
        return round(self.output_rms_dbfs - self.target_loudness_dbfs, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "line_number": self.line_number,
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "utterance_id": self.utterance_id,
            "duration_seconds": self.duration_seconds,
            "source_rms_dbfs": self.source_rms_dbfs,
            "output_rms_dbfs": self.output_rms_dbfs,
            "target_loudness_dbfs": self.target_loudness_dbfs,
            "target_error_db": self.target_error_db,
            "requested_gain_db": self.requested_gain_db,
            "applied_gain_db": self.applied_gain_db,
            "gain_clamped": self.gain_clamped,
            "peak_limited": self.peak_limited,
            "applied": self.applied,
            "skip_reason": self.skip_reason,
            "alignment_error": self.alignment_error,
            "degradation_check_passed": self.degradation_check_passed,
        }


@dataclass(frozen=True, slots=True)
class LoudnessComparisonSummary:
    row_count: int
    changed_row_count: int
    target_reached_row_count: int
    gain_clamped_row_count: int
    peak_limited_row_count: int
    degradation_check_failed_row_count: int
    mean_source_rms_dbfs: float | None
    mean_output_rms_dbfs: float | None
    mean_applied_gain_db: float
    max_alignment_error: float
    max_abs_target_error_db: float

    @property
    def baseline_guard_passed(self) -> bool:
        return self.degradation_check_failed_row_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "changed_row_count": self.changed_row_count,
            "target_reached_row_count": self.target_reached_row_count,
            "gain_clamped_row_count": self.gain_clamped_row_count,
            "peak_limited_row_count": self.peak_limited_row_count,
            "degradation_check_failed_row_count": self.degradation_check_failed_row_count,
            "baseline_guard_passed": self.baseline_guard_passed,
            "mean_source_rms_dbfs": self.mean_source_rms_dbfs,
            "mean_output_rms_dbfs": self.mean_output_rms_dbfs,
            "mean_applied_gain_db": self.mean_applied_gain_db,
            "max_alignment_error": self.max_alignment_error,
            "max_abs_target_error_db": self.max_abs_target_error_db,
        }


@dataclass(frozen=True, slots=True)
class LoudnessComparisonReport:
    project_root: str
    manifest_path: str
    loudness_mode: str
    target_loudness_dbfs: float
    max_loudness_gain_db: float
    max_loudness_attenuation_db: float
    vad_mode: str
    limit: int | None
    summary: LoudnessComparisonSummary
    records: list[LoudnessComparisonRecord]

    def to_dict(self, *, include_records: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "project_root": self.project_root,
            "manifest_path": self.manifest_path,
            "loudness_mode": self.loudness_mode,
            "target_loudness_dbfs": self.target_loudness_dbfs,
            "max_loudness_gain_db": self.max_loudness_gain_db,
            "max_loudness_attenuation_db": self.max_loudness_attenuation_db,
            "vad_mode": self.vad_mode,
            "limit": self.limit,
            "summary": self.summary.to_dict(),
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenLoudnessComparisonReport:
    output_root: str
    json_path: str
    markdown_path: str
    rows_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
            "rows_path": self.rows_path,
        }


def build_loudness_normalization_report(
    *,
    project_root: Path | str,
    manifest_path: Path | str,
    normalization: NormalizationConfig,
    vad: VADConfig | None = None,
    limit: int | None = None,
) -> LoudnessComparisonReport:
    baseline_request = AudioLoadRequest.from_config(
        replace(normalization, loudness_mode="none"),
        vad=vad,
    )
    normalized_request = AudioLoadRequest.from_config(normalization, vad=vad)
    records: list[LoudnessComparisonRecord] = []

    for index, loaded in enumerate(
        iter_manifest_audio(
            manifest_path,
            project_root=project_root,
            request=baseline_request,
        )
    ):
        if limit is not None and index >= limit:
            break

        normalized = load_audio(
            loaded.row.audio_path,
            project_root=project_root,
            request=normalized_request,
        )
        records.append(
            LoudnessComparisonRecord(
                manifest_path=loaded.manifest_path or str(manifest_path),
                line_number=loaded.line_number,
                audio_path=loaded.row.audio_path,
                speaker_id=loaded.row.speaker_id,
                utterance_id=loaded.row.utterance_id,
                duration_seconds=loaded.audio.duration_seconds,
                source_rms_dbfs=loaded.audio.post_loudness_rms_dbfs,
                output_rms_dbfs=normalized.post_loudness_rms_dbfs,
                target_loudness_dbfs=normalized.loudness_target_dbfs,
                requested_gain_db=round(
                    0.0
                    if loaded.audio.post_loudness_rms_dbfs is None
                    else normalized.loudness_target_dbfs - loaded.audio.post_loudness_rms_dbfs,
                    6,
                ),
                applied_gain_db=normalized.loudness_gain_db,
                gain_clamped=normalized.loudness_gain_clamped,
                peak_limited=normalized.loudness_peak_limited,
                applied=normalized.loudness_applied,
                skip_reason=normalized.loudness_skip_reason,
                alignment_error=normalized.loudness_alignment_error,
                degradation_check_passed=normalized.loudness_degradation_check_passed,
            )
        )

    report = LoudnessComparisonReport(
        project_root=str(resolve_project_path(str(project_root), ".")),
        manifest_path=str(resolve_project_path(str(project_root), str(manifest_path))),
        loudness_mode=normalization.loudness_mode,
        target_loudness_dbfs=normalization.target_loudness_dbfs,
        max_loudness_gain_db=normalization.max_loudness_gain_db,
        max_loudness_attenuation_db=normalization.max_loudness_attenuation_db,
        vad_mode="none" if vad is None else vad.mode,
        limit=limit,
        summary=_build_summary(records),
        records=records,
    )
    return report


def write_loudness_normalization_report(
    *,
    report: LoudnessComparisonReport,
    output_root: Path | str,
) -> WrittenLoudnessComparisonReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "loudness_normalization_report.json"
    markdown_path = output_root_path / "loudness_normalization_report.md"
    rows_path = output_root_path / "loudness_normalization_rows.jsonl"

    json_path.write_text(
        json.dumps(report.to_dict(include_records=True), indent=2, sort_keys=True) + "\n"
    )
    markdown_path.write_text(render_loudness_normalization_markdown(report))
    rows_path.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in report.records)
    )
    return WrittenLoudnessComparisonReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        rows_path=str(rows_path),
    )


def render_loudness_normalization_markdown(report: LoudnessComparisonReport) -> str:
    summary = report.summary
    lines = [
        "# Loudness Normalization Report",
        "",
        "## Scope",
        "",
        f"- manifest: `{report.manifest_path}`",
        f"- rows analyzed: `{summary.row_count}`",
        f"- loudness mode: `{report.loudness_mode}`",
        f"- target loudness: `{report.target_loudness_dbfs:.2f} dBFS`",
        f"- max gain: `{report.max_loudness_gain_db:.2f} dB`",
        f"- max attenuation: `{report.max_loudness_attenuation_db:.2f} dB`",
        f"- vad mode: `{report.vad_mode}`",
        "",
        "## Summary",
        "",
        f"- changed rows: `{summary.changed_row_count}`",
        (
            f"- target reached within ±{_TARGET_TOLERANCE_DB:.1f} dB: "
            f"`{summary.target_reached_row_count}`"
        ),
        f"- gain-clamped rows: `{summary.gain_clamped_row_count}`",
        f"- peak-limited rows: `{summary.peak_limited_row_count}`",
        f"- degradation-check failures: `{summary.degradation_check_failed_row_count}`",
        f"- baseline guard passed: `{summary.baseline_guard_passed}`",
        f"- mean source loudness: `{_format_optional_dbfs(summary.mean_source_rms_dbfs)}`",
        f"- mean output loudness: `{_format_optional_dbfs(summary.mean_output_rms_dbfs)}`",
        f"- mean applied gain: `{summary.mean_applied_gain_db:.2f} dB`",
        f"- max alignment error: `{summary.max_alignment_error:.12f}`",
        f"- max |target error|: `{summary.max_abs_target_error_db:.2f} dB`",
        "",
    ]

    top_gain = sorted(report.records, key=lambda record: abs(record.applied_gain_db), reverse=True)[
        :5
    ]
    if top_gain:
        lines.extend(
            [
                "## Largest Gain Changes",
                "",
                *[
                    (
                        f"- `{record.audio_path}`: applied `{record.applied_gain_db:.2f} dB`, "
                        f"source `{_format_optional_dbfs(record.source_rms_dbfs)}`, "
                        f"output `{_format_optional_dbfs(record.output_rms_dbfs)}`"
                    )
                    for record in top_gain
                ],
                "",
            ]
        )
    return "\n".join(lines)


def _build_summary(records: list[LoudnessComparisonRecord]) -> LoudnessComparisonSummary:
    target_errors = [
        abs(error) for error in (record.target_error_db for record in records) if error is not None
    ]
    return LoudnessComparisonSummary(
        row_count=len(records),
        changed_row_count=sum(1 for record in records if record.applied),
        target_reached_row_count=sum(
            1
            for record in records
            if record.target_error_db is not None
            and abs(record.target_error_db) <= _TARGET_TOLERANCE_DB
        ),
        gain_clamped_row_count=sum(1 for record in records if record.gain_clamped),
        peak_limited_row_count=sum(1 for record in records if record.peak_limited),
        degradation_check_failed_row_count=sum(
            1 for record in records if not record.degradation_check_passed
        ),
        mean_source_rms_dbfs=_mean_optional(record.source_rms_dbfs for record in records),
        mean_output_rms_dbfs=_mean_optional(record.output_rms_dbfs for record in records),
        mean_applied_gain_db=round(
            sum(record.applied_gain_db for record in records) / len(records),
            6,
        )
        if records
        else 0.0,
        max_alignment_error=max((record.alignment_error for record in records), default=0.0),
        max_abs_target_error_db=max(target_errors, default=0.0),
    )


def _mean_optional(values: Iterable[float | None]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return round(sum(finite_values) / len(finite_values), 6)


def _format_optional_dbfs(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} dBFS"
