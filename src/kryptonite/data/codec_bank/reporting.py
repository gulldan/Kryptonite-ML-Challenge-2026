"""Build and render reproducible codec/channel simulation artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .ffmpeg import CodecSimulationError, apply_codec_preset, inspect_ffmpeg_tools
from .models import (
    ALLOWED_CODEC_FAMILIES,
    ALLOWED_CODEC_SEVERITIES,
    FAILURES_JSONL_NAME,
    MANIFEST_JSONL_NAME,
    PROBE_AUDIO_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    CodecBankEntry,
    CodecBankFailureRecord,
    CodecBankPlan,
    CodecBankReport,
    CodecBankSummary,
    WrittenCodecBankArtifacts,
)
from .probe import analyze_audio_file, write_probe_audio


def build_codec_bank(
    *,
    project_root: Path | str,
    output_root: Path | str,
    plan: CodecBankPlan,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    plan_path: Path | str | None = None,
) -> CodecBankReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    output_root_path = resolve_project_path(str(project_root_path), str(output_root))
    probe_path = output_root_path / "probe" / PROBE_AUDIO_NAME
    probe_metrics = write_probe_audio(probe_path, settings=plan.probe)
    ffmpeg_metadata = inspect_ffmpeg_tools(ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path)

    entries: list[CodecBankEntry] = []
    failures: list[CodecBankFailureRecord] = []
    for preset in plan.presets:
        preview_path = output_root_path / "previews" / f"{preset.id}.wav"
        if not ffmpeg_metadata.ffmpeg_available:
            failures.append(
                CodecBankFailureRecord(
                    preset_id=preset.id,
                    name=preset.name,
                    family=preset.family,
                    severity=preset.severity,
                    issue_code="ffmpeg_unavailable",
                    reason=ffmpeg_metadata.ffmpeg_error or "ffmpeg is not available in PATH.",
                )
            )
            continue
        try:
            command_trace = apply_codec_preset(
                input_path=probe_path,
                output_path=preview_path,
                preset=preset,
                final_sample_rate_hz=plan.probe.sample_rate_hz,
                ffmpeg_path=ffmpeg_path,
            )
        except CodecSimulationError as exc:
            failures.append(
                CodecBankFailureRecord(
                    preset_id=preset.id,
                    name=preset.name,
                    family=preset.family,
                    severity=preset.severity,
                    issue_code="ffmpeg_apply_error",
                    reason=str(exc),
                )
            )
            continue

        output_metrics = analyze_audio_file(preview_path)
        entries.append(
            CodecBankEntry(
                preset_id=preset.id,
                name=preset.name,
                family=preset.family,
                severity=preset.severity,
                description=preset.description,
                tags=preset.tags,
                sampling_weight=preset.sampling_weight(plan.severity_profiles),
                probe_audio_path=_relative_to_project(probe_path, project_root_path),
                preview_audio_path=_relative_to_project(preview_path, project_root_path),
                preview_sha256=_sha256_file(preview_path),
                ffmpeg_pre_filter_graph=",".join(preset.filters) or None,
                ffmpeg_post_filter_graph=",".join(preset.post_filters) or None,
                ffmpeg_encode_codec=preset.codec_name,
                ffmpeg_container=preset.container_extension if preset.uses_codec_stage else None,
                ffmpeg_encode_sample_rate_hz=preset.encode_sample_rate_hz,
                ffmpeg_encode_bitrate=preset.encode_bitrate,
                ffmpeg_options=preset.ffmpeg_options,
                encode_command=command_trace.encode_command,
                decode_command=command_trace.decode_command,
                source_metrics=probe_metrics,
                output_metrics=output_metrics,
            )
        )

    summary = _build_summary(plan=plan, entries=entries, failures=failures)
    resolved_plan_path = (
        str(resolve_project_path(str(project_root_path), str(plan_path)))
        if plan_path is not None
        else None
    )
    return CodecBankReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        output_root=str(output_root_path),
        plan_path=resolved_plan_path,
        notes=plan.notes,
        probe=plan.probe,
        probe_audio_path=_relative_to_project(probe_path, project_root_path),
        probe_metrics=probe_metrics,
        ffmpeg=ffmpeg_metadata,
        severity_profiles=plan.severity_profiles,
        entries=tuple(entries),
        failures=tuple(failures),
        summary=summary,
    )


def render_codec_bank_markdown(report: CodecBankReport) -> str:
    missing_families = report.summary.missing_family_coverage
    family_coverage = "complete" if not missing_families else ", ".join(missing_families)
    lines = [
        "# Codec Bank Report",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Output root: `{report.output_root}`",
        f"- Plan path: `{report.plan_path or '-'}`",
        f"- Probe audio: `{report.probe_audio_path}`",
        "",
    ]
    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    lines.extend(
        [
            "## FFmpeg Environment",
            "",
            _markdown_table(
                ["Field", "Value"],
                [
                    ["ffmpeg available", str(report.ffmpeg.ffmpeg_available)],
                    ["ffprobe available", str(report.ffmpeg.ffprobe_available)],
                    ["ffmpeg path", report.ffmpeg.ffmpeg_path],
                    ["ffprobe path", report.ffmpeg.ffprobe_path],
                    ["ffmpeg version", report.ffmpeg.version_line or "-"],
                    ["ffprobe version", report.ffmpeg.ffprobe_version_line or "-"],
                    ["configuration", report.ffmpeg.configuration or "-"],
                    ["ffmpeg error", report.ffmpeg.ffmpeg_error or "-"],
                    ["ffprobe error", report.ffmpeg.ffprobe_error or "-"],
                ],
            ),
            "",
            "## Overview",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Presets", str(report.summary.preset_count)],
                    ["Rendered previews", str(report.summary.rendered_preview_count)],
                    ["Failures", str(report.summary.failure_count)],
                    ["Codec stages", str(report.summary.codec_stage_count)],
                    ["Family coverage", family_coverage],
                    ["Family counts", _format_counts(report.summary.family_counts)],
                    ["Severity counts", _format_counts(report.summary.severity_counts)],
                ],
            ),
            "",
            "## Probe Metrics",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Sample rate (Hz)", str(report.probe_metrics.sample_rate_hz)],
                    ["Duration (s)", f"{report.probe_metrics.duration_seconds:.3f}"],
                    ["Peak amplitude", f"{report.probe_metrics.peak_amplitude:.4f}"],
                    ["RMS (dBFS)", f"{report.probe_metrics.rms_dbfs:.3f}"],
                    ["Clipped sample ratio", f"{report.probe_metrics.clipped_sample_ratio:.6f}"],
                    ["Spectral centroid (Hz)", f"{report.probe_metrics.spectral_centroid_hz:.1f}"],
                    [
                        "Spectral rolloff 95% (Hz)",
                        f"{report.probe_metrics.spectral_rolloff_95_hz:.1f}",
                    ],
                ],
            ),
            "",
            "## Severity Profiles",
            "",
            _markdown_table(
                ["Severity", "Weight", "Description"],
                [
                    [
                        name,
                        f"{report.severity_profiles[name].weight_multiplier:.2f}",
                        report.severity_profiles[name].description,
                    ]
                    for name in ALLOWED_CODEC_SEVERITIES
                ],
            ),
            "",
            "## Preset Coverage",
            "",
            _markdown_table(
                [
                    "Preset",
                    "Family",
                    "Severity",
                    "Codec",
                    "Encode rate",
                    "Bitrate",
                    "Filters",
                ],
                [
                    [
                        entry.name,
                        entry.family,
                        entry.severity,
                        entry.ffmpeg_encode_codec or "filter-only",
                        str(entry.ffmpeg_encode_sample_rate_hz or report.probe.sample_rate_hz),
                        entry.ffmpeg_encode_bitrate or "-",
                        entry.ffmpeg_pre_filter_graph or "-",
                    ]
                    for entry in report.entries
                ],
            ),
            "",
        ]
    )

    if report.entries:
        lines.extend(
            [
                "## Preview Metrics",
                "",
                _markdown_table(
                    [
                        "Preset",
                        "RMS (dBFS)",
                        "Rolloff 95% (Hz)",
                        "Rolloff delta (Hz)",
                        "Clipped ratio",
                        "Preview",
                    ],
                    [
                        [
                            entry.preset_id,
                            f"{entry.output_metrics.rms_dbfs:.3f}",
                            f"{entry.output_metrics.spectral_rolloff_95_hz:.1f}",
                            f"{entry.rolloff_delta_hz:.1f}",
                            f"{entry.output_metrics.clipped_sample_ratio:.6f}",
                            entry.preview_audio_path,
                        ]
                        for entry in report.entries
                    ],
                ),
                "",
            ]
        )

    if report.failures:
        lines.extend(["## Failures", ""])
        lines.extend(
            f"- `{record.preset_id}`: `{record.issue_code}` ({record.reason})"
            for record in report.failures
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_codec_bank_report(
    *,
    report: CodecBankReport,
    output_root: Path | str | None = None,
) -> WrittenCodecBankArtifacts:
    output_root_path = (
        Path(report.output_root)
        if output_root is None
        else resolve_project_path(report.project_root, str(output_root))
    )
    manifests_root = output_root_path / "manifests"
    reports_root = output_root_path / "reports"
    manifests_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_root / MANIFEST_JSONL_NAME
    failures_path = manifests_root / FAILURES_JSONL_NAME
    json_path = reports_root / REPORT_JSON_NAME
    markdown_path = reports_root / REPORT_MARKDOWN_NAME
    probe_path = resolve_project_path(report.project_root, report.probe_audio_path)

    manifest_path.write_text(
        "".join(json.dumps(entry.to_dict(), sort_keys=True) + "\n" for entry in report.entries)
    )
    failures_path.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in report.failures)
    )
    json_path.write_text(
        json.dumps(
            report.to_dict(include_entries=True, include_failures=True),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_path.write_text(render_codec_bank_markdown(report))
    return WrittenCodecBankArtifacts(
        output_root=str(output_root_path),
        probe_path=str(probe_path),
        manifest_path=str(manifest_path),
        failures_path=str(failures_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def _build_summary(
    *,
    plan: CodecBankPlan,
    entries: list[CodecBankEntry],
    failures: list[CodecBankFailureRecord],
) -> CodecBankSummary:
    family_counts = Counter(preset.family for preset in plan.presets)
    severity_counts = Counter(preset.severity for preset in plan.presets)
    return CodecBankSummary(
        preset_count=len(plan.presets),
        rendered_preview_count=len(entries),
        failure_count=len(failures),
        codec_stage_count=sum(preset.uses_codec_stage for preset in plan.presets),
        family_counts={
            family: family_counts.get(family, 0)
            for family in ALLOWED_CODEC_FAMILIES
            if family_counts.get(family, 0)
        },
        severity_counts={
            severity: severity_counts.get(severity, 0)
            for severity in ALLOWED_CODEC_SEVERITIES
            if severity_counts.get(severity, 0)
        },
    )


def _relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = ["| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


__all__ = [
    "build_codec_bank",
    "render_codec_bank_markdown",
    "write_codec_bank_report",
]
