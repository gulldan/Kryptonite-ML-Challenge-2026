"""Report building and rendering for audio-quality EDA."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .constants import (
    KNOWN_DATA_SPLITS,
    LONG_DURATION_SECONDS,
    LOW_LOUDNESS_DBFS,
    MODERATE_SILENCE_RATIO,
    SILENCE_CHUNK_MS,
    SILENCE_THRESHOLD_DBFS,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE_HZ,
)
from .formatting import (
    format_counts,
    format_dbfs,
    format_duration,
    format_ratio,
    format_seconds,
    markdown_table,
    render_histogram,
)
from .manifests import (
    collect_manifest_inputs,
    deduplicate_records,
    group_records,
    summarize_records,
)
from .models import (
    AudioQualityPattern,
    DatasetAudioQualityReport,
    NamedSummary,
    QualitySummary,
)
from .selection import select_examples


def build_dataset_audio_quality_report(
    *,
    project_root: Path | str,
    manifests_root: Path | str,
) -> DatasetAudioQualityReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifests_root_path = resolve_project_path(str(project_root_path), str(manifests_root))
    generated_at = utc_now()

    collected = collect_manifest_inputs(
        project_root=project_root_path,
        manifests_root=manifests_root_path,
    )
    unique_records = deduplicate_records(collected.all_records)
    total_summary = summarize_records(list(unique_records.values()))

    report = DatasetAudioQualityReport(
        generated_at=generated_at,
        project_root=str(project_root_path),
        manifests_root=str(manifests_root_path),
        raw_entry_count=len(collected.all_records),
        duplicate_entry_count=max(0, len(collected.all_records) - len(unique_records)),
        invalid_line_count=collected.invalid_line_count,
        total_summary=total_summary,
        split_summaries=build_named_summaries(
            unique_records.values(),
            key=lambda record: record.split_name,
        ),
        dataset_summaries=build_named_summaries(
            unique_records.values(),
            key=lambda record: record.dataset_name,
        ),
        source_summaries=build_named_summaries(
            unique_records.values(),
            key=lambda record: record.source_label or "unknown",
        ),
        manifest_profiles=collected.manifest_profiles,
        ignored_manifests=collected.ignored_manifests,
        patterns=build_patterns(total_summary),
        examples=select_examples(list(unique_records.values())),
        warnings=[],
        records=sorted(
            unique_records.values(),
            key=lambda record: (record.audio_path or record.identity_key, record.identity_key),
        ),
    )
    report.warnings = build_warnings(report)
    return report


def render_dataset_audio_quality_markdown(report: DatasetAudioQualityReport) -> str:
    lines = [
        "# Dataset Audio Quality Report",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Manifests root: `{report.manifests_root}`",
        "",
    ]

    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")

    lines.extend(
        [
            "## Overview",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["Profiled manifests", str(report.manifest_count)],
                    ["Ignored JSONL files", str(len(report.ignored_manifests))],
                    ["Raw manifest rows", str(report.raw_entry_count)],
                    ["Deduplicated dataset rows", str(report.total_summary.entry_count)],
                    ["Duplicate rows collapsed", str(report.duplicate_entry_count)],
                    ["Invalid JSON lines", str(report.invalid_line_count)],
                    ["Rows with audio paths", str(report.total_summary.entries_with_audio_path)],
                    ["Resolved audio files", str(report.total_summary.resolved_audio_count)],
                    [
                        "Waveform-derived metrics",
                        str(report.total_summary.waveform_metrics_count),
                    ],
                    ["Unique speakers", str(report.total_summary.unique_speakers)],
                    ["Unique sessions", str(report.total_summary.unique_sessions)],
                    [
                        "Total duration",
                        format_duration(report.total_summary.duration_summary.total),
                    ],
                    [
                        "Duration p95",
                        format_seconds(report.total_summary.duration_summary.p95),
                    ],
                    [
                        "Mean loudness",
                        format_dbfs(report.total_summary.loudness_summary.mean),
                    ],
                    [
                        "Silence ratio p95",
                        format_ratio(report.total_summary.silence_summary.p95),
                    ],
                    ["Flags", format_counts(report.total_summary.flag_counts, limit=8)],
                ],
            ),
            "",
            "## Split Quality",
            "",
            markdown_table(
                [
                    "Split",
                    "Rows",
                    "Total duration",
                    "Mean loudness",
                    "Silence p95",
                    "Sample rates",
                    "Flags",
                ],
                [
                    [
                        summary.name,
                        str(summary.summary.entry_count),
                        format_duration(summary.summary.duration_summary.total),
                        format_dbfs(summary.summary.loudness_summary.mean),
                        format_ratio(summary.summary.silence_summary.p95),
                        format_counts(summary.summary.sample_rate_counts, limit=4),
                        format_counts(summary.summary.flag_counts, limit=4),
                    ]
                    for summary in report.split_summaries
                ]
                or [["-", "0", "0.00 s", "-", "-", "-", "-"]],
            ),
            "",
            "## Dataset Quality",
            "",
            markdown_table(
                [
                    "Dataset",
                    "Rows",
                    "Split mix",
                    "Source mix",
                    "Sample rates",
                    "Channels",
                    "Flags",
                ],
                [
                    [
                        summary.name,
                        str(summary.summary.entry_count),
                        format_counts(summary.summary.split_counts, limit=4),
                        format_counts(summary.summary.source_counts, limit=4),
                        format_counts(summary.summary.sample_rate_counts, limit=4),
                        format_counts(summary.summary.channel_counts, limit=4),
                        format_counts(summary.summary.flag_counts, limit=4),
                    ]
                    for summary in report.dataset_summaries
                ]
                or [["-", "0", "-", "-", "-", "-", "-"]],
            ),
            "",
            "## Manifest Inputs",
            "",
            markdown_table(
                [
                    "Manifest",
                    "Dataset",
                    "Rows",
                    "Mean loudness",
                    "Silence p95",
                    "Flags",
                    "Invalid lines",
                ],
                [
                    [
                        profile.manifest_path,
                        profile.primary_dataset,
                        str(profile.summary.entry_count),
                        format_dbfs(profile.summary.loudness_summary.mean),
                        format_ratio(profile.summary.silence_summary.p95),
                        format_counts(profile.summary.flag_counts, limit=4),
                        str(profile.invalid_line_count),
                    ]
                    for profile in report.manifest_profiles
                ]
                or [["-", "-", "0", "-", "-", "-", "0"]],
            ),
            "",
            "## Observed Distributions",
            "",
            markdown_table(
                ["Category", "Counts"],
                [
                    ["Split", format_counts(report.total_summary.split_counts, limit=6)],
                    ["Role", format_counts(report.total_summary.role_counts, limit=6)],
                    ["Dataset", format_counts(report.total_summary.dataset_counts, limit=6)],
                    ["Source", format_counts(report.total_summary.source_counts, limit=8)],
                    [
                        "Capture conditions",
                        format_counts(report.total_summary.condition_counts, limit=8),
                    ],
                    [
                        "Sample rates",
                        format_counts(report.total_summary.sample_rate_counts, limit=6),
                    ],
                    ["Channels", format_counts(report.total_summary.channel_counts, limit=6)],
                    ["Formats", format_counts(report.total_summary.audio_format_counts, limit=6)],
                ],
            ),
            "",
            "## Quality Flags",
            "",
            markdown_table(
                ["Flag", "Rows", "Share"],
                [
                    [flag, str(count), format_ratio(count / report.total_summary.entry_count)]
                    for flag, count in report.total_summary.flag_counts.items()
                ]
                or [["-", "0", "-"]],
            ),
            "",
            "## Key Patterns",
            "",
        ]
    )

    if report.patterns:
        lines.extend(
            [
                f"- `{pattern.code}`: {pattern.summary} {pattern.implication}"
                for pattern in report.patterns
            ]
        )
    else:
        lines.append("_No actionable audio-quality patterns were detected._")
    lines.append("")

    lines.extend(["## Flagged Examples", ""])
    if report.examples:
        lines.append(
            markdown_table(
                [
                    "Audio",
                    "Split",
                    "Dataset",
                    "Source",
                    "Condition",
                    "Duration",
                    "Loudness",
                    "Silence",
                    "Flags",
                ],
                [
                    [
                        example.audio_path,
                        example.split_name,
                        example.dataset_name,
                        example.source_label or "-",
                        example.condition_label or "-",
                        format_seconds(example.duration_seconds),
                        format_dbfs(example.rms_dbfs),
                        format_ratio(example.silence_ratio),
                        ", ".join(example.flags),
                    ]
                    for example in report.examples
                ],
            )
        )
    else:
        lines.append("_No flagged examples._")
    lines.append("")

    lines.extend(
        [
            "## Graphs",
            "",
            "### Duration Histogram",
            "",
            render_histogram(report.total_summary.duration_histogram),
            "",
            "### Loudness Histogram (dBFS)",
            "",
            render_histogram(report.total_summary.loudness_histogram),
            "",
            "### Silence Ratio Histogram",
            "",
            render_histogram(report.total_summary.silence_histogram),
            "",
        ]
    )

    if report.ignored_manifests:
        lines.extend(
            [
                "## Ignored JSONL Files",
                "",
                markdown_table(
                    ["Path", "Reason"],
                    [
                        [manifest.manifest_path, manifest.reason]
                        for manifest in report.ignored_manifests
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            (
                "- Aggregated tables are deduplicated by canonical row identity so that "
                "`all/train/dev` manifest overlap does not inflate the quality summary."
            ),
            (
                f"- Silence is estimated on {SILENCE_CHUNK_MS} ms waveform windows using a "
                f"{SILENCE_THRESHOLD_DBFS:.0f} dBFS RMS threshold."
            ),
            (
                "- Loudness, peak, DC offset, clipping, and silence metrics are derived from "
                "decoded WAV/FLAC/MP3 waveforms via the shared audio I/O layer."
            ),
            (
                "- JSONL files whose name contains `trial` or `quarantine` are excluded "
                "from active audio-quality profiling."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_named_summaries(records, *, key) -> list[NamedSummary]:
    return [
        NamedSummary(name=name, summary=summarize_records(grouped_records))
        for name, grouped_records in group_records(records, key=key)
        if name
    ]


def build_patterns(summary: QualitySummary) -> list[AudioQualityPattern]:
    patterns: list[AudioQualityPattern] = []
    if summary.entry_count == 0:
        return patterns

    if set(summary.sample_rate_counts) != {str(TARGET_SAMPLE_RATE_HZ)}:
        patterns.append(
            AudioQualityPattern(
                code="mixed_sample_rates",
                summary=(
                    "The active manifests span multiple sample rates "
                    f"({format_counts(summary.sample_rate_counts, limit=6)})."
                ),
                implication=(
                    "Resampling to 16 kHz must be an explicit preprocessing step before "
                    "feature extraction or augmentation."
                ),
            )
        )
    if set(summary.channel_counts) - {str(TARGET_CHANNELS)}:
        patterns.append(
            AudioQualityPattern(
                code="non_mono_audio",
                summary=(
                    f"Some rows are not mono ({format_counts(summary.channel_counts, limit=6)})."
                ),
                implication=(
                    "The loader should fold channels down deterministically before scoring."
                ),
            )
        )

    silence_flag_count = summary.flag_counts.get("high_silence_ratio", 0) + summary.flag_counts.get(
        "moderate_silence_ratio", 0
    )
    if silence_flag_count:
        patterns.append(
            AudioQualityPattern(
                code="silence_heavy_tail",
                summary=(
                    f"{silence_flag_count} rows have at least {MODERATE_SILENCE_RATIO:.0%} silent "
                    f"windows; silence ratio p95 is {format_ratio(summary.silence_summary.p95)}."
                ),
                implication=(
                    "Optional VAD/trimming and silence-aware augmentation need to be part of the "
                    "preprocessing policy."
                ),
            )
        )

    loudness_flag_count = summary.flag_counts.get("very_low_loudness", 0) + summary.flag_counts.get(
        "low_loudness", 0
    )
    if loudness_flag_count:
        patterns.append(
            AudioQualityPattern(
                code="low_level_recordings",
                summary=(
                    f"{loudness_flag_count} rows are quieter than {LOW_LOUDNESS_DBFS:.0f} dBFS; "
                    f"mean loudness is {format_dbfs(summary.loudness_summary.mean)}."
                ),
                implication=(
                    "The preprocessing stack should define loudness normalization or gain limits "
                    "before robust training."
                ),
            )
        )

    zero_signal_count = summary.flag_counts.get("zero_signal", 0)
    if zero_signal_count:
        patterns.append(
            AudioQualityPattern(
                code="zero_signal_rows",
                summary=(
                    f"{zero_signal_count} rows decode to an all-zero waveform and carry no usable "
                    "speaker signal."
                ),
                implication=(
                    "Those rows should be quarantined or explicitly excluded before training and "
                    "verification benchmarking."
                ),
            )
        )

    clipping_count = summary.flag_counts.get("clipping_risk", 0)
    if clipping_count:
        patterns.append(
            AudioQualityPattern(
                code="clipping_present",
                summary=(
                    f"{clipping_count} rows reach near-full-scale peaks; peak max is "
                    f"{format_dbfs(summary.peak_summary.maximum)}."
                ),
                implication=(
                    "Clipping should be tracked as a quality flag and considered when defining "
                    "normalization or corruption policies."
                ),
            )
        )

    if (
        summary.duration_summary.maximum is not None
        and summary.duration_summary.median is not None
        and summary.duration_summary.maximum
        > max(LONG_DURATION_SECONDS, summary.duration_summary.median * 3.0)
    ):
        patterns.append(
            AudioQualityPattern(
                code="duration_long_tail",
                summary=(
                    "Duration has a long tail: "
                    f"median {format_seconds(summary.duration_summary.median)}, "
                    f"p95 {format_seconds(summary.duration_summary.p95)}, max "
                    f"{format_seconds(summary.duration_summary.maximum)}."
                ),
                implication=(
                    "Chunking/truncation policy should be explicit so training batches do not "
                    "inherit uncontrolled sequence-length variance."
                ),
            )
        )

    if summary.missing_audio_file_count or summary.audio_inspection_error_count:
        patterns.append(
            AudioQualityPattern(
                code="inspection_gaps",
                summary=(
                    f"{summary.missing_audio_file_count} rows are missing audio files and "
                    f"{summary.audio_inspection_error_count} rows could not be fully inspected."
                ),
                implication=(
                    "Manifest validation should gate preprocessing so bad paths do not silently "
                    "enter downstream jobs."
                ),
            )
        )
    return patterns


def build_warnings(report: DatasetAudioQualityReport) -> list[str]:
    warnings: list[str] = []
    if not Path(report.manifests_root).exists():
        warnings.append("Configured manifests root does not exist.")
        return warnings
    if not report.manifest_profiles:
        warnings.append("No data manifests were discovered under the manifests root.")
        return warnings

    missing_splits = [
        split_name
        for split_name in KNOWN_DATA_SPLITS
        if split_name not in report.total_summary.split_counts
    ]
    if missing_splits:
        warnings.append(
            "Expected dataset coverage is incomplete. Missing splits: "
            + ", ".join(missing_splits)
            + "."
        )
    if report.total_summary.missing_audio_path_count:
        warnings.append(
            f"{report.total_summary.missing_audio_path_count} rows do not define `audio_path`."
        )
    if report.total_summary.missing_audio_file_count:
        warnings.append(
            f"{report.total_summary.missing_audio_file_count} rows point to missing audio files."
        )
    if report.total_summary.audio_inspection_error_count:
        warnings.append(
            f"{report.total_summary.audio_inspection_error_count} audio files could not "
            "be fully inspected."
        )
    if report.invalid_line_count:
        warnings.append(f"{report.invalid_line_count} invalid JSONL lines were skipped.")
    if report.duplicate_entry_count:
        warnings.append(
            f"{report.duplicate_entry_count} overlapping rows were deduplicated across manifests."
        )
    return warnings


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
