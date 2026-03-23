"""Reproducible audio-quality EDA for manifests-backed audio corpora."""

from __future__ import annotations

import audioop
import json
import math
import wave
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kryptonite.deployment import resolve_project_path

KNOWN_DATA_SPLITS: tuple[str, ...] = ("train", "dev", "demo")
DURATION_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-5s", 2.0, 5.0),
    ("5-10s", 5.0, 10.0),
    ("10-30s", 10.0, 30.0),
    ("30-60s", 30.0, 60.0),
    ("60s+", 60.0, None),
)
LOUDNESS_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<-35", None, -35.0),
    ("-35:-30", -35.0, -30.0),
    ("-30:-25", -30.0, -25.0),
    ("-25:-20", -25.0, -20.0),
    ("-20:-15", -20.0, -15.0),
    ("-15:-10", -15.0, -10.0),
    (">=-10", -10.0, None),
)
SILENCE_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("0-5%", 0.0, 0.05),
    ("5-20%", 0.05, 0.20),
    ("20-50%", 0.20, 0.50),
    ("50-80%", 0.50, 0.80),
    ("80%+", 0.80, None),
)
SILENCE_CHUNK_MS = 100
SILENCE_THRESHOLD_DBFS = -45.0
TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_CHANNELS = 1
SHORT_DURATION_SECONDS = 1.0
LONG_DURATION_SECONDS = 15.0
LOW_LOUDNESS_DBFS = -30.0
VERY_LOW_LOUDNESS_DBFS = -35.0
CLIPPING_PEAK_DBFS = -0.1
HIGH_SILENCE_RATIO = 0.50
MODERATE_SILENCE_RATIO = 0.20
HIGH_DC_OFFSET_RATIO = 0.01
MAX_EXAMPLES = 12


@dataclass(frozen=True, slots=True)
class NumericDistribution:
    count: int
    total: float
    minimum: float | None
    mean: float | None
    median: float | None
    p95: float | None
    maximum: float | None

    @classmethod
    def from_values(cls, values: list[float]) -> NumericDistribution:
        if not values:
            return cls(
                count=0,
                total=0.0,
                minimum=None,
                mean=None,
                median=None,
                p95=None,
                maximum=None,
            )

        ordered = sorted(values)
        return cls(
            count=len(ordered),
            total=float(sum(ordered)),
            minimum=float(ordered[0]),
            mean=float(sum(ordered) / len(ordered)),
            median=float(_percentile(ordered, 0.5)),
            p95=float(_percentile(ordered, 0.95)),
            maximum=float(ordered[-1]),
        )

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total": self.total,
            "minimum": self.minimum,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class HistogramBucket:
    label: str
    count: int

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "count": self.count}


@dataclass(frozen=True, slots=True)
class AudioQualityInspection:
    exists: bool
    file_size_bytes: int | None
    duration_seconds: float | None
    sample_rate_hz: int | None
    channels: int | None
    audio_format: str | None
    sample_width_bytes: int | None
    rms_dbfs: float | None
    peak_dbfs: float | None
    silence_ratio: float | None
    dc_offset_ratio: float | None
    clipped_chunk_ratio: float | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestQualityRecord:
    identity_key: str
    audio_path: str | None
    audio_exists: bool
    audio_error: str | None
    dataset_name: str
    split_name: str
    speaker_id: str | None
    session_key: str | None
    role: str | None
    source_label: str | None
    condition_label: str | None
    sample_rate_hz: int | None
    channels: int | None
    audio_format: str | None
    duration_seconds: float | None
    file_size_bytes: int | None
    rms_dbfs: float | None
    peak_dbfs: float | None
    silence_ratio: float | None
    dc_offset_ratio: float | None
    clipped_chunk_ratio: float | None
    quality_flags: tuple[str, ...]


@dataclass(slots=True)
class QualitySummary:
    entry_count: int
    entries_with_audio_path: int
    resolved_audio_count: int
    waveform_metrics_count: int
    missing_audio_path_count: int
    missing_audio_file_count: int
    audio_inspection_error_count: int
    unique_speakers: int
    unique_sessions: int
    duration_summary: NumericDistribution
    loudness_summary: NumericDistribution
    peak_summary: NumericDistribution
    silence_summary: NumericDistribution
    dc_offset_summary: NumericDistribution
    clipped_chunk_summary: NumericDistribution
    duration_histogram: list[HistogramBucket] = field(default_factory=list)
    loudness_histogram: list[HistogramBucket] = field(default_factory=list)
    silence_histogram: list[HistogramBucket] = field(default_factory=list)
    flag_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    role_counts: dict[str, int] = field(default_factory=dict)
    dataset_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    condition_counts: dict[str, int] = field(default_factory=dict)
    sample_rate_counts: dict[str, int] = field(default_factory=dict)
    channel_counts: dict[str, int] = field(default_factory=dict)
    audio_format_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_count": self.entry_count,
            "entries_with_audio_path": self.entries_with_audio_path,
            "resolved_audio_count": self.resolved_audio_count,
            "waveform_metrics_count": self.waveform_metrics_count,
            "missing_audio_path_count": self.missing_audio_path_count,
            "missing_audio_file_count": self.missing_audio_file_count,
            "audio_inspection_error_count": self.audio_inspection_error_count,
            "unique_speakers": self.unique_speakers,
            "unique_sessions": self.unique_sessions,
            "duration_summary": self.duration_summary.to_dict(),
            "loudness_summary": self.loudness_summary.to_dict(),
            "peak_summary": self.peak_summary.to_dict(),
            "silence_summary": self.silence_summary.to_dict(),
            "dc_offset_summary": self.dc_offset_summary.to_dict(),
            "clipped_chunk_summary": self.clipped_chunk_summary.to_dict(),
            "duration_histogram": [bucket.to_dict() for bucket in self.duration_histogram],
            "loudness_histogram": [bucket.to_dict() for bucket in self.loudness_histogram],
            "silence_histogram": [bucket.to_dict() for bucket in self.silence_histogram],
            "flag_counts": dict(self.flag_counts),
            "split_counts": dict(self.split_counts),
            "role_counts": dict(self.role_counts),
            "dataset_counts": dict(self.dataset_counts),
            "source_counts": dict(self.source_counts),
            "condition_counts": dict(self.condition_counts),
            "sample_rate_counts": dict(self.sample_rate_counts),
            "channel_counts": dict(self.channel_counts),
            "audio_format_counts": dict(self.audio_format_counts),
        }


@dataclass(slots=True)
class NamedSummary:
    name: str
    summary: QualitySummary

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "summary": self.summary.to_dict()}


@dataclass(slots=True)
class ManifestQualityProfile:
    manifest_path: str
    primary_dataset: str
    invalid_line_count: int
    summary: QualitySummary

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "primary_dataset": self.primary_dataset,
            "invalid_line_count": self.invalid_line_count,
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class IgnoredManifest:
    manifest_path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"manifest_path": self.manifest_path, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class AudioQualityPattern:
    code: str
    summary: str
    implication: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "summary": self.summary,
            "implication": self.implication,
        }


@dataclass(frozen=True, slots=True)
class FlaggedExample:
    audio_path: str
    split_name: str
    dataset_name: str
    source_label: str | None
    condition_label: str | None
    duration_seconds: float | None
    rms_dbfs: float | None
    peak_dbfs: float | None
    silence_ratio: float | None
    flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "split_name": self.split_name,
            "dataset_name": self.dataset_name,
            "source_label": self.source_label,
            "condition_label": self.condition_label,
            "duration_seconds": self.duration_seconds,
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "silence_ratio": self.silence_ratio,
            "flags": list(self.flags),
        }


@dataclass(slots=True)
class DatasetAudioQualityReport:
    generated_at: str
    project_root: str
    manifests_root: str
    raw_entry_count: int
    duplicate_entry_count: int
    invalid_line_count: int
    total_summary: QualitySummary
    split_summaries: list[NamedSummary]
    dataset_summaries: list[NamedSummary]
    source_summaries: list[NamedSummary]
    manifest_profiles: list[ManifestQualityProfile]
    ignored_manifests: list[IgnoredManifest]
    patterns: list[AudioQualityPattern]
    examples: list[FlaggedExample]
    warnings: list[str]

    @property
    def manifest_count(self) -> int:
        return len(self.manifest_profiles)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "manifests_root": self.manifests_root,
            "manifest_count": self.manifest_count,
            "raw_entry_count": self.raw_entry_count,
            "duplicate_entry_count": self.duplicate_entry_count,
            "invalid_line_count": self.invalid_line_count,
            "warnings": list(self.warnings),
            "total_summary": self.total_summary.to_dict(),
            "split_summaries": [summary.to_dict() for summary in self.split_summaries],
            "dataset_summaries": [summary.to_dict() for summary in self.dataset_summaries],
            "source_summaries": [summary.to_dict() for summary in self.source_summaries],
            "manifest_profiles": [profile.to_dict() for profile in self.manifest_profiles],
            "ignored_manifests": [manifest.to_dict() for manifest in self.ignored_manifests],
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "examples": [example.to_dict() for example in self.examples],
        }


@dataclass(frozen=True, slots=True)
class WrittenDatasetAudioQualityReport:
    output_root: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def build_dataset_audio_quality_report(
    *,
    project_root: Path | str,
    manifests_root: Path | str,
) -> DatasetAudioQualityReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifests_root_path = resolve_project_path(str(project_root_path), str(manifests_root))
    generated_at = _utc_now()

    manifest_profiles: list[ManifestQualityProfile] = []
    ignored_manifests: list[IgnoredManifest] = []
    all_records: list[ManifestQualityRecord] = []
    audio_cache: dict[Path, AudioQualityInspection] = {}

    if manifests_root_path.exists():
        for manifest_path in sorted(manifests_root_path.rglob("*.jsonl")):
            relative_manifest_path = _relative_to_project(manifest_path, project_root_path)
            manifest_name = manifest_path.name.lower()
            if "trial" in manifest_name:
                ignored_manifests.append(
                    IgnoredManifest(
                        manifest_path=relative_manifest_path,
                        reason="trial-only JSONL does not contain dataset rows",
                    )
                )
                continue
            if "quarantine" in manifest_name:
                ignored_manifests.append(
                    IgnoredManifest(
                        manifest_path=relative_manifest_path,
                        reason="quarantine JSONL is excluded from active dataset profiling",
                    )
                )
                continue

            objects, invalid_line_count = _load_jsonl_objects(manifest_path)
            if "manifest" not in manifest_name and not any("audio_path" in row for row in objects):
                ignored_manifests.append(
                    IgnoredManifest(
                        manifest_path=relative_manifest_path,
                        reason="JSONL does not look like a data manifest",
                    )
                )
                continue

            manifest_records = [
                _build_quality_record(
                    manifest_path=manifest_path,
                    entry=row,
                    line_number=index,
                    project_root=project_root_path,
                    manifests_root=manifests_root_path,
                    audio_cache=audio_cache,
                )
                for index, row in enumerate(objects, start=1)
            ]
            manifest_profiles.append(
                ManifestQualityProfile(
                    manifest_path=relative_manifest_path,
                    primary_dataset=_primary_dataset_name(manifest_records, manifest_path),
                    invalid_line_count=invalid_line_count,
                    summary=_summarize_records(manifest_records),
                )
            )
            all_records.extend(manifest_records)

    unique_records = _deduplicate_records(all_records)
    total_summary = _summarize_records(list(unique_records.values()))

    split_summaries = [
        NamedSummary(name=name, summary=_summarize_records(records))
        for name, records in _group_records(
            unique_records.values(),
            key=lambda record: record.split_name,
        )
        if name
    ]
    dataset_summaries = [
        NamedSummary(name=name, summary=_summarize_records(records))
        for name, records in _group_records(
            unique_records.values(),
            key=lambda record: record.dataset_name,
        )
        if name
    ]
    source_summaries = [
        NamedSummary(name=name, summary=_summarize_records(records))
        for name, records in _group_records(
            unique_records.values(),
            key=lambda record: record.source_label or "unknown",
        )
        if name
    ]

    report = DatasetAudioQualityReport(
        generated_at=generated_at,
        project_root=str(project_root_path),
        manifests_root=str(manifests_root_path),
        raw_entry_count=len(all_records),
        duplicate_entry_count=max(0, len(all_records) - len(unique_records)),
        invalid_line_count=sum(profile.invalid_line_count for profile in manifest_profiles),
        total_summary=total_summary,
        split_summaries=split_summaries,
        dataset_summaries=dataset_summaries,
        source_summaries=source_summaries,
        manifest_profiles=manifest_profiles,
        ignored_manifests=ignored_manifests,
        patterns=_build_patterns(total_summary),
        examples=_select_examples(list(unique_records.values())),
        warnings=[],
    )
    report.warnings = _build_warnings(report)
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
            _markdown_table(
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
                        _format_duration(report.total_summary.duration_summary.total),
                    ],
                    [
                        "Duration p95",
                        _format_seconds(report.total_summary.duration_summary.p95),
                    ],
                    [
                        "Mean loudness",
                        _format_dbfs(report.total_summary.loudness_summary.mean),
                    ],
                    [
                        "Silence ratio p95",
                        _format_ratio(report.total_summary.silence_summary.p95),
                    ],
                    ["Flags", _format_counts(report.total_summary.flag_counts, limit=8)],
                ],
            ),
            "",
            "## Split Quality",
            "",
            _markdown_table(
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
                        _format_duration(summary.summary.duration_summary.total),
                        _format_dbfs(summary.summary.loudness_summary.mean),
                        _format_ratio(summary.summary.silence_summary.p95),
                        _format_counts(summary.summary.sample_rate_counts, limit=4),
                        _format_counts(summary.summary.flag_counts, limit=4),
                    ]
                    for summary in report.split_summaries
                ]
                or [["-", "0", "0.00 s", "-", "-", "-", "-"]],
            ),
            "",
            "## Dataset Quality",
            "",
            _markdown_table(
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
                        _format_counts(summary.summary.split_counts, limit=4),
                        _format_counts(summary.summary.source_counts, limit=4),
                        _format_counts(summary.summary.sample_rate_counts, limit=4),
                        _format_counts(summary.summary.channel_counts, limit=4),
                        _format_counts(summary.summary.flag_counts, limit=4),
                    ]
                    for summary in report.dataset_summaries
                ]
                or [["-", "0", "-", "-", "-", "-", "-"]],
            ),
            "",
            "## Manifest Inputs",
            "",
            _markdown_table(
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
                        _format_dbfs(profile.summary.loudness_summary.mean),
                        _format_ratio(profile.summary.silence_summary.p95),
                        _format_counts(profile.summary.flag_counts, limit=4),
                        str(profile.invalid_line_count),
                    ]
                    for profile in report.manifest_profiles
                ]
                or [["-", "-", "0", "-", "-", "-", "0"]],
            ),
            "",
            "## Observed Distributions",
            "",
            _markdown_table(
                ["Category", "Counts"],
                [
                    ["Split", _format_counts(report.total_summary.split_counts, limit=6)],
                    ["Role", _format_counts(report.total_summary.role_counts, limit=6)],
                    ["Dataset", _format_counts(report.total_summary.dataset_counts, limit=6)],
                    ["Source", _format_counts(report.total_summary.source_counts, limit=8)],
                    [
                        "Capture conditions",
                        _format_counts(report.total_summary.condition_counts, limit=8),
                    ],
                    [
                        "Sample rates",
                        _format_counts(report.total_summary.sample_rate_counts, limit=6),
                    ],
                    ["Channels", _format_counts(report.total_summary.channel_counts, limit=6)],
                    ["Formats", _format_counts(report.total_summary.audio_format_counts, limit=6)],
                ],
            ),
            "",
            "## Quality Flags",
            "",
            _markdown_table(
                ["Flag", "Rows", "Share"],
                [
                    [flag, str(count), _format_ratio(count / report.total_summary.entry_count)]
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
            _markdown_table(
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
                        _format_seconds(example.duration_seconds),
                        _format_dbfs(example.rms_dbfs),
                        _format_ratio(example.silence_ratio),
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
            _render_histogram(report.total_summary.duration_histogram),
            "",
            "### Loudness Histogram (dBFS)",
            "",
            _render_histogram(report.total_summary.loudness_histogram),
            "",
            "### Silence Ratio Histogram",
            "",
            _render_histogram(report.total_summary.silence_histogram),
            "",
        ]
    )

    if report.ignored_manifests:
        lines.extend(
            [
                "## Ignored JSONL Files",
                "",
                _markdown_table(
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
                "- Loudness, peak, DC offset, and clipping signals are waveform-derived for "
                "WAV files; non-WAV files fall back to manifest/header metadata only."
            ),
            (
                "- JSONL files whose name contains `trial` or `quarantine` are excluded "
                "from active audio-quality profiling."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_dataset_audio_quality_report(
    *,
    report: DatasetAudioQualityReport,
    output_root: Path | str,
) -> WrittenDatasetAudioQualityReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "dataset_audio_quality.json"
    markdown_path = output_root_path / "dataset_audio_quality.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_dataset_audio_quality_markdown(report))
    return WrittenDatasetAudioQualityReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def _build_quality_record(
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    line_number: int,
    project_root: Path,
    manifests_root: Path,
    audio_cache: dict[Path, AudioQualityInspection],
) -> ManifestQualityRecord:
    audio_path = _coerce_str(entry.get("audio_path"))
    resolved_audio_path = (
        resolve_project_path(str(project_root), audio_path) if audio_path is not None else None
    )
    inspection = (
        _inspect_audio_file(resolved_audio_path, audio_cache)
        if resolved_audio_path is not None
        else AudioQualityInspection(
            exists=False,
            file_size_bytes=None,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=None,
            sample_width_bytes=None,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
        )
    )

    manifest_duration = _coerce_float(entry.get("duration_seconds"))
    duration_seconds = inspection.duration_seconds or manifest_duration
    sample_rate_hz = inspection.sample_rate_hz or _coerce_int(entry.get("sample_rate_hz"))
    channels = inspection.channels or _coerce_int(entry.get("channels"))
    dataset_name = _infer_dataset_name(
        entry=entry,
        audio_path=audio_path,
        manifest_path=manifest_path,
        manifests_root=manifests_root,
    )
    split_name = _infer_split_name(entry=entry, manifest_path=manifest_path)
    speaker_id = _coerce_str(entry.get("speaker_id"))
    session_key = _infer_session_key(entry=entry, speaker_id=speaker_id)
    role = _coerce_str(entry.get("role"))
    source_label = _infer_source_label(entry=entry, dataset_name=dataset_name)
    condition_label = _infer_condition_label(entry)
    audio_format = inspection.audio_format or _audio_format_from_path(audio_path)
    quality_flags = _build_quality_flags(
        audio_path=audio_path,
        inspection=inspection,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_seconds=duration_seconds,
    )
    identity_key = _build_identity_key(
        audio_path=audio_path,
        entry=entry,
        manifest_path=manifest_path,
        line_number=line_number,
    )

    return ManifestQualityRecord(
        identity_key=identity_key,
        audio_path=audio_path,
        audio_exists=inspection.exists,
        audio_error=inspection.error,
        dataset_name=dataset_name,
        split_name=split_name,
        speaker_id=speaker_id,
        session_key=session_key,
        role=role,
        source_label=source_label,
        condition_label=condition_label,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        audio_format=audio_format,
        duration_seconds=duration_seconds,
        file_size_bytes=inspection.file_size_bytes,
        rms_dbfs=inspection.rms_dbfs,
        peak_dbfs=inspection.peak_dbfs,
        silence_ratio=inspection.silence_ratio,
        dc_offset_ratio=inspection.dc_offset_ratio,
        clipped_chunk_ratio=inspection.clipped_chunk_ratio,
        quality_flags=quality_flags,
    )


def _build_quality_flags(
    *,
    audio_path: str | None,
    inspection: AudioQualityInspection,
    sample_rate_hz: int | None,
    channels: int | None,
    duration_seconds: float | None,
) -> tuple[str, ...]:
    flags: list[str] = []
    if audio_path is None:
        flags.append("missing_audio_path")
    elif not inspection.exists:
        flags.append("missing_audio_file")

    if inspection.error is not None:
        flags.append("audio_read_error")
    elif inspection.exists and inspection.rms_dbfs is None and inspection.audio_format != "wav":
        flags.append("unsupported_signal_analysis")

    if sample_rate_hz is not None and sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        flags.append("non_16k_sample_rate")
    if channels is not None and channels != TARGET_CHANNELS:
        flags.append("non_mono_audio")
    if duration_seconds is not None and duration_seconds < SHORT_DURATION_SECONDS:
        flags.append("short_duration")
    if duration_seconds is not None and duration_seconds > LONG_DURATION_SECONDS:
        flags.append("long_duration")
    if inspection.rms_dbfs is not None:
        if inspection.rms_dbfs <= VERY_LOW_LOUDNESS_DBFS:
            flags.append("very_low_loudness")
        elif inspection.rms_dbfs <= LOW_LOUDNESS_DBFS:
            flags.append("low_loudness")
    if inspection.silence_ratio is not None:
        if inspection.silence_ratio >= HIGH_SILENCE_RATIO:
            flags.append("high_silence_ratio")
        elif inspection.silence_ratio >= MODERATE_SILENCE_RATIO:
            flags.append("moderate_silence_ratio")
    if (
        inspection.dc_offset_ratio is not None
        and inspection.dc_offset_ratio >= HIGH_DC_OFFSET_RATIO
    ):
        flags.append("dc_offset_risk")
    if inspection.peak_dbfs is not None and inspection.peak_dbfs >= CLIPPING_PEAK_DBFS:
        flags.append("clipping_risk")
    return tuple(flags)


def _build_patterns(summary: QualitySummary) -> list[AudioQualityPattern]:
    patterns: list[AudioQualityPattern] = []
    if summary.entry_count == 0:
        return patterns

    if set(summary.sample_rate_counts) != {str(TARGET_SAMPLE_RATE_HZ)}:
        patterns.append(
            AudioQualityPattern(
                code="mixed_sample_rates",
                summary=(
                    "The active manifests span multiple sample rates "
                    f"({_format_counts(summary.sample_rate_counts, limit=6)})."
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
                    f"Some rows are not mono ({_format_counts(summary.channel_counts, limit=6)})."
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
                    f"windows; silence ratio p95 is {_format_ratio(summary.silence_summary.p95)}."
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
                    f"mean loudness is {_format_dbfs(summary.loudness_summary.mean)}."
                ),
                implication=(
                    "The preprocessing stack should define loudness normalization or gain limits "
                    "before robust training."
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
                    f"{_format_dbfs(summary.peak_summary.maximum)}."
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
                    f"median {_format_seconds(summary.duration_summary.median)}, "
                    f"p95 {_format_seconds(summary.duration_summary.p95)}, max "
                    f"{_format_seconds(summary.duration_summary.maximum)}."
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


def _select_examples(records: list[ManifestQualityRecord]) -> list[FlaggedExample]:
    flagged_records = [record for record in records if record.quality_flags]
    flagged_records.sort(
        key=lambda record: (-_example_score(record), record.audio_path or record.identity_key)
    )
    return [
        FlaggedExample(
            audio_path=record.audio_path or record.identity_key,
            split_name=record.split_name,
            dataset_name=record.dataset_name,
            source_label=record.source_label,
            condition_label=record.condition_label,
            duration_seconds=record.duration_seconds,
            rms_dbfs=record.rms_dbfs,
            peak_dbfs=record.peak_dbfs,
            silence_ratio=record.silence_ratio,
            flags=record.quality_flags,
        )
        for record in flagged_records[:MAX_EXAMPLES]
    ]


def _example_score(record: ManifestQualityRecord) -> int:
    priority = {
        "missing_audio_file": 100,
        "audio_read_error": 90,
        "clipping_risk": 80,
        "high_silence_ratio": 75,
        "very_low_loudness": 70,
        "non_16k_sample_rate": 60,
        "non_mono_audio": 55,
        "dc_offset_risk": 50,
        "moderate_silence_ratio": 40,
        "low_loudness": 35,
        "long_duration": 20,
        "short_duration": 15,
        "unsupported_signal_analysis": 10,
        "missing_audio_path": 5,
    }
    return sum(priority.get(flag, 1) for flag in record.quality_flags)


def _build_warnings(report: DatasetAudioQualityReport) -> list[str]:
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
    unresolved_waveform_count = (
        report.total_summary.resolved_audio_count - report.total_summary.waveform_metrics_count
    )
    if unresolved_waveform_count > 0:
        warnings.append(
            f"{unresolved_waveform_count} resolved audio files do not expose waveform-derived "
            "quality metrics (for example, non-WAV inputs)."
        )
    if report.invalid_line_count:
        warnings.append(f"{report.invalid_line_count} invalid JSONL lines were skipped.")
    if report.duplicate_entry_count:
        warnings.append(
            f"{report.duplicate_entry_count} overlapping rows were deduplicated across manifests."
        )
    return warnings


def _deduplicate_records(
    records: list[ManifestQualityRecord],
) -> dict[str, ManifestQualityRecord]:
    unique_records: dict[str, ManifestQualityRecord] = {}
    for record in records:
        existing = unique_records.get(record.identity_key)
        if existing is None or _record_score(record) > _record_score(existing):
            unique_records[record.identity_key] = record
    return unique_records


def _record_score(record: ManifestQualityRecord) -> int:
    score = 0
    if record.audio_exists:
        score += 20
    if record.rms_dbfs is not None:
        score += 10
    if record.duration_seconds is not None:
        score += 6
    if record.sample_rate_hz is not None:
        score += 4
    if record.channels is not None:
        score += 2
    if record.speaker_id is not None:
        score += 1
    if record.session_key is not None:
        score += 1
    if record.split_name != "unknown":
        score += 1
    return score


def _group_records(
    records: Any,
    *,
    key: Any,
) -> list[tuple[str, list[ManifestQualityRecord]]]:
    grouped: dict[str, list[ManifestQualityRecord]] = {}
    for record in records:
        name = key(record)
        grouped.setdefault(name, []).append(record)

    if grouped and set(grouped).issubset(set(KNOWN_DATA_SPLITS) | {"unknown"}):
        ordering = {name: index for index, name in enumerate((*KNOWN_DATA_SPLITS, "unknown"))}
        return sorted(grouped.items(), key=lambda item: ordering.get(item[0], len(ordering)))
    return sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))


def _summarize_records(records: list[ManifestQualityRecord]) -> QualitySummary:
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    audio_format_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    speakers: set[str] = set()
    sessions: set[str] = set()
    durations: list[float] = []
    loudness_values: list[float] = []
    peak_values: list[float] = []
    silence_values: list[float] = []
    dc_offset_values: list[float] = []
    clipped_values: list[float] = []

    entries_with_audio_path = 0
    resolved_audio_count = 0
    waveform_metrics_count = 0
    missing_audio_path_count = 0
    missing_audio_file_count = 0
    audio_inspection_error_count = 0

    for record in records:
        split_counts[record.split_name] += 1
        dataset_counts[record.dataset_name] += 1

        if record.role is not None:
            role_counts[record.role] += 1
        if record.source_label is not None:
            source_counts[record.source_label] += 1
        if record.condition_label is not None:
            condition_counts[record.condition_label] += 1
        if record.sample_rate_hz is not None:
            sample_rate_counts[str(record.sample_rate_hz)] += 1
        if record.channels is not None:
            channel_counts[str(record.channels)] += 1
        if record.audio_format is not None:
            audio_format_counts[record.audio_format] += 1
        if record.speaker_id is not None:
            speakers.add(record.speaker_id)
        if record.session_key is not None:
            sessions.add(record.session_key)
        if record.duration_seconds is not None:
            durations.append(record.duration_seconds)
        if record.rms_dbfs is not None:
            loudness_values.append(record.rms_dbfs)
            waveform_metrics_count += 1
        if record.peak_dbfs is not None:
            peak_values.append(record.peak_dbfs)
        if record.silence_ratio is not None:
            silence_values.append(record.silence_ratio)
        if record.dc_offset_ratio is not None:
            dc_offset_values.append(record.dc_offset_ratio)
        if record.clipped_chunk_ratio is not None:
            clipped_values.append(record.clipped_chunk_ratio)

        if record.audio_path is None:
            missing_audio_path_count += 1
        else:
            entries_with_audio_path += 1
            if record.audio_exists:
                resolved_audio_count += 1
            else:
                missing_audio_file_count += 1
        if record.audio_error is not None:
            audio_inspection_error_count += 1
        for flag in record.quality_flags:
            flag_counts[flag] += 1

    return QualitySummary(
        entry_count=len(records),
        entries_with_audio_path=entries_with_audio_path,
        resolved_audio_count=resolved_audio_count,
        waveform_metrics_count=waveform_metrics_count,
        missing_audio_path_count=missing_audio_path_count,
        missing_audio_file_count=missing_audio_file_count,
        audio_inspection_error_count=audio_inspection_error_count,
        unique_speakers=len(speakers),
        unique_sessions=len(sessions),
        duration_summary=NumericDistribution.from_values(durations),
        loudness_summary=NumericDistribution.from_values(loudness_values),
        peak_summary=NumericDistribution.from_values(peak_values),
        silence_summary=NumericDistribution.from_values(silence_values),
        dc_offset_summary=NumericDistribution.from_values(dc_offset_values),
        clipped_chunk_summary=NumericDistribution.from_values(clipped_values),
        duration_histogram=_build_histogram(durations, DURATION_BUCKETS),
        loudness_histogram=_build_histogram(loudness_values, LOUDNESS_BUCKETS),
        silence_histogram=_build_histogram(silence_values, SILENCE_BUCKETS),
        flag_counts=_sorted_counts(flag_counts),
        split_counts=_sorted_counts(split_counts),
        role_counts=_sorted_counts(role_counts),
        dataset_counts=_sorted_counts(dataset_counts),
        source_counts=_sorted_counts(source_counts),
        condition_counts=_sorted_counts(condition_counts),
        sample_rate_counts=_sorted_counts(sample_rate_counts),
        channel_counts=_sorted_counts(channel_counts),
        audio_format_counts=_sorted_counts(audio_format_counts),
    )


def _build_histogram(
    values: list[float],
    buckets: tuple[tuple[str, float | None, float | None], ...],
) -> list[HistogramBucket]:
    bucket_counts = [0 for _ in buckets]
    for value in values:
        for index, (_, lower, upper) in enumerate(buckets):
            lower_ok = lower is None or value >= lower
            upper_ok = upper is None or value < upper
            if lower_ok and upper_ok:
                bucket_counts[index] += 1
                break
    return [
        HistogramBucket(label=label, count=count)
        for (label, _, _), count in zip(buckets, bucket_counts, strict=True)
    ]


def _primary_dataset_name(records: list[ManifestQualityRecord], manifest_path: Path) -> str:
    dataset_counts = Counter(record.dataset_name for record in records if record.dataset_name)
    if dataset_counts:
        return next(iter(_sorted_counts(dataset_counts)), manifest_path.parent.name)
    return manifest_path.parent.name


def _inspect_audio_file(
    path: Path,
    cache: dict[Path, AudioQualityInspection],
) -> AudioQualityInspection:
    cached = cache.get(path)
    if cached is not None:
        return cached

    audio_format = _audio_format_from_path(path.as_posix())
    if not path.exists():
        inspection = AudioQualityInspection(
            exists=False,
            file_size_bytes=None,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
            sample_width_bytes=None,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
        )
        cache[path] = inspection
        return inspection

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        inspection = AudioQualityInspection(
            exists=True,
            file_size_bytes=None,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
            sample_width_bytes=None,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    if path.suffix.lower() != ".wav":
        inspection = AudioQualityInspection(
            exists=True,
            file_size_bytes=size_bytes,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
            sample_width_bytes=None,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
        )
        cache[path] = inspection
        return inspection

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate_hz = handle.getframerate()
            channels = handle.getnchannels()
            sample_width_bytes = handle.getsampwidth()
            frame_count = handle.getnframes()

            chunk_frames = max(1, round(sample_rate_hz * SILENCE_CHUNK_MS / 1000))
            peak_max = 0
            total_samples = 0
            energy_sum = 0.0
            signed_sum = 0.0
            chunk_count = 0
            silent_chunks = 0
            clipped_chunks = 0
            max_possible_amplitude = _max_possible_amplitude(sample_width_bytes)

            while True:
                frames = handle.readframes(chunk_frames)
                if not frames:
                    break
                if sample_width_bytes == 1:
                    frames = audioop.bias(frames, sample_width_bytes, -128)
                sample_count = len(frames) // sample_width_bytes
                if sample_count == 0:
                    continue

                chunk_count += 1
                rms = audioop.rms(frames, sample_width_bytes)
                peak = audioop.max(frames, sample_width_bytes)
                average = audioop.avg(frames, sample_width_bytes)

                peak_max = max(peak_max, peak)
                total_samples += sample_count
                energy_sum += float(rms * rms * sample_count)
                signed_sum += float(average * sample_count)

                chunk_rms_dbfs = _amplitude_to_dbfs(rms, max_possible_amplitude)
                if rms == 0 or (
                    chunk_rms_dbfs is not None and chunk_rms_dbfs <= SILENCE_THRESHOLD_DBFS
                ):
                    silent_chunks += 1
                if peak >= max_possible_amplitude - 1:
                    clipped_chunks += 1
    except (OSError, wave.Error, audioop.error) as exc:
        inspection = AudioQualityInspection(
            exists=True,
            file_size_bytes=size_bytes,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
            sample_width_bytes=None,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    duration_seconds = frame_count / sample_rate_hz if sample_rate_hz > 0 else None
    if total_samples > 0:
        rms_amplitude = math.sqrt(energy_sum / total_samples)
        dc_offset_ratio = abs(signed_sum / total_samples) / max_possible_amplitude
        rms_dbfs = _amplitude_to_dbfs(rms_amplitude, max_possible_amplitude)
        peak_dbfs = _amplitude_to_dbfs(float(peak_max), max_possible_amplitude)
    else:
        dc_offset_ratio = None
        rms_dbfs = None
        peak_dbfs = None

    silence_ratio = silent_chunks / chunk_count if chunk_count > 0 else None
    clipped_chunk_ratio = clipped_chunks / chunk_count if chunk_count > 0 else None
    inspection = AudioQualityInspection(
        exists=True,
        file_size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        audio_format=audio_format,
        sample_width_bytes=sample_width_bytes,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        silence_ratio=silence_ratio,
        dc_offset_ratio=dc_offset_ratio,
        clipped_chunk_ratio=clipped_chunk_ratio,
    )
    cache[path] = inspection
    return inspection


def _load_jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], int]:
    objects: list[dict[str, Any]] = []
    invalid_line_count = 0
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            invalid_line_count += 1
            continue
        if not isinstance(payload, dict):
            invalid_line_count += 1
            continue
        objects.append(payload)
    return objects, invalid_line_count


def _infer_dataset_name(
    *,
    entry: dict[str, Any],
    audio_path: str | None,
    manifest_path: Path,
    manifests_root: Path,
) -> str:
    for field_name in ("dataset", "source_dataset"):
        value = _coerce_str(entry.get(field_name))
        if value:
            return value

    if audio_path:
        parts = PurePosixPath(audio_path).parts
        if len(parts) >= 2 and parts[0] == "datasets":
            return parts[1]

    if manifest_path.parent != manifests_root:
        return manifest_path.parent.name

    stem = manifest_path.stem
    for suffix in ("_manifest", "-manifest"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem


def _infer_split_name(*, entry: dict[str, Any], manifest_path: Path) -> str:
    split = _coerce_str(entry.get("split"))
    if split:
        return split
    if _coerce_str(entry.get("role")) is not None:
        return "demo"

    stem = manifest_path.stem.lower()
    for candidate in ("train", "dev", "demo"):
        if candidate in stem:
            return candidate
    return "unknown"


def _infer_session_key(*, entry: dict[str, Any], speaker_id: str | None) -> str | None:
    session_value = _coerce_str(entry.get("session_id")) or _coerce_str(entry.get("session_index"))
    if session_value is None:
        return None
    if speaker_id:
        return f"{speaker_id}:{session_value}"
    return session_value


def _infer_source_label(*, entry: dict[str, Any], dataset_name: str) -> str | None:
    for field_name in ("source_prefix", "source", "domain", "device"):
        value = _coerce_str(entry.get(field_name))
        if value:
            return value
    return dataset_name or None


def _infer_condition_label(entry: dict[str, Any]) -> str | None:
    for field_name in ("capture_condition", "condition", "environment"):
        value = _coerce_str(entry.get(field_name))
        if value:
            return value
    return None


def _build_identity_key(
    *,
    audio_path: str | None,
    entry: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> str:
    if audio_path is not None:
        return f"audio:{audio_path}"
    for field_name in ("demo_subset_path", "utterance_id", "recording_id", "id"):
        value = _coerce_str(entry.get(field_name))
        if value is not None:
            return f"{field_name}:{value}"
    return f"manifest:{manifest_path.as_posix()}:{line_number}"


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter, key=lambda item: (-counter[item], item))}


def _format_counts(counts: dict[str, int], *, limit: int | None = None) -> str:
    if not counts:
        return "-"
    items = list(counts.items())
    if limit is not None:
        visible_items = items[:limit]
        remaining_count = len(items) - len(visible_items)
    else:
        visible_items = items
        remaining_count = 0
    rendered = ", ".join(f"{name}={count}" for name, count in visible_items)
    if remaining_count > 0:
        rendered = f"{rendered}, +{remaining_count} more"
    return rendered


def _format_duration(total_seconds: float) -> str:
    return f"{total_seconds:.2f} s"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} s"


def _format_dbfs(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} dBFS"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def _render_histogram(histogram: list[HistogramBucket]) -> str:
    counts = {bucket.label: bucket.count for bucket in histogram if bucket.count > 0}
    if not counts:
        return "_No data available._"
    return _render_text_chart(counts)


def _render_text_chart(counts: dict[str, int]) -> str:
    if not counts:
        return "_No data available._"

    max_count = max(counts.values())
    lines = ["```text"]
    for label, count in counts.items():
        bar_width = 0 if max_count == 0 else max(1, round((count / max_count) * 28))
        lines.append(f"{label:<12} {'#' * bar_width:<28} {count}")
    lines.append("```")
    return "\n".join(lines)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _audio_format_from_path(path: str | None) -> str | None:
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    return suffix.removeprefix(".") or None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Percentile requires at least one value.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _amplitude_to_dbfs(value: float, max_possible_amplitude: int) -> float | None:
    if value <= 0.0 or max_possible_amplitude <= 0:
        return None
    return 20.0 * math.log10(value / max_possible_amplitude)


def _max_possible_amplitude(sample_width_bytes: int) -> int:
    return (1 << ((sample_width_bytes * 8) - 1)) - 1


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
