"""Datamodels for audio-quality EDA reports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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
