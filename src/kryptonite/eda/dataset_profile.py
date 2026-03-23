"""Reproducible dataset profiling for manifests-backed audio corpora."""

from __future__ import annotations

import json
import statistics
import wave
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kryptonite.data.schema import normalize_manifest_entry
from kryptonite.deployment import resolve_project_path

DURATION_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-5s", 2.0, 5.0),
    ("5-10s", 5.0, 10.0),
    ("10-30s", 10.0, 30.0),
    ("30-60s", 30.0, 60.0),
    ("60s+", 60.0, None),
)
KNOWN_DATA_SPLITS: tuple[str, ...] = ("train", "dev", "demo")


@dataclass(frozen=True, slots=True)
class NumericSummary:
    count: int
    total: float
    minimum: float | None
    mean: float | None
    median: float | None
    maximum: float | None

    @classmethod
    def from_values(cls, values: list[float]) -> NumericSummary:
        if not values:
            return cls(
                count=0,
                total=0.0,
                minimum=None,
                mean=None,
                median=None,
                maximum=None,
            )

        ordered = sorted(values)
        return cls(
            count=len(ordered),
            total=float(sum(ordered)),
            minimum=float(ordered[0]),
            mean=float(statistics.fmean(ordered)),
            median=float(statistics.median(ordered)),
            maximum=float(ordered[-1]),
        )

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total": self.total,
            "minimum": self.minimum,
            "mean": self.mean,
            "median": self.median,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class HistogramBucket:
    label: str
    count: int

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "count": self.count}


@dataclass(frozen=True, slots=True)
class AudioInspection:
    exists: bool
    file_size_bytes: int | None
    duration_seconds: float | None
    sample_rate_hz: int | None
    channels: int | None
    audio_format: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestRecord:
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
    sample_rate_hz: int | None
    channels: int | None
    audio_format: str | None
    duration_seconds: float | None
    duration_source: str
    file_size_bytes: int | None


@dataclass(slots=True)
class ProfileSummary:
    entry_count: int
    entries_with_audio_path: int
    resolved_audio_count: int
    missing_audio_path_count: int
    missing_audio_file_count: int
    audio_inspection_error_count: int
    unique_speakers: int
    unique_sessions: int
    duration_summary: NumericSummary
    file_size_summary: NumericSummary
    duration_histogram: list[HistogramBucket] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    role_counts: dict[str, int] = field(default_factory=dict)
    dataset_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    sample_rate_counts: dict[str, int] = field(default_factory=dict)
    channel_counts: dict[str, int] = field(default_factory=dict)
    audio_format_counts: dict[str, int] = field(default_factory=dict)
    duration_source_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_count": self.entry_count,
            "entries_with_audio_path": self.entries_with_audio_path,
            "resolved_audio_count": self.resolved_audio_count,
            "missing_audio_path_count": self.missing_audio_path_count,
            "missing_audio_file_count": self.missing_audio_file_count,
            "audio_inspection_error_count": self.audio_inspection_error_count,
            "unique_speakers": self.unique_speakers,
            "unique_sessions": self.unique_sessions,
            "duration_summary": self.duration_summary.to_dict(),
            "file_size_summary": self.file_size_summary.to_dict(),
            "duration_histogram": [bucket.to_dict() for bucket in self.duration_histogram],
            "split_counts": dict(self.split_counts),
            "role_counts": dict(self.role_counts),
            "dataset_counts": dict(self.dataset_counts),
            "source_counts": dict(self.source_counts),
            "sample_rate_counts": dict(self.sample_rate_counts),
            "channel_counts": dict(self.channel_counts),
            "audio_format_counts": dict(self.audio_format_counts),
            "duration_source_counts": dict(self.duration_source_counts),
        }


@dataclass(slots=True)
class NamedSummary:
    name: str
    summary: ProfileSummary

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "summary": self.summary.to_dict()}


@dataclass(slots=True)
class ManifestProfile:
    manifest_path: str
    primary_dataset: str
    invalid_line_count: int
    summary: ProfileSummary

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


@dataclass(slots=True)
class DatasetProfileReport:
    generated_at: str
    project_root: str
    manifests_root: str
    raw_entry_count: int
    duplicate_entry_count: int
    invalid_line_count: int
    total_summary: ProfileSummary
    split_summaries: list[NamedSummary]
    dataset_summaries: list[NamedSummary]
    manifest_profiles: list[ManifestProfile]
    ignored_manifests: list[IgnoredManifest]
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
            "manifest_profiles": [profile.to_dict() for profile in self.manifest_profiles],
            "ignored_manifests": [manifest.to_dict() for manifest in self.ignored_manifests],
        }


@dataclass(frozen=True, slots=True)
class WrittenDatasetProfile:
    output_root: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def build_dataset_profile_report(
    *,
    project_root: Path | str,
    manifests_root: Path | str,
) -> DatasetProfileReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifests_root_path = resolve_project_path(str(project_root_path), str(manifests_root))
    generated_at = _utc_now()

    manifest_profiles: list[ManifestProfile] = []
    ignored_manifests: list[IgnoredManifest] = []
    all_records: list[ManifestRecord] = []
    audio_cache: dict[Path, AudioInspection] = {}

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
                _build_manifest_record(
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
                ManifestProfile(
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
        NamedSummary(name=split_name, summary=_summarize_records(records))
        for split_name, records in _group_records(
            unique_records.values(),
            key=lambda record: record.split_name,
        )
        if split_name
    ]
    dataset_summaries = [
        NamedSummary(name=dataset_name, summary=_summarize_records(records))
        for dataset_name, records in _group_records(
            unique_records.values(), key=lambda record: record.dataset_name
        )
        if dataset_name
    ]

    report = DatasetProfileReport(
        generated_at=generated_at,
        project_root=str(project_root_path),
        manifests_root=str(manifests_root_path),
        raw_entry_count=len(all_records),
        duplicate_entry_count=max(0, len(all_records) - len(unique_records)),
        invalid_line_count=sum(profile.invalid_line_count for profile in manifest_profiles),
        total_summary=total_summary,
        split_summaries=split_summaries,
        dataset_summaries=dataset_summaries,
        manifest_profiles=manifest_profiles,
        ignored_manifests=ignored_manifests,
        warnings=[],
    )
    report.warnings = _build_warnings(report)
    return report


def render_dataset_profile_markdown(report: DatasetProfileReport) -> str:
    lines = [
        "# Dataset Profile Report",
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
                    ["Unique speakers", str(report.total_summary.unique_speakers)],
                    ["Unique sessions", str(report.total_summary.unique_sessions)],
                    [
                        "Total duration",
                        _format_duration(report.total_summary.duration_summary.total),
                    ],
                    ["Datasets", _format_counts(report.total_summary.dataset_counts)],
                ],
            ),
            "",
            "## Split Coverage",
            "",
            _markdown_table(
                [
                    "Split",
                    "Rows",
                    "Speakers",
                    "Sessions",
                    "Total duration",
                    "Mean duration (s)",
                    "Missing audio",
                ],
                [
                    [
                        summary.name,
                        str(summary.summary.entry_count),
                        str(summary.summary.unique_speakers),
                        str(summary.summary.unique_sessions),
                        _format_duration(summary.summary.duration_summary.total),
                        _format_float(summary.summary.duration_summary.mean),
                        str(summary.summary.missing_audio_file_count),
                    ]
                    for summary in report.split_summaries
                ]
                or [["-", "0", "0", "0", "0.00 s", "-", "0"]],
            ),
            "",
            "## Dataset Coverage",
            "",
            _markdown_table(
                [
                    "Dataset",
                    "Rows",
                    "Speakers",
                    "Sessions",
                    "Split mix",
                    "Sample rates",
                    "Formats",
                ],
                [
                    [
                        summary.name,
                        str(summary.summary.entry_count),
                        str(summary.summary.unique_speakers),
                        str(summary.summary.unique_sessions),
                        _format_counts(summary.summary.split_counts),
                        _format_counts(summary.summary.sample_rate_counts),
                        _format_counts(summary.summary.audio_format_counts),
                    ]
                    for summary in report.dataset_summaries
                ]
                or [["-", "0", "0", "0", "-", "-", "-"]],
            ),
            "",
            "## Manifest Summaries",
            "",
            _markdown_table(
                [
                    "Manifest",
                    "Dataset",
                    "Rows",
                    "Split mix",
                    "Speakers",
                    "Total duration",
                    "Missing audio",
                    "Invalid lines",
                ],
                [
                    [
                        profile.manifest_path,
                        profile.primary_dataset,
                        str(profile.summary.entry_count),
                        _format_counts(profile.summary.split_counts),
                        str(profile.summary.unique_speakers),
                        _format_duration(profile.summary.duration_summary.total),
                        str(profile.summary.missing_audio_file_count),
                        str(profile.invalid_line_count),
                    ]
                    for profile in report.manifest_profiles
                ]
                or [["-", "-", "0", "-", "0", "0.00 s", "0", "0"]],
            ),
            "",
            "## Observed Distributions",
            "",
            _markdown_table(
                ["Category", "Counts"],
                [
                    ["Split", _format_counts(report.total_summary.split_counts)],
                    ["Role", _format_counts(report.total_summary.role_counts)],
                    ["Source", _format_counts(report.total_summary.source_counts)],
                    ["Sample rates", _format_counts(report.total_summary.sample_rate_counts)],
                    ["Channels", _format_counts(report.total_summary.channel_counts)],
                    ["Formats", _format_counts(report.total_summary.audio_format_counts)],
                    [
                        "Duration source",
                        _format_counts(report.total_summary.duration_source_counts),
                    ],
                ],
            ),
            "",
            "## Graphs",
            "",
            "### Rows By Split",
            "",
            _render_text_chart(
                report.total_summary.split_counts,
                order=list(KNOWN_DATA_SPLITS) + ["unknown"],
            ),
            "",
            "### Duration Histogram",
            "",
            _render_histogram(report.total_summary.duration_histogram),
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
                "- Aggregated split and dataset tables are deduplicated by canonical row "
                "identity so that `all/train/dev` manifest overlaps do not inflate totals."
            ),
            (
                "- WAV duration, sample rate, and channel counts come from waveform "
                "inspection when possible; otherwise the report falls back to manifest "
                "fields."
            ),
            (
                "- JSONL files whose name contains `trial` are ignored in the dataset "
                "profile because they describe evaluation pairs, not corpus rows."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_dataset_profile_report(
    *,
    report: DatasetProfileReport,
    output_root: Path | str,
) -> WrittenDatasetProfile:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "dataset_profile.json"
    markdown_path = output_root_path / "dataset_profile.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_dataset_profile_markdown(report))
    return WrittenDatasetProfile(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def _build_manifest_record(
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    line_number: int,
    project_root: Path,
    manifests_root: Path,
    audio_cache: dict[Path, AudioInspection],
) -> ManifestRecord:
    normalized_entry = normalize_manifest_entry(entry)
    audio_path = _coerce_str(normalized_entry.get("audio_path"))
    resolved_audio_path = (
        resolve_project_path(str(project_root), audio_path) if audio_path is not None else None
    )
    inspection = (
        _inspect_audio_file(resolved_audio_path, audio_cache)
        if resolved_audio_path is not None
        else AudioInspection(
            exists=False,
            file_size_bytes=None,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=None,
        )
    )

    manifest_duration = _coerce_float(normalized_entry.get("duration_seconds"))
    duration_seconds: float | None
    duration_source: str
    if inspection.duration_seconds is not None:
        duration_seconds = inspection.duration_seconds
        duration_source = "audio"
    elif manifest_duration is not None:
        duration_seconds = manifest_duration
        duration_source = "manifest"
    else:
        duration_seconds = None
        duration_source = "missing"

    sample_rate_hz = inspection.sample_rate_hz or _coerce_int(
        normalized_entry.get("sample_rate_hz")
    )
    channels = (
        inspection.channels
        or _coerce_int(normalized_entry.get("num_channels"))
        or _coerce_int(normalized_entry.get("channels"))
    )
    dataset_name = _infer_dataset_name(
        entry=normalized_entry,
        audio_path=audio_path,
        manifest_path=manifest_path,
        manifests_root=manifests_root,
    )
    split_name = _infer_split_name(entry=normalized_entry, manifest_path=manifest_path)
    speaker_id = _coerce_str(normalized_entry.get("speaker_id"))
    session_key = _infer_session_key(entry=normalized_entry, speaker_id=speaker_id)
    role = _coerce_str(normalized_entry.get("role"))
    source_label = _infer_source_label(entry=normalized_entry, dataset_name=dataset_name)
    audio_format = inspection.audio_format or _audio_format_from_path(audio_path)
    identity_key = _build_identity_key(
        audio_path=audio_path,
        entry=normalized_entry,
        manifest_path=manifest_path,
        line_number=line_number,
    )

    return ManifestRecord(
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
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        audio_format=audio_format,
        duration_seconds=duration_seconds,
        duration_source=duration_source,
        file_size_bytes=inspection.file_size_bytes,
    )


def _build_warnings(report: DatasetProfileReport) -> list[str]:
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


def _deduplicate_records(records: list[ManifestRecord]) -> dict[str, ManifestRecord]:
    unique_records: dict[str, ManifestRecord] = {}
    for record in records:
        existing = unique_records.get(record.identity_key)
        if existing is None or _record_score(record) > _record_score(existing):
            unique_records[record.identity_key] = record
    return unique_records


def _group_records(
    records: Any,
    *,
    key: Any,
) -> list[tuple[str, list[ManifestRecord]]]:
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        name = key(record)
        grouped.setdefault(name, []).append(record)

    if key is not None and grouped and set(grouped).issubset(set(KNOWN_DATA_SPLITS) | {"unknown"}):
        ordering = {name: index for index, name in enumerate((*KNOWN_DATA_SPLITS, "unknown"))}
        return sorted(grouped.items(), key=lambda item: ordering.get(item[0], len(ordering)))
    return sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))


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
    session_id = _coerce_str(entry.get("session_id"))
    if session_id is not None:
        if speaker_id and ":" not in session_id:
            return f"{speaker_id}:{session_id}"
        return session_id

    session_index = _coerce_str(entry.get("session_index"))
    if session_index is None:
        return None
    if speaker_id:
        return f"{speaker_id}:{session_index}"
    return session_index


def _infer_source_label(*, entry: dict[str, Any], dataset_name: str) -> str | None:
    for field_name in ("source_prefix", "source", "domain", "device"):
        value = _coerce_str(entry.get(field_name))
        if value:
            return value
    return dataset_name or None


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


def _summarize_records(records: list[ManifestRecord]) -> ProfileSummary:
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    audio_format_counts: Counter[str] = Counter()
    duration_source_counts: Counter[str] = Counter()
    speakers: set[str] = set()
    sessions: set[str] = set()
    durations: list[float] = []
    file_sizes: list[float] = []

    entries_with_audio_path = 0
    resolved_audio_count = 0
    missing_audio_path_count = 0
    missing_audio_file_count = 0
    audio_inspection_error_count = 0

    for record in records:
        split_counts[record.split_name] += 1
        dataset_counts[record.dataset_name] += 1
        duration_source_counts[record.duration_source] += 1

        if record.role is not None:
            role_counts[record.role] += 1
        if record.source_label is not None:
            source_counts[record.source_label] += 1
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
        if record.file_size_bytes is not None:
            file_sizes.append(float(record.file_size_bytes))
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

    return ProfileSummary(
        entry_count=len(records),
        entries_with_audio_path=entries_with_audio_path,
        resolved_audio_count=resolved_audio_count,
        missing_audio_path_count=missing_audio_path_count,
        missing_audio_file_count=missing_audio_file_count,
        audio_inspection_error_count=audio_inspection_error_count,
        unique_speakers=len(speakers),
        unique_sessions=len(sessions),
        duration_summary=NumericSummary.from_values(durations),
        file_size_summary=NumericSummary.from_values(file_sizes),
        duration_histogram=_build_histogram(durations),
        split_counts=_sorted_counts(split_counts),
        role_counts=_sorted_counts(role_counts),
        dataset_counts=_sorted_counts(dataset_counts),
        source_counts=_sorted_counts(source_counts),
        sample_rate_counts=_sorted_counts(sample_rate_counts),
        channel_counts=_sorted_counts(channel_counts),
        audio_format_counts=_sorted_counts(audio_format_counts),
        duration_source_counts=_sorted_counts(duration_source_counts),
    )


def _primary_dataset_name(records: list[ManifestRecord], manifest_path: Path) -> str:
    dataset_counts = Counter(record.dataset_name for record in records if record.dataset_name)
    if dataset_counts:
        return next(iter(_sorted_counts(dataset_counts)), manifest_path.parent.name)
    return manifest_path.parent.name


def _inspect_audio_file(path: Path, cache: dict[Path, AudioInspection]) -> AudioInspection:
    cached = cache.get(path)
    if cached is not None:
        return cached

    audio_format = _audio_format_from_path(path.as_posix())
    if not path.exists():
        inspection = AudioInspection(
            exists=False,
            file_size_bytes=None,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
        )
        cache[path] = inspection
        return inspection

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        inspection = AudioInspection(
            exists=True,
            file_size_bytes=None,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    if path.suffix.lower() != ".wav":
        inspection = AudioInspection(
            exists=True,
            file_size_bytes=size_bytes,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
        )
        cache[path] = inspection
        return inspection

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate_hz = handle.getframerate()
            channels = handle.getnchannels()
            frame_count = handle.getnframes()
    except (OSError, wave.Error) as exc:
        inspection = AudioInspection(
            exists=True,
            file_size_bytes=size_bytes,
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            audio_format=audio_format,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    duration_seconds = frame_count / sample_rate_hz if sample_rate_hz > 0 else None
    inspection = AudioInspection(
        exists=True,
        file_size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        audio_format=audio_format,
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


def _record_score(record: ManifestRecord) -> int:
    score = 0
    if record.audio_exists:
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
    if record.role is not None:
        score += 1
    return score


def _build_histogram(values: list[float]) -> list[HistogramBucket]:
    buckets = [0 for _ in DURATION_BUCKETS]
    for value in values:
        for index, (_, lower, upper) in enumerate(DURATION_BUCKETS):
            if upper is None and value >= lower:
                buckets[index] += 1
                break
            if upper is not None and lower <= value < upper:
                buckets[index] += 1
                break
    return [
        HistogramBucket(label=label, count=count)
        for (label, _, _), count in zip(DURATION_BUCKETS, buckets, strict=True)
    ]


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter, key=lambda item: (-counter[item], item))}


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _format_duration(total_seconds: float) -> str:
    return f"{total_seconds:.2f} s"


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def _render_text_chart(counts: dict[str, int], *, order: list[str] | None = None) -> str:
    if not counts:
        return "_No rows available._"

    labels = order or list(counts)
    rows = [(label, counts[label]) for label in labels if label in counts]
    max_count = max(count for _, count in rows)
    lines = ["```text"]
    for label, count in rows:
        bar_width = 0 if max_count == 0 else max(1, round((count / max_count) * 28))
        lines.append(f"{label:<12} {'#' * bar_width:<28} {count}")
    lines.append("```")
    return "\n".join(lines)


def _render_histogram(histogram: list[HistogramBucket]) -> str:
    counts = {bucket.label: bucket.count for bucket in histogram if bucket.count > 0}
    if not counts:
        return "_No duration data available._"
    return _render_text_chart(counts)


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


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
