"""Manifest traversal and summarization for audio-quality EDA."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kryptonite.deployment import resolve_project_path

from .constants import (
    CLIPPING_PEAK_DBFS,
    DURATION_BUCKETS,
    HIGH_DC_OFFSET_RATIO,
    HIGH_SILENCE_RATIO,
    KNOWN_DATA_SPLITS,
    LONG_DURATION_SECONDS,
    LOUDNESS_BUCKETS,
    LOW_LOUDNESS_DBFS,
    MODERATE_SILENCE_RATIO,
    SHORT_DURATION_SECONDS,
    SILENCE_BUCKETS,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE_HZ,
    VERY_LOW_LOUDNESS_DBFS,
)
from .inspection import audio_format_from_path, inspect_audio_file
from .models import (
    AudioQualityInspection,
    HistogramBucket,
    IgnoredManifest,
    ManifestQualityProfile,
    ManifestQualityRecord,
    NumericDistribution,
    QualitySummary,
)


@dataclass(slots=True)
class CollectedManifestInputs:
    manifest_profiles: list[ManifestQualityProfile]
    ignored_manifests: list[IgnoredManifest]
    all_records: list[ManifestQualityRecord]
    invalid_line_count: int


def collect_manifest_inputs(
    *,
    project_root: Path,
    manifests_root: Path,
) -> CollectedManifestInputs:
    manifest_profiles: list[ManifestQualityProfile] = []
    ignored_manifests: list[IgnoredManifest] = []
    all_records: list[ManifestQualityRecord] = []
    audio_cache: dict[Path, AudioQualityInspection] = {}
    invalid_line_count = 0

    if manifests_root.exists():
        for manifest_path in sorted(manifests_root.rglob("*.jsonl")):
            relative_manifest_path = relative_to_project(manifest_path, project_root)
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

            objects, manifest_invalid_line_count = load_jsonl_objects(manifest_path)
            invalid_line_count += manifest_invalid_line_count
            if "manifest" not in manifest_name and not any("audio_path" in row for row in objects):
                ignored_manifests.append(
                    IgnoredManifest(
                        manifest_path=relative_manifest_path,
                        reason="JSONL does not look like a data manifest",
                    )
                )
                continue

            manifest_records = [
                build_quality_record(
                    manifest_path=manifest_path,
                    entry=row,
                    line_number=index,
                    project_root=project_root,
                    manifests_root=manifests_root,
                    audio_cache=audio_cache,
                )
                for index, row in enumerate(objects, start=1)
            ]
            manifest_profiles.append(
                ManifestQualityProfile(
                    manifest_path=relative_manifest_path,
                    primary_dataset=primary_dataset_name(manifest_records, manifest_path),
                    invalid_line_count=manifest_invalid_line_count,
                    summary=summarize_records(manifest_records),
                )
            )
            all_records.extend(manifest_records)

    return CollectedManifestInputs(
        manifest_profiles=manifest_profiles,
        ignored_manifests=ignored_manifests,
        all_records=all_records,
        invalid_line_count=invalid_line_count,
    )


def build_quality_record(
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    line_number: int,
    project_root: Path,
    manifests_root: Path,
    audio_cache: dict[Path, AudioQualityInspection],
) -> ManifestQualityRecord:
    audio_path = coerce_str(entry.get("audio_path"))
    resolved_audio_path = (
        resolve_project_path(str(project_root), audio_path) if audio_path is not None else None
    )
    inspection = (
        inspect_audio_file(resolved_audio_path, audio_cache)
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

    manifest_duration = coerce_float(entry.get("duration_seconds"))
    duration_seconds = inspection.duration_seconds or manifest_duration
    sample_rate_hz = inspection.sample_rate_hz or coerce_int(entry.get("sample_rate_hz"))
    channels = inspection.channels or coerce_int(entry.get("channels"))
    dataset_name = infer_dataset_name(
        entry=entry,
        audio_path=audio_path,
        manifest_path=manifest_path,
        manifests_root=manifests_root,
    )
    split_name = infer_split_name(entry=entry, manifest_path=manifest_path)
    speaker_id = coerce_str(entry.get("speaker_id"))
    session_key = infer_session_key(entry=entry, speaker_id=speaker_id)
    role = coerce_str(entry.get("role"))
    source_label = infer_source_label(entry=entry, dataset_name=dataset_name)
    condition_label = infer_condition_label(entry)
    audio_format = inspection.audio_format or audio_format_from_path(audio_path)
    quality_flags = build_quality_flags(
        audio_path=audio_path,
        inspection=inspection,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_seconds=duration_seconds,
    )
    identity_key = build_identity_key(
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


def build_quality_flags(
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


def deduplicate_records(
    records: list[ManifestQualityRecord],
) -> dict[str, ManifestQualityRecord]:
    unique_records: dict[str, ManifestQualityRecord] = {}
    for record in records:
        existing = unique_records.get(record.identity_key)
        if existing is None or record_score(record) > record_score(existing):
            unique_records[record.identity_key] = record
    return unique_records


def record_score(record: ManifestQualityRecord) -> int:
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


def group_records(
    records: Iterable[ManifestQualityRecord],
    *,
    key: Callable[[ManifestQualityRecord], str],
) -> list[tuple[str, list[ManifestQualityRecord]]]:
    grouped: dict[str, list[ManifestQualityRecord]] = {}
    for record in records:
        name = key(record)
        grouped.setdefault(name, []).append(record)

    if grouped and set(grouped).issubset(set(KNOWN_DATA_SPLITS) | {"unknown"}):
        ordering = {name: index for index, name in enumerate((*KNOWN_DATA_SPLITS, "unknown"))}
        return sorted(grouped.items(), key=lambda item: ordering.get(item[0], len(ordering)))
    return sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))


def summarize_records(records: list[ManifestQualityRecord]) -> QualitySummary:
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
        duration_histogram=build_histogram(durations, DURATION_BUCKETS),
        loudness_histogram=build_histogram(loudness_values, LOUDNESS_BUCKETS),
        silence_histogram=build_histogram(silence_values, SILENCE_BUCKETS),
        flag_counts=sorted_counts(flag_counts),
        split_counts=sorted_counts(split_counts),
        role_counts=sorted_counts(role_counts),
        dataset_counts=sorted_counts(dataset_counts),
        source_counts=sorted_counts(source_counts),
        condition_counts=sorted_counts(condition_counts),
        sample_rate_counts=sorted_counts(sample_rate_counts),
        channel_counts=sorted_counts(channel_counts),
        audio_format_counts=sorted_counts(audio_format_counts),
    )


def build_histogram(
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


def primary_dataset_name(records: list[ManifestQualityRecord], manifest_path: Path) -> str:
    dataset_counts = Counter(record.dataset_name for record in records if record.dataset_name)
    if dataset_counts:
        return next(iter(sorted_counts(dataset_counts)), manifest_path.parent.name)
    return manifest_path.parent.name


def load_jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], int]:
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


def infer_dataset_name(
    *,
    entry: dict[str, Any],
    audio_path: str | None,
    manifest_path: Path,
    manifests_root: Path,
) -> str:
    for field_name in ("dataset", "source_dataset"):
        value = coerce_str(entry.get(field_name))
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


def infer_split_name(*, entry: dict[str, Any], manifest_path: Path) -> str:
    split = coerce_str(entry.get("split"))
    if split:
        return split
    if coerce_str(entry.get("role")) is not None:
        return "demo"

    stem = manifest_path.stem.lower()
    for candidate in ("train", "dev", "demo"):
        if candidate in stem:
            return candidate
    return "unknown"


def infer_session_key(*, entry: dict[str, Any], speaker_id: str | None) -> str | None:
    session_value = coerce_str(entry.get("session_id")) or coerce_str(entry.get("session_index"))
    if session_value is None:
        return None
    if speaker_id:
        return f"{speaker_id}:{session_value}"
    return session_value


def infer_source_label(*, entry: dict[str, Any], dataset_name: str) -> str | None:
    for field_name in ("source_prefix", "source", "domain", "device"):
        value = coerce_str(entry.get(field_name))
        if value:
            return value
    return dataset_name or None


def infer_condition_label(entry: dict[str, Any]) -> str | None:
    for field_name in ("capture_condition", "condition", "environment"):
        value = coerce_str(entry.get(field_name))
        if value:
            return value
    return None


def build_identity_key(
    *,
    audio_path: str | None,
    entry: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> str:
    if audio_path is not None:
        return f"audio:{audio_path}"
    for field_name in ("demo_subset_path", "utterance_id", "recording_id", "id"):
        value = coerce_str(entry.get(field_name))
        if value is not None:
            return f"{field_name}:{value}"
    return f"manifest:{manifest_path.as_posix()}:{line_number}"


def sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter, key=lambda item: (-counter[item], item))}


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def coerce_float(value: object) -> float | None:
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


def coerce_int(value: object) -> int | None:
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
