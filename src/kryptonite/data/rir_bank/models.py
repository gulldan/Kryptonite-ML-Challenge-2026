"""Datamodels for reproducible RIR-bank assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from kryptonite.data.normalization import AudioNormalizationPolicy

RIRDirectCondition = Literal["high", "medium", "low"]
RIRFamily = Literal["real", "simulated"]
RIRField = Literal["near", "mid", "far"]
RIRRoomSize = Literal["small", "medium", "large"]
RIRRT60Bucket = Literal["short", "medium", "long"]

ALLOWED_RIR_DIRECT_CONDITIONS: tuple[RIRDirectCondition, ...] = ("high", "medium", "low")
ALLOWED_RIR_FAMILIES: tuple[RIRFamily, ...] = ("real", "simulated")
ALLOWED_RIR_FIELDS: tuple[RIRField, ...] = ("near", "mid", "far")
ALLOWED_RIR_ROOM_SIZES: tuple[RIRRoomSize, ...] = ("small", "medium", "large")
ALLOWED_RIR_RT60_BUCKETS: tuple[RIRRT60Bucket, ...] = ("short", "medium", "long")
MANIFEST_JSONL_NAME = "rir_bank_manifest.jsonl"
QUARANTINE_JSONL_NAME = "rir_bank_quarantine.jsonl"
REPORT_JSON_NAME = "rir_bank_report.json"
REPORT_MARKDOWN_NAME = "rir_bank_report.md"
ROOM_CONFIG_JSONL_NAME = "room_simulation_configs.jsonl"
SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3"})


@dataclass(frozen=True, slots=True)
class NumericRange:
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError("NumericRange must define at least one bound.")
        if self.minimum is not None and self.maximum is not None and self.maximum <= self.minimum:
            raise ValueError("NumericRange maximum must be greater than minimum.")

    def contains(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value >= self.maximum:
            return False
        return True

    def to_dict(self) -> dict[str, float | None]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class RIRAnalysisSettings:
    direct_window_ms: float
    reverb_start_ms: float
    preview_duration_ms: float
    preview_bins: int
    rt60_buckets: dict[RIRRT60Bucket, NumericRange]
    field_buckets: dict[RIRField, NumericRange]
    direct_buckets: dict[RIRDirectCondition, NumericRange]

    def __post_init__(self) -> None:
        if self.direct_window_ms <= 0.0:
            raise ValueError("direct_window_ms must be positive.")
        if self.reverb_start_ms <= 0.0:
            raise ValueError("reverb_start_ms must be positive.")
        if self.reverb_start_ms <= self.direct_window_ms:
            raise ValueError("reverb_start_ms must be greater than direct_window_ms.")
        if self.preview_duration_ms <= 0.0:
            raise ValueError("preview_duration_ms must be positive.")
        if self.preview_bins <= 0:
            raise ValueError("preview_bins must be positive.")

    def bucket_rt60(self, seconds: float) -> RIRRT60Bucket:
        return _bucket_from_ranges(
            value=seconds,
            ranges=self.rt60_buckets,
            order=ALLOWED_RIR_RT60_BUCKETS,
            label="rt60",
        )

    def bucket_field(self, drr_db: float) -> RIRField:
        return _bucket_from_ranges(
            value=drr_db,
            ranges=self.field_buckets,
            order=ALLOWED_RIR_FIELDS,
            label="field",
        )

    def bucket_direct_condition(self, drr_db: float) -> RIRDirectCondition:
        return _bucket_from_ranges(
            value=drr_db,
            ranges=self.direct_buckets,
            order=ALLOWED_RIR_DIRECT_CONDITIONS,
            label="direct condition",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "direct_window_ms": self.direct_window_ms,
            "reverb_start_ms": self.reverb_start_ms,
            "preview_duration_ms": self.preview_duration_ms,
            "preview_bins": self.preview_bins,
            "rt60_buckets": {
                name: self.rt60_buckets[name].to_dict() for name in ALLOWED_RIR_RT60_BUCKETS
            },
            "field_buckets": {
                name: self.field_buckets[name].to_dict() for name in ALLOWED_RIR_FIELDS
            },
            "direct_buckets": {
                name: self.direct_buckets[name].to_dict() for name in ALLOWED_RIR_DIRECT_CONDITIONS
            },
        }


@dataclass(frozen=True, slots=True)
class RIRClassificationRule:
    match_any: tuple[str, ...]
    room_size: RIRRoomSize | None = None
    field: RIRField | None = None
    rt60_bucket: RIRRT60Bucket | None = None
    direct_condition: RIRDirectCondition | None = None
    tags: tuple[str, ...] = ()

    def matches(self, normalized_path: str) -> bool:
        return any(token in normalized_path for token in self.match_any)

    def to_dict(self) -> dict[str, object]:
        return {
            "match_any": list(self.match_any),
            "room_size": self.room_size,
            "field": self.field,
            "rt60_bucket": self.rt60_bucket,
            "direct_condition": self.direct_condition,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class RIRClassificationOverride:
    room_size: RIRRoomSize | None = None
    field: RIRField | None = None
    rt60_bucket: RIRRT60Bucket | None = None
    direct_condition: RIRDirectCondition | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "room_size": self.room_size,
            "field": self.field,
            "rt60_bucket": self.rt60_bucket,
            "direct_condition": self.direct_condition,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class RIRClassification:
    room_size: RIRRoomSize
    field: RIRField
    rt60_bucket: RIRRT60Bucket
    direct_condition: RIRDirectCondition
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "room_size": self.room_size,
            "field": self.field,
            "rt60_bucket": self.rt60_bucket,
            "direct_condition": self.direct_condition,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class RIRSourcePlan:
    id: str
    name: str
    inventory_source_id: str
    room_family: RIRFamily
    root_candidates: tuple[str, ...]
    default_room_size: RIRRoomSize
    base_weight: float = 1.0
    tags: tuple[str, ...] = ()
    classification_rules: tuple[RIRClassificationRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.root_candidates:
            raise ValueError(f"RIR source '{self.id}' must define at least one root candidate.")
        if self.base_weight <= 0.0:
            raise ValueError(f"RIR source '{self.id}' must define a positive base_weight.")

    def classify(self, relative_path: str) -> RIRClassificationOverride:
        normalized_path = relative_path.lower().replace("-", " ").replace("_", " ")
        room_size: RIRRoomSize | None = None
        field: RIRField | None = None
        rt60_bucket: RIRRT60Bucket | None = None
        direct_condition: RIRDirectCondition | None = None
        tags = list(self.tags)
        for rule in self.classification_rules:
            if not rule.matches(normalized_path):
                continue
            if rule.room_size is not None:
                room_size = rule.room_size
            if rule.field is not None:
                field = rule.field
            if rule.rt60_bucket is not None:
                rt60_bucket = rule.rt60_bucket
            if rule.direct_condition is not None:
                direct_condition = rule.direct_condition
            tags.extend(rule.tags)
        return RIRClassificationOverride(
            room_size=room_size,
            field=field,
            rt60_bucket=rt60_bucket,
            direct_condition=direct_condition,
            tags=tuple(sorted(set(tags))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "inventory_source_id": self.inventory_source_id,
            "room_family": self.room_family,
            "root_candidates": list(self.root_candidates),
            "default_room_size": self.default_room_size,
            "base_weight": self.base_weight,
            "tags": list(self.tags),
            "classification_rules": [rule.to_dict() for rule in self.classification_rules],
        }


@dataclass(frozen=True, slots=True)
class RIRBankPlan:
    notes: tuple[str, ...]
    analysis: RIRAnalysisSettings
    sources: tuple[RIRSourcePlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "notes": list(self.notes),
            "analysis": self.analysis.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class RIRSourceStatus:
    source_id: str
    name: str
    inventory_source_id: str
    room_family: RIRFamily
    configured_roots: tuple[str, ...]
    resolved_root: str | None
    status: str
    discovered_audio_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "inventory_source_id": self.inventory_source_id,
            "room_family": self.room_family,
            "configured_roots": list(self.configured_roots),
            "resolved_root": self.resolved_root,
            "status": self.status,
            "discovered_audio_count": self.discovered_audio_count,
        }


@dataclass(frozen=True, slots=True)
class RIRAcousticMetrics:
    peak_time_ms: float
    tail_duration_ms: float
    energy_centroid_ms: float
    estimated_rt60_seconds: float
    estimated_drr_db: float
    envelope_preview: str

    def to_dict(self) -> dict[str, object]:
        return {
            "peak_time_ms": self.peak_time_ms,
            "tail_duration_ms": self.tail_duration_ms,
            "energy_centroid_ms": self.energy_centroid_ms,
            "estimated_rt60_seconds": self.estimated_rt60_seconds,
            "estimated_drr_db": self.estimated_drr_db,
            "envelope_preview": self.envelope_preview,
        }


@dataclass(frozen=True, slots=True)
class RIRBankEntry:
    rir_id: str
    source_id: str
    source_name: str
    inventory_source_id: str
    room_family: RIRFamily
    source_audio_path: str
    normalized_audio_path: str
    relative_path: str
    room_size: RIRRoomSize
    field: RIRField
    rt60_bucket: RIRRT60Bucket
    direct_condition: RIRDirectCondition
    tags: tuple[str, ...]
    sampling_weight: float
    peak_time_ms: float
    tail_duration_ms: float
    energy_centroid_ms: float
    estimated_rt60_seconds: float
    estimated_drr_db: float
    envelope_preview: str
    source_sample_rate_hz: int
    source_num_channels: int
    source_duration_seconds: float
    normalized_duration_seconds: float
    normalization_profile: str
    normalization_resampled: bool
    normalization_downmixed: bool
    normalization_peak_scaled: bool
    normalization_loudness_applied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "rir_id": self.rir_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "inventory_source_id": self.inventory_source_id,
            "room_family": self.room_family,
            "source_audio_path": self.source_audio_path,
            "normalized_audio_path": self.normalized_audio_path,
            "relative_path": self.relative_path,
            "room_size": self.room_size,
            "field": self.field,
            "rt60_bucket": self.rt60_bucket,
            "direct_condition": self.direct_condition,
            "tags": list(self.tags),
            "sampling_weight": self.sampling_weight,
            "peak_time_ms": self.peak_time_ms,
            "tail_duration_ms": self.tail_duration_ms,
            "energy_centroid_ms": self.energy_centroid_ms,
            "estimated_rt60_seconds": self.estimated_rt60_seconds,
            "estimated_drr_db": self.estimated_drr_db,
            "envelope_preview": self.envelope_preview,
            "source_sample_rate_hz": self.source_sample_rate_hz,
            "source_num_channels": self.source_num_channels,
            "source_duration_seconds": self.source_duration_seconds,
            "normalized_duration_seconds": self.normalized_duration_seconds,
            "normalization_profile": self.normalization_profile,
            "normalization_resampled": self.normalization_resampled,
            "normalization_downmixed": self.normalization_downmixed,
            "normalization_peak_scaled": self.normalization_peak_scaled,
            "normalization_loudness_applied": self.normalization_loudness_applied,
        }


@dataclass(frozen=True, slots=True)
class RoomSimulationConfig:
    config_id: str
    room_size: RIRRoomSize
    field: RIRField
    rt60_bucket: RIRRT60Bucket
    direct_condition: RIRDirectCondition
    rir_count: int
    sample_rir_ids: tuple[str, ...]
    min_rt60_seconds: float
    max_rt60_seconds: float
    min_drr_db: float
    max_drr_db: float
    room_families: tuple[RIRFamily, ...]
    source_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "room_size": self.room_size,
            "field": self.field,
            "rt60_bucket": self.rt60_bucket,
            "direct_condition": self.direct_condition,
            "rir_count": self.rir_count,
            "sample_rir_ids": list(self.sample_rir_ids),
            "min_rt60_seconds": self.min_rt60_seconds,
            "max_rt60_seconds": self.max_rt60_seconds,
            "min_drr_db": self.min_drr_db,
            "max_drr_db": self.max_drr_db,
            "room_families": list(self.room_families),
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True, slots=True)
class RIRBankQuarantineRecord:
    source_id: str
    source_name: str
    inventory_source_id: str
    room_family: RIRFamily
    source_audio_path: str
    relative_path: str
    room_size: RIRRoomSize
    issue_code: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "inventory_source_id": self.inventory_source_id,
            "room_family": self.room_family,
            "source_audio_path": self.source_audio_path,
            "relative_path": self.relative_path,
            "room_size": self.room_size,
            "issue_code": self.issue_code,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RIRBankSummary:
    source_count: int
    present_source_count: int
    missing_source_count: int
    entry_count: int
    config_count: int
    quarantine_count: int
    total_duration_seconds: float
    room_size_counts: dict[str, int]
    field_counts: dict[str, int]
    rt60_counts: dict[str, int]
    direct_condition_counts: dict[str, int]
    room_family_counts: dict[str, int]

    @property
    def missing_field_coverage(self) -> tuple[RIRField, ...]:
        return tuple(field for field in ALLOWED_RIR_FIELDS if self.field_counts.get(field, 0) == 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_count": self.source_count,
            "present_source_count": self.present_source_count,
            "missing_source_count": self.missing_source_count,
            "entry_count": self.entry_count,
            "config_count": self.config_count,
            "quarantine_count": self.quarantine_count,
            "total_duration_seconds": self.total_duration_seconds,
            "room_size_counts": dict(self.room_size_counts),
            "field_counts": dict(self.field_counts),
            "rt60_counts": dict(self.rt60_counts),
            "direct_condition_counts": dict(self.direct_condition_counts),
            "room_family_counts": dict(self.room_family_counts),
            "missing_field_coverage": list(self.missing_field_coverage),
        }


@dataclass(frozen=True, slots=True)
class RIRBankReport:
    generated_at: str
    project_root: str
    dataset_root: str
    output_root: str
    plan_path: str | None
    policy: AudioNormalizationPolicy
    analysis: RIRAnalysisSettings
    notes: tuple[str, ...]
    sources: tuple[RIRSourceStatus, ...]
    entries: tuple[RIRBankEntry, ...]
    room_configs: tuple[RoomSimulationConfig, ...]
    quarantined: tuple[RIRBankQuarantineRecord, ...]
    summary: RIRBankSummary

    def to_dict(
        self,
        *,
        include_entries: bool = False,
        include_configs: bool = False,
        include_quarantine: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "dataset_root": self.dataset_root,
            "output_root": self.output_root,
            "plan_path": self.plan_path,
            "policy": self.policy.to_dict(),
            "analysis": self.analysis.to_dict(),
            "notes": list(self.notes),
            "sources": [source.to_dict() for source in self.sources],
            "summary": self.summary.to_dict(),
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        if include_configs:
            payload["room_configs"] = [config.to_dict() for config in self.room_configs]
        if include_quarantine:
            payload["quarantined"] = [record.to_dict() for record in self.quarantined]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenRIRBankArtifacts:
    output_root: str
    manifest_path: str
    room_config_path: str
    quarantine_path: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "manifest_path": self.manifest_path,
            "room_config_path": self.room_config_path,
            "quarantine_path": self.quarantine_path,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def _bucket_from_ranges[BucketName: str](
    *,
    value: float,
    ranges: Mapping[BucketName, NumericRange],
    order: tuple[BucketName, ...],
    label: str,
) -> BucketName:
    for name in order:
        if ranges[name].contains(value):
            return name
    raise ValueError(f"Unable to classify {label} value {value!r} with configured ranges.")


__all__ = [
    "ALLOWED_RIR_DIRECT_CONDITIONS",
    "ALLOWED_RIR_FAMILIES",
    "ALLOWED_RIR_FIELDS",
    "ALLOWED_RIR_ROOM_SIZES",
    "ALLOWED_RIR_RT60_BUCKETS",
    "MANIFEST_JSONL_NAME",
    "NumericRange",
    "QUARANTINE_JSONL_NAME",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "RIRAcousticMetrics",
    "RIRAnalysisSettings",
    "RIRBankEntry",
    "RIRBankPlan",
    "RIRBankQuarantineRecord",
    "RIRBankReport",
    "RIRBankSummary",
    "RIRClassification",
    "RIRClassificationOverride",
    "RIRClassificationRule",
    "RIRDirectCondition",
    "RIRFamily",
    "RIRField",
    "RIRRoomSize",
    "RIRRT60Bucket",
    "RIRSourcePlan",
    "RIRSourceStatus",
    "ROOM_CONFIG_JSONL_NAME",
    "RoomSimulationConfig",
    "SUPPORTED_AUDIO_SUFFIXES",
    "WrittenRIRBankArtifacts",
]
