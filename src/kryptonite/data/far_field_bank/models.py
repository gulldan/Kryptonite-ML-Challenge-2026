"""Datamodels for reproducible far-field and distance simulation presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DistanceField = Literal["near", "mid", "far"]

ALLOWED_DISTANCE_FIELDS: tuple[DistanceField, ...] = ("near", "mid", "far")
MANIFEST_JSONL_NAME = "far_field_bank_manifest.jsonl"
REPORT_JSON_NAME = "far_field_bank_report.json"
REPORT_MARKDOWN_NAME = "far_field_bank_report.md"
PROBE_AUDIO_NAME = "far_field_probe.wav"


@dataclass(frozen=True, slots=True)
class FarFieldProbeSettings:
    sample_rate_hz: int
    duration_seconds: float
    peak_amplitude: float = 0.85

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("Far-field probe sample_rate_hz must be positive.")
        if self.duration_seconds <= 0.0:
            raise ValueError("Far-field probe duration_seconds must be positive.")
        if not 0.0 < self.peak_amplitude <= 1.0:
            raise ValueError("Far-field probe peak_amplitude must be within (0.0, 1.0].")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "duration_seconds": self.duration_seconds,
            "peak_amplitude": self.peak_amplitude,
        }


@dataclass(frozen=True, slots=True)
class FarFieldRenderSettings:
    kernel_duration_seconds: float
    speed_of_sound_mps: float = 343.0
    output_peak_limit: float = 0.92
    high_shelf_pivot_hz: float = 1_800.0

    def __post_init__(self) -> None:
        if self.kernel_duration_seconds <= 0.0:
            raise ValueError("Far-field kernel_duration_seconds must be positive.")
        if self.speed_of_sound_mps <= 0.0:
            raise ValueError("Far-field speed_of_sound_mps must be positive.")
        if not 0.0 < self.output_peak_limit <= 1.0:
            raise ValueError("Far-field output_peak_limit must be within (0.0, 1.0].")
        if self.high_shelf_pivot_hz <= 0.0:
            raise ValueError("Far-field high_shelf_pivot_hz must be positive.")

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_duration_seconds": self.kernel_duration_seconds,
            "speed_of_sound_mps": self.speed_of_sound_mps,
            "output_peak_limit": self.output_peak_limit,
            "high_shelf_pivot_hz": self.high_shelf_pivot_hz,
        }


@dataclass(frozen=True, slots=True)
class FarFieldSimulationPreset:
    id: str
    name: str
    field: DistanceField
    description: str
    distance_meters: float
    off_axis_angle_deg: float
    attenuation_db: float
    target_drr_db: float
    reverb_rt60_seconds: float
    late_reverb_start_ms: float
    lowpass_hz: float
    high_shelf_db: float = 0.0
    base_weight: float = 1.0
    early_reflection_delays_ms: tuple[float, ...] = ()
    early_reflection_gains_db: tuple[float, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Far-field preset id must be non-empty.")
        if not self.name.strip():
            raise ValueError("Far-field preset name must be non-empty.")
        if not self.description.strip():
            raise ValueError("Far-field preset description must be non-empty.")
        if self.distance_meters <= 0.0:
            raise ValueError("Far-field preset distance_meters must be positive.")
        if not 0.0 <= self.off_axis_angle_deg <= 90.0:
            raise ValueError("Far-field preset off_axis_angle_deg must be within [0, 90].")
        if self.reverb_rt60_seconds <= 0.0:
            raise ValueError("Far-field preset reverb_rt60_seconds must be positive.")
        if self.late_reverb_start_ms <= 0.0:
            raise ValueError("Far-field preset late_reverb_start_ms must be positive.")
        if self.lowpass_hz <= 0.0:
            raise ValueError("Far-field preset lowpass_hz must be positive.")
        if self.base_weight <= 0.0:
            raise ValueError("Far-field preset base_weight must be positive.")
        if len(self.early_reflection_delays_ms) != len(self.early_reflection_gains_db):
            raise ValueError(
                "Far-field preset early_reflection_delays_ms and early_reflection_gains_db "
                "must have the same length."
            )
        if any(delay_ms <= 0.0 for delay_ms in self.early_reflection_delays_ms):
            raise ValueError("Far-field preset reflection delays must be positive.")

    @property
    def reflection_count(self) -> int:
        return len(self.early_reflection_delays_ms)

    @property
    def sampling_weight(self) -> float:
        return round(self.base_weight, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "field": self.field,
            "description": self.description,
            "distance_meters": self.distance_meters,
            "off_axis_angle_deg": self.off_axis_angle_deg,
            "attenuation_db": self.attenuation_db,
            "target_drr_db": self.target_drr_db,
            "reverb_rt60_seconds": self.reverb_rt60_seconds,
            "late_reverb_start_ms": self.late_reverb_start_ms,
            "lowpass_hz": self.lowpass_hz,
            "high_shelf_db": self.high_shelf_db,
            "base_weight": self.base_weight,
            "early_reflection_delays_ms": list(self.early_reflection_delays_ms),
            "early_reflection_gains_db": list(self.early_reflection_gains_db),
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class FarFieldBankPlan:
    notes: tuple[str, ...]
    probe: FarFieldProbeSettings
    render: FarFieldRenderSettings
    presets: tuple[FarFieldSimulationPreset, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "notes": list(self.notes),
            "probe": self.probe.to_dict(),
            "render": self.render.to_dict(),
            "presets": [preset.to_dict() for preset in self.presets],
        }


@dataclass(frozen=True, slots=True)
class FarFieldAudioMetrics:
    duration_seconds: float
    sample_rate_hz: int
    num_channels: int
    peak_amplitude: float
    rms_dbfs: float
    clipped_sample_ratio: float
    spectral_centroid_hz: float
    spectral_rolloff_95_hz: float

    def to_dict(self) -> dict[str, object]:
        return {
            "duration_seconds": self.duration_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "num_channels": self.num_channels,
            "peak_amplitude": self.peak_amplitude,
            "rms_dbfs": self.rms_dbfs,
            "clipped_sample_ratio": self.clipped_sample_ratio,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_rolloff_95_hz": self.spectral_rolloff_95_hz,
        }


@dataclass(frozen=True, slots=True)
class FarFieldKernelMetrics:
    arrival_delay_ms: float
    actual_drr_db: float
    late_reverb_start_ms: float
    reflection_count: int
    kernel_duration_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "arrival_delay_ms": self.arrival_delay_ms,
            "actual_drr_db": self.actual_drr_db,
            "late_reverb_start_ms": self.late_reverb_start_ms,
            "reflection_count": self.reflection_count,
            "kernel_duration_ms": self.kernel_duration_ms,
        }


@dataclass(frozen=True, slots=True)
class FarFieldBankEntry:
    preset_id: str
    name: str
    field: DistanceField
    description: str
    distance_meters: float
    off_axis_angle_deg: float
    attenuation_db: float
    target_drr_db: float
    reverb_rt60_seconds: float
    lowpass_hz: float
    high_shelf_db: float
    tags: tuple[str, ...]
    sampling_weight: float
    probe_audio_path: str
    kernel_audio_path: str
    preview_audio_path: str
    kernel_sha256: str
    preview_sha256: str
    source_metrics: FarFieldAudioMetrics
    output_metrics: FarFieldAudioMetrics
    kernel_metrics: FarFieldKernelMetrics

    @property
    def rolloff_delta_hz(self) -> float:
        return round(
            self.output_metrics.spectral_rolloff_95_hz - self.source_metrics.spectral_rolloff_95_hz,
            6,
        )

    @property
    def rms_delta_db(self) -> float:
        return round(self.output_metrics.rms_dbfs - self.source_metrics.rms_dbfs, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "field": self.field,
            "description": self.description,
            "distance_meters": self.distance_meters,
            "off_axis_angle_deg": self.off_axis_angle_deg,
            "attenuation_db": self.attenuation_db,
            "target_drr_db": self.target_drr_db,
            "reverb_rt60_seconds": self.reverb_rt60_seconds,
            "lowpass_hz": self.lowpass_hz,
            "high_shelf_db": self.high_shelf_db,
            "tags": list(self.tags),
            "sampling_weight": self.sampling_weight,
            "probe_audio_path": self.probe_audio_path,
            "kernel_audio_path": self.kernel_audio_path,
            "preview_audio_path": self.preview_audio_path,
            "kernel_sha256": self.kernel_sha256,
            "preview_sha256": self.preview_sha256,
            "source_metrics": self.source_metrics.to_dict(),
            "output_metrics": self.output_metrics.to_dict(),
            "kernel_metrics": self.kernel_metrics.to_dict(),
            "rolloff_delta_hz": self.rolloff_delta_hz,
            "rms_delta_db": self.rms_delta_db,
        }


@dataclass(frozen=True, slots=True)
class FarFieldBankSummary:
    preset_count: int
    rendered_preview_count: int
    field_counts: dict[str, int]

    @property
    def missing_field_coverage(self) -> tuple[DistanceField, ...]:
        return tuple(
            field for field in ALLOWED_DISTANCE_FIELDS if self.field_counts.get(field, 0) == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_count": self.preset_count,
            "rendered_preview_count": self.rendered_preview_count,
            "field_counts": dict(self.field_counts),
            "missing_field_coverage": list(self.missing_field_coverage),
        }


@dataclass(frozen=True, slots=True)
class FarFieldBankReport:
    generated_at: str
    project_root: str
    output_root: str
    plan_path: str | None
    notes: tuple[str, ...]
    probe: FarFieldProbeSettings
    render: FarFieldRenderSettings
    probe_audio_path: str
    probe_metrics: FarFieldAudioMetrics
    entries: tuple[FarFieldBankEntry, ...]
    summary: FarFieldBankSummary

    def to_dict(self, *, include_entries: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "plan_path": self.plan_path,
            "notes": list(self.notes),
            "probe": self.probe.to_dict(),
            "render": self.render.to_dict(),
            "probe_audio_path": self.probe_audio_path,
            "probe_metrics": self.probe_metrics.to_dict(),
            "summary": self.summary.to_dict(),
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenFarFieldArtifacts:
    output_root: str
    probe_path: str
    manifest_path: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "probe_path": self.probe_path,
            "manifest_path": self.manifest_path,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


__all__ = [
    "ALLOWED_DISTANCE_FIELDS",
    "DistanceField",
    "FarFieldAudioMetrics",
    "FarFieldBankEntry",
    "FarFieldBankPlan",
    "FarFieldBankReport",
    "FarFieldBankSummary",
    "FarFieldKernelMetrics",
    "FarFieldProbeSettings",
    "FarFieldRenderSettings",
    "FarFieldSimulationPreset",
    "MANIFEST_JSONL_NAME",
    "PROBE_AUDIO_NAME",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "WrittenFarFieldArtifacts",
]
