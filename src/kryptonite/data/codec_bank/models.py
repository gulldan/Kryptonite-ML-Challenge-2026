"""Datamodels for reproducible codec/channel simulation presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

CodecFamily = Literal["band_limit", "telephony", "voip", "compression", "channel"]
CodecSeverity = Literal["light", "medium", "heavy"]

ALLOWED_CODEC_FAMILIES: tuple[CodecFamily, ...] = (
    "band_limit",
    "telephony",
    "voip",
    "compression",
    "channel",
)
ALLOWED_CODEC_SEVERITIES: tuple[CodecSeverity, ...] = ("light", "medium", "heavy")
REPORT_JSON_NAME = "codec_bank_report.json"
REPORT_MARKDOWN_NAME = "codec_bank_report.md"
MANIFEST_JSONL_NAME = "codec_bank_manifest.jsonl"
FAILURES_JSONL_NAME = "codec_bank_failures.jsonl"
PROBE_AUDIO_NAME = "codec_bank_probe.wav"


@dataclass(frozen=True, slots=True)
class CodecSeverityProfile:
    description: str
    weight_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("Codec severity profile description must be non-empty.")
        if self.weight_multiplier <= 0.0:
            raise ValueError("Codec severity profile weight_multiplier must be positive.")

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "weight_multiplier": self.weight_multiplier,
        }


@dataclass(frozen=True, slots=True)
class CodecProbeSettings:
    sample_rate_hz: int
    duration_seconds: float
    peak_amplitude: float = 0.85

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("Codec probe sample_rate_hz must be positive.")
        if self.duration_seconds <= 0.0:
            raise ValueError("Codec probe duration_seconds must be positive.")
        if not 0.0 < self.peak_amplitude <= 1.0:
            raise ValueError("Codec probe peak_amplitude must be within (0.0, 1.0].")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "duration_seconds": self.duration_seconds,
            "peak_amplitude": self.peak_amplitude,
        }


@dataclass(frozen=True, slots=True)
class CodecEQBand:
    frequency_hz: float
    width_hz: float
    gain_db: float

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("Codec EQ band frequency_hz must be positive.")
        if self.width_hz <= 0.0:
            raise ValueError("Codec EQ band width_hz must be positive.")

    def to_dict(self) -> dict[str, float]:
        return {
            "frequency_hz": self.frequency_hz,
            "width_hz": self.width_hz,
            "gain_db": self.gain_db,
        }

    def to_ffmpeg_filter(self) -> str:
        return (
            "equalizer="
            f"f={_format_float(self.frequency_hz)}:"
            "width_type=h:"
            f"width={_format_float(self.width_hz)}:"
            f"g={_format_float(self.gain_db)}"
        )


@dataclass(frozen=True, slots=True)
class CodecSimulationPreset:
    id: str
    name: str
    family: CodecFamily
    severity: CodecSeverity
    description: str
    base_weight: float = 1.0
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    pre_gain_db: float = 0.0
    post_gain_db: float = 0.0
    bitcrusher_bits: int | None = None
    bitcrusher_mix: float = 1.0
    soft_clip: bool = False
    codec_name: str | None = None
    container_extension: str = "wav"
    encode_sample_rate_hz: int | None = None
    encode_bitrate: str | None = None
    ffmpeg_options: tuple[str, ...] = ()
    eq_bands: tuple[CodecEQBand, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Codec preset id must be non-empty.")
        if not self.name.strip():
            raise ValueError("Codec preset name must be non-empty.")
        if not self.description.strip():
            raise ValueError("Codec preset description must be non-empty.")
        if self.base_weight <= 0.0:
            raise ValueError("Codec preset base_weight must be positive.")
        if self.highpass_hz is not None and self.highpass_hz <= 0.0:
            raise ValueError("Codec preset highpass_hz must be positive when provided.")
        if self.lowpass_hz is not None and self.lowpass_hz <= 0.0:
            raise ValueError("Codec preset lowpass_hz must be positive when provided.")
        if (
            self.highpass_hz is not None
            and self.lowpass_hz is not None
            and self.lowpass_hz <= self.highpass_hz
        ):
            raise ValueError("Codec preset lowpass_hz must exceed highpass_hz.")
        if self.bitcrusher_bits is not None and self.bitcrusher_bits <= 0:
            raise ValueError("Codec preset bitcrusher_bits must be positive when provided.")
        if not 0.0 <= self.bitcrusher_mix <= 1.0:
            raise ValueError("Codec preset bitcrusher_mix must be within [0.0, 1.0].")
        if self.encode_sample_rate_hz is not None and self.encode_sample_rate_hz <= 0:
            raise ValueError("Codec preset encode_sample_rate_hz must be positive when provided.")
        if self.codec_name is None and self.encode_bitrate is not None:
            raise ValueError("Codec preset encode_bitrate requires codec_name.")
        if self.codec_name is None and self.encode_sample_rate_hz is None and not self.filters:
            raise ValueError("Codec preset must define at least one filter or codec stage.")
        if self.codec_name is None and self.ffmpeg_options:
            raise ValueError("Codec preset ffmpeg_options require codec_name.")
        if not self.container_extension.strip():
            raise ValueError("Codec preset container_extension must be non-empty.")
        invalid_options = [option for option in self.ffmpeg_options if "=" not in option]
        if invalid_options:
            raise ValueError("Codec preset ffmpeg_options must be key=value pairs.")

    @property
    def filters(self) -> tuple[str, ...]:
        filters: list[str] = []
        if self.highpass_hz is not None:
            filters.append(f"highpass=f={_format_float(self.highpass_hz)}")
        if self.lowpass_hz is not None:
            filters.append(f"lowpass=f={_format_float(self.lowpass_hz)}")
        filters.extend(band.to_ffmpeg_filter() for band in self.eq_bands)
        if self.bitcrusher_bits is not None:
            filters.append(
                f"acrusher=bits={self.bitcrusher_bits}:mix={_format_float(self.bitcrusher_mix)}"
            )
        if self.pre_gain_db:
            filters.append(f"volume={_format_float(self.pre_gain_db)}dB")
        if self.soft_clip:
            filters.append("asoftclip=type=tanh")
        return tuple(filters)

    @property
    def post_filters(self) -> tuple[str, ...]:
        filters: list[str] = []
        if self.post_gain_db:
            filters.append(f"volume={_format_float(self.post_gain_db)}dB")
        return tuple(filters)

    @property
    def uses_codec_stage(self) -> bool:
        return self.codec_name is not None

    def sampling_weight(
        self, severity_profiles: dict[CodecSeverity, CodecSeverityProfile]
    ) -> float:
        return round(self.base_weight * severity_profiles[self.severity].weight_multiplier, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "severity": self.severity,
            "description": self.description,
            "base_weight": self.base_weight,
            "highpass_hz": self.highpass_hz,
            "lowpass_hz": self.lowpass_hz,
            "pre_gain_db": self.pre_gain_db,
            "post_gain_db": self.post_gain_db,
            "bitcrusher_bits": self.bitcrusher_bits,
            "bitcrusher_mix": self.bitcrusher_mix,
            "soft_clip": self.soft_clip,
            "codec_name": self.codec_name,
            "container_extension": self.container_extension,
            "encode_sample_rate_hz": self.encode_sample_rate_hz,
            "encode_bitrate": self.encode_bitrate,
            "ffmpeg_options": list(self.ffmpeg_options),
            "eq_bands": [band.to_dict() for band in self.eq_bands],
            "tags": list(self.tags),
            "filters": list(self.filters),
            "post_filters": list(self.post_filters),
        }


@dataclass(frozen=True, slots=True)
class CodecBankPlan:
    notes: tuple[str, ...]
    probe: CodecProbeSettings
    severity_profiles: dict[CodecSeverity, CodecSeverityProfile]
    presets: tuple[CodecSimulationPreset, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "notes": list(self.notes),
            "probe": self.probe.to_dict(),
            "severity_profiles": {
                name: profile.to_dict()
                for name, profile in sorted(self.severity_profiles.items(), key=_severity_sort_key)
            },
            "presets": [preset.to_dict() for preset in self.presets],
        }


@dataclass(frozen=True, slots=True)
class FFmpegToolMetadata:
    ffmpeg_path: str
    ffprobe_path: str
    ffmpeg_available: bool
    ffprobe_available: bool
    version_line: str | None
    configuration: str | None
    ffprobe_version_line: str | None
    ffmpeg_error: str | None = None
    ffprobe_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ffmpeg_path": self.ffmpeg_path,
            "ffprobe_path": self.ffprobe_path,
            "ffmpeg_available": self.ffmpeg_available,
            "ffprobe_available": self.ffprobe_available,
            "version_line": self.version_line,
            "configuration": self.configuration,
            "ffprobe_version_line": self.ffprobe_version_line,
            "ffmpeg_error": self.ffmpeg_error,
            "ffprobe_error": self.ffprobe_error,
        }


@dataclass(frozen=True, slots=True)
class ProbeAudioMetrics:
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
class CodecBankEntry:
    preset_id: str
    name: str
    family: CodecFamily
    severity: CodecSeverity
    description: str
    tags: tuple[str, ...]
    sampling_weight: float
    probe_audio_path: str
    preview_audio_path: str
    preview_sha256: str
    ffmpeg_pre_filter_graph: str | None
    ffmpeg_post_filter_graph: str | None
    ffmpeg_encode_codec: str | None
    ffmpeg_container: str | None
    ffmpeg_encode_sample_rate_hz: int | None
    ffmpeg_encode_bitrate: str | None
    ffmpeg_options: tuple[str, ...]
    encode_command: str | None
    decode_command: str
    source_metrics: ProbeAudioMetrics
    output_metrics: ProbeAudioMetrics

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
            "family": self.family,
            "severity": self.severity,
            "description": self.description,
            "tags": list(self.tags),
            "sampling_weight": self.sampling_weight,
            "probe_audio_path": self.probe_audio_path,
            "preview_audio_path": self.preview_audio_path,
            "preview_sha256": self.preview_sha256,
            "ffmpeg_pre_filter_graph": self.ffmpeg_pre_filter_graph,
            "ffmpeg_post_filter_graph": self.ffmpeg_post_filter_graph,
            "ffmpeg_encode_codec": self.ffmpeg_encode_codec,
            "ffmpeg_container": self.ffmpeg_container,
            "ffmpeg_encode_sample_rate_hz": self.ffmpeg_encode_sample_rate_hz,
            "ffmpeg_encode_bitrate": self.ffmpeg_encode_bitrate,
            "ffmpeg_options": list(self.ffmpeg_options),
            "encode_command": self.encode_command,
            "decode_command": self.decode_command,
            "source_metrics": self.source_metrics.to_dict(),
            "output_metrics": self.output_metrics.to_dict(),
            "rolloff_delta_hz": self.rolloff_delta_hz,
            "rms_delta_db": self.rms_delta_db,
        }


@dataclass(frozen=True, slots=True)
class CodecBankFailureRecord:
    preset_id: str
    name: str
    family: CodecFamily
    severity: CodecSeverity
    issue_code: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "family": self.family,
            "severity": self.severity,
            "issue_code": self.issue_code,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CodecBankSummary:
    preset_count: int
    rendered_preview_count: int
    failure_count: int
    codec_stage_count: int
    family_counts: dict[str, int]
    severity_counts: dict[str, int]

    @property
    def missing_family_coverage(self) -> tuple[CodecFamily, ...]:
        return tuple(
            family for family in ALLOWED_CODEC_FAMILIES if self.family_counts.get(family, 0) == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_count": self.preset_count,
            "rendered_preview_count": self.rendered_preview_count,
            "failure_count": self.failure_count,
            "codec_stage_count": self.codec_stage_count,
            "family_counts": dict(self.family_counts),
            "severity_counts": dict(self.severity_counts),
            "missing_family_coverage": list(self.missing_family_coverage),
        }


@dataclass(frozen=True, slots=True)
class CodecBankReport:
    generated_at: str
    project_root: str
    output_root: str
    plan_path: str | None
    notes: tuple[str, ...]
    probe: CodecProbeSettings
    probe_audio_path: str
    probe_metrics: ProbeAudioMetrics
    ffmpeg: FFmpegToolMetadata
    severity_profiles: dict[CodecSeverity, CodecSeverityProfile]
    entries: tuple[CodecBankEntry, ...]
    failures: tuple[CodecBankFailureRecord, ...]
    summary: CodecBankSummary

    def to_dict(
        self,
        *,
        include_entries: bool = False,
        include_failures: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "plan_path": self.plan_path,
            "notes": list(self.notes),
            "probe": self.probe.to_dict(),
            "probe_audio_path": self.probe_audio_path,
            "probe_metrics": self.probe_metrics.to_dict(),
            "ffmpeg": self.ffmpeg.to_dict(),
            "severity_profiles": {
                name: profile.to_dict()
                for name, profile in sorted(self.severity_profiles.items(), key=_severity_sort_key)
            },
            "summary": self.summary.to_dict(),
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        if include_failures:
            payload["failures"] = [record.to_dict() for record in self.failures]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenCodecBankArtifacts:
    output_root: str
    probe_path: str
    manifest_path: str
    failures_path: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "probe_path": self.probe_path,
            "manifest_path": self.manifest_path,
            "failures_path": self.failures_path,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def _severity_sort_key(item: tuple[str, CodecSeverityProfile]) -> int:
    return ALLOWED_CODEC_SEVERITIES.index(cast(CodecSeverity, item[0]))


def _format_float(value: float) -> str:
    return format(value, "g")


__all__ = [
    "ALLOWED_CODEC_FAMILIES",
    "ALLOWED_CODEC_SEVERITIES",
    "CodecBankEntry",
    "CodecBankFailureRecord",
    "CodecBankPlan",
    "CodecBankReport",
    "CodecBankSummary",
    "CodecEQBand",
    "CodecFamily",
    "CodecProbeSettings",
    "CodecSeverity",
    "CodecSeverityProfile",
    "CodecSimulationPreset",
    "FAILURES_JSONL_NAME",
    "FFmpegToolMetadata",
    "MANIFEST_JSONL_NAME",
    "PROBE_AUDIO_NAME",
    "ProbeAudioMetrics",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "WrittenCodecBankArtifacts",
]
