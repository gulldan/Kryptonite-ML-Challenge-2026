"""Datamodels for audio normalization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kryptonite.config import NormalizationConfig


@dataclass(frozen=True, slots=True)
class AudioNormalizationPolicy:
    target_sample_rate_hz: int
    target_channels: int
    output_format: str
    output_pcm_bits_per_sample: int
    peak_headroom_db: float
    dc_offset_threshold: float
    clipped_sample_threshold: float

    @classmethod
    def from_config(cls, config: NormalizationConfig) -> AudioNormalizationPolicy:
        return cls(
            target_sample_rate_hz=config.target_sample_rate_hz,
            target_channels=config.target_channels,
            output_format=config.output_format,
            output_pcm_bits_per_sample=config.output_pcm_bits_per_sample,
            peak_headroom_db=config.peak_headroom_db,
            dc_offset_threshold=config.dc_offset_threshold,
            clipped_sample_threshold=config.clipped_sample_threshold,
        )

    @property
    def output_suffix(self) -> str:
        return f".{self.output_format.lower()}"

    @property
    def normalization_profile(self) -> str:
        return (
            f"{self.target_sample_rate_hz}hz-"
            f"{self.target_channels}ch-"
            f"pcm{self.output_pcm_bits_per_sample}-"
            f"{self.output_format.lower()}"
        )

    @property
    def target_peak_amplitude(self) -> float:
        return float(10 ** (-self.peak_headroom_db / 20.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "target_sample_rate_hz": self.target_sample_rate_hz,
            "target_channels": self.target_channels,
            "output_format": self.output_format,
            "output_pcm_bits_per_sample": self.output_pcm_bits_per_sample,
            "peak_headroom_db": self.peak_headroom_db,
            "dc_offset_threshold": self.dc_offset_threshold,
            "clipped_sample_threshold": self.clipped_sample_threshold,
            "normalization_profile": self.normalization_profile,
        }


@dataclass(frozen=True, slots=True)
class SourceManifestTable:
    name: str
    path: Path
    rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class AuxiliaryTable:
    name: str
    path: Path
    rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class NormalizedAudioRecord:
    source_audio_path: str
    normalized_audio_path: str
    source_sample_rate_hz: int
    source_num_channels: int
    source_duration_seconds: float
    normalized_duration_seconds: float
    source_peak_amplitude: float
    source_dc_offset_ratio: float
    source_clipped_sample_ratio: float
    resampled: bool
    downmixed: bool
    dc_offset_removed: bool
    peak_scaled: bool


@dataclass(frozen=True, slots=True)
class QuarantineDecision:
    issue_code: str
    reason: str
    source_audio_path: str | None


@dataclass(slots=True)
class AudioNormalizationSummary:
    dataset: str
    source_manifests_root: str
    output_root: str
    output_manifests_root: str
    output_audio_root: str
    manifest_inventory_file: str
    report_json_file: str
    report_markdown_file: str
    source_manifest_count: int
    source_row_count: int
    normalized_row_count: int
    normalized_audio_count: int
    generated_quarantine_row_count: int
    carried_quarantine_row_count: int
    auxiliary_table_count: int
    copied_metadata_file_count: int
    resampled_row_count: int
    downmixed_row_count: int
    dc_offset_fixed_row_count: int
    peak_scaled_row_count: int
    source_clipping_row_count: int
    quarantine_issue_counts: dict[str, int]
    policy: AudioNormalizationPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "source_manifests_root": self.source_manifests_root,
            "output_root": self.output_root,
            "output_manifests_root": self.output_manifests_root,
            "output_audio_root": self.output_audio_root,
            "manifest_inventory_file": self.manifest_inventory_file,
            "report_json_file": self.report_json_file,
            "report_markdown_file": self.report_markdown_file,
            "source_manifest_count": self.source_manifest_count,
            "source_row_count": self.source_row_count,
            "normalized_row_count": self.normalized_row_count,
            "normalized_audio_count": self.normalized_audio_count,
            "generated_quarantine_row_count": self.generated_quarantine_row_count,
            "carried_quarantine_row_count": self.carried_quarantine_row_count,
            "auxiliary_table_count": self.auxiliary_table_count,
            "copied_metadata_file_count": self.copied_metadata_file_count,
            "resampled_row_count": self.resampled_row_count,
            "downmixed_row_count": self.downmixed_row_count,
            "dc_offset_fixed_row_count": self.dc_offset_fixed_row_count,
            "peak_scaled_row_count": self.peak_scaled_row_count,
            "source_clipping_row_count": self.source_clipping_row_count,
            "quarantine_issue_counts": dict(self.quarantine_issue_counts),
            "policy": self.policy.to_dict(),
        }
