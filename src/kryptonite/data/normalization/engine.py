"""Per-file normalization engine with cached source-audio decisions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from ..loudness import LoudnessNormalizationSettings, apply_loudness_normalization
from .audio_io import (
    all_finite,
    channel_dc_offset_ratio,
    clipped_sample_ratio,
    peak_amplitude,
    read_audio_file,
    resample_waveform,
    write_audio_file,
)
from .common import coerce_str, relative_to_project
from .models import AudioNormalizationPolicy, NormalizedAudioRecord, QuarantineDecision


class ManifestAudioNormalizer:
    def __init__(
        self,
        *,
        project_root: Path,
        dataset_root: Path,
        audio_output_root: Path,
        policy: AudioNormalizationPolicy,
    ) -> None:
        self.project_root = project_root
        self.dataset_root = dataset_root
        self.audio_output_root = audio_output_root
        self.policy = policy
        self.success_by_source_audio_path: dict[str, NormalizedAudioRecord] = {}
        self.failure_by_source_audio_path: dict[str, QuarantineDecision] = {}
        self.output_basename_by_source_basename: dict[str, str] = {}

    def normalize_row(
        self, row: Mapping[str, object]
    ) -> NormalizedAudioRecord | QuarantineDecision:
        source_audio_path = coerce_str(row.get("audio_path"))
        if source_audio_path is None:
            return QuarantineDecision(
                issue_code="missing_audio_path",
                reason="manifest row does not define a non-empty audio_path",
                source_audio_path=None,
            )

        cached_success = self.success_by_source_audio_path.get(source_audio_path)
        if cached_success is not None:
            return cached_success

        cached_failure = self.failure_by_source_audio_path.get(source_audio_path)
        if cached_failure is not None:
            return cached_failure

        decision = self._normalize_source_audio(source_audio_path)
        if isinstance(decision, QuarantineDecision):
            self.failure_by_source_audio_path[source_audio_path] = decision
        else:
            self.success_by_source_audio_path[source_audio_path] = decision
            self.output_basename_by_source_basename[Path(source_audio_path).name] = Path(
                decision.normalized_audio_path
            ).name
        return decision

    def _normalize_source_audio(
        self,
        source_audio_path: str,
    ) -> NormalizedAudioRecord | QuarantineDecision:
        source_path = resolve_project_path(str(self.project_root), source_audio_path)
        if not source_path.exists():
            return QuarantineDecision(
                issue_code="missing_audio_file",
                reason=f"audio file is missing from disk: {source_audio_path}",
                source_audio_path=source_audio_path,
            )

        try:
            waveform, sample_rate_hz = read_audio_file(source_path)
        except Exception as exc:
            return QuarantineDecision(
                issue_code="audio_decode_error",
                reason=f"{type(exc).__name__}: {exc}",
                source_audio_path=source_audio_path,
            )

        if waveform.ndim != 2 or int(waveform.shape[-1]) == 0:
            return QuarantineDecision(
                issue_code="empty_audio_signal",
                reason="decoded audio tensor is empty",
                source_audio_path=source_audio_path,
            )
        if not all_finite(waveform).all():
            return QuarantineDecision(
                issue_code="invalid_audio_signal",
                reason="decoded audio tensor contains non-finite values",
                source_audio_path=source_audio_path,
            )
        if sample_rate_hz <= 0:
            return QuarantineDecision(
                issue_code="invalid_sample_rate",
                reason=f"decoded audio reports a non-positive sample rate: {sample_rate_hz}",
                source_audio_path=source_audio_path,
            )

        source_num_channels = int(waveform.shape[0])
        source_duration_seconds = round(float(waveform.shape[-1]) / float(sample_rate_hz), 6)
        source_peak_value = peak_amplitude(waveform)
        source_dc_offset_value = channel_dc_offset_ratio(waveform)
        source_clipped_ratio = clipped_sample_ratio(
            waveform,
            threshold=self.policy.clipped_sample_threshold,
        )

        normalized = waveform
        downmixed = False
        if self.policy.target_channels == 1 and source_num_channels != 1:
            normalized = normalized.mean(axis=0, keepdims=True)
            downmixed = True
        elif source_num_channels != self.policy.target_channels:
            return QuarantineDecision(
                issue_code="unsupported_channel_layout",
                reason=(
                    f"cannot normalize {source_num_channels} channels into "
                    f"{self.policy.target_channels} channels"
                ),
                source_audio_path=source_audio_path,
            )

        resampled = False
        if sample_rate_hz != self.policy.target_sample_rate_hz:
            normalized = resample_waveform(
                normalized,
                orig_freq=sample_rate_hz,
                new_freq=self.policy.target_sample_rate_hz,
            )
            resampled = True

        dc_offset_removed = False
        if channel_dc_offset_ratio(normalized) >= self.policy.dc_offset_threshold:
            normalized = normalized - normalized.mean(axis=-1, keepdims=True)
            dc_offset_removed = True

        peak_scaled = False
        normalized, loudness_decision = apply_loudness_normalization(
            normalized,
            settings=LoudnessNormalizationSettings(
                mode=self.policy.loudness_mode,
                target_loudness_dbfs=self.policy.target_loudness_dbfs,
                max_gain_db=self.policy.max_loudness_gain_db,
                max_attenuation_db=self.policy.max_loudness_attenuation_db,
                peak_headroom_db=self.policy.peak_headroom_db,
            ),
        )

        normalized_peak = peak_amplitude(normalized)
        if normalized_peak > self.policy.target_peak_amplitude and normalized_peak > 0.0:
            normalized = normalized * (self.policy.target_peak_amplitude / normalized_peak)
            peak_scaled = True

        normalized_audio_path = self._build_normalized_audio_path(source_path)
        try:
            normalized_audio_path.parent.mkdir(parents=True, exist_ok=True)
            write_audio_file(
                path=normalized_audio_path,
                waveform=normalized,
                sample_rate_hz=self.policy.target_sample_rate_hz,
                output_format=self.policy.output_format,
                pcm_bits_per_sample=self.policy.output_pcm_bits_per_sample,
            )
        except Exception as exc:
            return QuarantineDecision(
                issue_code="audio_write_error",
                reason=f"{type(exc).__name__}: {exc}",
                source_audio_path=source_audio_path,
            )

        return NormalizedAudioRecord(
            source_audio_path=source_audio_path,
            normalized_audio_path=relative_to_project(normalized_audio_path, self.project_root),
            source_sample_rate_hz=sample_rate_hz,
            source_num_channels=source_num_channels,
            source_duration_seconds=source_duration_seconds,
            normalized_duration_seconds=round(
                float(normalized.shape[-1]) / float(self.policy.target_sample_rate_hz),
                6,
            ),
            source_peak_amplitude=source_peak_value,
            source_rms_dbfs=loudness_decision.source_rms_dbfs,
            normalized_rms_dbfs=loudness_decision.output_rms_dbfs,
            source_dc_offset_ratio=source_dc_offset_value,
            source_clipped_sample_ratio=source_clipped_ratio,
            resampled=resampled,
            downmixed=downmixed,
            dc_offset_removed=dc_offset_removed,
            peak_scaled=peak_scaled,
            loudness_mode=loudness_decision.mode,
            loudness_applied=loudness_decision.applied,
            loudness_gain_db=loudness_decision.applied_gain_db,
            loudness_gain_clamped=loudness_decision.gain_clamped,
            loudness_peak_limited=loudness_decision.peak_limited,
            loudness_degradation_check_passed=loudness_decision.degradation_check_passed,
        )

    def _build_normalized_audio_path(self, source_path: Path) -> Path:
        if source_path.is_relative_to(self.dataset_root):
            relative_source = source_path.relative_to(self.dataset_root)
        elif source_path.is_relative_to(self.project_root):
            relative_source = source_path.relative_to(self.project_root)
        else:
            relative_source = Path(source_path.name)
        return (self.audio_output_root / relative_source).with_suffix(self.policy.output_suffix)
