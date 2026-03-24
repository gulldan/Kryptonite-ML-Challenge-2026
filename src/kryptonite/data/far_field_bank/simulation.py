"""Signal generation and rendering helpers for far-field distance simulation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.data.audio_io import read_audio_file, write_audio_file
from kryptonite.data.convolution import fft_convolve_1d

from .models import (
    FarFieldAudioMetrics,
    FarFieldKernelMetrics,
    FarFieldProbeSettings,
    FarFieldRenderSettings,
    FarFieldSimulationPreset,
)


@dataclass(frozen=True, slots=True)
class RenderedFarFieldPreset:
    preview_waveform: np.ndarray
    kernel_waveform: np.ndarray
    output_metrics: FarFieldAudioMetrics
    kernel_metrics: FarFieldKernelMetrics


def generate_probe_waveform(settings: FarFieldProbeSettings) -> np.ndarray:
    frame_count = int(round(settings.sample_rate_hz * settings.duration_seconds))
    timeline = np.arange(frame_count, dtype=np.float64) / float(settings.sample_rate_hz)

    voiced = _voiced_component(timeline, sample_rate_hz=settings.sample_rate_hz)
    chirp = _chirp_component(
        timeline,
        start_hz=120.0,
        end_hz=min(7_200.0, 0.45 * settings.sample_rate_hz),
        duration_seconds=settings.duration_seconds,
    )
    transients = _transient_component(timeline, sample_rate_hz=settings.sample_rate_hz)

    envelope = np.minimum(
        1.0, np.sin(np.pi * np.clip(timeline / settings.duration_seconds, 0.0, 1.0))
    )
    waveform = (0.55 * voiced + 0.3 * chirp + 0.15 * transients) * envelope
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = waveform * (settings.peak_amplitude / peak)
    return waveform[np.newaxis, :].astype("float32")


def analyze_waveform_metrics(waveform: Any, *, sample_rate_hz: int) -> FarFieldAudioMetrics:
    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("Expected channel-first waveform with non-empty frames.")

    mono = array.mean(axis=0)
    duration_seconds = round(float(mono.shape[0]) / float(sample_rate_hz), 6)
    peak_amplitude = round(float(np.max(np.abs(mono))), 6)
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    rms_dbfs = round(20.0 * math.log10(max(rms, 1e-12)), 6)
    clipped_sample_ratio = round(float(np.mean(np.abs(mono) >= 0.98)), 6)

    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    frequencies = np.fft.rfftfreq(mono.shape[0], d=1.0 / float(sample_rate_hz))
    spectral_centroid_hz = 0.0
    spectral_rolloff_95_hz = 0.0
    total_energy = float(spectrum.sum())
    if total_energy > 0.0:
        spectral_centroid_hz = round(
            float(np.sum(frequencies * spectrum) / total_energy),
            6,
        )
        cumulative_energy = np.cumsum(spectrum)
        threshold_index = int(np.searchsorted(cumulative_energy, total_energy * 0.95))
        threshold_index = min(threshold_index, len(frequencies) - 1)
        spectral_rolloff_95_hz = round(float(frequencies[threshold_index]), 6)

    return FarFieldAudioMetrics(
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        num_channels=int(array.shape[0]),
        peak_amplitude=peak_amplitude,
        rms_dbfs=rms_dbfs,
        clipped_sample_ratio=clipped_sample_ratio,
        spectral_centroid_hz=spectral_centroid_hz,
        spectral_rolloff_95_hz=spectral_rolloff_95_hz,
    )


def analyze_audio_file(path: Path) -> FarFieldAudioMetrics:
    waveform, info = read_audio_file(path)
    return analyze_waveform_metrics(waveform, sample_rate_hz=info.sample_rate_hz)


def write_probe_audio(path: Path, *, settings: FarFieldProbeSettings) -> FarFieldAudioMetrics:
    waveform = generate_probe_waveform(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_audio_file(
        path=path,
        waveform=waveform,
        sample_rate_hz=settings.sample_rate_hz,
        output_format="wav",
        pcm_bits_per_sample=16,
    )
    return analyze_waveform_metrics(waveform, sample_rate_hz=settings.sample_rate_hz)


def render_far_field_preset(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    preset: FarFieldSimulationPreset,
    render_settings: FarFieldRenderSettings,
) -> RenderedFarFieldPreset:
    mono = np.asarray(waveform, dtype=np.float64).mean(axis=0)
    kernel, kernel_metrics = _build_impulse_response(
        sample_rate_hz=sample_rate_hz,
        preset=preset,
        render_settings=render_settings,
    )

    preview = fft_convolve_1d(mono, kernel)
    preview = _apply_frequency_shaping(
        preview,
        sample_rate_hz=sample_rate_hz,
        lowpass_hz=preset.lowpass_hz,
        high_shelf_db=_effective_high_shelf_db(preset),
        shelf_pivot_hz=render_settings.high_shelf_pivot_hz,
    )
    preview = _limit_peak(preview, peak_limit=render_settings.output_peak_limit)

    kernel_audio = _normalize_for_preview(kernel)
    preview_waveform = preview[np.newaxis, :].astype("float32")
    output_metrics = analyze_waveform_metrics(preview_waveform, sample_rate_hz=sample_rate_hz)
    return RenderedFarFieldPreset(
        preview_waveform=preview_waveform,
        kernel_waveform=kernel_audio[np.newaxis, :].astype("float32"),
        output_metrics=output_metrics,
        kernel_metrics=kernel_metrics,
    )


def _build_impulse_response(
    *,
    sample_rate_hz: int,
    preset: FarFieldSimulationPreset,
    render_settings: FarFieldRenderSettings,
) -> tuple[np.ndarray, FarFieldKernelMetrics]:
    frame_count = int(round(render_settings.kernel_duration_seconds * sample_rate_hz))
    frame_count = max(frame_count, 2)
    kernel = np.zeros(frame_count, dtype=np.float64)
    direct_gain = _db_to_amplitude(-preset.attenuation_db)

    arrival_delay_ms = round(
        1000.0 * preset.distance_meters / render_settings.speed_of_sound_mps,
        6,
    )
    arrival_index = min(frame_count - 1, int(round(arrival_delay_ms * sample_rate_hz / 1000.0)))
    kernel[arrival_index] = direct_gain

    late_start_index = arrival_index + int(
        round(preset.late_reverb_start_ms * sample_rate_hz / 1000.0)
    )
    late_start_index = min(max(late_start_index, arrival_index + 1), frame_count - 1)

    reverb_components = np.zeros_like(kernel)
    for delay_ms, gain_db in zip(
        preset.early_reflection_delays_ms,
        preset.early_reflection_gains_db,
        strict=True,
    ):
        reflection_index = arrival_index + int(round(delay_ms * sample_rate_hz / 1000.0))
        if reflection_index >= frame_count:
            continue
        reverb_components[reflection_index] += _db_to_amplitude(gain_db)

    tail_frame_count = frame_count - late_start_index
    if tail_frame_count > 0:
        rng = np.random.default_rng(_stable_seed(preset.id))
        timeline = np.arange(tail_frame_count, dtype=np.float64) / float(sample_rate_hz)
        tau = preset.reverb_rt60_seconds / math.log(1000.0)
        decay = np.exp(-timeline / tau)
        tail = rng.normal(size=tail_frame_count) * decay
        tail = _apply_frequency_shaping(
            tail,
            sample_rate_hz=sample_rate_hz,
            lowpass_hz=max(350.0, min(preset.lowpass_hz, 0.9 * preset.lowpass_hz)),
            high_shelf_db=min(-1.0, preset.high_shelf_db - 1.5),
            shelf_pivot_hz=1_400.0,
        )
        reverb_components[late_start_index:] += tail

    target_reverb_energy = (direct_gain * direct_gain) / (10.0 ** (preset.target_drr_db / 10.0))
    current_reverb_energy = float(np.square(reverb_components, dtype=np.float64).sum())
    if current_reverb_energy <= 0.0:
        raise ValueError(f"Far-field preset {preset.id!r} generated zero reverberant energy.")
    reverb_components *= math.sqrt(target_reverb_energy / current_reverb_energy)
    kernel += reverb_components

    actual_reverb_energy = float(np.square(reverb_components, dtype=np.float64).sum())
    actual_drr_db = round(
        10.0 * math.log10(max((direct_gain * direct_gain) / actual_reverb_energy, 1e-12)),
        6,
    )
    kernel_metrics = FarFieldKernelMetrics(
        arrival_delay_ms=arrival_delay_ms,
        actual_drr_db=actual_drr_db,
        late_reverb_start_ms=round(
            float(late_start_index - arrival_index) * 1000.0 / float(sample_rate_hz),
            6,
        ),
        reflection_count=preset.reflection_count,
        kernel_duration_ms=round(float(frame_count) * 1000.0 / float(sample_rate_hz), 6),
    )
    return kernel, kernel_metrics


def _apply_frequency_shaping(
    signal: np.ndarray,
    *,
    sample_rate_hz: int,
    lowpass_hz: float,
    high_shelf_db: float,
    shelf_pivot_hz: float,
) -> np.ndarray:
    if signal.size == 0:
        return signal
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.shape[0], d=1.0 / float(sample_rate_hz))

    lowpass_magnitude = 1.0 / np.sqrt(1.0 + np.power(frequencies / max(lowpass_hz, 1.0), 8.0))
    if high_shelf_db == 0.0:
        shelf_magnitude = np.ones_like(lowpass_magnitude)
    else:
        shelf_gain = _db_to_amplitude(high_shelf_db)
        transition = np.clip(
            (frequencies - shelf_pivot_hz) / max(float(sample_rate_hz) / 2.0 - shelf_pivot_hz, 1.0),
            0.0,
            1.0,
        )
        shelf_magnitude = 1.0 + (shelf_gain - 1.0) * np.power(transition, 0.75)

    shaped = np.fft.irfft(spectrum * lowpass_magnitude * shelf_magnitude, n=signal.shape[0])
    return shaped.astype(np.float64, copy=False)


def _normalize_for_preview(kernel: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(kernel)))
    if peak <= 0.0:
        return kernel.astype(np.float64, copy=True)
    return (kernel / peak) * 0.92


def _limit_peak(signal: np.ndarray, *, peak_limit: float) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak <= peak_limit or peak <= 0.0:
        return signal
    return signal * (peak_limit / peak)


def _effective_high_shelf_db(preset: FarFieldSimulationPreset) -> float:
    angle_component_db = -6.0 * (preset.off_axis_angle_deg / 90.0)
    return preset.high_shelf_db + angle_component_db


def _db_to_amplitude(value_db: float) -> float:
    return float(10.0 ** (value_db / 20.0))


def _stable_seed(identifier: str) -> int:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _voiced_component(timeline: np.ndarray, *, sample_rate_hz: int) -> np.ndarray:
    fundamental_hz = 110.0 + 18.0 * np.sin(2.0 * np.pi * 0.9 * timeline)
    phase = 2.0 * np.pi * np.cumsum(fundamental_hz) / float(sample_rate_hz)
    harmonic_weights = np.array([1.0, 0.5, 0.33, 0.25, 0.2], dtype=np.float64)
    harmonics = [
        weight * np.sin((index + 1) * phase + 0.1 * index)
        for index, weight in enumerate(harmonic_weights)
    ]
    modulation = 0.55 + 0.45 * np.sin(2.0 * np.pi * 2.1 * timeline) ** 2
    return np.sum(harmonics, axis=0) * modulation


def _chirp_component(
    timeline: np.ndarray,
    *,
    start_hz: float,
    end_hz: float,
    duration_seconds: float,
) -> np.ndarray:
    sweep = (end_hz - start_hz) / max(duration_seconds, 1e-12)
    phase = 2.0 * np.pi * (start_hz * timeline + 0.5 * sweep * np.square(timeline))
    return np.sin(phase)


def _transient_component(timeline: np.ndarray, *, sample_rate_hz: int) -> np.ndarray:
    waveform = np.zeros_like(timeline)
    event_positions = (0.22, 0.61, 1.11, 1.47)
    decay_samples = max(1, int(round(0.035 * sample_rate_hz)))
    envelope = np.exp(-np.arange(decay_samples, dtype=np.float64) / (0.009 * sample_rate_hz))
    for event_seconds in event_positions:
        start = int(round(event_seconds * sample_rate_hz))
        end = min(start + decay_samples, waveform.shape[0])
        if start >= waveform.shape[0]:
            continue
        waveform[start:end] += envelope[: end - start] * np.sin(
            2.0 * np.pi * 2_400.0 * timeline[start:end]
        )
    return waveform


__all__ = [
    "RenderedFarFieldPreset",
    "analyze_audio_file",
    "analyze_waveform_metrics",
    "generate_probe_waveform",
    "render_far_field_preset",
    "write_probe_audio",
]
