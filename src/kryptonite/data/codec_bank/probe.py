"""Synthetic probe generation and preview metrics for codec-bank reporting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.data.audio_io import read_audio_file, write_audio_file

from .models import CodecProbeSettings, ProbeAudioMetrics


def generate_probe_waveform(settings: CodecProbeSettings) -> np.ndarray:
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


def analyze_waveform_metrics(waveform: Any, *, sample_rate_hz: int) -> ProbeAudioMetrics:
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

    return ProbeAudioMetrics(
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        num_channels=int(array.shape[0]),
        peak_amplitude=peak_amplitude,
        rms_dbfs=rms_dbfs,
        clipped_sample_ratio=clipped_sample_ratio,
        spectral_centroid_hz=spectral_centroid_hz,
        spectral_rolloff_95_hz=spectral_rolloff_95_hz,
    )


def analyze_audio_file(path: Path) -> ProbeAudioMetrics:
    waveform, info = read_audio_file(path)
    return analyze_waveform_metrics(waveform, sample_rate_hz=info.sample_rate_hz)


def write_probe_audio(path: Path, *, settings: CodecProbeSettings) -> ProbeAudioMetrics:
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
            2.0 * np.pi * 2400.0 * timeline[start:end]
        )
    return waveform


__all__ = [
    "analyze_audio_file",
    "analyze_waveform_metrics",
    "generate_probe_waveform",
    "write_probe_audio",
]
