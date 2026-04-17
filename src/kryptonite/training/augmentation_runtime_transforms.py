"""Waveform transform helpers for scheduled training augmentations."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kryptonite.config import SilenceAugmentationConfig
from kryptonite.data.audio_io import read_audio_file
from kryptonite.data.convolution import fft_convolve_1d

from .augmentation_runtime_catalog import _coerce_float, _coerce_str


def _apply_noise(
    waveform: NDArray[np.float32],
    *,
    project_root: Path,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    del sample_rate_hz
    noise_path = _metadata_str(metadata, "normalized_audio_path")
    if not noise_path:
        raise ValueError("missing_noise_audio_path")
    noise = _read_mono_audio(project_root / noise_path)
    noise_segment = _sample_or_tile(noise, target_samples=int(waveform.shape[-1]), rng=rng)
    signal_rms = _rms(waveform)
    noise_rms = _rms(noise_segment)
    if signal_rms <= 1e-8 or noise_rms <= 1e-8:
        return waveform, {"skip_reason": "zero_rms"}
    snr_min = _metadata_float(metadata, "snr_db_min", 8.0)
    snr_max = _metadata_float(metadata, "snr_db_max", 18.0)
    target_snr_db = rng.uniform(min(snr_min, snr_max), max(snr_min, snr_max))
    target_noise_rms = signal_rms / (10.0 ** (target_snr_db / 20.0))
    mixed = waveform + (noise_segment.reshape(1, -1) * (target_noise_rms / noise_rms))
    return mixed.astype(np.float32, copy=False), {
        "target_snr_db": round(float(target_snr_db), 4),
        "noise_path": noise_path,
        "category": metadata.get("category", ""),
    }


def _apply_reverb(
    waveform: NDArray[np.float32],
    *,
    project_root: Path,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    del sample_rate_hz
    raw_paths = metadata.get("rir_audio_paths", ())
    paths = (
        tuple(str(path) for path in raw_paths if str(path))
        if isinstance(raw_paths, tuple | list)
        else ()
    )
    if not paths:
        return waveform, {"skip_reason": "empty_rir_paths"}
    rir_path = rng.choice(paths)
    rir = _read_mono_audio(project_root / rir_path)
    if rir.size == 0:
        return waveform, {"skip_reason": "empty_rir"}
    peak = max(float(np.max(np.abs(rir))), 1e-8)
    kernel = rir / peak
    convolved = _convolve_channels(waveform, kernel, output_samples=int(waveform.shape[-1]))
    convolved = _match_rms(convolved, reference=waveform)
    return convolved, {
        "rir_path": rir_path,
        "direct_condition": metadata.get("direct_condition", ""),
        "rt60_bucket": metadata.get("rt60_bucket", ""),
    }


def _apply_distance(
    waveform: NDArray[np.float32],
    *,
    project_root: Path,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    del rng
    transformed = waveform
    kernel_path = _metadata_str(metadata, "kernel_audio_path")
    if kernel_path:
        kernel = _read_mono_audio(project_root / kernel_path)
        if kernel.size:
            peak = max(float(np.max(np.abs(kernel))), 1e-8)
            transformed = _convolve_channels(
                transformed,
                kernel / peak,
                output_samples=int(waveform.shape[-1]),
            )
    lowpass_hz = min(_metadata_float(metadata, "lowpass_hz", 5600.0), sample_rate_hz / 2.0 - 50.0)
    transformed = _bandpass_fft(
        transformed,
        sample_rate_hz=sample_rate_hz,
        low_hz=80.0,
        high_hz=lowpass_hz,
    )
    attenuation_db = _metadata_float(metadata, "attenuation_db", 3.0)
    transformed = transformed * (10.0 ** (-attenuation_db / 20.0))
    return _match_rms(transformed, reference=waveform, max_gain_db=3.0), {
        "field": metadata.get("field", ""),
        "lowpass_hz": round(float(lowpass_hz), 3),
        "attenuation_db": round(float(attenuation_db), 3),
        "kernel_audio_path": kernel_path,
    }


def _apply_codec(
    waveform: NDArray[np.float32],
    *,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    family = _metadata_str(metadata, "codec_family") or "codec"
    severity = _metadata_str(metadata, "severity") or ""
    low_hz, high_hz = _codec_band(family=family, severity=severity)
    filtered = _bandpass_fft(
        waveform,
        sample_rate_hz=sample_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    filtered = _random_eq(filtered, sample_rate_hz=sample_rate_hz, rng=rng, family=family)
    if family in {"telephony", "compression"} or severity == "heavy":
        bits = 8 if severity == "heavy" else 10
        filtered = _bit_crush(filtered, bits=bits)
    if severity == "heavy":
        filtered = np.tanh(filtered * 1.25).astype(np.float32, copy=False)
        filtered = _apply_packet_loss(filtered, sample_rate_hz=sample_rate_hz, rng=rng)
    return filtered, {
        "codec_family": family,
        "codec_name": metadata.get("codec_name", ""),
        "low_hz": low_hz,
        "high_hz": high_hz,
        "simulation": "fft_filter_eq_quantize_packet_loss",
    }


def _apply_silence(
    waveform: NDArray[np.float32],
    *,
    base_config: SilenceAugmentationConfig,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    scale = _metadata_float(metadata, "scale", 1.0)
    leading = _sample_seconds(base_config.max_leading_padding_seconds * scale, rng=rng)
    trailing = _sample_seconds(base_config.max_trailing_padding_seconds * scale, rng=rng)
    leading_frames = int(round(leading * sample_rate_hz))
    trailing_frames = int(round(trailing * sample_rate_hz))
    pieces: list[NDArray[np.float32]] = []
    if leading_frames > 0:
        pieces.append(np.zeros((waveform.shape[0], leading_frames), dtype=np.float32))
    body = waveform
    inserted = 0
    dropped = 0
    max_pauses = max(0, int(round(base_config.max_inserted_pauses * scale)))
    if max_pauses > 0 and body.shape[-1] > sample_rate_hz:
        body, inserted = _insert_random_pauses(
            body,
            sample_rate_hz=sample_rate_hz,
            max_pauses=max_pauses,
            min_seconds=base_config.min_inserted_pause_seconds,
            max_seconds=base_config.max_inserted_pause_seconds * max(scale, 0.25),
            rng=rng,
        )
    if scale >= 0.7:
        body, dropped = _vad_drop(
            body,
            sample_rate_hz=sample_rate_hz,
            max_drops=1 if scale < 1.0 else 2,
            rng=rng,
        )
    pieces.append(body)
    if trailing_frames > 0:
        pieces.append(np.zeros((waveform.shape[0], trailing_frames), dtype=np.float32))
    return np.concatenate(pieces, axis=-1), {
        "leading_padding_seconds": round(float(leading), 4),
        "trailing_padding_seconds": round(float(trailing), 4),
        "inserted_pause_count": inserted,
        "vad_drop_count": dropped,
    }


def _apply_speed(
    waveform: NDArray[np.float32],
    *,
    metadata: dict[str, object],
) -> tuple[NDArray[np.float32], dict[str, object]]:
    factor = _metadata_float(metadata, "speed_factor", 1.0)
    if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-6):
        return waveform, {"speed_factor": 1.0}
    source_samples = int(waveform.shape[-1])
    target_samples = max(1, int(round(source_samples / factor)))
    source_positions = np.linspace(0.0, 1.0, num=source_samples, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_samples, dtype=np.float32)
    channels = [
        np.interp(target_positions, source_positions, channel).astype(np.float32, copy=False)
        for channel in waveform
    ]
    return np.stack(channels, axis=0), {
        "speed_factor": round(float(factor), 4),
        "source_samples": source_samples,
        "target_samples": target_samples,
    }


def _coerce_audio(waveform: Any) -> NDArray[np.float32]:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[-1] == 0:
        raise ValueError("augmentation waveform must be shaped [channels, samples]")
    return array


def _read_mono_audio(path: Path) -> NDArray[np.float32]:
    waveform, _ = read_audio_file(path)
    array = _coerce_audio(waveform)
    return array.mean(axis=0, dtype=np.float32)


def _sample_or_tile(
    waveform: NDArray[np.float32],
    *,
    target_samples: int,
    rng: random.Random,
) -> NDArray[np.float32]:
    if waveform.size >= target_samples:
        max_start = waveform.size - target_samples
        start = 0 if max_start <= 0 else rng.randint(0, max_start)
        return waveform[start : start + target_samples]
    repeats = int(math.ceil(target_samples / max(waveform.size, 1)))
    return np.tile(waveform, repeats)[:target_samples].astype(np.float32, copy=False)


def _convolve_channels(
    waveform: NDArray[np.float32],
    kernel: NDArray[np.float32],
    *,
    output_samples: int,
) -> NDArray[np.float32]:
    convolved = [fft_convolve_1d(channel, kernel)[:output_samples] for channel in waveform]
    return np.stack(convolved, axis=0).astype(np.float32, copy=False)


def _match_rms(
    waveform: NDArray[np.float32],
    *,
    reference: NDArray[np.float32],
    max_gain_db: float = 6.0,
) -> NDArray[np.float32]:
    source = _rms(waveform)
    target = _rms(reference)
    if source <= 1e-8 or target <= 1e-8:
        return waveform.astype(np.float32, copy=False)
    gain = min(target / source, 10.0 ** (max_gain_db / 20.0))
    return (waveform * gain).astype(np.float32, copy=False)


def _bandpass_fft(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    low_hz: float,
    high_hz: float,
) -> NDArray[np.float32]:
    sample_count = int(waveform.shape[-1])
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / float(sample_rate_hz))
    mask = (freqs >= max(0.0, low_hz)) & (freqs <= min(high_hz, sample_rate_hz / 2.0))
    spectrum = np.fft.rfft(waveform.astype(np.float64), axis=-1)
    spectrum *= mask.reshape(1, -1)
    return np.fft.irfft(spectrum, n=sample_count, axis=-1).astype(np.float32, copy=False)


def _random_eq(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    rng: random.Random,
    family: str,
) -> NDArray[np.float32]:
    sample_count = int(waveform.shape[-1])
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / float(sample_rate_hz))
    max_gain = 5.0 if family in {"channel", "telephony"} else 2.5
    anchor_freqs = np.asarray([0.0, 300.0, 1000.0, 2500.0, 5000.0, sample_rate_hz / 2.0])
    anchor_db = np.asarray([rng.uniform(-max_gain, max_gain) for _ in anchor_freqs])
    gains = 10.0 ** (np.interp(freqs, anchor_freqs, anchor_db) / 20.0)
    spectrum = np.fft.rfft(waveform.astype(np.float64), axis=-1)
    spectrum *= gains.reshape(1, -1)
    return np.fft.irfft(spectrum, n=sample_count, axis=-1).astype(np.float32, copy=False)


def _bit_crush(waveform: NDArray[np.float32], *, bits: int) -> NDArray[np.float32]:
    levels = float(2 ** max(2, bits - 1))
    return (np.round(np.clip(waveform, -1.0, 1.0) * levels) / levels).astype(
        np.float32,
        copy=False,
    )


def _apply_packet_loss(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    rng: random.Random,
) -> NDArray[np.float32]:
    output = waveform.copy()
    drop_count = rng.randint(1, 3)
    total = int(output.shape[-1])
    for _ in range(drop_count):
        span = rng.randint(int(0.02 * sample_rate_hz), int(0.08 * sample_rate_hz))
        if total <= span:
            continue
        start = rng.randint(0, total - span)
        output[:, start : start + span] *= rng.uniform(0.0, 0.15)
    return output


def _insert_random_pauses(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    max_pauses: int,
    min_seconds: float,
    max_seconds: float,
    rng: random.Random,
) -> tuple[NDArray[np.float32], int]:
    total = int(waveform.shape[-1])
    pause_count = rng.randint(0, max_pauses)
    if pause_count == 0:
        return waveform, 0
    points = sorted(rng.randint(0, total) for _ in range(pause_count))
    pieces: list[NDArray[np.float32]] = []
    cursor = 0
    inserted = 0
    for point in points:
        pieces.append(waveform[:, cursor:point])
        pause_seconds = rng.uniform(min_seconds, max(max_seconds, min_seconds))
        pause_frames = max(1, int(round(pause_seconds * sample_rate_hz)))
        pieces.append(np.zeros((waveform.shape[0], pause_frames), dtype=np.float32))
        cursor = point
        inserted += 1
    pieces.append(waveform[:, cursor:])
    return np.concatenate(pieces, axis=-1), inserted


def _vad_drop(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    max_drops: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], int]:
    output = waveform.copy()
    total = int(output.shape[-1])
    drop_count = rng.randint(0, max_drops)
    for _ in range(drop_count):
        span = rng.randint(int(0.06 * sample_rate_hz), int(0.18 * sample_rate_hz))
        if total <= span:
            continue
        start = rng.randint(0, total - span)
        output[:, start : start + span] = 0.0
    return output, drop_count


def _codec_band(*, family: str, severity: str) -> tuple[float, float]:
    if family == "telephony":
        return 300.0, 3400.0 if severity != "heavy" else 3200.0
    if family == "compression":
        return 180.0, 5400.0
    if family == "channel":
        return 90.0, 7600.0 if severity != "heavy" else 5000.0
    if family == "voip":
        return 140.0, 6800.0
    return 120.0, 7200.0


def _rms(waveform: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32), dtype=np.float32)))


def _sample_seconds(max_seconds: float, *, rng: random.Random) -> float:
    if max_seconds <= 0.0:
        return 0.0
    return rng.uniform(0.0, max_seconds)


def _metadata_str(metadata: dict[str, object], key: str) -> str:
    return _coerce_str(metadata.get(key))


def _metadata_float(metadata: dict[str, object], key: str, default: float) -> float:
    return _coerce_float(metadata.get(key), default)
