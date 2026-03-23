"""Waveform inspection helpers for audio-quality EDA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kryptonite.data.audio_io import inspect_audio_file as inspect_audio_header
from kryptonite.data.audio_io import read_audio_file as decode_audio_file

from .constants import CLIPPING_PEAK_DBFS, SILENCE_CHUNK_MS, SILENCE_THRESHOLD_DBFS
from .models import AudioQualityInspection


@dataclass(frozen=True, slots=True)
class PCMChunkStats:
    sample_count: int
    sum_of_squares: float
    signed_sum: float
    peak_amplitude: int
    is_silent: bool
    is_clipped: bool


@dataclass(frozen=True, slots=True)
class PCMInspectionStats:
    sample_count: int
    sum_of_squares: float
    signed_sum: float
    peak_amplitude: int
    chunk_count: int
    silent_chunk_count: int
    clipped_chunk_count: int


@dataclass(frozen=True, slots=True)
class WaveformInspectionStats:
    rms_dbfs: float | None
    peak_dbfs: float | None
    silence_ratio: float | None
    dc_offset_ratio: float | None
    clipped_chunk_ratio: float | None


def inspect_audio_file(
    path: Path,
    cache: dict[Path, AudioQualityInspection],
) -> AudioQualityInspection:
    cached = cache.get(path)
    if cached is not None:
        return cached

    audio_format = audio_format_from_path(path.as_posix())
    if not path.exists():
        inspection = _empty_inspection(exists=False, audio_format=audio_format)
        cache[path] = inspection
        return inspection

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        inspection = _empty_inspection(
            exists=True,
            audio_format=audio_format,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    try:
        header_info = inspect_audio_header(path)
    except Exception as exc:
        inspection = _empty_inspection(
            exists=True,
            audio_format=audio_format,
            file_size_bytes=size_bytes,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    audio_format = header_info.format.lower()
    sample_width_bytes = _sample_width_bytes_from_subtype(header_info.subtype)
    try:
        waveform, _ = decode_audio_file(path)
    except Exception as exc:
        inspection = AudioQualityInspection(
            exists=True,
            file_size_bytes=size_bytes,
            duration_seconds=header_info.duration_seconds,
            sample_rate_hz=header_info.sample_rate_hz,
            channels=header_info.num_channels,
            audio_format=audio_format,
            sample_width_bytes=sample_width_bytes,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    if waveform.ndim != 2 or int(waveform.shape[-1]) == 0:
        inspection = AudioQualityInspection(
            exists=True,
            file_size_bytes=size_bytes,
            duration_seconds=header_info.duration_seconds,
            sample_rate_hz=header_info.sample_rate_hz,
            channels=header_info.num_channels,
            audio_format=audio_format,
            sample_width_bytes=sample_width_bytes,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
            error="ValueError: decoded audio signal is empty",
        )
        cache[path] = inspection
        return inspection

    if not np.isfinite(waveform).all():
        inspection = AudioQualityInspection(
            exists=True,
            file_size_bytes=size_bytes,
            duration_seconds=header_info.duration_seconds,
            sample_rate_hz=header_info.sample_rate_hz,
            channels=header_info.num_channels,
            audio_format=audio_format,
            sample_width_bytes=sample_width_bytes,
            rms_dbfs=None,
            peak_dbfs=None,
            silence_ratio=None,
            dc_offset_ratio=None,
            clipped_chunk_ratio=None,
            error="ValueError: decoded audio signal contains non-finite values",
        )
        cache[path] = inspection
        return inspection

    waveform_stats = inspect_decoded_waveform(
        waveform=waveform,
        sample_rate_hz=header_info.sample_rate_hz,
    )
    inspection = AudioQualityInspection(
        exists=True,
        file_size_bytes=size_bytes,
        duration_seconds=header_info.duration_seconds,
        sample_rate_hz=header_info.sample_rate_hz,
        channels=header_info.num_channels,
        audio_format=audio_format,
        sample_width_bytes=sample_width_bytes,
        rms_dbfs=waveform_stats.rms_dbfs,
        peak_dbfs=waveform_stats.peak_dbfs,
        silence_ratio=waveform_stats.silence_ratio,
        dc_offset_ratio=waveform_stats.dc_offset_ratio,
        clipped_chunk_ratio=waveform_stats.clipped_chunk_ratio,
    )
    cache[path] = inspection
    return inspection


def inspect_decoded_waveform(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
) -> WaveformInspectionStats:
    waveform64 = waveform.astype(np.float64, copy=False)
    rms_amplitude = float(np.sqrt(np.mean(np.square(waveform64))))
    peak_amplitude_value = float(np.abs(waveform64).max(initial=0.0))
    dc_offset_ratio = float(np.abs(waveform64.mean(axis=-1)).max(initial=0.0))

    chunk_frames = max(1, round(sample_rate_hz * SILENCE_CHUNK_MS / 1000))
    silence_threshold = _silence_threshold_amplitude(1)
    clipping_threshold = 10.0 ** (CLIPPING_PEAK_DBFS / 20.0)

    chunk_count = 0
    silent_chunk_count = 0
    clipped_chunk_count = 0
    for start in range(0, int(waveform64.shape[-1]), chunk_frames):
        stop = min(start + chunk_frames, int(waveform64.shape[-1]))
        chunk = waveform64[:, start:stop]
        if chunk.size == 0:
            continue
        chunk_count += 1
        chunk_rms = float(np.sqrt(np.mean(np.square(chunk))))
        chunk_peak = float(np.abs(chunk).max(initial=0.0))
        silent_chunk_count += int(chunk_rms == 0.0 or chunk_rms <= silence_threshold)
        clipped_chunk_count += int(chunk_peak >= clipping_threshold)

    return WaveformInspectionStats(
        rms_dbfs=amplitude_to_dbfs(rms_amplitude, 1),
        peak_dbfs=amplitude_to_dbfs(peak_amplitude_value, 1),
        silence_ratio=silent_chunk_count / chunk_count if chunk_count > 0 else None,
        dc_offset_ratio=dc_offset_ratio,
        clipped_chunk_ratio=clipped_chunk_count / chunk_count if chunk_count > 0 else None,
    )


def analyze_pcm_signal(
    *,
    frames: bytes,
    sample_width_bytes: int,
    chunk_sample_count: int,
    max_possible_amplitude: int,
) -> PCMInspectionStats:
    samples = pcm_samples(frames=frames, sample_width_bytes=sample_width_bytes)
    sample_count = int(samples.size)
    if sample_count == 0:
        return PCMInspectionStats(
            sample_count=0,
            sum_of_squares=0.0,
            signed_sum=0.0,
            peak_amplitude=0,
            chunk_count=0,
            silent_chunk_count=0,
            clipped_chunk_count=0,
        )

    if sample_width_bytes == 4:
        return _analyze_pcm_signal_fallback(
            samples=samples,
            chunk_sample_count=chunk_sample_count,
            max_possible_amplitude=max_possible_amplitude,
        )

    samples64 = samples.astype(np.int64, copy=False)
    chunk_starts = np.arange(0, sample_count, chunk_sample_count, dtype=np.int64)
    chunk_lengths = np.diff(np.append(chunk_starts, sample_count))
    chunk_sums = np.add.reduceat(samples64, chunk_starts)
    chunk_sum_of_squares = np.add.reduceat(samples64 * samples64, chunk_starts)
    chunk_peaks = np.maximum.reduceat(np.abs(samples64), chunk_starts)
    chunk_rms = np.fromiter(
        (
            math.isqrt(int(sum_of_squares // chunk_length))
            for sum_of_squares, chunk_length in zip(
                chunk_sum_of_squares.tolist(),
                chunk_lengths.tolist(),
                strict=True,
            )
        ),
        dtype=np.int64,
        count=int(chunk_starts.size),
    )
    chunk_averages = np.floor_divide(chunk_sums, chunk_lengths)
    chunk_rms_threshold = _silence_threshold_amplitude(max_possible_amplitude)
    silent_chunks = (chunk_rms == 0) | (chunk_rms.astype(np.float64) <= chunk_rms_threshold)
    clipped_chunks = chunk_peaks >= (max_possible_amplitude - 1)

    return PCMInspectionStats(
        sample_count=sample_count,
        sum_of_squares=float(np.dot(chunk_rms * chunk_rms, chunk_lengths)),
        signed_sum=float(np.dot(chunk_averages, chunk_lengths)),
        peak_amplitude=int(chunk_peaks.max()),
        chunk_count=int(chunk_starts.size),
        silent_chunk_count=int(silent_chunks.sum()),
        clipped_chunk_count=int(clipped_chunks.sum()),
    )


def analyze_pcm_chunk(
    *,
    frames: bytes,
    sample_width_bytes: int,
    max_possible_amplitude: int,
) -> PCMChunkStats | None:
    signal_stats = analyze_pcm_signal(
        frames=frames,
        sample_width_bytes=sample_width_bytes,
        chunk_sample_count=max(1, int(len(frames) / sample_width_bytes)),
        max_possible_amplitude=max_possible_amplitude,
    )
    if signal_stats.sample_count == 0:
        return None

    return PCMChunkStats(
        sample_count=signal_stats.sample_count,
        sum_of_squares=signal_stats.sum_of_squares,
        signed_sum=signal_stats.signed_sum,
        peak_amplitude=signal_stats.peak_amplitude,
        is_silent=signal_stats.silent_chunk_count > 0,
        is_clipped=signal_stats.clipped_chunk_count > 0,
    )


def pcm_samples(*, frames: bytes, sample_width_bytes: int) -> np.ndarray:
    if sample_width_bytes == 1:
        return np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128
    if sample_width_bytes == 2:
        return np.frombuffer(frames, dtype="<i2")
    if sample_width_bytes == 4:
        return np.frombuffer(frames, dtype="<i4")
    if sample_width_bytes == 3:
        return _decode_pcm_24bit_le(frames)
    raise ValueError(f"Unsupported PCM sample width: {sample_width_bytes}")


def _decode_pcm_24bit_le(frames: bytes) -> np.ndarray:
    raw = np.frombuffer(frames, dtype=np.uint8)
    usable_bytes = raw.size - (raw.size % 3)
    if usable_bytes == 0:
        return np.empty(0, dtype=np.int32)

    triplets = raw[:usable_bytes].reshape(-1, 3).astype(np.int32)
    values = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
    sign_mask = 1 << 23
    return (values ^ sign_mask) - sign_mask


def _analyze_pcm_signal_fallback(
    *,
    samples: np.ndarray,
    chunk_sample_count: int,
    max_possible_amplitude: int,
) -> PCMInspectionStats:
    peak_amplitude = 0
    total_sum_of_squares = 0.0
    total_signed_sum = 0.0
    chunk_count = 0
    silent_chunk_count = 0
    clipped_chunk_count = 0
    sample_count = int(samples.size)

    for start in range(0, sample_count, chunk_sample_count):
        stop = min(start + chunk_sample_count, sample_count)
        chunk_samples = samples[start:stop]
        if chunk_samples.size == 0:
            continue

        sample_values = [int(value) for value in chunk_samples.tolist()]
        signed_sum = sum(sample_values)
        sum_of_squares = int(math.sumprod(sample_values, sample_values))
        chunk_size = len(sample_values)
        chunk_peak = max(abs(min(sample_values)), abs(max(sample_values)))
        chunk_rms = math.isqrt(sum_of_squares // chunk_size)
        chunk_average = signed_sum // chunk_size
        chunk_rms_dbfs = amplitude_to_dbfs(float(chunk_rms), max_possible_amplitude)

        peak_amplitude = max(peak_amplitude, chunk_peak)
        total_sum_of_squares += float(chunk_rms * chunk_rms * chunk_size)
        total_signed_sum += float(chunk_average * chunk_size)
        chunk_count += 1
        silent_chunk_count += int(
            chunk_rms == 0
            or (chunk_rms_dbfs is not None and chunk_rms_dbfs <= SILENCE_THRESHOLD_DBFS)
        )
        clipped_chunk_count += int(chunk_peak >= max_possible_amplitude - 1)

    return PCMInspectionStats(
        sample_count=sample_count,
        sum_of_squares=total_sum_of_squares,
        signed_sum=total_signed_sum,
        peak_amplitude=peak_amplitude,
        chunk_count=chunk_count,
        silent_chunk_count=silent_chunk_count,
        clipped_chunk_count=clipped_chunk_count,
    )


def _silence_threshold_amplitude(max_possible_amplitude: int) -> float:
    return max_possible_amplitude * (10.0 ** (SILENCE_THRESHOLD_DBFS / 20.0))


def amplitude_to_dbfs(value: float, max_possible_amplitude: int) -> float | None:
    if value <= 0.0 or max_possible_amplitude <= 0:
        return None
    return 20.0 * math.log10(value / max_possible_amplitude)


def maximum_possible_amplitude(sample_width_bytes: int) -> int:
    return (1 << ((sample_width_bytes * 8) - 1)) - 1


def audio_format_from_path(path: str | None) -> str | None:
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    return suffix.removeprefix(".") or None


def _sample_width_bytes_from_subtype(subtype: str | None) -> int | None:
    if subtype is None:
        return None
    normalized = subtype.upper()
    if normalized.startswith("PCM_"):
        suffix = normalized.removeprefix("PCM_")
        if suffix.isdigit():
            bits_per_sample = int(suffix)
            if bits_per_sample > 0 and bits_per_sample % 8 == 0:
                return bits_per_sample // 8
    if normalized in {"FLOAT", "PCM_FLOAT"}:
        return 4
    if normalized == "DOUBLE":
        return 8
    return None


def _empty_inspection(
    *,
    exists: bool,
    audio_format: str | None,
    file_size_bytes: int | None = None,
    error: str | None = None,
) -> AudioQualityInspection:
    return AudioQualityInspection(
        exists=exists,
        file_size_bytes=file_size_bytes,
        duration_seconds=None,
        sample_rate_hz=None,
        channels=None,
        audio_format=audio_format,
        sample_width_bytes=None,
        rms_dbfs=None,
        peak_dbfs=None,
        silence_ratio=None,
        dc_offset_ratio=None,
        clipped_chunk_ratio=None,
        error=error,
    )
