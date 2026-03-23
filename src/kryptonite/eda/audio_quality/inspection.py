"""Waveform inspection helpers for audio-quality EDA."""

from __future__ import annotations

import math
import sys
import wave
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .constants import SILENCE_CHUNK_MS, SILENCE_THRESHOLD_DBFS
from .models import AudioQualityInspection

_NATIVE_LITTLE_ENDIAN = sys.byteorder == "little"

@dataclass(frozen=True, slots=True)
class PCMChunkStats:
    sample_count: int
    sum_of_squares: float
    signed_sum: float
    peak_amplitude: int
    is_silent: bool
    is_clipped: bool


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

    if path.suffix.lower() != ".wav":
        inspection = _empty_inspection(
            exists=True,
            audio_format=audio_format,
            file_size_bytes=size_bytes,
        )
        cache[path] = inspection
        return inspection

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate_hz = handle.getframerate()
            channels = handle.getnchannels()
            sample_width_bytes = handle.getsampwidth()
            frame_count = handle.getnframes()

            chunk_frames = max(1, round(sample_rate_hz * SILENCE_CHUNK_MS / 1000))
            peak_max = 0
            total_samples = 0
            total_sum_of_squares = 0.0
            total_signed_sum = 0.0
            chunk_count = 0
            silent_chunks = 0
            clipped_chunks = 0
            max_amplitude = maximum_possible_amplitude(sample_width_bytes)

            while True:
                frames = handle.readframes(chunk_frames)
                if not frames:
                    break
                chunk_stats = analyze_pcm_chunk(
                    frames=frames,
                    sample_width_bytes=sample_width_bytes,
                    max_possible_amplitude=max_amplitude,
                )
                if chunk_stats is None:
                    continue

                chunk_count += 1
                peak_max = max(peak_max, chunk_stats.peak_amplitude)
                total_samples += chunk_stats.sample_count
                total_sum_of_squares += chunk_stats.sum_of_squares
                total_signed_sum += chunk_stats.signed_sum
                silent_chunks += int(chunk_stats.is_silent)
                clipped_chunks += int(chunk_stats.is_clipped)
    except (OSError, ValueError, wave.Error) as exc:
        inspection = _empty_inspection(
            exists=True,
            audio_format=audio_format,
            file_size_bytes=size_bytes,
            error=f"{type(exc).__name__}: {exc}",
        )
        cache[path] = inspection
        return inspection

    duration_seconds = frame_count / sample_rate_hz if sample_rate_hz > 0 else None
    if total_samples > 0:
        rms_amplitude = math.sqrt(total_sum_of_squares / total_samples)
        dc_offset_ratio = abs(total_signed_sum / total_samples) / max_amplitude
        rms_dbfs = amplitude_to_dbfs(rms_amplitude, max_amplitude)
        peak_dbfs = amplitude_to_dbfs(float(peak_max), max_amplitude)
    else:
        dc_offset_ratio = None
        rms_dbfs = None
        peak_dbfs = None

    silence_ratio = silent_chunks / chunk_count if chunk_count > 0 else None
    clipped_chunk_ratio = clipped_chunks / chunk_count if chunk_count > 0 else None
    inspection = AudioQualityInspection(
        exists=True,
        file_size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        audio_format=audio_format,
        sample_width_bytes=sample_width_bytes,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        silence_ratio=silence_ratio,
        dc_offset_ratio=dc_offset_ratio,
        clipped_chunk_ratio=clipped_chunk_ratio,
    )
    cache[path] = inspection
    return inspection


def analyze_pcm_chunk(
    *,
    frames: bytes,
    sample_width_bytes: int,
    max_possible_amplitude: int,
) -> PCMChunkStats | None:
    samples = pcm_samples(frames=frames, sample_width_bytes=sample_width_bytes)
    sample_count = len(samples)
    if sample_count == 0:
        return None

    sum_of_squares = math.sumprod(samples, samples)
    signed_sum = float(sum(samples))
    peak_amplitude = max(abs(min(samples)), abs(max(samples)))
    rms_amplitude = math.sqrt(sum_of_squares / sample_count)
    chunk_rms_dbfs = amplitude_to_dbfs(rms_amplitude, max_possible_amplitude)
    is_silent = rms_amplitude == 0 or (
        chunk_rms_dbfs is not None and chunk_rms_dbfs <= SILENCE_THRESHOLD_DBFS
    )
    is_clipped = peak_amplitude >= max_possible_amplitude - 1
    return PCMChunkStats(
        sample_count=sample_count,
        sum_of_squares=float(sum_of_squares),
        signed_sum=signed_sum,
        peak_amplitude=peak_amplitude,
        is_silent=is_silent,
        is_clipped=is_clipped,
    )


def pcm_samples(*, frames: bytes, sample_width_bytes: int) -> Sequence[int]:
    if sample_width_bytes == 1:
        return [value - 128 for value in frames]
    if sample_width_bytes == 2 and _NATIVE_LITTLE_ENDIAN:
        return memoryview(frames).cast("h")
    if sample_width_bytes == 4 and _NATIVE_LITTLE_ENDIAN:
        return memoryview(frames).cast("i")
    if sample_width_bytes == 2:
        samples = array("h")
        samples.frombytes(frames)
        samples.byteswap()
        return samples
    if sample_width_bytes == 4:
        samples = array("i")
        samples.frombytes(frames)
        samples.byteswap()
        return samples
    if sample_width_bytes == 3:
        return _decode_pcm_24bit_le(frames)
    raise ValueError(f"Unsupported PCM sample width: {sample_width_bytes}")


def _decode_pcm_24bit_le(frames: bytes) -> list[int]:
    samples: list[int] = []
    for index in range(0, len(frames), 3):
        chunk = frames[index : index + 3]
        if len(chunk) < 3:
            continue
        value = chunk[0] | (chunk[1] << 8) | (chunk[2] << 16)
        if value & 0x800000:
            value -= 1 << 24
        samples.append(value)
    return samples


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
