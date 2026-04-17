"""Offline audio quality profiling for EDA artifacts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.data.audio_io import inspect_audio_file, read_audio_file

EPSILON = 1e-12
SILENCE_THRESHOLD_DBFS = -40.0


def compute_audio_stats_table(
    manifest: pl.DataFrame,
    *,
    analysis_seconds: float | None = 30.0,
    workers: int = 1,
) -> pl.DataFrame:
    """Compute one audio-quality row per manifest row.

    The function reads waveform data only inside this offline batch step. Dashboard code should
    consume the resulting Parquet table and open audio only for selected examples.
    """

    jobs = [
        {
            "row_index": int(row["row_index"]),
            "split": str(row["split"]),
            "speaker_id": row["speaker_id"],
            "filepath": str(row["filepath"]),
            "resolved_path": str(row["resolved_path"]),
            "analysis_seconds": analysis_seconds,
        }
        for row in manifest.iter_rows(named=True)
    ]
    if workers <= 1:
        rows = [_compute_one_audio_stats(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_compute_one_audio_stats, jobs))
    return pl.DataFrame(rows, infer_schema_length=None)


def _compute_one_audio_stats(job: dict[str, Any]) -> dict[str, Any]:
    row = {
        "row_index": job["row_index"],
        "split": job["split"],
        "speaker_id": job["speaker_id"],
        "filepath": job["filepath"],
        "resolved_path": job["resolved_path"],
        "exists": False,
        "error": None,
        "format": None,
        "subtype": None,
        "sample_rate_hz": None,
        "num_channels": None,
        "frame_count": None,
        "duration_s": None,
        "analysis_duration_s": None,
        "rms_dbfs": None,
        "peak_dbfs": None,
        "crest_factor_db": None,
        "clipping_frac": None,
        "silence_ratio_40db": None,
        "leading_silence_s": None,
        "trailing_silence_s": None,
        "zcr": None,
        "spectral_centroid_hz": None,
        "rolloff95_hz": None,
        "spectral_bandwidth_hz": None,
        "spectral_flatness": None,
        "band_energy_0_300": None,
        "band_energy_300_3400": None,
        "band_energy_3400_8000": None,
        "narrowband_proxy": None,
    }
    path = Path(str(job["resolved_path"]))
    try:
        info = inspect_audio_file(path)
        row.update(
            {
                "exists": True,
                "format": info.format,
                "subtype": info.subtype,
                "sample_rate_hz": info.sample_rate_hz,
                "num_channels": info.num_channels,
                "frame_count": info.frame_count,
                "duration_s": info.duration_seconds,
            }
        )
        max_seconds = job["analysis_seconds"]
        frame_count = None
        if max_seconds is not None:
            frame_count = max(1, int(float(max_seconds) * info.sample_rate_hz))
        waveform, _ = read_audio_file(path, frame_count=frame_count)
        mono = _to_mono(np.asarray(waveform, dtype=np.float32))
        row.update(_measure_waveform(mono, sample_rate_hz=info.sample_rate_hz))
    except (OSError, RuntimeError, ValueError) as exc:
        row["error"] = str(exc)
    return row


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform.astype(np.float32, copy=False)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError(f"Expected non-empty audio array, got shape {tuple(waveform.shape)}.")
    return waveform.mean(axis=0, dtype=np.float32)


def _measure_waveform(waveform: np.ndarray, *, sample_rate_hz: int) -> dict[str, float]:
    if waveform.size == 0:
        raise ValueError("Decoded audio is empty.")

    abs_wave = np.abs(waveform)
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    peak = float(np.max(abs_wave))
    rms_dbfs = _amplitude_to_dbfs(rms)
    peak_dbfs = _amplitude_to_dbfs(peak)
    frame_rms_dbfs = _frame_rms_dbfs(waveform, sample_rate_hz=sample_rate_hz)
    silent_frames = frame_rms_dbfs <= SILENCE_THRESHOLD_DBFS
    band_metrics = _spectral_metrics(waveform, sample_rate_hz=sample_rate_hz)

    signs = np.signbit(waveform)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if waveform.size > 1 else 0.0
    return {
        "analysis_duration_s": round(float(waveform.size) / float(sample_rate_hz), 6),
        "rms_dbfs": round(rms_dbfs, 6),
        "peak_dbfs": round(peak_dbfs, 6),
        "crest_factor_db": round(max(0.0, peak_dbfs - rms_dbfs), 6),
        "clipping_frac": round(float(np.mean(abs_wave >= 0.999)), 8),
        "silence_ratio_40db": round(float(np.mean(silent_frames)), 6),
        "leading_silence_s": round(_edge_silence_seconds(silent_frames, sample_rate_hz), 6),
        "trailing_silence_s": round(
            _edge_silence_seconds(silent_frames[::-1], sample_rate_hz),
            6,
        ),
        "zcr": round(zcr, 8),
        **band_metrics,
    }


def _amplitude_to_dbfs(value: float) -> float:
    return float(max(-120.0, 20.0 * math.log10(max(value, EPSILON))))


def _frame_rms_dbfs(waveform: np.ndarray, *, sample_rate_hz: int) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate_hz * 0.02)))
    frame_count = max(1, waveform.size // frame_length)
    trimmed = waveform[: frame_count * frame_length]
    if trimmed.size == 0:
        trimmed = waveform
        frame_count = 1
    frames = trimmed.reshape(frame_count, -1)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return np.maximum(-120.0, 20.0 * np.log10(np.maximum(rms, EPSILON)))


def _edge_silence_seconds(mask: np.ndarray, sample_rate_hz: int) -> float:
    silent_count = 0
    for is_silent in mask:
        if not bool(is_silent):
            break
        silent_count += 1
    return silent_count * 0.02 if sample_rate_hz > 0 else 0.0


def _spectral_metrics(waveform: np.ndarray, *, sample_rate_hz: int) -> dict[str, float]:
    centered = waveform - float(np.mean(waveform))
    if centered.size < 4 or float(np.max(np.abs(centered))) <= EPSILON:
        return {
            "spectral_centroid_hz": 0.0,
            "rolloff95_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_flatness": 0.0,
            "band_energy_0_300": 0.0,
            "band_energy_300_3400": 0.0,
            "band_energy_3400_8000": 0.0,
            "narrowband_proxy": 0.0,
        }

    max_samples = min(centered.size, sample_rate_hz * 10)
    clipped = centered[:max_samples]
    window = np.hanning(clipped.size)
    spectrum = np.fft.rfft(clipped * window)
    magnitude = np.abs(spectrum).astype(np.float64)
    power = np.square(magnitude)
    freqs = np.fft.rfftfreq(clipped.size, d=1.0 / sample_rate_hz)
    mag_sum = float(np.sum(magnitude)) + EPSILON
    power_sum = float(np.sum(power)) + EPSILON

    centroid = float(np.sum(freqs * magnitude) / mag_sum)
    bandwidth = float(np.sqrt(np.sum(np.square(freqs - centroid) * magnitude) / mag_sum))
    cumulative = np.cumsum(power)
    rolloff_index = int(np.searchsorted(cumulative, 0.95 * power_sum, side="left"))
    rolloff = float(freqs[min(rolloff_index, freqs.size - 1)])
    flatness = float(np.exp(np.mean(np.log(magnitude + EPSILON))) / (np.mean(magnitude) + EPSILON))
    low = _band_energy_ratio(freqs, power, lower_hz=0.0, upper_hz=300.0, total=power_sum)
    mid = _band_energy_ratio(freqs, power, lower_hz=300.0, upper_hz=3400.0, total=power_sum)
    high = _band_energy_ratio(freqs, power, lower_hz=3400.0, upper_hz=8000.0, total=power_sum)
    narrowband_proxy = _narrowband_proxy(rolloff95_hz=rolloff, high_ratio=high)
    return {
        "spectral_centroid_hz": round(centroid, 6),
        "rolloff95_hz": round(rolloff, 6),
        "spectral_bandwidth_hz": round(bandwidth, 6),
        "spectral_flatness": round(flatness, 8),
        "band_energy_0_300": round(low, 8),
        "band_energy_300_3400": round(mid, 8),
        "band_energy_3400_8000": round(high, 8),
        "narrowband_proxy": round(narrowband_proxy, 6),
    }


def _band_energy_ratio(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    lower_hz: float,
    upper_hz: float,
    total: float,
) -> float:
    mask = (freqs >= lower_hz) & (freqs < upper_hz)
    return float(np.sum(power[mask]) / max(total, EPSILON))


def _narrowband_proxy(*, rolloff95_hz: float, high_ratio: float) -> float:
    rolloff_score = np.clip((4200.0 - rolloff95_hz) / 2200.0, 0.0, 1.0)
    high_score = np.clip((0.08 - high_ratio) / 0.08, 0.0, 1.0)
    return float(rolloff_score * high_score)


def iter_existing_audio_paths(frame: pl.DataFrame) -> Iterable[Path]:
    for value in frame.filter(pl.col("exists")).get_column("resolved_path").to_list():
        yield Path(str(value))
