"""Acoustic analysis helpers for normalized RIR waveforms."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from kryptonite.data.normalization.audio_io import read_audio_file

from .models import RIRAcousticMetrics, RIRAnalysisSettings

_PREVIEW_CHARSET = " .:-=+*#%@"


def analyze_rir_file(path: Path, settings: RIRAnalysisSettings) -> RIRAcousticMetrics:
    waveform, sample_rate_hz = read_audio_file(path)
    mono = _coerce_mono_waveform(waveform)
    peak_index = _peak_index(mono)

    direct_energy = _slice_energy(
        mono,
        start=peak_index,
        stop=peak_index + _milliseconds_to_samples(settings.direct_window_ms, sample_rate_hz),
    )
    if direct_energy <= 0.0:
        raise ValueError("direct-path energy is zero.")

    reverb_start = peak_index + _milliseconds_to_samples(settings.reverb_start_ms, sample_rate_hz)
    reverb_energy = _slice_energy(mono, start=reverb_start, stop=mono.shape[0])
    estimated_drr_db = (
        60.0 if reverb_energy <= 1e-12 else 10.0 * math.log10(direct_energy / reverb_energy)
    )

    tail = mono[peak_index:]
    estimated_rt60_seconds = _estimate_rt60_seconds(tail, sample_rate_hz)
    tail_duration_ms = round(float(tail.shape[0]) / float(sample_rate_hz) * 1000.0, 6)
    energy_centroid_ms = _energy_centroid_ms(tail, sample_rate_hz)
    preview = _build_preview(
        tail,
        sample_rate_hz=sample_rate_hz,
        preview_duration_ms=settings.preview_duration_ms,
        preview_bins=settings.preview_bins,
    )

    return RIRAcousticMetrics(
        peak_time_ms=round(float(peak_index) / float(sample_rate_hz) * 1000.0, 6),
        tail_duration_ms=tail_duration_ms,
        energy_centroid_ms=energy_centroid_ms,
        estimated_rt60_seconds=estimated_rt60_seconds,
        estimated_drr_db=round(estimated_drr_db, 6),
        envelope_preview=preview,
    )


def _coerce_mono_waveform(waveform: Any) -> Any:
    import numpy as np

    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim != 2 or array.shape[-1] == 0:
        raise ValueError("decoded RIR waveform is empty.")
    if not np.isfinite(array).all():
        raise ValueError("decoded RIR waveform contains non-finite values.")
    mono = array.mean(axis=0)
    if mono.size == 0:
        raise ValueError("decoded RIR waveform is empty after downmix.")
    return mono


def _peak_index(waveform: Any) -> int:
    import numpy as np

    peak_index = int(np.argmax(np.abs(waveform)))
    if float(np.abs(waveform[peak_index])) <= 0.0:
        raise ValueError("RIR waveform peak amplitude is zero.")
    return peak_index


def _estimate_rt60_seconds(waveform: Any, sample_rate_hz: int) -> float:
    import numpy as np

    energy = np.square(waveform, dtype=np.float64)
    if float(energy.sum()) <= 0.0:
        raise ValueError("RIR waveform has zero tail energy.")

    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder_db = 10.0 * np.log10(np.maximum(schroeder, 1e-12) / schroeder[0])
    times = np.arange(schroeder_db.shape[0], dtype=np.float64) / float(sample_rate_hz)

    fit_mask = (schroeder_db <= -5.0) & (schroeder_db >= -35.0)
    if int(fit_mask.sum()) < 2:
        fit_mask = (schroeder_db <= -5.0) & (schroeder_db >= -25.0)
    if int(fit_mask.sum()) < 2:
        fallback_index = np.flatnonzero(schroeder_db <= -20.0)
        if fallback_index.size == 0:
            raise ValueError("insufficient decay range for RT60 estimation.")
        estimate = float(times[int(fallback_index[0])] * 3.0)
        if estimate <= 0.0:
            raise ValueError("invalid fallback RT60 estimate.")
        return round(estimate, 6)

    slope, intercept = np.polyfit(times[fit_mask], schroeder_db[fit_mask], deg=1)
    del intercept
    if slope >= 0.0:
        raise ValueError("RIR energy envelope is not decaying.")
    return round(float(-60.0 / slope), 6)


def _energy_centroid_ms(waveform: Any, sample_rate_hz: int) -> float:
    import numpy as np

    energy = np.square(waveform, dtype=np.float64)
    total_energy = float(energy.sum())
    if total_energy <= 0.0:
        return 0.0
    indices = np.arange(energy.shape[0], dtype=np.float64)
    centroid = float((indices * energy).sum() / total_energy)
    return round(centroid / float(sample_rate_hz) * 1000.0, 6)


def _build_preview(
    waveform: Any,
    *,
    sample_rate_hz: int,
    preview_duration_ms: float,
    preview_bins: int,
) -> str:
    import numpy as np

    preview_samples = _milliseconds_to_samples(preview_duration_ms, sample_rate_hz)
    preview = np.abs(np.asarray(waveform[:preview_samples], dtype=np.float64))
    if preview.size == 0:
        return ""

    edges = np.linspace(0, preview.size, num=preview_bins + 1, dtype=int)
    bins = []
    for start, stop in zip(edges[:-1], edges[1:], strict=False):
        segment = preview[start:stop]
        bins.append(float(segment.max()) if segment.size else 0.0)

    peak = max(bins, default=0.0)
    if peak <= 0.0:
        return _PREVIEW_CHARSET[0] * preview_bins

    preview_chars: list[str] = []
    scale = len(_PREVIEW_CHARSET) - 1
    for value in bins:
        normalized = min(max(value / peak, 0.0), 1.0)
        preview_chars.append(_PREVIEW_CHARSET[int(round(normalized * scale))])
    return "".join(preview_chars)


def _slice_energy(waveform: Any, *, start: int, stop: int) -> float:
    import numpy as np

    bounded_start = max(start, 0)
    bounded_stop = min(stop, waveform.shape[0])
    if bounded_start >= bounded_stop:
        return 0.0
    return float(np.square(waveform[bounded_start:bounded_stop], dtype=np.float64).sum())


def _milliseconds_to_samples(milliseconds: float, sample_rate_hz: int) -> int:
    return max(1, int(round(milliseconds / 1000.0 * float(sample_rate_hz))))


__all__ = ["analyze_rir_file"]
