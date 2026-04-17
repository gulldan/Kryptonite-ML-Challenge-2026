"""Bounded loudness normalization helpers shared across data workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SUPPORTED_LOUDNESS_MODES = frozenset({"none", "rms"})
DEFAULT_DEGRADATION_ALIGNMENT_TOLERANCE = 1e-6
_RMS_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class LoudnessNormalizationSettings:
    mode: str = "none"
    target_loudness_dbfs: float = -27.0
    max_gain_db: float = 20.0
    max_attenuation_db: float = 12.0
    peak_headroom_db: float = 1.0

    def __post_init__(self) -> None:
        mode = self.mode.lower()
        if mode not in SUPPORTED_LOUDNESS_MODES:
            raise ValueError(f"mode must be one of {sorted(SUPPORTED_LOUDNESS_MODES)}")
        if self.max_gain_db < 0.0:
            raise ValueError("max_gain_db must be non-negative")
        if self.max_attenuation_db < 0.0:
            raise ValueError("max_attenuation_db must be non-negative")
        if self.peak_headroom_db < 0.0:
            raise ValueError("peak_headroom_db must be non-negative")

    @property
    def normalized_mode(self) -> str:
        return self.mode.lower()

    @property
    def target_peak_amplitude(self) -> float:
        return float(10 ** (-self.peak_headroom_db / 20.0))


@dataclass(frozen=True, slots=True)
class LoudnessNormalizationDecision:
    mode: str
    applied: bool
    source_rms_dbfs: float | None
    output_rms_dbfs: float | None
    target_loudness_dbfs: float
    requested_gain_db: float
    applied_gain_db: float
    gain_clamped: bool
    peak_limited: bool
    skip_reason: str
    alignment_error: float
    degradation_check_passed: bool


def measure_rms_dbfs(waveform: Any) -> float | None:
    waveform64 = np.asarray(waveform, dtype=np.float64)
    if waveform64.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(waveform64))))
    if rms <= _RMS_EPSILON:
        return None
    return round(20.0 * math.log10(rms), 6)


def apply_loudness_normalization(
    waveform: Any,
    *,
    settings: LoudnessNormalizationSettings,
    alignment_tolerance: float = DEFAULT_DEGRADATION_ALIGNMENT_TOLERANCE,
) -> tuple[Any, LoudnessNormalizationDecision]:
    waveform32 = np.asarray(waveform, dtype=np.float32)
    source_rms_dbfs = measure_rms_dbfs(waveform32)
    mode = settings.normalized_mode

    if mode == "none":
        return waveform32, LoudnessNormalizationDecision(
            mode=mode,
            applied=False,
            source_rms_dbfs=source_rms_dbfs,
            output_rms_dbfs=source_rms_dbfs,
            target_loudness_dbfs=settings.target_loudness_dbfs,
            requested_gain_db=0.0,
            applied_gain_db=0.0,
            gain_clamped=False,
            peak_limited=False,
            skip_reason="disabled",
            alignment_error=0.0,
            degradation_check_passed=True,
        )

    if source_rms_dbfs is None:
        return waveform32, LoudnessNormalizationDecision(
            mode=mode,
            applied=False,
            source_rms_dbfs=None,
            output_rms_dbfs=None,
            target_loudness_dbfs=settings.target_loudness_dbfs,
            requested_gain_db=0.0,
            applied_gain_db=0.0,
            gain_clamped=False,
            peak_limited=False,
            skip_reason="zero_signal",
            alignment_error=0.0,
            degradation_check_passed=True,
        )

    requested_gain_db = settings.target_loudness_dbfs - source_rms_dbfs
    clamped_gain_db = min(
        settings.max_gain_db,
        max(-settings.max_attenuation_db, requested_gain_db),
    )
    gain_clamped = not math.isclose(clamped_gain_db, requested_gain_db, abs_tol=1e-6)
    gain_factor = float(10 ** (clamped_gain_db / 20.0))
    normalized = waveform32 * gain_factor

    peak_limited = False
    peak = float(np.abs(normalized).max(initial=0.0))
    if peak > settings.target_peak_amplitude and peak > 0.0:
        limited_factor = settings.target_peak_amplitude / peak
        normalized = normalized * limited_factor
        gain_factor *= float(limited_factor)
        peak_limited = True

    applied_gain_db = 20.0 * math.log10(gain_factor) if gain_factor > 0.0 else 0.0
    output_rms_dbfs = measure_rms_dbfs(normalized)
    alignment_error = _scale_invariant_alignment_error(reference=waveform32, transformed=normalized)
    return normalized.astype(np.float32, copy=False), LoudnessNormalizationDecision(
        mode=mode,
        applied=not math.isclose(applied_gain_db, 0.0, abs_tol=1e-6),
        source_rms_dbfs=source_rms_dbfs,
        output_rms_dbfs=output_rms_dbfs,
        target_loudness_dbfs=settings.target_loudness_dbfs,
        requested_gain_db=round(requested_gain_db, 6),
        applied_gain_db=round(applied_gain_db, 6),
        gain_clamped=gain_clamped,
        peak_limited=peak_limited,
        skip_reason="normalized",
        alignment_error=round(alignment_error, 12),
        degradation_check_passed=alignment_error <= alignment_tolerance,
    )


def _scale_invariant_alignment_error(*, reference: Any, transformed: Any) -> float:
    reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    transformed64 = np.asarray(transformed, dtype=np.float64).reshape(-1)
    if reference64.shape != transformed64.shape:
        raise ValueError("reference and transformed waveforms must share the same shape")

    reference_energy = float(np.dot(reference64, reference64))
    if reference_energy <= _RMS_EPSILON:
        return 0.0

    scale = float(np.dot(transformed64, reference64) / reference_energy)
    baseline = reference64 * scale
    denominator = float(np.linalg.norm(baseline))
    if denominator <= _RMS_EPSILON:
        return 0.0
    residual = transformed64 - baseline
    return float(np.linalg.norm(residual) / denominator)
