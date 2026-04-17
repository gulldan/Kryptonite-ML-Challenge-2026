"""Audio transforms for dense-gallery EDA validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from kryptonite.data.audio_io import read_audio_file, resample_waveform


@dataclass(frozen=True, slots=True)
class SyntheticShiftProfile:
    leading_silence_s: np.ndarray
    trailing_silence_s: np.ndarray
    band_limit_probability: float
    peak_limit_probability: float
    seed: int

    @classmethod
    def from_file_stats(cls, path: Path, *, seed: int) -> SyntheticShiftProfile:
        public = pl.read_parquet(path).filter(pl.col("split") == "test_public")
        narrowband = (
            (public["narrowband_proxy"] >= 0.5) | (public["rolloff95_hz"] <= 3800.0)
        ).mean()
        peak_limited = (public["peak_dbfs"] >= -0.1).mean()
        return cls(
            leading_silence_s=public["leading_silence_s"].fill_null(0.0).to_numpy(),
            trailing_silence_s=public["trailing_silence_s"].fill_null(0.0).to_numpy(),
            band_limit_probability=float(cast(float, narrowband)),
            peak_limit_probability=float(cast(float, peak_limited)),
            seed=seed,
        )

    def sample_padding(self, key: int) -> tuple[float, float]:
        rng = np.random.default_rng(self.seed + 1_000_003 * int(key))
        lead = float(self.leading_silence_s[rng.integers(0, self.leading_silence_s.size)])
        trail = float(self.trailing_silence_s[rng.integers(0, self.trailing_silence_s.size)])
        return lead, trail

    def sample_channel_flags(self, key: int) -> tuple[bool, bool]:
        rng = np.random.default_rng(self.seed + 7_000_001 * int(key))
        band_limit = bool(rng.random() < min(self.band_limit_probability, 0.50))
        peak_limit = bool(rng.random() < self.peak_limit_probability)
        return band_limit, peak_limit


@dataclass(frozen=True, slots=True)
class ChannelShiftCondition:
    band_limit: bool
    peak_limit: bool


def load_eval_waveform(
    path: Path,
    *,
    trim: bool,
    shift_profile: SyntheticShiftProfile | None = None,
    shift_mode: str = "none",
    shift_key: int = 0,
) -> np.ndarray:
    waveform, info = read_audio_file(path)
    mono = np.asarray(waveform, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=0, dtype=np.float32)
    if info.sample_rate_hz != 16_000:
        mono = resample_waveform(mono, orig_freq=info.sample_rate_hz, new_freq=16_000)
    if shift_profile is not None and shift_mode in {"edge_silence", "v2"}:
        mono = add_empirical_edge_silence(mono, shift_profile, shift_key=shift_key)
    return trim_silence(mono) if trim else mono


def sample_channel_condition(
    profile: SyntheticShiftProfile, *, shift_key: int
) -> ChannelShiftCondition:
    band_limit, peak_limit = profile.sample_channel_flags(shift_key)
    return ChannelShiftCondition(band_limit=band_limit, peak_limit=peak_limit)


def apply_channel_condition(waveform: np.ndarray, condition: ChannelShiftCondition) -> np.ndarray:
    if condition.band_limit:
        waveform = band_limit_telephone(waveform)
    if condition.peak_limit:
        waveform = mild_peak_limit(waveform)
    return waveform


def add_empirical_edge_silence(
    waveform: np.ndarray, profile: SyntheticShiftProfile, *, shift_key: int
) -> np.ndarray:
    lead_s, trail_s = profile.sample_padding(shift_key)
    lead = np.zeros(max(0, int(round(lead_s * 16_000))), dtype=np.float32)
    trail = np.zeros(max(0, int(round(trail_s * 16_000))), dtype=np.float32)
    if lead.size == 0 and trail.size == 0:
        return waveform
    return np.concatenate([lead, waveform, trail]).astype(np.float32, copy=False)


def band_limit_telephone(
    waveform: np.ndarray,
    *,
    _sample_rate: int = 16_000,
    _low_hz: float = 300.0,
    _high_hz: float = 3400.0,
) -> np.ndarray:
    if waveform.size < 4:
        return waveform
    source_x = np.arange(0, waveform.size, 2, dtype=np.float32)
    target_x = np.arange(waveform.size, dtype=np.float32)
    filtered = np.interp(target_x, source_x, waveform[::2]).astype(np.float32)
    peak = float(np.max(np.abs(filtered)))
    if peak > 1.0:
        filtered = filtered / peak
    return filtered


def mild_peak_limit(waveform: np.ndarray, *, target_peak: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak <= 1e-6:
        return waveform
    boosted = waveform * (target_peak / peak)
    return (np.tanh(1.6 * boosted) / np.tanh(1.6)).astype(np.float32)


def trim_silence(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16_000,
    top_db: float = 38.0,
    context_s: float = 0.15,
    min_duration_s: float = 1.5,
) -> np.ndarray:
    frame = int(round(0.025 * sample_rate))
    hop = int(round(0.010 * sample_rate))
    if waveform.size < frame:
        return waveform
    starts = np.arange(0, waveform.size - frame + 1, hop)
    frames = np.stack([waveform[start : start + frame] for start in starts])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    threshold_db = float(np.max(rms_db)) - top_db
    voiced = np.flatnonzero(rms_db >= threshold_db)
    if voiced.size == 0:
        return waveform
    context = int(round(context_s * sample_rate))
    start = max(0, int(starts[int(voiced[0])]) - context)
    end = min(waveform.size, int(starts[int(voiced[-1])] + frame) + context)
    if end - start < int(round(min_duration_s * sample_rate)):
        return waveform
    return waveform[start:end]


def eval_crops(waveform: np.ndarray, *, crop_samples: int, n_crops: int) -> list[np.ndarray]:
    if waveform.size == 0:
        waveform = np.zeros(crop_samples, dtype=np.float32)
    if waveform.size < crop_samples:
        repeats = crop_samples // max(1, waveform.size) + 1
        waveform = np.tile(waveform, repeats)[:crop_samples]
    if n_crops == 1:
        starts = [max(0, (waveform.size - crop_samples) // 2)]
    else:
        starts = np.linspace(0, waveform.size - crop_samples, num=n_crops).astype(np.int64).tolist()
    return [
        waveform[start : start + crop_samples].astype(np.float32, copy=False) for start in starts
    ]


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)
