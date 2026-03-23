"""Audio read/write and waveform transforms for normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audio_io import (
    read_audio_file as _read_audio_file,
)
from ..audio_io import (
    resample_waveform,
    write_audio_file,
)
from .models import AudioNormalizationPolicy

__all__ = [
    "all_finite",
    "channel_dc_offset_ratio",
    "clipped_sample_ratio",
    "peak_amplitude",
    "read_audio_file",
    "resample_waveform",
    "validate_audio_normalization_policy",
    "write_audio_file",
]


def validate_audio_normalization_policy(policy: AudioNormalizationPolicy) -> None:
    output_format = policy.output_format.lower()
    if output_format not in {"wav", "flac"}:
        raise ValueError(f"Unsupported output format: {policy.output_format!r}")
    if policy.target_sample_rate_hz <= 0:
        raise ValueError("target_sample_rate_hz must be positive")
    if policy.target_channels <= 0:
        raise ValueError("target_channels must be positive")
    if not 0.0 < policy.clipped_sample_threshold <= 1.0:
        raise ValueError("clipped_sample_threshold must be within (0, 1]")
    if policy.dc_offset_threshold < 0.0:
        raise ValueError("dc_offset_threshold must be non-negative")
    if policy.peak_headroom_db < 0.0:
        raise ValueError("peak_headroom_db must be non-negative")


def channel_dc_offset_ratio(waveform: Any) -> float:
    import numpy as np

    channel_means = np.abs(waveform.mean(axis=-1))
    return float(channel_means.max(initial=0.0))


def clipped_sample_ratio(waveform: Any, *, threshold: float) -> float:
    import numpy as np

    sample_count = int(waveform.size)
    if sample_count == 0:
        return 0.0
    clipped_count = int(np.count_nonzero(np.abs(waveform) >= threshold))
    return clipped_count / sample_count


def peak_amplitude(waveform: Any) -> float:
    import numpy as np

    if waveform.size == 0:
        return 0.0
    return float(np.abs(waveform).max(initial=0.0))


def read_audio_file(path: Path) -> tuple[Any, int]:
    waveform, info = _read_audio_file(path)
    return waveform, info.sample_rate_hz


def all_finite(waveform: Any) -> Any:
    import numpy as np

    return np.isfinite(waveform)
