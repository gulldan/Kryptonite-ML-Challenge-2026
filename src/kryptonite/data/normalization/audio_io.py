"""Audio read/write and waveform transforms for normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AudioNormalizationPolicy


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
    import soundfile as sf

    waveform, sample_rate_hz = sf.read(
        str(path),
        always_2d=True,
        dtype="float32",
    )
    return waveform.T, int(sample_rate_hz)


def resample_waveform(waveform: Any, *, orig_freq: int, new_freq: int) -> Any:
    import numpy as np
    import soxr

    if waveform.ndim == 1:
        return soxr.resample(waveform, orig_freq, new_freq, quality="HQ").astype(
            "float32",
            copy=False,
        )
    channels = [soxr.resample(channel, orig_freq, new_freq, quality="HQ") for channel in waveform]
    return np.stack(channels, axis=0).astype("float32", copy=False)


def write_audio_file(
    *,
    path: Path,
    waveform: Any,
    sample_rate_hz: int,
    output_format: str,
    pcm_bits_per_sample: int,
) -> None:
    import numpy as np
    import soundfile as sf

    output_format = output_format.lower()
    sf.write(
        str(path),
        np.clip(waveform, -1.0, 1.0).T,
        sample_rate_hz,
        format=output_format.upper(),
        subtype=pcm_subtype(bits_per_sample=pcm_bits_per_sample),
    )


def all_finite(waveform: Any) -> Any:
    import numpy as np

    return np.isfinite(waveform)


def pcm_subtype(*, bits_per_sample: int) -> str:
    return {
        8: "PCM_U8",
        16: "PCM_16",
        24: "PCM_24",
        32: "PCM_32",
    }.get(bits_per_sample) or _raise_unsupported_pcm_bits(bits_per_sample)


def _raise_unsupported_pcm_bits(bits_per_sample: int) -> str:
    raise ValueError(f"Unsupported PCM bits per sample: {bits_per_sample}")
