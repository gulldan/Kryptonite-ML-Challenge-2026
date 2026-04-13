"""Official 3D-Speaker CAM++ frontend helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def resolve_audio_path(raw_path: str, data_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return data_root / path


def load_official_campp_waveform(
    raw_path: str,
    *,
    data_root: Path,
    sample_rate_hz: int,
) -> Any:
    import soundfile as sf
    import torch
    import torchaudio

    samples, sample_rate = sf.read(
        resolve_audio_path(raw_path, data_root),
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(samples.T.copy())
    if int(sample_rate) != sample_rate_hz:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), sample_rate_hz)
    return waveform.mean(dim=0)


def repeat_or_pad_waveform(waveform: Any, *, target_samples: int, pad_mode: str = "repeat") -> Any:
    import torch

    values = waveform.flatten().float()
    if values.numel() >= target_samples:
        return values[:target_samples]
    if values.numel() == 0:
        return torch.zeros(target_samples, dtype=torch.float32)
    if pad_mode == "repeat":
        repeats = int(np.ceil(target_samples / values.numel()))
        return values.repeat(repeats)[:target_samples]
    return torch.nn.functional.pad(values, (0, target_samples - values.numel()))


def even_waveform_segments(
    waveform: Any,
    *,
    sample_rate_hz: int,
    chunk_seconds: float,
    segment_count: int,
    pad_mode: str = "repeat",
) -> list[Any]:
    values = waveform.flatten().float()
    chunk_samples = int(round(sample_rate_hz * chunk_seconds))
    if values.numel() <= chunk_samples:
        return [repeat_or_pad_waveform(values, target_samples=chunk_samples, pad_mode=pad_mode)]

    max_start = int(values.numel()) - chunk_samples
    if segment_count <= 1:
        starts = [max_start // 2]
    else:
        starts = np.linspace(0, max_start, num=segment_count, dtype=int).tolist()
    return [values[start : start + chunk_samples] for start in starts]


def official_campp_fbank(
    waveform: Any,
    *,
    sample_rate_hz: int,
    num_mel_bins: int,
) -> Any:
    import torchaudio.compliance.kaldi as kaldi

    features = kaldi.fbank(
        waveform.flatten().float().unsqueeze(0),
        num_mel_bins=int(num_mel_bins),
        sample_frequency=float(sample_rate_hz),
        dither=0.0,
    )
    return features - features.mean(dim=0, keepdim=True)


__all__ = [
    "even_waveform_segments",
    "load_official_campp_waveform",
    "official_campp_fbank",
    "repeat_or_pad_waveform",
    "resolve_audio_path",
]
