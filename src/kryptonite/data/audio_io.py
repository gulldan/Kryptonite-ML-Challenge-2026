"""Shared audio I/O helpers for manifest-backed loaders and preprocessors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_AUDIO_FORMATS = frozenset({"WAV", "FLAC", "MP3"})


@dataclass(frozen=True, slots=True)
class AudioFileInfo:
    format: str
    subtype: str | None
    sample_rate_hz: int
    num_channels: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return round(float(self.frame_count) / float(self.sample_rate_hz), 6)


def inspect_audio_file(path: Path) -> AudioFileInfo:
    import soundfile as sf

    with sf.SoundFile(str(path)) as handle:
        format_name = (handle.format or "").upper()
        if format_name not in SUPPORTED_AUDIO_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
            raise ValueError(
                f"Unsupported audio format {format_name or '<unknown>'!r} for {path}; "
                f"expected one of: {supported}"
            )
        if handle.samplerate <= 0:
            raise ValueError(f"Audio file {path} reports a non-positive sample rate.")
        if handle.channels <= 0:
            raise ValueError(f"Audio file {path} reports a non-positive channel count.")
        return AudioFileInfo(
            format=format_name,
            subtype=handle.subtype,
            sample_rate_hz=int(handle.samplerate),
            num_channels=int(handle.channels),
            frame_count=int(handle.frames),
        )


def read_audio_file(
    path: Path,
    *,
    frame_offset: int = 0,
    frame_count: int | None = None,
) -> tuple[Any, AudioFileInfo]:
    import soundfile as sf

    if frame_offset < 0:
        raise ValueError("frame_offset must be non-negative")
    if frame_count is not None and frame_count <= 0:
        raise ValueError("frame_count must be positive when provided")

    with sf.SoundFile(str(path)) as handle:
        info = AudioFileInfo(
            format=(handle.format or "").upper(),
            subtype=handle.subtype,
            sample_rate_hz=int(handle.samplerate),
            num_channels=int(handle.channels),
            frame_count=int(handle.frames),
        )
        if info.format not in SUPPORTED_AUDIO_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
            raise ValueError(
                f"Unsupported audio format {info.format or '<unknown>'!r} for {path}; "
                f"expected one of: {supported}"
            )
        if frame_offset >= info.frame_count:
            raise ValueError(
                f"Requested frame_offset={frame_offset} is past EOF for {path} "
                f"(total frames={info.frame_count})."
            )
        handle.seek(frame_offset)
        waveform = handle.read(
            frames=-1 if frame_count is None else frame_count,
            always_2d=True,
            dtype="float32",
        )
    return waveform.T, info


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


def pcm_subtype(*, bits_per_sample: int) -> str:
    return {
        8: "PCM_U8",
        16: "PCM_16",
        24: "PCM_24",
        32: "PCM_32",
    }.get(bits_per_sample) or _raise_unsupported_pcm_bits(bits_per_sample)


def _raise_unsupported_pcm_bits(bits_per_sample: int) -> str:
    raise ValueError(f"Unsupported PCM bits per sample: {bits_per_sample}")
