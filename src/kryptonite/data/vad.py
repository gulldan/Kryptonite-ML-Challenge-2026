"""Energy-based speech trimming for manifest-driven audio loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

VADMode = Literal["none", "light", "aggressive"]
SUPPORTED_VAD_MODES: tuple[VADMode, ...] = ("none", "light", "aggressive")


@dataclass(frozen=True, slots=True)
class VADSettings:
    mode: VADMode
    window_ms: int
    energy_threshold_dbfs: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    padding_ms: int
    minimum_retained_duration_ms: int

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_VAD_MODES:
            raise ValueError(
                f"Unsupported VAD mode {self.mode!r}; expected one of {SUPPORTED_VAD_MODES}"
            )
        for name in (
            "window_ms",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "padding_ms",
            "minimum_retained_duration_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class TrimDecision:
    mode: VADMode
    applied: bool
    speech_detected: bool
    reason: str
    original_frame_count: int
    output_frame_count: int
    start_frame: int
    end_frame: int
    leading_trim_frames: int
    trailing_trim_frames: int


def resolve_vad_settings(mode: str) -> VADSettings:
    normalized_mode = mode.lower()
    if normalized_mode == "none":
        return VADSettings(
            mode="none",
            window_ms=30,
            energy_threshold_dbfs=-45.0,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            padding_ms=0,
            minimum_retained_duration_ms=0,
        )
    if normalized_mode == "light":
        return VADSettings(
            mode="light",
            window_ms=30,
            energy_threshold_dbfs=-45.0,
            min_speech_duration_ms=120,
            min_silence_duration_ms=180,
            padding_ms=120,
            minimum_retained_duration_ms=300,
        )
    if normalized_mode == "aggressive":
        return VADSettings(
            mode="aggressive",
            window_ms=30,
            energy_threshold_dbfs=-38.0,
            min_speech_duration_ms=80,
            min_silence_duration_ms=120,
            padding_ms=40,
            minimum_retained_duration_ms=250,
        )
    raise ValueError(f"Unsupported VAD mode {mode!r}; expected one of {SUPPORTED_VAD_MODES}")


def apply_vad_policy(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    mode: str,
) -> tuple[np.ndarray, TrimDecision]:
    settings = resolve_vad_settings(mode)
    if waveform.ndim != 2:
        raise ValueError("waveform must be channel-first with shape (channels, frames)")
    total_frames = int(waveform.shape[-1])
    if total_frames == 0:
        decision = TrimDecision(
            mode=settings.mode,
            applied=False,
            speech_detected=False,
            reason="empty_waveform",
            original_frame_count=0,
            output_frame_count=0,
            start_frame=0,
            end_frame=0,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    if settings.mode == "none":
        decision = TrimDecision(
            mode="none",
            applied=False,
            speech_detected=True,
            reason="disabled",
            original_frame_count=total_frames,
            output_frame_count=total_frames,
            start_frame=0,
            end_frame=total_frames,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    frames_per_window = max(1, _frames_from_ms(sample_rate_hz, settings.window_ms))
    speech_mask = _speech_mask(
        waveform,
        frames_per_window=frames_per_window,
        energy_threshold_dbfs=settings.energy_threshold_dbfs,
    )
    speech_mask = _fill_short_silence_runs(
        speech_mask,
        max_run=max(0, round(settings.min_silence_duration_ms / settings.window_ms)),
    )
    speech_mask = _drop_short_speech_runs(
        speech_mask,
        min_run=max(1, round(settings.min_speech_duration_ms / settings.window_ms)),
    )
    if not speech_mask.any():
        decision = TrimDecision(
            mode=settings.mode,
            applied=False,
            speech_detected=False,
            reason="no_speech_detected",
            original_frame_count=total_frames,
            output_frame_count=total_frames,
            start_frame=0,
            end_frame=total_frames,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    first_window = int(np.flatnonzero(speech_mask)[0])
    last_window = int(np.flatnonzero(speech_mask)[-1])
    padding_frames = _frames_from_ms(sample_rate_hz, settings.padding_ms)
    start_frame = max(0, first_window * frames_per_window - padding_frames)
    end_frame = min(total_frames, (last_window + 1) * frames_per_window + padding_frames)

    retained_frames = end_frame - start_frame
    minimum_retained_frames = _frames_from_ms(
        sample_rate_hz,
        settings.minimum_retained_duration_ms,
    )
    if retained_frames < minimum_retained_frames:
        decision = TrimDecision(
            mode=settings.mode,
            applied=False,
            speech_detected=True,
            reason="below_minimum_retained_duration",
            original_frame_count=total_frames,
            output_frame_count=total_frames,
            start_frame=0,
            end_frame=total_frames,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    if start_frame == 0 and end_frame == total_frames:
        decision = TrimDecision(
            mode=settings.mode,
            applied=False,
            speech_detected=True,
            reason="no_boundary_change",
            original_frame_count=total_frames,
            output_frame_count=total_frames,
            start_frame=0,
            end_frame=total_frames,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    trimmed = waveform[:, start_frame:end_frame]
    decision = TrimDecision(
        mode=settings.mode,
        applied=True,
        speech_detected=True,
        reason="trimmed",
        original_frame_count=total_frames,
        output_frame_count=int(trimmed.shape[-1]),
        start_frame=start_frame,
        end_frame=end_frame,
        leading_trim_frames=start_frame,
        trailing_trim_frames=total_frames - end_frame,
    )
    return trimmed, decision


def _speech_mask(
    waveform: np.ndarray,
    *,
    frames_per_window: int,
    energy_threshold_dbfs: float,
) -> np.ndarray:
    mono = waveform.astype(np.float64, copy=False).mean(axis=0)
    decisions: list[bool] = []
    for start in range(0, int(mono.shape[0]), frames_per_window):
        stop = min(start + frames_per_window, int(mono.shape[0]))
        chunk = mono[start:stop]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        decisions.append(_amplitude_to_dbfs(rms) > energy_threshold_dbfs)
    return np.asarray(decisions, dtype=bool)


def _fill_short_silence_runs(mask: np.ndarray, *, max_run: int) -> np.ndarray:
    if max_run <= 0 or mask.size == 0:
        return mask
    result = mask.copy()
    for start, stop, value in _runs(mask):
        if value or start == 0 or stop == mask.size:
            continue
        if stop - start <= max_run:
            result[start:stop] = True
    return result


def _drop_short_speech_runs(mask: np.ndarray, *, min_run: int) -> np.ndarray:
    if min_run <= 1 or mask.size == 0:
        return mask
    result = mask.copy()
    for start, stop, value in _runs(mask):
        if value and stop - start < min_run:
            result[start:stop] = False
    return result


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if mask.size == 0:
        return []
    runs: list[tuple[int, int, bool]] = []
    start = 0
    current = bool(mask[0])
    for index in range(1, int(mask.size)):
        value = bool(mask[index])
        if value != current:
            runs.append((start, index, current))
            start = index
            current = value
    runs.append((start, int(mask.size), current))
    return runs


def _frames_from_ms(sample_rate_hz: int, milliseconds: int) -> int:
    return max(0, round(sample_rate_hz * milliseconds / 1000))


def _amplitude_to_dbfs(value: float) -> float:
    clamped = max(value, 1e-12)
    return 20.0 * float(np.log10(clamped))
