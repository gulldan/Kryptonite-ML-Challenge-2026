"""Silence and pause robustness augmentation primitives."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from kryptonite.config import SilenceAugmentationConfig


@dataclass(frozen=True, slots=True)
class PauseSpan:
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def duration_seconds(self, *, sample_rate_hz: int) -> float:
        return round(float(self.duration_frames) / float(sample_rate_hz), 6)


@dataclass(frozen=True, slots=True)
class SilenceProfile:
    duration_seconds: float
    silence_ratio: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    interior_pause_count: int
    interior_pause_total_seconds: float
    mean_interior_pause_seconds: float
    max_interior_pause_seconds: float
    interior_pause_spans: tuple[PauseSpan, ...] = ()

    @property
    def interior_pause_ratio(self) -> float:
        if self.duration_seconds <= 0.0:
            return 0.0
        return round(self.interior_pause_total_seconds / self.duration_seconds, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "duration_seconds": self.duration_seconds,
            "silence_ratio": self.silence_ratio,
            "leading_silence_seconds": self.leading_silence_seconds,
            "trailing_silence_seconds": self.trailing_silence_seconds,
            "interior_pause_count": self.interior_pause_count,
            "interior_pause_total_seconds": self.interior_pause_total_seconds,
            "interior_pause_ratio": self.interior_pause_ratio,
            "mean_interior_pause_seconds": self.mean_interior_pause_seconds,
            "max_interior_pause_seconds": self.max_interior_pause_seconds,
        }


@dataclass(frozen=True, slots=True)
class SilenceAugmentationDecision:
    applied: bool
    skip_reason: str
    original_pause_count: int
    perturbed_pause_count: int
    stretched_pause_count: int
    compressed_pause_count: int
    inserted_pause_count: int
    inserted_pause_total_seconds: float
    leading_padding_seconds: float
    trailing_padding_seconds: float
    output_duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "skip_reason": self.skip_reason,
            "original_pause_count": self.original_pause_count,
            "perturbed_pause_count": self.perturbed_pause_count,
            "stretched_pause_count": self.stretched_pause_count,
            "compressed_pause_count": self.compressed_pause_count,
            "inserted_pause_count": self.inserted_pause_count,
            "inserted_pause_total_seconds": self.inserted_pause_total_seconds,
            "leading_padding_seconds": self.leading_padding_seconds,
            "trailing_padding_seconds": self.trailing_padding_seconds,
            "output_duration_seconds": self.output_duration_seconds,
        }


def analyze_silence_profile(
    waveform: Any,
    *,
    sample_rate_hz: int,
    config: SilenceAugmentationConfig,
) -> SilenceProfile:
    _validate_waveform(waveform, sample_rate_hz=sample_rate_hz)
    total_frames = int(waveform.shape[-1])
    if total_frames == 0:
        return SilenceProfile(
            duration_seconds=0.0,
            silence_ratio=0.0,
            leading_silence_seconds=0.0,
            trailing_silence_seconds=0.0,
            interior_pause_count=0,
            interior_pause_total_seconds=0.0,
            mean_interior_pause_seconds=0.0,
            max_interior_pause_seconds=0.0,
        )

    frame_size = _analysis_frame_count(
        sample_rate_hz=sample_rate_hz,
        analysis_frame_ms=config.analysis_frame_ms,
    )
    frame_rms = _frame_rms(waveform, frame_size=frame_size)
    threshold = _silence_threshold_amplitude(config.silence_threshold_dbfs)
    silent_mask = frame_rms <= threshold
    silence_ratio = round(float(silent_mask.mean()) if silent_mask.size else 0.0, 6)

    silent_spans = _sample_spans_from_mask(
        silent_mask=silent_mask,
        frame_size=frame_size,
        total_frames=total_frames,
    )
    leading_silence_seconds = (
        silent_spans[0].duration_seconds(sample_rate_hz=sample_rate_hz)
        if silent_spans and silent_spans[0].start_frame == 0
        else 0.0
    )
    trailing_silence_seconds = (
        silent_spans[-1].duration_seconds(sample_rate_hz=sample_rate_hz)
        if silent_spans and silent_spans[-1].end_frame == total_frames
        else 0.0
    )
    min_pause_frames = _duration_to_frames(
        config.min_detected_pause_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    interior_pause_spans = tuple(
        span
        for span in silent_spans
        if span.start_frame > 0
        and span.end_frame < total_frames
        and span.duration_frames >= min_pause_frames
    )
    interior_pause_durations = [
        span.duration_seconds(sample_rate_hz=sample_rate_hz) for span in interior_pause_spans
    ]
    interior_pause_total_seconds = round(sum(interior_pause_durations), 6)
    mean_interior_pause_seconds = (
        round(
            interior_pause_total_seconds / len(interior_pause_durations),
            6,
        )
        if interior_pause_durations
        else 0.0
    )
    max_interior_pause_seconds = (
        round(max(interior_pause_durations), 6) if interior_pause_durations else 0.0
    )

    return SilenceProfile(
        duration_seconds=round(float(total_frames) / float(sample_rate_hz), 6),
        silence_ratio=silence_ratio,
        leading_silence_seconds=leading_silence_seconds,
        trailing_silence_seconds=trailing_silence_seconds,
        interior_pause_count=len(interior_pause_spans),
        interior_pause_total_seconds=interior_pause_total_seconds,
        mean_interior_pause_seconds=mean_interior_pause_seconds,
        max_interior_pause_seconds=max_interior_pause_seconds,
        interior_pause_spans=interior_pause_spans,
    )


def apply_silence_augmentation(
    waveform: Any,
    *,
    sample_rate_hz: int,
    config: SilenceAugmentationConfig,
    rng: random.Random | None = None,
) -> tuple[Any, SilenceAugmentationDecision]:
    _validate_waveform(waveform, sample_rate_hz=sample_rate_hz)
    base_waveform = np.asarray(waveform)
    profile = analyze_silence_profile(
        base_waveform,
        sample_rate_hz=sample_rate_hz,
        config=config,
    )
    if not config.enabled:
        return base_waveform, _build_decision(
            applied=False,
            skip_reason="disabled",
            original_pause_count=profile.interior_pause_count,
            output_frames=int(base_waveform.shape[-1]),
            sample_rate_hz=sample_rate_hz,
        )
    if not _has_effective_silence_augmentation(config):
        return base_waveform, _build_decision(
            applied=False,
            skip_reason="no_effective_configuration",
            original_pause_count=profile.interior_pause_count,
            output_frames=int(base_waveform.shape[-1]),
            sample_rate_hz=sample_rate_hz,
        )

    active_rng = rng or random.Random()
    augmented = base_waveform
    perturbed_pause_count = 0
    stretched_pause_count = 0
    compressed_pause_count = 0
    inserted_pause_count = 0
    inserted_pause_frames = 0

    if profile.interior_pause_spans and (
        config.pause_ratio_min != 1.0 or config.pause_ratio_max != 1.0
    ):
        augmented, perturbed_pause_count, stretched_pause_count, compressed_pause_count = (
            _perturb_interior_pauses(
                augmented,
                sample_rate_hz=sample_rate_hz,
                pause_spans=profile.interior_pause_spans,
                config=config,
                rng=active_rng,
            )
        )

    if config.max_inserted_pauses > 0 and config.max_inserted_pause_seconds > 0.0:
        augmented, inserted_pause_count, inserted_pause_frames = _insert_pauses(
            augmented,
            sample_rate_hz=sample_rate_hz,
            config=config,
            rng=active_rng,
        )

    leading_padding_frames = _sample_padding_frames(
        config.max_leading_padding_seconds,
        sample_rate_hz=sample_rate_hz,
        rng=active_rng,
    )
    trailing_padding_frames = _sample_padding_frames(
        config.max_trailing_padding_seconds,
        sample_rate_hz=sample_rate_hz,
        rng=active_rng,
    )
    if leading_padding_frames > 0:
        augmented = np.concatenate(
            [_zeros_like(base_waveform, leading_padding_frames), augmented],
            axis=1,
        )
    if trailing_padding_frames > 0:
        augmented = np.concatenate(
            [augmented, _zeros_like(base_waveform, trailing_padding_frames)],
            axis=1,
        )

    applied = any(
        (
            perturbed_pause_count > 0,
            inserted_pause_count > 0,
            leading_padding_frames > 0,
            trailing_padding_frames > 0,
        )
    )
    skip_reason = "augmented" if applied else "no_effect_applied"
    return augmented, _build_decision(
        applied=applied,
        skip_reason=skip_reason,
        original_pause_count=profile.interior_pause_count,
        perturbed_pause_count=perturbed_pause_count,
        stretched_pause_count=stretched_pause_count,
        compressed_pause_count=compressed_pause_count,
        inserted_pause_count=inserted_pause_count,
        inserted_pause_frames=inserted_pause_frames,
        leading_padding_frames=leading_padding_frames,
        trailing_padding_frames=trailing_padding_frames,
        output_frames=int(augmented.shape[-1]),
        sample_rate_hz=sample_rate_hz,
    )


def _build_decision(
    *,
    applied: bool,
    skip_reason: str,
    original_pause_count: int,
    output_frames: int,
    sample_rate_hz: int,
    perturbed_pause_count: int = 0,
    stretched_pause_count: int = 0,
    compressed_pause_count: int = 0,
    inserted_pause_count: int = 0,
    inserted_pause_frames: int = 0,
    leading_padding_frames: int = 0,
    trailing_padding_frames: int = 0,
) -> SilenceAugmentationDecision:
    return SilenceAugmentationDecision(
        applied=applied,
        skip_reason=skip_reason,
        original_pause_count=original_pause_count,
        perturbed_pause_count=perturbed_pause_count,
        stretched_pause_count=stretched_pause_count,
        compressed_pause_count=compressed_pause_count,
        inserted_pause_count=inserted_pause_count,
        inserted_pause_total_seconds=round(
            float(inserted_pause_frames) / float(sample_rate_hz),
            6,
        ),
        leading_padding_seconds=round(
            float(leading_padding_frames) / float(sample_rate_hz),
            6,
        ),
        trailing_padding_seconds=round(
            float(trailing_padding_frames) / float(sample_rate_hz),
            6,
        ),
        output_duration_seconds=round(float(output_frames) / float(sample_rate_hz), 6),
    )


def _has_effective_silence_augmentation(config: SilenceAugmentationConfig) -> bool:
    return any(
        (
            config.max_leading_padding_seconds > 0.0,
            config.max_trailing_padding_seconds > 0.0,
            config.max_inserted_pauses > 0 and config.max_inserted_pause_seconds > 0.0,
            config.pause_ratio_min != 1.0,
            config.pause_ratio_max != 1.0,
        )
    )


def _perturb_interior_pauses(
    waveform: Any,
    *,
    sample_rate_hz: int,
    pause_spans: tuple[PauseSpan, ...],
    config: SilenceAugmentationConfig,
    rng: random.Random,
) -> tuple[Any, int, int, int]:
    min_pause_frames = _duration_to_frames(
        config.min_detected_pause_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    max_perturbed_frames = _duration_to_frames(
        config.max_perturbed_pause_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    pieces: list[Any] = []
    cursor = 0
    perturbed_pause_count = 0
    stretched_pause_count = 0
    compressed_pause_count = 0

    for pause_span in pause_spans:
        pieces.append(waveform[:, cursor : pause_span.start_frame])
        original_pause = waveform[:, pause_span.start_frame : pause_span.end_frame]
        ratio = rng.uniform(config.pause_ratio_min, config.pause_ratio_max)
        target_frames = max(min_pause_frames, int(round(pause_span.duration_frames * ratio)))
        if target_frames > pause_span.duration_frames:
            target_frames = min(
                target_frames,
                max(pause_span.duration_frames, max_perturbed_frames),
            )
        if target_frames == pause_span.duration_frames:
            pieces.append(original_pause)
        else:
            pieces.append(_resize_pause_segment(original_pause, target_frames=target_frames))
            perturbed_pause_count += 1
            if target_frames > pause_span.duration_frames:
                stretched_pause_count += 1
            else:
                compressed_pause_count += 1
        cursor = pause_span.end_frame

    pieces.append(waveform[:, cursor:])
    return (
        np.concatenate(pieces, axis=1),
        perturbed_pause_count,
        stretched_pause_count,
        compressed_pause_count,
    )


def _insert_pauses(
    waveform: Any,
    *,
    sample_rate_hz: int,
    config: SilenceAugmentationConfig,
    rng: random.Random,
) -> tuple[Any, int, int]:
    insertion_points = _find_insertion_points(
        waveform,
        sample_rate_hz=sample_rate_hz,
        config=config,
        rng=rng,
    )
    if not insertion_points:
        return waveform, 0, 0

    pieces: list[Any] = []
    cursor = 0
    inserted_pause_count = 0
    inserted_pause_frames = 0
    for insertion_point in insertion_points:
        insertion_frame = min(max(cursor, insertion_point), int(waveform.shape[-1]))
        pieces.append(waveform[:, cursor:insertion_frame])
        pause_frames = _draw_pause_frames(
            config.min_inserted_pause_seconds,
            config.max_inserted_pause_seconds,
            sample_rate_hz=sample_rate_hz,
            rng=rng,
        )
        if pause_frames > 0:
            pieces.append(_zeros_like(waveform, pause_frames))
            inserted_pause_count += 1
            inserted_pause_frames += pause_frames
        cursor = insertion_frame
    pieces.append(waveform[:, cursor:])
    return np.concatenate(pieces, axis=1), inserted_pause_count, inserted_pause_frames


def _find_insertion_points(
    waveform: Any,
    *,
    sample_rate_hz: int,
    config: SilenceAugmentationConfig,
    rng: random.Random,
) -> list[int]:
    total_frames = int(waveform.shape[-1])
    if total_frames <= 2:
        return []
    frame_size = _analysis_frame_count(
        sample_rate_hz=sample_rate_hz,
        analysis_frame_ms=config.analysis_frame_ms,
    )
    frame_rms = _frame_rms(waveform, frame_size=frame_size)
    if frame_rms.size < 3:
        return []

    threshold = _silence_threshold_amplitude(config.silence_threshold_dbfs)
    silent_mask = frame_rms <= threshold
    candidate_frames = [
        index
        for index in range(1, int(frame_rms.size) - 1)
        if not silent_mask[index]
        and frame_rms[index] <= frame_rms[index - 1]
        and frame_rms[index] <= frame_rms[index + 1]
    ]
    if not candidate_frames:
        candidate_frames = [
            index for index in range(1, int(frame_rms.size) - 1) if not silent_mask[index]
        ]
    if not candidate_frames:
        return []

    candidate_frames.sort(key=lambda index: (float(frame_rms[index]), index))
    top_k = max(config.max_inserted_pauses * 6, config.max_inserted_pauses)
    candidate_pool = candidate_frames[:top_k]
    rng.shuffle(candidate_pool)
    candidate_pool.sort(key=lambda index: (float(frame_rms[index]), index))

    min_spacing_frames = max(
        1,
        math.ceil(
            _duration_to_frames(
                config.min_inserted_pause_seconds,
                sample_rate_hz=sample_rate_hz,
            )
            / max(frame_size, 1)
        ),
    )
    selected: list[int] = []
    for frame_index in candidate_pool:
        if all(abs(frame_index - existing) >= min_spacing_frames for existing in selected):
            selected.append(frame_index)
            if len(selected) >= config.max_inserted_pauses:
                break
    if not selected:
        return []
    return sorted(min(frame_index * frame_size, total_frames) for frame_index in selected)


def _resize_pause_segment(segment: Any, *, target_frames: int) -> Any:
    current_frames = int(segment.shape[-1])
    if target_frames == current_frames:
        return segment
    if target_frames < current_frames:
        return segment[:, :target_frames]
    repeats = max(1, math.ceil(target_frames / max(current_frames, 1)))
    tiled = np.tile(segment, (1, repeats))
    return tiled[:, :target_frames]


def _sample_spans_from_mask(
    *,
    silent_mask: np.ndarray,
    frame_size: int,
    total_frames: int,
) -> list[PauseSpan]:
    spans: list[PauseSpan] = []
    start_index: int | None = None
    for frame_index, is_silent in enumerate(silent_mask.tolist()):
        if is_silent and start_index is None:
            start_index = frame_index
        elif not is_silent and start_index is not None:
            spans.append(
                PauseSpan(
                    start_frame=start_index * frame_size,
                    end_frame=min(frame_index * frame_size, total_frames),
                )
            )
            start_index = None
    if start_index is not None:
        spans.append(
            PauseSpan(
                start_frame=start_index * frame_size,
                end_frame=total_frames,
            )
        )
    return spans


def _frame_rms(waveform: Any, *, frame_size: int) -> np.ndarray:
    waveform64 = np.asarray(waveform, dtype=np.float64)
    frame_count = math.ceil(int(waveform64.shape[-1]) / frame_size)
    frame_rms = np.empty(frame_count, dtype=np.float64)
    for frame_index in range(frame_count):
        start = frame_index * frame_size
        stop = min(start + frame_size, int(waveform64.shape[-1]))
        chunk = waveform64[:, start:stop]
        frame_rms[frame_index] = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
    return frame_rms


def _zeros_like(waveform: Any, frame_count: int) -> Any:
    return np.zeros((int(waveform.shape[0]), frame_count), dtype=np.asarray(waveform).dtype)


def _sample_padding_frames(
    max_padding_seconds: float,
    *,
    sample_rate_hz: int,
    rng: random.Random,
) -> int:
    if max_padding_seconds <= 0.0:
        return 0
    return _duration_to_frames(
        rng.uniform(0.0, max_padding_seconds),
        sample_rate_hz=sample_rate_hz,
    )


def _draw_pause_frames(
    min_pause_seconds: float,
    max_pause_seconds: float,
    *,
    sample_rate_hz: int,
    rng: random.Random,
) -> int:
    if max_pause_seconds <= 0.0:
        return 0
    if math.isclose(min_pause_seconds, max_pause_seconds):
        duration_seconds = max_pause_seconds
    else:
        duration_seconds = rng.uniform(min_pause_seconds, max_pause_seconds)
    return _duration_to_frames(duration_seconds, sample_rate_hz=sample_rate_hz)


def _duration_to_frames(duration_seconds: float, *, sample_rate_hz: int) -> int:
    return max(0, int(round(duration_seconds * sample_rate_hz)))


def _analysis_frame_count(*, sample_rate_hz: int, analysis_frame_ms: float) -> int:
    return max(1, int(round(sample_rate_hz * analysis_frame_ms / 1000.0)))


def _silence_threshold_amplitude(threshold_dbfs: float) -> float:
    return 10.0 ** (threshold_dbfs / 20.0)


def _validate_waveform(waveform: Any, *, sample_rate_hz: int) -> None:
    array = np.asarray(waveform)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if array.ndim != 2:
        raise ValueError("waveform must be channel-first with shape (channels, frames)")
    if int(array.shape[-1]) <= 0:
        raise ValueError("waveform must contain at least one frame")
