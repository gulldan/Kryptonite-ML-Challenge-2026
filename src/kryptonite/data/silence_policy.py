"""Shared helpers for silence augmentation scaling across training and evaluation."""

from __future__ import annotations

from kryptonite.config import SilenceAugmentationConfig


def has_effective_silence_profile(config: SilenceAugmentationConfig) -> bool:
    return any(
        (
            config.max_leading_padding_seconds > 0.0,
            config.max_trailing_padding_seconds > 0.0,
            config.max_inserted_pauses > 0 and config.max_inserted_pause_seconds > 0.0,
            config.pause_ratio_min < 1.0,
            config.pause_ratio_max > 1.0,
        )
    )


def scale_pause_ratio(value: float, scale: float) -> float:
    if value >= 1.0:
        return 1.0 + ((value - 1.0) * scale)
    return 1.0 - ((1.0 - value) * scale)


def build_scaled_silence_config(
    config: SilenceAugmentationConfig,
    *,
    scale: float,
) -> SilenceAugmentationConfig:
    max_inserted_pauses = 0
    if config.max_inserted_pauses > 0:
        max_inserted_pauses = max(1, round(config.max_inserted_pauses * scale))
    return SilenceAugmentationConfig(
        enabled=True,
        max_leading_padding_seconds=round(config.max_leading_padding_seconds * scale, 6),
        max_trailing_padding_seconds=round(config.max_trailing_padding_seconds * scale, 6),
        max_inserted_pauses=max_inserted_pauses,
        min_inserted_pause_seconds=round(config.min_inserted_pause_seconds, 6),
        max_inserted_pause_seconds=round(config.max_inserted_pause_seconds * scale, 6),
        pause_ratio_min=round(scale_pause_ratio(config.pause_ratio_min, scale), 6),
        pause_ratio_max=round(scale_pause_ratio(config.pause_ratio_max, scale), 6),
        min_detected_pause_seconds=round(config.min_detected_pause_seconds, 6),
        max_perturbed_pause_seconds=round(
            config.min_detected_pause_seconds
            + (config.max_perturbed_pause_seconds - config.min_detected_pause_seconds) * scale,
            6,
        ),
        analysis_frame_ms=round(config.analysis_frame_ms, 6),
        silence_threshold_dbfs=round(config.silence_threshold_dbfs, 6),
    )


def build_scaled_silence_profile(
    config: SilenceAugmentationConfig,
    *,
    scale: float,
) -> dict[str, object]:
    scaled_config = build_scaled_silence_config(config, scale=scale)
    return {
        "enabled": scaled_config.enabled,
        "max_leading_padding_seconds": scaled_config.max_leading_padding_seconds,
        "max_trailing_padding_seconds": scaled_config.max_trailing_padding_seconds,
        "max_inserted_pauses": scaled_config.max_inserted_pauses,
        "min_inserted_pause_seconds": scaled_config.min_inserted_pause_seconds,
        "max_inserted_pause_seconds": scaled_config.max_inserted_pause_seconds,
        "pause_ratio_min": scaled_config.pause_ratio_min,
        "pause_ratio_max": scaled_config.pause_ratio_max,
        "min_detected_pause_seconds": scaled_config.min_detected_pause_seconds,
        "max_perturbed_pause_seconds": scaled_config.max_perturbed_pause_seconds,
        "analysis_frame_ms": scaled_config.analysis_frame_ms,
        "silence_threshold_dbfs": scaled_config.silence_threshold_dbfs,
    }


__all__ = [
    "build_scaled_silence_config",
    "build_scaled_silence_profile",
    "has_effective_silence_profile",
    "scale_pause_ratio",
]
