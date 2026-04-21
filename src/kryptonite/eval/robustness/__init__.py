"""Robustness benchmark orchestration, distortions, and reporting."""

from .audio import (
    TARGET_SAMPLE_RATE_HZ,
    DistortionCondition,
    apply_distortion,
    build_distorted_plan,
    build_frozen_clean_subset,
    collect_audio_stats,
    default_distortion_conditions,
    materialize_condition_audio,
)
from .benchmark import BenchmarkPaths, ModelSpec

__all__ = [
    "BenchmarkPaths",
    "DistortionCondition",
    "ModelSpec",
    "TARGET_SAMPLE_RATE_HZ",
    "apply_distortion",
    "build_distorted_plan",
    "build_frozen_clean_subset",
    "collect_audio_stats",
    "default_distortion_conditions",
    "materialize_condition_audio",
]
