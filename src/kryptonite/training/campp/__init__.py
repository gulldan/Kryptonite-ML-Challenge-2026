"""CAM++ baseline training entrypoints."""

from .config import (
    CAMPPlusBaselineConfig,
    CAMPPlusDataConfig,
    CAMPPlusObjectiveConfig,
    CAMPPlusOptimizationConfig,
    load_campp_baseline_config,
)
from .pipeline import CAMPPlusRunArtifacts, run_campp_baseline

__all__ = [
    "CAMPPlusBaselineConfig",
    "CAMPPlusDataConfig",
    "CAMPPlusObjectiveConfig",
    "CAMPPlusOptimizationConfig",
    "CAMPPlusRunArtifacts",
    "load_campp_baseline_config",
    "run_campp_baseline",
]
