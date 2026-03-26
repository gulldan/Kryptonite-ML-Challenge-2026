"""CAM++ baseline and stage-2 training entrypoints."""

from .config import (
    CAMPPlusBaselineConfig,
    CAMPPlusDataConfig,
    CAMPPlusObjectiveConfig,
    CAMPPlusOptimizationConfig,
    load_campp_baseline_config,
)
from .pipeline import CAMPPlusRunArtifacts, run_campp_baseline
from .stage2_config import (
    CAMPPlusStage2Config,
    Stage2Config,
    Stage2HardNegativeConfig,
    Stage2UtteranceCurriculumConfig,
    load_campp_stage2_config,
)
from .stage2_pipeline import CAMPPlusStage2RunArtifacts, run_campp_stage2

__all__ = [
    "CAMPPlusBaselineConfig",
    "CAMPPlusDataConfig",
    "CAMPPlusObjectiveConfig",
    "CAMPPlusOptimizationConfig",
    "CAMPPlusRunArtifacts",
    "CAMPPlusStage2Config",
    "CAMPPlusStage2RunArtifacts",
    "Stage2Config",
    "Stage2HardNegativeConfig",
    "Stage2UtteranceCurriculumConfig",
    "load_campp_baseline_config",
    "load_campp_stage2_config",
    "run_campp_baseline",
    "run_campp_stage2",
]
