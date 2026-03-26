"""CAM++ baseline and staged fine-tuning entrypoints."""

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
from .stage3_config import (
    CAMPPlusStage3Config,
    Stage3Config,
    Stage3CropCurriculumConfig,
    Stage3HardNegativeConfig,
    Stage3MarginScheduleConfig,
    load_campp_stage3_config,
)
from .stage3_pipeline import CAMPPlusStage3RunArtifacts, run_campp_stage3

__all__ = [
    "CAMPPlusBaselineConfig",
    "CAMPPlusDataConfig",
    "CAMPPlusObjectiveConfig",
    "CAMPPlusOptimizationConfig",
    "CAMPPlusRunArtifacts",
    "CAMPPlusStage2Config",
    "CAMPPlusStage2RunArtifacts",
    "CAMPPlusStage3Config",
    "CAMPPlusStage3RunArtifacts",
    "Stage2Config",
    "Stage2HardNegativeConfig",
    "Stage2UtteranceCurriculumConfig",
    "Stage3Config",
    "Stage3CropCurriculumConfig",
    "Stage3HardNegativeConfig",
    "Stage3MarginScheduleConfig",
    "load_campp_baseline_config",
    "load_campp_stage2_config",
    "load_campp_stage3_config",
    "run_campp_baseline",
    "run_campp_stage2",
    "run_campp_stage3",
]
