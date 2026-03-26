"""CAM++ baseline and staged fine-tuning entrypoints."""

from .config import (
    CAMPPlusBaselineConfig,
    CAMPPlusDataConfig,
    CAMPPlusObjectiveConfig,
    CAMPPlusOptimizationConfig,
    load_campp_baseline_config,
)
from .model_selection import (
    ModelSelectionArtifacts,
    ModelSelectionSummary,
    ModelSelectionVariant,
    render_campp_model_selection_markdown,
    run_campp_model_selection,
)
from .model_selection_config import (
    CAMPPlusModelSelectionConfig,
    CheckpointAveragingConfig,
    load_campp_model_selection_config,
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
from .sweep_shortlist import (
    SweepCandidateResult,
    SweepShortlistRunArtifacts,
    SweepShortlistSummary,
    SweepSuiteEvaluation,
    load_campp_sweep_shortlist_report,
    render_campp_sweep_shortlist_markdown,
    run_campp_sweep_shortlist,
)
from .sweep_shortlist_config import (
    CAMPPlusSweepShortlistConfig,
    CorruptedSuitesConfig,
    SweepBudgetConfig,
    SweepCandidateConfig,
    SweepCropCurriculumOverride,
    SweepMarginScheduleOverride,
    SweepSelectionConfig,
    load_campp_sweep_shortlist_config,
)

__all__ = [
    "CAMPPlusBaselineConfig",
    "CAMPPlusDataConfig",
    "CAMPPlusModelSelectionConfig",
    "CAMPPlusObjectiveConfig",
    "CAMPPlusOptimizationConfig",
    "CAMPPlusRunArtifacts",
    "CAMPPlusSweepShortlistConfig",
    "CAMPPlusStage2Config",
    "CAMPPlusStage2RunArtifacts",
    "CAMPPlusStage3Config",
    "CAMPPlusStage3RunArtifacts",
    "CheckpointAveragingConfig",
    "CorruptedSuitesConfig",
    "ModelSelectionArtifacts",
    "ModelSelectionSummary",
    "ModelSelectionVariant",
    "Stage2Config",
    "Stage2HardNegativeConfig",
    "Stage2UtteranceCurriculumConfig",
    "Stage3Config",
    "Stage3CropCurriculumConfig",
    "Stage3HardNegativeConfig",
    "Stage3MarginScheduleConfig",
    "load_campp_baseline_config",
    "load_campp_model_selection_config",
    "load_campp_sweep_shortlist_report",
    "load_campp_sweep_shortlist_config",
    "load_campp_stage2_config",
    "load_campp_stage3_config",
    "render_campp_model_selection_markdown",
    "SweepBudgetConfig",
    "SweepCandidateConfig",
    "SweepCandidateResult",
    "SweepCropCurriculumOverride",
    "SweepMarginScheduleOverride",
    "SweepSelectionConfig",
    "SweepShortlistRunArtifacts",
    "SweepShortlistSummary",
    "SweepSuiteEvaluation",
    "render_campp_sweep_shortlist_markdown",
    "run_campp_baseline",
    "run_campp_model_selection",
    "run_campp_sweep_shortlist",
    "run_campp_stage2",
    "run_campp_stage3",
]
