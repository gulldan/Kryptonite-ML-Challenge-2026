"""Compatibility barrel for training recipes and orchestration helpers.

The canonical architecture lives in the subpackages; this module re-exports the
stable surface used by scripts and thin app entrypoints.
"""

from .augmentation_scheduler import (
    AugmentationScheduler,
    build_augmentation_scheduler_report,
    render_augmentation_scheduler_markdown,
    write_augmentation_scheduler_report,
)
from .campp import (
    CAMPPlusConsistencyRunArtifacts,
    CAMPPlusDistillationRunArtifacts,
    CAMPPlusRunArtifacts,
    load_campp_baseline_config,
    load_campp_consistency_config,
    load_campp_distillation_config,
    run_campp_baseline,
    run_campp_consistency,
    run_campp_distillation,
)
from .deployment import build_training_artifact_report
from .environment import build_training_environment_report, render_training_environment_report
from .eres2netv2 import (
    ERes2NetV2RunArtifacts,
    load_eres2netv2_baseline_config,
    run_eres2netv2_baseline,
)
from .production_dataloader import BalancedSpeakerBatchSampler, build_production_train_dataloader
from .teacher_peft import TeacherPeftRunArtifacts, load_teacher_peft_config, run_teacher_peft

__all__ = [
    "AugmentationScheduler",
    "BalancedSpeakerBatchSampler",
    "CAMPPlusConsistencyRunArtifacts",
    "CAMPPlusDistillationRunArtifacts",
    "CAMPPlusRunArtifacts",
    "ERes2NetV2RunArtifacts",
    "TeacherPeftRunArtifacts",
    "build_augmentation_scheduler_report",
    "build_production_train_dataloader",
    "build_training_artifact_report",
    "build_training_environment_report",
    "load_campp_baseline_config",
    "load_campp_consistency_config",
    "load_campp_distillation_config",
    "load_eres2netv2_baseline_config",
    "load_teacher_peft_config",
    "render_augmentation_scheduler_markdown",
    "render_training_environment_report",
    "run_campp_baseline",
    "run_campp_consistency",
    "run_campp_distillation",
    "run_eres2netv2_baseline",
    "run_teacher_peft",
    "write_augmentation_scheduler_report",
]
