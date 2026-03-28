"""Stretch teacher training entrypoints backed by Hugging Face PEFT."""

from .config import (
    TeacherPeftAdapterConfig,
    TeacherPeftConfig,
    TeacherPeftDataConfig,
    TeacherPeftModelConfig,
    TeacherPeftObjectiveConfig,
    TeacherPeftOptimizationConfig,
    load_teacher_peft_config,
)
from .pipeline import TeacherPeftRunArtifacts, run_teacher_peft

__all__ = [
    "TeacherPeftAdapterConfig",
    "TeacherPeftConfig",
    "TeacherPeftDataConfig",
    "TeacherPeftModelConfig",
    "TeacherPeftObjectiveConfig",
    "TeacherPeftOptimizationConfig",
    "TeacherPeftRunArtifacts",
    "load_teacher_peft_config",
    "run_teacher_peft",
]
