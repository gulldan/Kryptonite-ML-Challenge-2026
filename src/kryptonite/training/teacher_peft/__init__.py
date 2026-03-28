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
from .model import (
    KNOWN_TEACHER_PEFT_CHECKPOINT_NAMES,
    TeacherPeftEncoder,
    load_teacher_peft_encoder_from_checkpoint,
    resolve_teacher_peft_checkpoint_path,
)
from .pipeline import TeacherPeftRunArtifacts, run_teacher_peft

__all__ = [
    "KNOWN_TEACHER_PEFT_CHECKPOINT_NAMES",
    "TeacherPeftAdapterConfig",
    "TeacherPeftConfig",
    "TeacherPeftDataConfig",
    "TeacherPeftEncoder",
    "TeacherPeftModelConfig",
    "TeacherPeftObjectiveConfig",
    "TeacherPeftOptimizationConfig",
    "TeacherPeftRunArtifacts",
    "load_teacher_peft_config",
    "load_teacher_peft_encoder_from_checkpoint",
    "resolve_teacher_peft_checkpoint_path",
    "run_teacher_peft",
]
