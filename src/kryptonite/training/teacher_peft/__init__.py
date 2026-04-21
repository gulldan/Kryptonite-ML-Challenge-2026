"""Hugging Face PEFT speaker-training helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "TeacherPeftAdapterConfig": (".config", "TeacherPeftAdapterConfig"),
    "TeacherPeftConfig": (".config", "TeacherPeftConfig"),
    "TeacherPeftDataConfig": (".config", "TeacherPeftDataConfig"),
    "TeacherPeftModelConfig": (".config", "TeacherPeftModelConfig"),
    "TeacherPeftObjectiveConfig": (".config", "TeacherPeftObjectiveConfig"),
    "TeacherPeftOptimizationConfig": (".config", "TeacherPeftOptimizationConfig"),
    "load_teacher_peft_config": (".config", "load_teacher_peft_config"),
    "KNOWN_TEACHER_PEFT_CHECKPOINT_NAMES": (
        ".model",
        "KNOWN_TEACHER_PEFT_CHECKPOINT_NAMES",
    ),
    "TeacherPeftEncoder": (".model", "TeacherPeftEncoder"),
    "load_teacher_checkpoint_payload": (".model", "load_teacher_checkpoint_payload"),
    "load_teacher_peft_encoder_from_checkpoint": (
        ".model",
        "load_teacher_peft_encoder_from_checkpoint",
    ),
    "merge_teacher_lora_backbone": (".model", "merge_teacher_lora_backbone"),
    "resolve_teacher_peft_checkpoint_path": (
        ".model",
        "resolve_teacher_peft_checkpoint_path",
    ),
    "TeacherPeftRunArtifacts": (".pipeline", "TeacherPeftRunArtifacts"),
    "run_teacher_peft": (".pipeline", "run_teacher_peft"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
