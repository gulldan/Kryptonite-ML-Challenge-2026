"""Training recipes and orchestration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "BaselineConfig": (".baseline_config", "BaselineConfig"),
    "run_speaker_baseline": (".baseline_pipeline", "run_speaker_baseline"),
    "CAMPPlusRunArtifacts": (".campp", "CAMPPlusRunArtifacts"),
    "load_campp_baseline_config": (".campp", "load_campp_baseline_config"),
    "run_campp_baseline": (".campp", "run_campp_baseline"),
    "build_training_environment_report": (
        ".environment",
        "build_training_environment_report",
    ),
    "render_training_environment_report": (
        ".environment",
        "render_training_environment_report",
    ),
    "ERes2NetV2RunArtifacts": (".eres2netv2", "ERes2NetV2RunArtifacts"),
    "load_eres2netv2_baseline_config": (
        ".eres2netv2",
        "load_eres2netv2_baseline_config",
    ),
    "run_eres2netv2_baseline": (".eres2netv2", "run_eres2netv2_baseline"),
    "BalancedSpeakerBatchSampler": (
        ".production_dataloader",
        "BalancedSpeakerBatchSampler",
    ),
    "build_production_train_dataloader": (
        ".production_dataloader",
        "build_production_train_dataloader",
    ),
    "TeacherPeftRunArtifacts": (".teacher_peft", "TeacherPeftRunArtifacts"),
    "load_teacher_peft_config": (".teacher_peft", "load_teacher_peft_config"),
    "run_teacher_peft": (".teacher_peft", "run_teacher_peft"),
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
