"""Typed config loader for the CAM++ teacher-student distillation recipe."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from kryptonite.models import CAMPPlusConfig
from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)

from .config import _coerce_int_list, _coerce_string_list, _load_provenance_config
from .stage3_config import (
    Stage3Config,
    load_campp_stage3_config,
)


@dataclass(frozen=True, slots=True)
class DistillationStudentConfig:
    checkpoint: str
    comparison_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint.strip():
            raise ValueError("student.checkpoint must not be empty.")
        if self.comparison_checkpoint is not None and not self.comparison_checkpoint.strip():
            raise ValueError("student.comparison_checkpoint must not be empty when provided.")

    @property
    def resolved_comparison_checkpoint(self) -> str:
        return self.checkpoint if self.comparison_checkpoint is None else self.comparison_checkpoint

    def to_dict(self) -> dict[str, str | None]:
        return {
            "checkpoint": self.checkpoint,
            "comparison_checkpoint": self.comparison_checkpoint,
        }


@dataclass(frozen=True, slots=True)
class DistillationTeacherConfig:
    checkpoint: str

    def __post_init__(self) -> None:
        if not self.checkpoint.strip():
            raise ValueError("teacher.checkpoint must not be empty.")


@dataclass(frozen=True, slots=True)
class DistillationLossConfig:
    classification_weight: float = 1.0
    embedding_weight: float = 0.35
    score_weight: float = 0.15

    def __post_init__(self) -> None:
        for field_name in ("classification_weight", "embedding_weight", "score_weight"):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.classification_weight <= 0.0:
            raise ValueError("distillation.classification_weight must be positive.")
        if self.embedding_weight <= 0.0 and self.score_weight <= 0.0:
            raise ValueError(
                "At least one teacher-driven loss must be enabled; set "
                "distillation.embedding_weight or distillation.score_weight above zero."
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CAMPPlusDistillationConfig:
    base_stage3_config_path: str
    project_overrides: tuple[str, ...]
    project: Any
    data: BaselineDataConfig
    model: CAMPPlusConfig
    objective: BaselineObjectiveConfig
    optimization: BaselineOptimizationConfig
    provenance: BaselineProvenanceConfig
    stage3: Stage3Config
    student: DistillationStudentConfig
    teacher: DistillationTeacherConfig
    distillation: DistillationLossConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_stage3_config_path": self.base_stage3_config_path,
            "project_overrides": list(self.project_overrides),
            "project": self.project.to_dict(mask_secrets=True),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "objective": asdict(self.objective),
            "optimization": asdict(self.optimization),
            "provenance": self.provenance.to_dict(),
            "stage3": self.stage3.to_dict(),
            "student": self.student.to_dict(),
            "teacher": asdict(self.teacher),
            "distillation": self.distillation.to_dict(),
        }


def load_campp_distillation_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> CAMPPlusDistillationConfig:
    config_file = Path(config_path)
    raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
    base_stage3_config_path = str(
        raw.get("base_stage3_config", "configs/training/campp-stage3.toml")
    ).strip()
    if not base_stage3_config_path:
        raise ValueError("base_stage3_config must not be empty.")

    merged_project_overrides = tuple(
        [*_coerce_string_list(raw.get("project_overrides")), *(project_overrides or [])]
    )
    base_config = load_campp_stage3_config(
        config_path=base_stage3_config_path,
        env_file=env_file,
        project_overrides=list(merged_project_overrides),
    )
    student = _load_student_config(raw)
    teacher = _load_teacher_config(raw)
    return CAMPPlusDistillationConfig(
        base_stage3_config_path=base_stage3_config_path,
        project_overrides=merged_project_overrides,
        project=base_config.project,
        data=_load_data_config(raw, base=base_config.data),
        model=_load_model_config(raw, base=base_config.model),
        objective=_load_objective_config(raw, base=base_config.objective),
        optimization=_load_optimization_config(raw, base=base_config.optimization),
        provenance=_apply_teacher_provenance(
            provenance=_load_provenance_config(
                {
                    **_provenance_defaults(base_config.provenance),
                    **_optional_dict(raw, "provenance"),
                }
            ),
            student=student,
            teacher=teacher,
        ),
        stage3=_load_stage3_override(raw, base=base_config.stage3),
        student=student,
        teacher=teacher,
        distillation=_load_distillation_config(raw),
    )


def _load_data_config(raw: dict[str, Any], *, base: BaselineDataConfig) -> BaselineDataConfig:
    defaults = asdict(
        replace(
            base,
            output_root="artifacts/baselines/campp-distillation",
            checkpoint_name="campp_distilled_encoder.pt",
        )
    )
    defaults.update(_optional_dict(raw, "data"))
    return BaselineDataConfig(**defaults)


def _load_model_config(raw: dict[str, Any], *, base: CAMPPlusConfig) -> CAMPPlusConfig:
    values = {**asdict(base), **_optional_dict(raw, "model")}
    for key in ("head_res_blocks", "block_layers", "block_kernel_sizes", "block_dilations"):
        value = values[key]
        if isinstance(value, list):
            values[key] = tuple(_coerce_int_list(value, key))
        else:
            values[key] = tuple(int(item) for item in value)
    return CAMPPlusConfig(**values)


def _load_objective_config(
    raw: dict[str, Any],
    *,
    base: BaselineObjectiveConfig,
) -> BaselineObjectiveConfig:
    values = {**asdict(base), **_optional_dict(raw, "objective")}
    return BaselineObjectiveConfig(**values)


def _load_optimization_config(
    raw: dict[str, Any],
    *,
    base: BaselineOptimizationConfig,
) -> BaselineOptimizationConfig:
    values = {**asdict(base), **_optional_dict(raw, "optimization")}
    return BaselineOptimizationConfig(**values)


def _load_stage3_override(raw: dict[str, Any], *, base: Stage3Config) -> Stage3Config:
    section = _optional_dict(raw, "stage3")
    return Stage3Config(
        stage2_checkpoint=base.stage2_checkpoint,
        hard_negative=(
            base.hard_negative
            if "hard_negative" not in section
            else type(base.hard_negative)(**section["hard_negative"])
        ),
        crop_curriculum=(
            base.crop_curriculum
            if "crop_curriculum" not in section
            else type(base.crop_curriculum)(**section["crop_curriculum"])
        ),
        margin_schedule=(
            base.margin_schedule
            if "margin_schedule" not in section
            else type(base.margin_schedule)(**section["margin_schedule"])
        ),
    )


def _load_student_config(raw: dict[str, Any]) -> DistillationStudentConfig:
    section = _optional_dict(raw, "student")
    if not section:
        raise ValueError("[student] table is required for CAM++ distillation.")
    return DistillationStudentConfig(
        checkpoint=str(section.get("checkpoint", "")).strip(),
        comparison_checkpoint=(
            None
            if section.get("comparison_checkpoint") in (None, "")
            else str(section.get("comparison_checkpoint")).strip()
        ),
    )


def _load_teacher_config(raw: dict[str, Any]) -> DistillationTeacherConfig:
    section = _optional_dict(raw, "teacher")
    if not section:
        raise ValueError("[teacher] table is required for CAM++ distillation.")
    return DistillationTeacherConfig(checkpoint=str(section.get("checkpoint", "")).strip())


def _load_distillation_config(raw: dict[str, Any]) -> DistillationLossConfig:
    return DistillationLossConfig(
        **{
            "classification_weight": 1.0,
            "embedding_weight": 0.35,
            "score_weight": 0.15,
            **_optional_dict(raw, "distillation"),
        }
    )


def _apply_teacher_provenance(
    *,
    provenance: BaselineProvenanceConfig,
    student: DistillationStudentConfig,
    teacher: DistillationTeacherConfig,
) -> BaselineProvenanceConfig:
    teacher_resources = tuple(dict.fromkeys([*provenance.teacher_resources, teacher.checkpoint]))
    pretrained_resources = tuple(
        dict.fromkeys([*provenance.pretrained_resources, student.checkpoint])
    )
    notes = tuple(
        dict.fromkeys(
            [
                *provenance.notes,
                "CAM++ distillation warm-started from a completed student checkpoint.",
                (
                    "Teacher guidance is loaded from the PEFT teacher checkpoint "
                    "and stays frozen during student updates."
                ),
            ]
        )
    )
    return BaselineProvenanceConfig(
        ruleset=provenance.ruleset,
        initialization=provenance.initialization,
        teacher_resources=teacher_resources,
        pretrained_resources=pretrained_resources,
        notes=notes,
    )


def _optional_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw.get(key)
    return dict(section) if isinstance(section, dict) else {}


def _provenance_defaults(provenance: BaselineProvenanceConfig) -> dict[str, Any]:
    values = asdict(provenance)
    for key in ("teacher_resources", "pretrained_resources", "notes"):
        values[key] = list(values[key])
    return values


__all__ = [
    "CAMPPlusDistillationConfig",
    "DistillationLossConfig",
    "DistillationStudentConfig",
    "DistillationTeacherConfig",
    "load_campp_distillation_config",
]
