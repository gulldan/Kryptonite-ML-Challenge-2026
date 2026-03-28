"""Typed config loader for the CAM++ clean/corrupted consistency recipe."""

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
from .stage3_config import Stage3Config, load_campp_stage3_config


@dataclass(frozen=True, slots=True)
class ConsistencyStudentConfig:
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
class ConsistencyLossConfig:
    clean_classification_weight: float = 1.0
    corrupted_classification_weight: float = 0.5
    embedding_weight: float = 0.25
    score_weight: float = 0.1

    def __post_init__(self) -> None:
        for field_name in (
            "clean_classification_weight",
            "corrupted_classification_weight",
            "embedding_weight",
            "score_weight",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.clean_classification_weight <= 0.0:
            raise ValueError("consistency.clean_classification_weight must be positive.")
        if self.embedding_weight <= 0.0 and self.score_weight <= 0.0:
            raise ValueError(
                "At least one consistency loss must be enabled; set "
                "consistency.embedding_weight or consistency.score_weight above zero."
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConsistencyRobustDevConfig:
    enabled: bool = True
    catalog_path: str = "artifacts/eval/corrupted-dev-suites/corrupted_dev_suites_catalog.json"
    suite_ids: tuple[str, ...] = ()
    clean_weight: float = 0.25
    corrupted_weight: float = 0.75

    def __post_init__(self) -> None:
        if self.enabled and not self.catalog_path.strip():
            raise ValueError("robust_dev.catalog_path must not be empty when enabled.")
        if self.clean_weight < 0.0:
            raise ValueError("robust_dev.clean_weight must be non-negative.")
        if self.corrupted_weight < 0.0:
            raise ValueError("robust_dev.corrupted_weight must be non-negative.")
        if self.enabled and (self.clean_weight + self.corrupted_weight) <= 0.0:
            raise ValueError("robust_dev weights must sum to a positive value.")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "catalog_path": self.catalog_path,
            "suite_ids": list(self.suite_ids),
            "clean_weight": self.clean_weight,
            "corrupted_weight": self.corrupted_weight,
        }


@dataclass(frozen=True, slots=True)
class CAMPPlusConsistencyConfig:
    base_stage3_config_path: str
    project_overrides: tuple[str, ...]
    project: Any
    data: BaselineDataConfig
    model: CAMPPlusConfig
    objective: BaselineObjectiveConfig
    optimization: BaselineOptimizationConfig
    provenance: BaselineProvenanceConfig
    stage3: Stage3Config
    student: ConsistencyStudentConfig
    consistency: ConsistencyLossConfig
    robust_dev: ConsistencyRobustDevConfig

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
            "consistency": self.consistency.to_dict(),
            "robust_dev": self.robust_dev.to_dict(),
        }


def load_campp_consistency_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> CAMPPlusConsistencyConfig:
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
    return CAMPPlusConsistencyConfig(
        base_stage3_config_path=base_stage3_config_path,
        project_overrides=merged_project_overrides,
        project=base_config.project,
        data=_load_data_config(raw, base=base_config.data),
        model=_load_model_config(raw, base=base_config.model),
        objective=_load_objective_config(raw, base=base_config.objective),
        optimization=_load_optimization_config(raw, base=base_config.optimization),
        provenance=_apply_student_provenance(
            provenance=_load_provenance_config(
                {
                    **_provenance_defaults(base_config.provenance),
                    **_optional_dict(raw, "provenance"),
                }
            ),
            student=student,
        ),
        stage3=_load_stage3_override(raw, base=base_config.stage3),
        student=student,
        consistency=_load_consistency_config(raw),
        robust_dev=_load_robust_dev_config(raw),
    )


def _load_data_config(raw: dict[str, Any], *, base: BaselineDataConfig) -> BaselineDataConfig:
    defaults = asdict(
        replace(
            base,
            output_root="artifacts/baselines/campp-consistency",
            checkpoint_name="campp_consistency_encoder.pt",
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


def _load_student_config(raw: dict[str, Any]) -> ConsistencyStudentConfig:
    section = _optional_dict(raw, "student")
    if not section:
        raise ValueError("[student] table is required for CAM++ consistency.")
    return ConsistencyStudentConfig(
        checkpoint=str(section.get("checkpoint", "")).strip(),
        comparison_checkpoint=(
            None
            if section.get("comparison_checkpoint") in (None, "")
            else str(section.get("comparison_checkpoint")).strip()
        ),
    )


def _load_consistency_config(raw: dict[str, Any]) -> ConsistencyLossConfig:
    return ConsistencyLossConfig(
        **{
            "clean_classification_weight": 1.0,
            "corrupted_classification_weight": 0.5,
            "embedding_weight": 0.25,
            "score_weight": 0.1,
            **_optional_dict(raw, "consistency"),
        }
    )


def _load_robust_dev_config(raw: dict[str, Any]) -> ConsistencyRobustDevConfig:
    section = _optional_dict(raw, "robust_dev")
    raw_suite_ids = section.get("suite_ids", [])
    suite_ids = (
        [item for item in raw_suite_ids if isinstance(item, str)]
        if isinstance(raw_suite_ids, (list, tuple))
        else []
    )
    return ConsistencyRobustDevConfig(
        enabled=bool(section.get("enabled", True)),
        catalog_path=str(
            section.get(
                "catalog_path",
                "artifacts/eval/corrupted-dev-suites/corrupted_dev_suites_catalog.json",
            )
        ).strip(),
        suite_ids=tuple(suite_ids),
        clean_weight=float(section.get("clean_weight", 0.25)),
        corrupted_weight=float(section.get("corrupted_weight", 0.75)),
    )


def _apply_student_provenance(
    *,
    provenance: BaselineProvenanceConfig,
    student: ConsistencyStudentConfig,
) -> BaselineProvenanceConfig:
    pretrained_resources = tuple(
        dict.fromkeys([*provenance.pretrained_resources, student.checkpoint])
    )
    notes = tuple(
        dict.fromkeys(
            [
                *provenance.notes,
                "CAM++ consistency fine-tuning warm-starts from a completed stage-3 checkpoint.",
                (
                    "Clean/corrupted pair supervision is applied on aligned "
                    "crops of the same utterance."
                ),
            ]
        )
    )
    return BaselineProvenanceConfig(
        ruleset=provenance.ruleset,
        initialization=provenance.initialization,
        teacher_resources=provenance.teacher_resources,
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
    "CAMPPlusConsistencyConfig",
    "ConsistencyLossConfig",
    "ConsistencyRobustDevConfig",
    "ConsistencyStudentConfig",
    "load_campp_consistency_config",
]
