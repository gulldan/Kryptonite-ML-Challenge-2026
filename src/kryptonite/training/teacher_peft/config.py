"""Typed config loader for the stretch teacher PEFT pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)

TeacherPeftDataConfig = BaselineDataConfig
TeacherPeftObjectiveConfig = BaselineObjectiveConfig
TeacherPeftOptimizationConfig = BaselineOptimizationConfig


@dataclass(frozen=True, slots=True)
class TeacherPeftModelConfig:
    model_id: str = "microsoft/wavlm-base-plus"
    feature_extractor_id: str | None = None
    revision: str | None = None
    embedding_dim: int = 256
    projection_dropout: float = 0.1
    pooling_mode: str = "mean"
    gradient_checkpointing: bool = True
    freeze_feature_encoder: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.feature_extractor_id is not None and not self.feature_extractor_id.strip():
            raise ValueError("feature_extractor_id must not be empty when provided")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision must not be empty when provided")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= self.projection_dropout < 1.0:
            raise ValueError("projection_dropout must be within [0.0, 1.0)")
        if self.pooling_mode != "mean":
            raise ValueError("pooling_mode must currently be 'mean'")

    @property
    def resolved_feature_extractor_id(self) -> str:
        return self.model_id if self.feature_extractor_id is None else self.feature_extractor_id


@dataclass(frozen=True, slots=True)
class TeacherPeftAdapterConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("all-linear",)
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0.0, 1.0)")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        for module_name in self.target_modules:
            if not module_name.strip():
                raise ValueError("target_modules entries must not be empty")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("bias must be one of: none, all, lora_only")


@dataclass(frozen=True, slots=True)
class TeacherPeftConfig:
    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: TeacherPeftDataConfig
    model: TeacherPeftModelConfig
    adapter: TeacherPeftAdapterConfig
    objective: TeacherPeftObjectiveConfig
    optimization: TeacherPeftOptimizationConfig
    provenance: BaselineProvenanceConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_config_path": self.base_config_path,
            "project_overrides": list(self.project_overrides),
            "project": self.project.to_dict(mask_secrets=True),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "adapter": {
                **asdict(self.adapter),
                "target_modules": list(self.adapter.target_modules),
            },
            "objective": asdict(self.objective),
            "optimization": asdict(self.optimization),
            "provenance": self.provenance.to_dict(),
        }


def load_teacher_peft_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> TeacherPeftConfig:
    config_file = Path(config_path)
    raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
    base_config_path = str(raw.get("base_config", "configs/base.toml"))

    merged_project_overrides = tuple(
        [*_coerce_string_list(raw.get("project_overrides")), *(project_overrides or [])]
    )
    project = load_project_config(
        config_path=base_config_path,
        overrides=list(merged_project_overrides),
        env_file=env_file,
    )

    data_section = _optional_section(
        raw,
        "data",
        {
            "train_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "dev_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "output_root": "artifacts/baselines/teacher-peft",
            "trials_manifest": None,
            "checkpoint_name": "teacher_peft",
            "generate_demo_artifacts_if_missing": True,
            "max_train_rows": None,
            "max_dev_rows": None,
        },
    )
    model = TeacherPeftModelConfig(**_optional_section(raw, "model", {}))
    adapter = _load_adapter_config(_optional_section(raw, "adapter", {}))
    objective = TeacherPeftObjectiveConfig(**_optional_section(raw, "objective", {}))
    optimization = TeacherPeftOptimizationConfig(
        **_optional_section(
            raw,
            "optimization",
            {
                "optimizer_name": "adamw",
                "scheduler_name": "cosine",
                "learning_rate": 0.0002,
                "min_learning_rate": 0.00001,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "warmup_epochs": 1,
                "gradient_accumulation_steps": 8,
                "grad_clip_norm": 1.0,
            },
        )
    )
    provenance = _load_provenance_config(
        _optional_section(
            raw,
            "provenance",
            {
                "ruleset": "standard",
                "initialization": "pretrained",
                "teacher_resources": [],
                "pretrained_resources": [f"huggingface://{model.model_id}"],
                "notes": [],
            },
        )
    )

    return TeacherPeftConfig(
        base_config_path=base_config_path,
        project_overrides=merged_project_overrides,
        project=project,
        data=TeacherPeftDataConfig(**data_section),
        model=model,
        adapter=adapter,
        objective=objective,
        optimization=optimization,
        provenance=provenance,
    )


def _optional_section(data: dict[str, Any], name: str, default: dict[str, Any]) -> dict[str, Any]:
    value = data.get(name)
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a table.")
    return {**default, **value}


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected a TOML array of strings")
    return cast(list[str], list(value))


def _load_adapter_config(section: dict[str, Any]) -> TeacherPeftAdapterConfig:
    values = dict(section)
    if "target_modules" in values:
        values["target_modules"] = tuple(_coerce_string_list(values["target_modules"]))
    return TeacherPeftAdapterConfig(**values)


def _load_provenance_config(section: dict[str, Any]) -> BaselineProvenanceConfig:
    values = dict(section)
    for key in ("ruleset", "initialization"):
        if key in values:
            values[key] = str(values[key]).strip().lower()
    for key in ("teacher_resources", "pretrained_resources", "notes"):
        if key in values:
            values[key] = tuple(_coerce_string_list(values[key]))
    return BaselineProvenanceConfig(**values)
