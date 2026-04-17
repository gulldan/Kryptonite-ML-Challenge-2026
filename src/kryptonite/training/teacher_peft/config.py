"""Typed config loader for Hugging Face PEFT speaker recipes."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)
from kryptonite.training.config_helpers import (
    _coerce_string_list,
    load_baseline_toml_sections,
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
    pooling_mode: Literal["mean", "asp"] = "mean"
    gradient_checkpointing: bool = True
    freeze_feature_encoder: bool = True
    mfa_num_layers: int = 1
    layer_adapter_enabled: bool = False
    adapter_dim: int = 128

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
        if self.mfa_num_layers == 0 or self.mfa_num_layers < -1:
            raise ValueError("mfa_num_layers must be -1 or a positive integer")
        if self.adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive")
        if self.layer_adapter_enabled and self.mfa_num_layers == 1 and self.pooling_mode == "mean":
            raise ValueError(
                "Layer adapters are only meaningful for multi-layer aggregation; "
                "use mfa_num_layers=-1 or >1, or switch pooling_mode to 'asp'."
            )

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
    project: Any
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
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    sections = load_baseline_toml_sections(
        config_path=config_path,
        env_file=env_file,
        project_overrides=project_overrides,
        data_defaults={
            "train_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "dev_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "output_root": "artifacts/baselines/teacher-peft",
            "trials_manifest": None,
            "checkpoint_name": "teacher_peft",
            "generate_demo_artifacts_if_missing": True,
            "max_train_rows": None,
            "max_dev_rows": None,
        },
        optimization_defaults={
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
    model = TeacherPeftModelConfig(**sections.model_section)
    adapter = _load_adapter_config(sections.model_section, config_path=config_path)
    return TeacherPeftConfig(
        base_config_path=sections.base_config_path,
        project_overrides=sections.project_overrides,
        project=sections.project,
        data=sections.data,
        model=model,
        adapter=adapter,
        objective=sections.objective,
        optimization=sections.optimization,
        provenance=_load_teacher_provenance(
            sections.provenance,
            model=model,
            provenance_explicit=isinstance(raw.get("provenance"), dict),
        ),
    )


def _load_adapter_config(
    model_section: dict[str, Any],
    *,
    config_path: Path | str,
) -> TeacherPeftAdapterConfig:
    import tomllib

    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    adapter_section = raw.get("adapter") or {}
    if not isinstance(adapter_section, dict):
        raise ValueError("Config section 'adapter' must be a table.")
    values = dict(adapter_section)
    if "target_modules" in values:
        values["target_modules"] = tuple(_coerce_string_list(values["target_modules"]))
    return TeacherPeftAdapterConfig(**values)


def _load_teacher_provenance(
    provenance: BaselineProvenanceConfig,
    *,
    model: TeacherPeftModelConfig,
    provenance_explicit: bool,
) -> BaselineProvenanceConfig:
    initialization = provenance.initialization if provenance_explicit else "pretrained"
    pretrained_resources = provenance.pretrained_resources
    if initialization == "pretrained" and not pretrained_resources:
        pretrained_resources = (f"huggingface://{model.model_id}",)
    return BaselineProvenanceConfig(
        ruleset=provenance.ruleset,
        initialization=initialization,
        teacher_resources=provenance.teacher_resources,
        pretrained_resources=pretrained_resources,
        notes=provenance.notes,
    )
