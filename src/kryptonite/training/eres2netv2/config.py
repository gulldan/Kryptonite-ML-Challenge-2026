"""Typed config loader for the ERes2NetV2 baseline pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.models import ERes2NetV2Config
from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)
from kryptonite.training.config_helpers import (
    _coerce_int_list,
    _coerce_string_list,
    _load_provenance_config,
    _optional_section,
)

ERes2NetV2DataConfig = BaselineDataConfig
ERes2NetV2ObjectiveConfig = BaselineObjectiveConfig
ERes2NetV2OptimizationConfig = BaselineOptimizationConfig


@dataclass(frozen=True, slots=True)
class ERes2NetV2BaselineConfig:
    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: ERes2NetV2DataConfig
    model: ERes2NetV2Config
    objective: ERes2NetV2ObjectiveConfig
    optimization: ERes2NetV2OptimizationConfig
    provenance: BaselineProvenanceConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_config_path": self.base_config_path,
            "project_overrides": list(self.project_overrides),
            "project": self.project.to_dict(mask_secrets=True),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "objective": asdict(self.objective),
            "optimization": asdict(self.optimization),
            "provenance": self.provenance.to_dict(),
        }


def load_eres2netv2_baseline_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> ERes2NetV2BaselineConfig:
    config_file = Path(config_path)
    raw = tomllib.loads(config_file.read_text())
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
            "output_root": "artifacts/baselines/eres2netv2",
            "trials_manifest": None,
            "checkpoint_name": "eres2netv2_encoder.pt",
            "generate_demo_artifacts_if_missing": True,
            "max_train_rows": None,
            "max_dev_rows": None,
        },
    )
    model_section = _optional_section(raw, "model", {})
    objective_section = _optional_section(
        raw,
        "objective",
        {
            "classifier_blocks": 0,
            "classifier_hidden_dim": 192,
            "scale": 32.0,
            "margin": 0.3,
            "easy_margin": False,
        },
    )
    optimization_section = _optional_section(
        raw,
        "optimization",
        {
            "learning_rate": 0.2,
            "min_learning_rate": 5e-5,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "warmup_epochs": 5,
            "grad_clip_norm": 5.0,
        },
    )
    provenance_section = _optional_section(
        raw,
        "provenance",
        {
            "ruleset": "standard",
            "initialization": "from_scratch",
            "teacher_resources": [],
            "pretrained_resources": [],
            "notes": [],
        },
    )

    return ERes2NetV2BaselineConfig(
        base_config_path=base_config_path,
        project_overrides=merged_project_overrides,
        project=project,
        data=ERes2NetV2DataConfig(**data_section),
        model=_load_model_config(model_section),
        objective=ERes2NetV2ObjectiveConfig(**objective_section),
        optimization=ERes2NetV2OptimizationConfig(**optimization_section),
        provenance=_load_provenance_config(provenance_section),
    )


def _load_model_config(section: dict[str, Any]) -> ERes2NetV2Config:
    values = dict(section)
    if "num_blocks" in values:
        values["num_blocks"] = tuple(_coerce_int_list(values["num_blocks"], "num_blocks"))
    if "pooling_func" in values:
        values["pooling_func"] = str(values["pooling_func"]).upper()
    return ERes2NetV2Config(**values)
