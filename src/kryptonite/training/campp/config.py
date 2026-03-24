"""Typed config loader for the CAM++ baseline pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.models import CAMPPlusConfig
from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)

CAMPPlusDataConfig = BaselineDataConfig
CAMPPlusObjectiveConfig = BaselineObjectiveConfig
CAMPPlusOptimizationConfig = BaselineOptimizationConfig


@dataclass(frozen=True, slots=True)
class CAMPPlusBaselineConfig:
    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: CAMPPlusDataConfig
    model: CAMPPlusConfig
    objective: CAMPPlusObjectiveConfig
    optimization: CAMPPlusOptimizationConfig
    provenance: BaselineProvenanceConfig

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "base_config_path": self.base_config_path,
            "project_overrides": list(self.project_overrides),
            "project": self.project.to_dict(mask_secrets=True),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "objective": asdict(self.objective),
            "optimization": asdict(self.optimization),
            "provenance": self.provenance.to_dict(),
        }
        return payload


def load_campp_baseline_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> CAMPPlusBaselineConfig:
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
            "output_root": "artifacts/baselines/campp",
            "trials_manifest": None,
            "checkpoint_name": "campp_encoder.pt",
            "generate_demo_artifacts_if_missing": True,
            "max_train_rows": None,
            "max_dev_rows": None,
        },
    )
    model_section = _optional_section(raw, "model", {})
    objective_section = _optional_section(raw, "objective", {})
    optimization_section = _optional_section(raw, "optimization", {})
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

    return CAMPPlusBaselineConfig(
        base_config_path=base_config_path,
        project_overrides=merged_project_overrides,
        project=project,
        data=CAMPPlusDataConfig(**data_section),
        model=_load_model_config(model_section),
        objective=CAMPPlusObjectiveConfig(**objective_section),
        optimization=CAMPPlusOptimizationConfig(**optimization_section),
        provenance=_load_provenance_config(provenance_section),
    )


def _load_model_config(section: dict[str, Any]) -> CAMPPlusConfig:
    values = dict(section)
    for key in ("head_res_blocks", "block_layers", "block_kernel_sizes", "block_dilations"):
        if key in values:
            values[key] = tuple(_coerce_int_list(values[key], key))
    return CAMPPlusConfig(**values)


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
        raise ValueError("project_overrides must be a list of strings")
    return cast(list[str], list(value))


def _coerce_int_list(value: object, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a TOML array")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"{field_name} must contain only integer values")
    return cast(list[int], list(value))


def _load_provenance_config(section: dict[str, Any]) -> BaselineProvenanceConfig:
    values = dict(section)
    for key in ("ruleset", "initialization"):
        if key in values:
            values[key] = str(values[key]).strip().lower()
    for key in ("teacher_resources", "pretrained_resources", "notes"):
        if key in values:
            values[key] = tuple(_coerce_string_list(values[key]))
    return BaselineProvenanceConfig(**values)
