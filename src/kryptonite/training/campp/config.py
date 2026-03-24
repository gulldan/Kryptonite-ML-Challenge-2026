"""Typed config loader for the CAM++ baseline pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.models import CAMPPlusConfig


@dataclass(frozen=True, slots=True)
class CAMPPlusDataConfig:
    train_manifest: str
    dev_manifest: str
    output_root: str = "artifacts/baselines/campp"
    trials_manifest: str | None = None
    checkpoint_name: str = "campp_encoder.pt"
    generate_demo_artifacts_if_missing: bool = True
    max_train_rows: int | None = None
    max_dev_rows: int | None = None

    def __post_init__(self) -> None:
        if not self.train_manifest.strip():
            raise ValueError("train_manifest must not be empty")
        if not self.dev_manifest.strip():
            raise ValueError("dev_manifest must not be empty")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty")
        if not self.checkpoint_name.strip():
            raise ValueError("checkpoint_name must not be empty")
        if self.max_train_rows is not None and self.max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive when provided")
        if self.max_dev_rows is not None and self.max_dev_rows <= 0:
            raise ValueError("max_dev_rows must be positive when provided")


@dataclass(frozen=True, slots=True)
class CAMPPlusObjectiveConfig:
    classifier_blocks: int = 0
    classifier_hidden_dim: int = 512
    scale: float = 32.0
    margin: float = 0.2
    easy_margin: bool = False

    def __post_init__(self) -> None:
        if self.classifier_blocks < 0:
            raise ValueError("classifier_blocks must be non-negative")
        if self.classifier_hidden_dim <= 0:
            raise ValueError("classifier_hidden_dim must be positive")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")


@dataclass(frozen=True, slots=True)
class CAMPPlusOptimizationConfig:
    learning_rate: float = 0.1
    min_learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    grad_clip_norm: float | None = 5.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if self.momentum < 0.0:
            raise ValueError("momentum must be non-negative")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided")


@dataclass(frozen=True, slots=True)
class CAMPPlusBaselineConfig:
    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: CAMPPlusDataConfig
    model: CAMPPlusConfig
    objective: CAMPPlusObjectiveConfig
    optimization: CAMPPlusOptimizationConfig

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "base_config_path": self.base_config_path,
            "project_overrides": list(self.project_overrides),
            "project": self.project.to_dict(mask_secrets=True),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "objective": asdict(self.objective),
            "optimization": asdict(self.optimization),
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

    return CAMPPlusBaselineConfig(
        base_config_path=base_config_path,
        project_overrides=merged_project_overrides,
        project=project,
        data=CAMPPlusDataConfig(**data_section),
        model=_load_model_config(model_section),
        objective=CAMPPlusObjectiveConfig(**objective_section),
        optimization=CAMPPlusOptimizationConfig(**optimization_section),
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
