"""Typed config loader for the CAM++ stage-3 large-margin fine-tuning pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.models import CAMPPlusConfig
from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)

from .config import _coerce_string_list, _load_model_config, _load_provenance_config, _optional_section


@dataclass(frozen=True, slots=True)
class Stage3HardNegativeConfig:
    """Optional hard-negative mining during stage-3."""

    enabled: bool = False
    mining_interval_epochs: int = 2
    top_k_per_speaker: int = 20
    hard_negative_fraction: float = 0.25
    max_train_rows_for_mining: int | None = None

    def __post_init__(self) -> None:
        if self.mining_interval_epochs < 1:
            raise ValueError("mining_interval_epochs must be at least 1")
        if self.top_k_per_speaker < 1:
            raise ValueError("top_k_per_speaker must be at least 1")
        if not (0.0 <= self.hard_negative_fraction <= 1.0):
            raise ValueError("hard_negative_fraction must be in [0, 1]")
        if self.max_train_rows_for_mining is not None and self.max_train_rows_for_mining <= 0:
            raise ValueError("max_train_rows_for_mining must be positive when provided")


@dataclass(frozen=True, slots=True)
class Stage3CropCurriculumConfig:
    """Long-crop curriculum used in stage-3 target-like fine-tuning."""

    enabled: bool = True
    start_crop_seconds: float = 4.0
    end_crop_seconds: float = 6.0
    curriculum_epochs: int = 3

    def __post_init__(self) -> None:
        if self.start_crop_seconds <= 0.0:
            raise ValueError("start_crop_seconds must be positive")
        if self.end_crop_seconds <= 0.0:
            raise ValueError("end_crop_seconds must be positive")
        if self.start_crop_seconds > self.end_crop_seconds:
            raise ValueError("start_crop_seconds must not exceed end_crop_seconds")
        if self.curriculum_epochs < 0:
            raise ValueError("curriculum_epochs must be non-negative")


@dataclass(frozen=True, slots=True)
class Stage3MarginScheduleConfig:
    """Large-margin schedule for stage-3 fine-tuning."""

    enabled: bool = True
    start_margin: float = 0.3
    end_margin: float = 0.45
    ramp_epochs: int = 6

    def __post_init__(self) -> None:
        if self.start_margin < 0.0:
            raise ValueError("start_margin must be non-negative")
        if self.end_margin < 0.0:
            raise ValueError("end_margin must be non-negative")
        if self.start_margin > self.end_margin:
            raise ValueError("start_margin must not exceed end_margin")
        if self.ramp_epochs < 0:
            raise ValueError("ramp_epochs must be non-negative")


@dataclass(frozen=True, slots=True)
class Stage3Config:
    """Stage-3 specific additions on top of the baseline configuration."""

    stage2_checkpoint: str
    hard_negative: Stage3HardNegativeConfig
    crop_curriculum: Stage3CropCurriculumConfig
    margin_schedule: Stage3MarginScheduleConfig

    def __post_init__(self) -> None:
        if not self.stage2_checkpoint.strip():
            raise ValueError("stage3.stage2_checkpoint must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage2_checkpoint": self.stage2_checkpoint,
            "hard_negative": asdict(self.hard_negative),
            "crop_curriculum": asdict(self.crop_curriculum),
            "margin_schedule": asdict(self.margin_schedule),
        }


@dataclass(frozen=True, slots=True)
class CAMPPlusStage3Config:
    """Full configuration for the CAM++ stage-3 large-margin fine-tuning run."""

    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: BaselineDataConfig
    model: CAMPPlusConfig
    objective: BaselineObjectiveConfig
    optimization: BaselineOptimizationConfig
    provenance: BaselineProvenanceConfig
    stage3: Stage3Config

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
            "stage3": self.stage3.to_dict(),
        }


def load_campp_stage3_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> CAMPPlusStage3Config:
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
            "output_root": "artifacts/baselines/campp-stage3",
            "trials_manifest": None,
            "checkpoint_name": "campp_stage3_encoder.pt",
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
            "initialization": "pretrained",
            "teacher_resources": [],
            "pretrained_resources": [],
            "notes": [],
        },
    )
    stage3_section = _load_stage3_section(raw)

    return CAMPPlusStage3Config(
        base_config_path=base_config_path,
        project_overrides=merged_project_overrides,
        project=project,
        data=BaselineDataConfig(**data_section),
        model=_load_model_config(model_section),
        objective=BaselineObjectiveConfig(**objective_section),
        optimization=BaselineOptimizationConfig(**optimization_section),
        provenance=_load_provenance_config(provenance_section),
        stage3=stage3_section,
    )


def _load_stage3_section(raw: dict[str, Any]) -> Stage3Config:
    section = raw.get("stage3")
    if not isinstance(section, dict):
        raise ValueError(
            "[stage3] table is required in the config; at minimum set stage3.stage2_checkpoint."
        )

    checkpoint = section.get("stage2_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("stage3.stage2_checkpoint must be a non-empty string path")

    hn_raw = section.get("hard_negative", {})
    hn_defaults: dict[str, Any] = {
        "enabled": False,
        "mining_interval_epochs": 2,
        "top_k_per_speaker": 20,
        "hard_negative_fraction": 0.25,
        "max_train_rows_for_mining": None,
    }
    hn_merged = {**hn_defaults, **(hn_raw if isinstance(hn_raw, dict) else {})}

    crop_raw = section.get("crop_curriculum", {})
    crop_defaults: dict[str, Any] = {
        "enabled": True,
        "start_crop_seconds": 4.0,
        "end_crop_seconds": 6.0,
        "curriculum_epochs": 3,
    }
    crop_merged = {**crop_defaults, **(crop_raw if isinstance(crop_raw, dict) else {})}

    margin_raw = section.get("margin_schedule", {})
    margin_defaults: dict[str, Any] = {
        "enabled": True,
        "start_margin": 0.3,
        "end_margin": 0.45,
        "ramp_epochs": 6,
    }
    margin_merged = {
        **margin_defaults,
        **(margin_raw if isinstance(margin_raw, dict) else {}),
    }

    return Stage3Config(
        stage2_checkpoint=checkpoint.strip(),
        hard_negative=Stage3HardNegativeConfig(**hn_merged),
        crop_curriculum=Stage3CropCurriculumConfig(**crop_merged),
        margin_schedule=Stage3MarginScheduleConfig(**margin_merged),
    )
__all__ = [
    "CAMPPlusStage3Config",
    "Stage3Config",
    "Stage3CropCurriculumConfig",
    "Stage3HardNegativeConfig",
    "Stage3MarginScheduleConfig",
    "load_campp_stage3_config",
]
