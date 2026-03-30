"""Typed config loader for the CAM++ stage-2 heavy multi-condition training pipeline."""

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

from .config import (
    _coerce_string_list,
    _load_model_config,
    _load_provenance_config,
    _optional_section,
)


@dataclass(frozen=True, slots=True)
class Stage2HardNegativeConfig:
    """Configuration for periodic hard-negative speaker mining."""

    enabled: bool = True
    mining_interval_epochs: int = 2
    top_k_per_speaker: int = 20
    hard_negative_fraction: float = 0.5
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
class Stage2UtteranceCurriculumConfig:
    """Short-to-long utterance curriculum configuration for stage-2."""

    enabled: bool = True
    short_crop_seconds: float = 1.5
    long_crop_seconds: float = 4.0
    curriculum_epochs: int = 5

    def __post_init__(self) -> None:
        if self.short_crop_seconds <= 0.0:
            raise ValueError("short_crop_seconds must be positive")
        if self.long_crop_seconds <= 0.0:
            raise ValueError("long_crop_seconds must be positive")
        if self.short_crop_seconds > self.long_crop_seconds:
            raise ValueError("short_crop_seconds must not exceed long_crop_seconds")
        if self.curriculum_epochs < 0:
            raise ValueError("curriculum_epochs must be non-negative")


@dataclass(frozen=True, slots=True)
class Stage2Config:
    """Stage-2 specific additions on top of the baseline configuration."""

    stage1_checkpoint: str
    hard_negative: Stage2HardNegativeConfig
    utterance_curriculum: Stage2UtteranceCurriculumConfig

    def __post_init__(self) -> None:
        if not self.stage1_checkpoint.strip():
            raise ValueError("stage2.stage1_checkpoint must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage1_checkpoint": self.stage1_checkpoint,
            "hard_negative": asdict(self.hard_negative),
            "utterance_curriculum": asdict(self.utterance_curriculum),
        }


@dataclass(frozen=True, slots=True)
class CAMPPlusStage2Config:
    """Full configuration for the CAM++ stage-2 heavy multi-condition training run."""

    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: BaselineDataConfig
    model: CAMPPlusConfig
    objective: BaselineObjectiveConfig
    optimization: BaselineOptimizationConfig
    provenance: BaselineProvenanceConfig
    stage2: Stage2Config

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
            "stage2": self.stage2.to_dict(),
        }


def load_campp_stage2_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> CAMPPlusStage2Config:
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
            "output_root": "artifacts/baselines/campp-stage2",
            "trials_manifest": None,
            "checkpoint_name": "campp_stage2_encoder.pt",
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
    stage2_section = _load_stage2_section(raw)

    return CAMPPlusStage2Config(
        base_config_path=base_config_path,
        project_overrides=merged_project_overrides,
        project=project,
        data=BaselineDataConfig(**data_section),
        model=_load_model_config(model_section),
        objective=BaselineObjectiveConfig(**objective_section),
        optimization=BaselineOptimizationConfig(**optimization_section),
        provenance=_load_provenance_config(provenance_section),
        stage2=stage2_section,
    )


def _load_stage2_section(raw: dict[str, Any]) -> Stage2Config:
    section = raw.get("stage2")
    if not isinstance(section, dict):
        raise ValueError(
            "[stage2] table is required in the config; at minimum set stage2.stage1_checkpoint."
        )

    checkpoint = section.get("stage1_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("stage2.stage1_checkpoint must be a non-empty string path")

    hn_raw = section.get("hard_negative", {})
    hn_defaults: dict[str, Any] = {
        "enabled": True,
        "mining_interval_epochs": 2,
        "top_k_per_speaker": 20,
        "hard_negative_fraction": 0.5,
        "max_train_rows_for_mining": None,
    }
    hn_merged = {**hn_defaults, **(hn_raw if isinstance(hn_raw, dict) else {})}
    hard_negative = Stage2HardNegativeConfig(**hn_merged)

    curr_raw = section.get("utterance_curriculum", {})
    curr_defaults: dict[str, Any] = {
        "enabled": True,
        "short_crop_seconds": 1.5,
        "long_crop_seconds": 4.0,
        "curriculum_epochs": 5,
    }
    curr_merged = {**curr_defaults, **(curr_raw if isinstance(curr_raw, dict) else {})}
    utterance_curriculum = Stage2UtteranceCurriculumConfig(**curr_merged)

    return Stage2Config(
        stage1_checkpoint=checkpoint.strip(),
        hard_negative=hard_negative,
        utterance_curriculum=utterance_curriculum,
    )


__all__ = [
    "CAMPPlusStage2Config",
    "Stage2Config",
    "Stage2HardNegativeConfig",
    "Stage2UtteranceCurriculumConfig",
    "load_campp_stage2_config",
]
