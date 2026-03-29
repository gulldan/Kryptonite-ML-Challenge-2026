"""Typed config loader for the CAM++ baseline pipeline."""

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
from kryptonite.training.config_helpers import (
    _coerce_int_list,
    _coerce_string_list,
    _load_provenance_config,
    _optional_section,
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
