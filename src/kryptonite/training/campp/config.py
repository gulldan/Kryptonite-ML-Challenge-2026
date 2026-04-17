"""Typed config loader for the CAM++ baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kryptonite.config import ProjectConfig
from kryptonite.models import CAMPPlusConfig
from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)
from kryptonite.training.config_helpers import (
    _coerce_int_list,
    _coerce_string_list,  # noqa: F401 — re-exported for advanced config loaders
    _load_provenance_config,  # noqa: F401 — re-exported for advanced config loaders
    _optional_section,  # noqa: F401 — re-exported for advanced config loaders
    load_baseline_toml_sections,
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


def load_campp_baseline_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> CAMPPlusBaselineConfig:
    sections = load_baseline_toml_sections(
        config_path=config_path,
        env_file=env_file,
        project_overrides=project_overrides,
        data_defaults={
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
    return CAMPPlusBaselineConfig(
        base_config_path=sections.base_config_path,
        project_overrides=sections.project_overrides,
        project=sections.project,
        data=sections.data,
        model=_load_model_config(sections.model_section),
        objective=sections.objective,
        optimization=sections.optimization,
        provenance=sections.provenance,
    )


def _load_model_config(section: dict[str, Any]) -> CAMPPlusConfig:
    values = dict(section)
    for key in ("head_res_blocks", "block_layers", "block_kernel_sizes", "block_dilations"):
        if key in values:
            values[key] = tuple(_coerce_int_list(values[key], key))
    return CAMPPlusConfig(**values)
