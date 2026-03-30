"""Shared helpers for typed training config loaders."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)


@dataclass(frozen=True, slots=True)
class ParsedBaselineSections:
    """Standard TOML sections shared by every baseline config loader."""

    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: BaselineDataConfig
    model_section: dict[str, Any]
    objective: BaselineObjectiveConfig
    optimization: BaselineOptimizationConfig
    provenance: BaselineProvenanceConfig


def load_baseline_toml_sections(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
    data_defaults: dict[str, Any],
    objective_defaults: dict[str, Any] | None = None,
    optimization_defaults: dict[str, Any] | None = None,
) -> ParsedBaselineSections:
    """Parse a baseline TOML file and return the standard sections.

    Caller-provided *data_defaults* supply model-specific defaults for the
    ``[data]`` section (output_root, checkpoint_name, etc.).  Optional
    *objective_defaults* and *optimization_defaults* let models override the
    dataclass defaults where the common defaults are not appropriate.
    """
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

    data_section = _optional_section(raw, "data", data_defaults)
    model_section = _optional_section(raw, "model", {})
    objective_section = _optional_section(raw, "objective", objective_defaults or {})
    optimization_section = _optional_section(raw, "optimization", optimization_defaults or {})
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

    return ParsedBaselineSections(
        base_config_path=base_config_path,
        project_overrides=merged_project_overrides,
        project=project,
        data=BaselineDataConfig(**data_section),
        model_section=model_section,
        objective=BaselineObjectiveConfig(**objective_section),
        optimization=BaselineOptimizationConfig(**optimization_section),
        provenance=_load_provenance_config(provenance_section),
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


__all__ = [
    "ParsedBaselineSections",
    "_coerce_int_list",
    "_coerce_string_list",
    "_load_provenance_config",
    "_optional_section",
    "load_baseline_toml_sections",
]
