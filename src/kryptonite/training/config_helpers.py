"""Shared helpers for typed training config loaders."""

from __future__ import annotations

from typing import Any, cast

from kryptonite.training.baseline_config import BaselineProvenanceConfig


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
    "_coerce_int_list",
    "_coerce_string_list",
    "_load_provenance_config",
    "_optional_section",
]
