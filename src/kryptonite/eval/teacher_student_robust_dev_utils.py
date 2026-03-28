"""Shared helpers for teacher-vs-student robust-dev evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return cast(dict[str, Any], payload)


def require_mapping(raw: object, field_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be an object.")
    return cast(dict[str, Any], dict(raw))


def optional_int(raw: object) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int):
        raise ValueError("Expected an integer value.")
    return raw


def optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)):
        raise ValueError("Expected a numeric value.")
    return float(raw)


def state_dict_parameter_count(raw: object) -> int:
    state = require_mapping(raw, "model_state_dict")
    return sum(int(tensor.numel()) for tensor in state.values())


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "load_json_object",
    "optional_float",
    "optional_int",
    "parameter_count",
    "require_mapping",
    "state_dict_parameter_count",
    "trainable_parameter_count",
]
