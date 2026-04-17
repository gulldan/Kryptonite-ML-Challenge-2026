"""Shared parsing helpers for typed config loaders."""

from __future__ import annotations


def coerce_table(raw: object, field_name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a table.")
    return {str(key): value for key, value in raw.items()}


def coerce_string_list(raw: object, field_name: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array of strings.")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string.")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{field_name}[{index}] must not be empty.")
        values.append(stripped)
    return values


def coerce_optional_string(raw: object, *, field_name: str = "value") -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be a string when provided.")
    stripped = raw.strip()
    return stripped or None


def coerce_optional_float(raw: object, *, field_name: str = "value") -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a number when provided.")
    if isinstance(raw, (int, float)):
        return float(raw)
    raise ValueError(f"{field_name} must be a number when provided.")


def coerce_required_float(raw: object, field_name: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a number.")
    if isinstance(raw, (int, float)):
        return float(raw)
    raise ValueError(f"{field_name} must be a number.")


def coerce_required_int(raw: object, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field_name} must be an integer.")
    return raw
