"""JSON Schema validation helpers for TOML project configs."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def load_schema(schema_path: Path | str) -> dict[str, object]:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def load_toml_config(config_path: Path | str) -> dict[str, object]:
    payload = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must decode to an object: {config_path}")
    return payload


def validate_config_dict(
    payload: dict[str, object],
    *,
    schema: dict[str, object],
) -> list[ValidationError]:
    validator = Draft202012Validator(schema)
    return sorted(validator.iter_errors(payload), key=_error_sort_key)


def validate_config_file(
    *,
    config_path: Path | str,
    schema_path: Path | str,
) -> list[ValidationError]:
    return validate_config_dict(
        load_toml_config(config_path),
        schema=load_schema(schema_path),
    )


def format_validation_error(error: ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    if location:
        return f"{location}: {error.message}"
    return error.message


def _error_sort_key(error: ValidationError) -> tuple[str, str]:
    return (".".join(str(part) for part in error.absolute_path), error.message)
