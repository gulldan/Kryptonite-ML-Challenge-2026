"""Plan loading for frozen corrupted dev suites."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from .models import (
    CorruptedDevSuiteSpec,
    CorruptedDevSuitesPlan,
    DistanceFieldWeights,
    ReverbDirectWeights,
    SeverityWeights,
    SuiteFamily,
)


def load_corrupted_dev_suites_plan(path: Path | str) -> CorruptedDevSuitesPlan:
    raw = tomllib.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Corrupted dev suites plan must be a TOML table.")

    suites_raw = raw.get("suites")
    if not isinstance(suites_raw, list) or not suites_raw:
        raise ValueError("Corrupted dev suites plan must define at least one [[suites]] table.")

    trial_manifest_paths = _optional_string_list(raw.get("trial_manifest_paths"))
    return CorruptedDevSuitesPlan(
        output_root=_require_string(raw, "output_root"),
        source_manifest_path=_require_string(raw, "source_manifest_path"),
        trial_manifest_paths=tuple(trial_manifest_paths),
        seed=_require_int(raw, "seed"),
        suites=tuple(_load_suite_spec(entry) for entry in suites_raw),
    )


def _load_suite_spec(entry: object) -> CorruptedDevSuiteSpec:
    if not isinstance(entry, dict):
        raise ValueError("Each [[suites]] entry must be a TOML table.")
    entry_data = cast(dict[str, object], entry)

    family = cast(SuiteFamily, _require_string(entry_data, "family"))
    return CorruptedDevSuiteSpec(
        suite_id=_require_string(entry_data, "suite_id"),
        family=family,
        description=_require_string(entry_data, "description"),
        severity_weights=_load_severity_weights(entry_data.get("severity_weights")),
        codec_families=tuple(_optional_string_list(entry_data.get("codec_families"))),
        reverb_direct_weights=_load_reverb_direct_weights(entry_data.get("reverb_direct_weights")),
        distance_field_weights=_load_distance_field_weights(
            entry_data.get("distance_field_weights")
        ),
    )


def _load_severity_weights(value: object) -> SeverityWeights:
    if value is None:
        return SeverityWeights()
    table = _require_table(value, "severity_weights")
    return SeverityWeights(
        light=_optional_float(table, "light", default=1.0),
        medium=_optional_float(table, "medium", default=1.0),
        heavy=_optional_float(table, "heavy", default=1.0),
    )


def _load_reverb_direct_weights(value: object) -> ReverbDirectWeights | None:
    if value is None:
        return None
    table = _require_table(value, "reverb_direct_weights")
    return ReverbDirectWeights(
        high=_optional_float(table, "high", default=1.0),
        medium=_optional_float(table, "medium", default=1.0),
        low=_optional_float(table, "low", default=1.0),
    )


def _load_distance_field_weights(value: object) -> DistanceFieldWeights | None:
    if value is None:
        return None
    table = _require_table(value, "distance_field_weights")
    return DistanceFieldWeights(
        near=_optional_float(table, "near", default=1.0),
        mid=_optional_float(table, "mid", default=1.0),
        far=_optional_float(table, "far", default=1.0),
    )


def _require_table(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table.")
    return cast(dict[str, object], value)


def _require_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Plan field '{key}' is missing or invalid.")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Plan field '{key}' must be an integer.")
    return value


def _optional_float(data: dict[str, object], key: str, *, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Plan field '{key}' must be numeric.")
    return float(value)


def _optional_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected a list of strings.")
    return [cast(str, item) for item in value if cast(str, item).strip()]
