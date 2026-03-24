"""Plan loading for reproducible RIR-bank assembly."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from .models import (
    ALLOWED_RIR_DIRECT_CONDITIONS,
    ALLOWED_RIR_FAMILIES,
    ALLOWED_RIR_FIELDS,
    ALLOWED_RIR_ROOM_SIZES,
    ALLOWED_RIR_RT60_BUCKETS,
    NumericRange,
    RIRAnalysisSettings,
    RIRBankPlan,
    RIRClassificationRule,
    RIRDirectCondition,
    RIRFamily,
    RIRField,
    RIRRoomSize,
    RIRRT60Bucket,
    RIRSourcePlan,
)


def load_rir_bank_plan(path: Path | str) -> RIRBankPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())
    analysis_data = data.get("analysis")
    if not isinstance(analysis_data, dict):
        raise ValueError("RIR bank plan must define an [analysis] table.")

    analysis = RIRAnalysisSettings(
        direct_window_ms=_require_float(analysis_data, "direct_window_ms"),
        reverb_start_ms=_require_float(analysis_data, "reverb_start_ms"),
        preview_duration_ms=_require_float(analysis_data, "preview_duration_ms"),
        preview_bins=_require_int(analysis_data, "preview_bins"),
        rt60_buckets=_load_ranges(
            analysis_data=analysis_data,
            table_name="rt60_buckets",
            allowed=ALLOWED_RIR_RT60_BUCKETS,
        ),
        field_buckets=_load_ranges(
            analysis_data=analysis_data,
            table_name="field_buckets",
            allowed=ALLOWED_RIR_FIELDS,
        ),
        direct_buckets=_load_ranges(
            analysis_data=analysis_data,
            table_name="direct_buckets",
            allowed=ALLOWED_RIR_DIRECT_CONDITIONS,
        ),
    )

    sources_data = data.get("sources")
    if not isinstance(sources_data, list) or not sources_data:
        raise ValueError("RIR bank plan must define at least one [[sources]] table.")

    sources: list[RIRSourcePlan] = []
    for source_data in sources_data:
        if not isinstance(source_data, dict):
            raise ValueError("RIR bank source entries must be TOML tables.")
        rules_data = source_data.get("classification_rules", [])
        if not isinstance(rules_data, list):
            raise ValueError("classification_rules must be a list of TOML tables.")
        rules: list[RIRClassificationRule] = []
        for rule_data in rules_data:
            if not isinstance(rule_data, dict):
                raise ValueError("Each RIR classification rule must be a TOML table.")
            rules.append(
                RIRClassificationRule(
                    match_any=tuple(
                        token.lower() for token in _require_str_list(rule_data, "match_any")
                    ),
                    room_size=_optional_room_size(rule_data, "room_size"),
                    field=_optional_field(rule_data, "field"),
                    rt60_bucket=_optional_rt60_bucket(rule_data, "rt60_bucket"),
                    direct_condition=_optional_direct_condition(rule_data, "direct_condition"),
                    tags=tuple(_optional_str_list(rule_data, "tags")),
                )
            )

        sources.append(
            RIRSourcePlan(
                id=_require_str(source_data, "id"),
                name=_require_str(source_data, "name"),
                inventory_source_id=_require_str(source_data, "inventory_source_id"),
                room_family=_require_family(source_data, "room_family"),
                root_candidates=tuple(_require_str_list(source_data, "root_candidates")),
                default_room_size=_require_room_size(source_data, "default_room_size"),
                base_weight=_require_float(source_data, "base_weight")
                if "base_weight" in source_data
                else 1.0,
                tags=tuple(_optional_str_list(source_data, "tags")),
                classification_rules=tuple(rules),
            )
        )

    return RIRBankPlan(
        notes=tuple(_optional_str_list(data, "notes")),
        analysis=analysis,
        sources=tuple(sources),
    )


def _load_ranges[BucketName: str](
    *,
    analysis_data: dict[str, object],
    table_name: str,
    allowed: tuple[BucketName, ...],
) -> dict[BucketName, NumericRange]:
    ranges_data = analysis_data.get(table_name)
    if not isinstance(ranges_data, dict):
        raise ValueError(f"RIR bank plan must define an [analysis.{table_name}] table.")
    typed_ranges_data = cast(dict[str, object], ranges_data)
    result: dict[BucketName, NumericRange] = {}
    for name in allowed:
        range_data = typed_ranges_data.get(name)
        if not isinstance(range_data, dict):
            raise ValueError(f"RIR bank plan must define [analysis.{table_name}.{name}] settings.")
        typed_range_data = cast(dict[str, object], range_data)
        result[name] = NumericRange(
            minimum=_optional_float(typed_range_data, "minimum", None),
            maximum=_optional_float(typed_range_data, "maximum", None),
        )
    return result


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RIR bank field '{key}' is missing or invalid.")
    return value


def _require_str_list(data: dict[str, object], key: str) -> list[str]:
    values = _optional_str_list(data, key)
    if not values:
        raise ValueError(f"RIR bank field '{key}' must contain at least one string.")
    return values


def _optional_str_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"RIR bank field '{key}' must be a list of strings.")
    return cast(list[str], list(value))


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"RIR bank field '{key}' must be an integer.")


def _require_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"RIR bank field '{key}' must be numeric.")


def _optional_float(
    data: dict[str, object],
    key: str,
    default: float | None,
) -> float | None:
    if key not in data:
        return default
    return _require_float(data, key)


def _require_literal(
    data: dict[str, object],
    key: str,
    allowed_values: tuple[str, ...],
) -> str:
    value = _require_str(data, key)
    if value not in allowed_values:
        raise ValueError(f"RIR bank field '{key}' must be one of: {', '.join(allowed_values)}.")
    return value


def _require_family(data: dict[str, object], key: str) -> RIRFamily:
    return cast(RIRFamily, _require_literal(data, key, ALLOWED_RIR_FAMILIES))


def _require_room_size(data: dict[str, object], key: str) -> RIRRoomSize:
    return cast(RIRRoomSize, _require_literal(data, key, ALLOWED_RIR_ROOM_SIZES))


def _optional_room_size(data: dict[str, object], key: str) -> RIRRoomSize | None:
    if key not in data:
        return None
    return cast(RIRRoomSize, _require_literal(data, key, ALLOWED_RIR_ROOM_SIZES))


def _optional_field(data: dict[str, object], key: str) -> RIRField | None:
    if key not in data:
        return None
    return cast(RIRField, _require_literal(data, key, ALLOWED_RIR_FIELDS))


def _optional_rt60_bucket(data: dict[str, object], key: str) -> RIRRT60Bucket | None:
    if key not in data:
        return None
    return cast(RIRRT60Bucket, _require_literal(data, key, ALLOWED_RIR_RT60_BUCKETS))


def _optional_direct_condition(
    data: dict[str, object],
    key: str,
) -> RIRDirectCondition | None:
    if key not in data:
        return None
    return cast(
        RIRDirectCondition,
        _require_literal(data, key, ALLOWED_RIR_DIRECT_CONDITIONS),
    )


__all__ = ["load_rir_bank_plan"]
