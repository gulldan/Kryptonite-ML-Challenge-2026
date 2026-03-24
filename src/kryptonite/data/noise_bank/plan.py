"""Plan loading for reproducible noise-bank assembly."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from .models import (
    ALLOWED_NOISE_CATEGORIES,
    ALLOWED_NOISE_SEVERITIES,
    NoiseBankPlan,
    NoiseCategory,
    NoiseClassificationRule,
    NoiseSeverity,
    NoiseSeverityProfile,
    NoiseSourcePlan,
)


def load_noise_bank_plan(path: Path | str) -> NoiseBankPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())
    severity_data = data.get("severity_profiles")
    if not isinstance(severity_data, dict):
        raise ValueError("Noise bank plan must define a [severity_profiles] table.")

    severity_profiles: dict[NoiseSeverity, NoiseSeverityProfile] = {}
    for severity_name in ALLOWED_NOISE_SEVERITIES:
        profile_data = severity_data.get(severity_name)
        if not isinstance(profile_data, dict):
            raise ValueError(
                f"Noise bank plan must define [severity_profiles.{severity_name}] settings."
            )
        severity_profiles[severity_name] = NoiseSeverityProfile(
            snr_db_min=_require_float(profile_data, "snr_db_min"),
            snr_db_max=_require_float(profile_data, "snr_db_max"),
            weight_multiplier=_optional_float(profile_data, "weight_multiplier", 1.0),
        )

    sources_data = data.get("sources")
    if not isinstance(sources_data, list) or not sources_data:
        raise ValueError("Noise bank plan must define at least one [[sources]] table.")

    sources: list[NoiseSourcePlan] = []
    for source_data in sources_data:
        if not isinstance(source_data, dict):
            raise ValueError("Noise bank source entries must be TOML tables.")
        rules_data = source_data.get("classification_rules", [])
        if not isinstance(rules_data, list):
            raise ValueError("classification_rules must be a list of TOML tables.")
        rules = []
        for rule_data in rules_data:
            if not isinstance(rule_data, dict):
                raise ValueError("Each classification rule must be a TOML table.")
            rules.append(
                NoiseClassificationRule(
                    match_any=tuple(
                        token.lower() for token in _require_str_list(rule_data, "match_any")
                    ),
                    category=_optional_noise_category(rule_data, "category"),
                    severity=_optional_noise_severity(rule_data, "severity"),
                    tags=tuple(_optional_str_list(rule_data, "tags")),
                )
            )
        sources.append(
            NoiseSourcePlan(
                id=_require_str(source_data, "id"),
                name=_require_str(source_data, "name"),
                inventory_source_id=_require_str(source_data, "inventory_source_id"),
                root_candidates=tuple(_require_str_list(source_data, "root_candidates")),
                default_category=_require_noise_category(source_data, "default_category"),
                default_severity=_require_noise_severity(source_data, "default_severity"),
                base_weight=_optional_float(source_data, "base_weight", 1.0),
                tags=tuple(_optional_str_list(source_data, "tags")),
                classification_rules=tuple(rules),
            )
        )
    return NoiseBankPlan(
        notes=tuple(_optional_str_list(data, "notes")),
        severity_profiles=severity_profiles,
        sources=tuple(sources),
    )


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Noise bank field '{key}' is missing or invalid.")
    return value


def _require_str_list(data: dict[str, object], key: str) -> list[str]:
    values = _optional_str_list(data, key)
    if not values:
        raise ValueError(f"Noise bank field '{key}' must contain at least one string.")
    return values


def _optional_str_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Noise bank field '{key}' must be a list of strings.")
    return cast(list[str], list(value))


def _require_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Noise bank field '{key}' must be numeric.")


def _optional_float(data: dict[str, object], key: str, default: float) -> float:
    if key not in data:
        return default
    return _require_float(data, key)


def _require_noise_category(data: dict[str, object], key: str) -> NoiseCategory:
    return cast(NoiseCategory, _require_literal(data, key, ALLOWED_NOISE_CATEGORIES))


def _optional_noise_category(data: dict[str, object], key: str) -> NoiseCategory | None:
    if key not in data:
        return None
    return cast(NoiseCategory, _require_literal(data, key, ALLOWED_NOISE_CATEGORIES))


def _require_noise_severity(data: dict[str, object], key: str) -> NoiseSeverity:
    return cast(NoiseSeverity, _require_literal(data, key, ALLOWED_NOISE_SEVERITIES))


def _optional_noise_severity(data: dict[str, object], key: str) -> NoiseSeverity | None:
    if key not in data:
        return None
    return cast(NoiseSeverity, _require_literal(data, key, ALLOWED_NOISE_SEVERITIES))


def _require_literal(
    data: dict[str, object],
    key: str,
    allowed_values: tuple[str, ...],
) -> str:
    value = _require_str(data, key)
    if value not in allowed_values:
        raise ValueError(f"Noise bank field '{key}' must be one of: {', '.join(allowed_values)}.")
    return value


__all__ = ["load_noise_bank_plan"]
