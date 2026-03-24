"""Plan loading for reproducible far-field and distance simulation presets."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from .models import (
    ALLOWED_DISTANCE_FIELDS,
    DistanceField,
    FarFieldBankPlan,
    FarFieldProbeSettings,
    FarFieldRenderSettings,
    FarFieldSimulationPreset,
)


def load_far_field_bank_plan(path: Path | str) -> FarFieldBankPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())

    probe_data = data.get("probe")
    if not isinstance(probe_data, dict):
        raise ValueError("Far-field bank plan must define a [probe] table.")
    probe = FarFieldProbeSettings(
        sample_rate_hz=_require_int(probe_data, "sample_rate_hz"),
        duration_seconds=_require_float(probe_data, "duration_seconds"),
        peak_amplitude=cast(float, _optional_float(probe_data, "peak_amplitude", 0.85)),
    )

    render_data = data.get("render")
    if not isinstance(render_data, dict):
        raise ValueError("Far-field bank plan must define a [render] table.")
    render = FarFieldRenderSettings(
        kernel_duration_seconds=_require_float(render_data, "kernel_duration_seconds"),
        speed_of_sound_mps=cast(
            float,
            _optional_float(render_data, "speed_of_sound_mps", 343.0),
        ),
        output_peak_limit=cast(
            float,
            _optional_float(render_data, "output_peak_limit", 0.92),
        ),
        high_shelf_pivot_hz=cast(
            float,
            _optional_float(render_data, "high_shelf_pivot_hz", 1_800.0),
        ),
    )

    presets_data = data.get("presets")
    if not isinstance(presets_data, list) or not presets_data:
        raise ValueError("Far-field bank plan must define at least one [[presets]] table.")

    presets: list[FarFieldSimulationPreset] = []
    for preset_data in presets_data:
        if not isinstance(preset_data, dict):
            raise ValueError("Far-field preset entries must be TOML tables.")
        presets.append(
            FarFieldSimulationPreset(
                id=_require_str(preset_data, "id"),
                name=_require_str(preset_data, "name"),
                field=_require_field(preset_data, "field"),
                description=_require_str(preset_data, "description"),
                distance_meters=_require_float(preset_data, "distance_meters"),
                off_axis_angle_deg=_require_float(preset_data, "off_axis_angle_deg"),
                attenuation_db=_require_float(preset_data, "attenuation_db"),
                target_drr_db=_require_float(preset_data, "target_drr_db"),
                reverb_rt60_seconds=_require_float(preset_data, "reverb_rt60_seconds"),
                late_reverb_start_ms=_require_float(preset_data, "late_reverb_start_ms"),
                lowpass_hz=_require_float(preset_data, "lowpass_hz"),
                high_shelf_db=cast(float, _optional_float(preset_data, "high_shelf_db", 0.0)),
                base_weight=cast(float, _optional_float(preset_data, "base_weight", 1.0)),
                early_reflection_delays_ms=tuple(
                    _optional_float_list(preset_data, "early_reflection_delays_ms")
                ),
                early_reflection_gains_db=tuple(
                    _optional_float_list(preset_data, "early_reflection_gains_db")
                ),
                tags=tuple(_optional_str_list(preset_data, "tags")),
            )
        )

    return FarFieldBankPlan(
        notes=tuple(_optional_str_list(data, "notes")),
        probe=probe,
        render=render,
        presets=tuple(presets),
    )


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Far-field bank field '{key}' is missing or invalid.")
    return value


def _optional_str_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Far-field bank field '{key}' must be a list of strings.")
    return cast(list[str], list(value))


def _optional_float_list(data: dict[str, object], key: str) -> list[float]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Far-field bank field '{key}' must be a list of numerics.")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Far-field bank field '{key}' must contain only numerics.")
        result.append(float(item))
    return result


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"Far-field bank field '{key}' must be an integer.")


def _require_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Far-field bank field '{key}' must be numeric.")


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
        raise ValueError(
            f"Far-field bank field '{key}' must be one of: {', '.join(allowed_values)}."
        )
    return value


def _require_field(data: dict[str, object], key: str) -> DistanceField:
    return cast(DistanceField, _require_literal(data, key, ALLOWED_DISTANCE_FIELDS))


__all__ = ["load_far_field_bank_plan"]
