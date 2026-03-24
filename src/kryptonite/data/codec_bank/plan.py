"""Plan loading for reproducible codec/channel simulation presets."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from .models import (
    ALLOWED_CODEC_FAMILIES,
    ALLOWED_CODEC_SEVERITIES,
    CodecBankPlan,
    CodecEQBand,
    CodecFamily,
    CodecProbeSettings,
    CodecSeverity,
    CodecSeverityProfile,
    CodecSimulationPreset,
)


def load_codec_bank_plan(path: Path | str) -> CodecBankPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())

    probe_data = data.get("probe")
    if not isinstance(probe_data, dict):
        raise ValueError("Codec bank plan must define a [probe] table.")
    probe = CodecProbeSettings(
        sample_rate_hz=_require_int(probe_data, "sample_rate_hz"),
        duration_seconds=_require_float(probe_data, "duration_seconds"),
        peak_amplitude=cast(float, _optional_float(probe_data, "peak_amplitude", 0.85)),
    )

    severity_data = data.get("severity_profiles")
    if not isinstance(severity_data, dict):
        raise ValueError("Codec bank plan must define a [severity_profiles] table.")

    severity_profiles: dict[CodecSeverity, CodecSeverityProfile] = {}
    for severity_name in ALLOWED_CODEC_SEVERITIES:
        profile_data = severity_data.get(severity_name)
        if not isinstance(profile_data, dict):
            raise ValueError(
                f"Codec bank plan must define [severity_profiles.{severity_name}] settings."
            )
        severity_profiles[severity_name] = CodecSeverityProfile(
            description=_require_str(profile_data, "description"),
            weight_multiplier=cast(
                float,
                _optional_float(profile_data, "weight_multiplier", 1.0),
            ),
        )

    presets_data = data.get("presets")
    if not isinstance(presets_data, list) or not presets_data:
        raise ValueError("Codec bank plan must define at least one [[presets]] table.")

    presets: list[CodecSimulationPreset] = []
    for preset_data in presets_data:
        if not isinstance(preset_data, dict):
            raise ValueError("Codec preset entries must be TOML tables.")
        eq_bands_data = preset_data.get("eq_bands", [])
        if not isinstance(eq_bands_data, list):
            raise ValueError("Codec preset eq_bands must be a list of TOML tables.")
        eq_bands: list[CodecEQBand] = []
        for band_data in eq_bands_data:
            if not isinstance(band_data, dict):
                raise ValueError("Each codec EQ band must be a TOML table.")
            eq_bands.append(
                CodecEQBand(
                    frequency_hz=_require_float(band_data, "frequency_hz"),
                    width_hz=_require_float(band_data, "width_hz"),
                    gain_db=_require_float(band_data, "gain_db"),
                )
            )

        presets.append(
            CodecSimulationPreset(
                id=_require_str(preset_data, "id"),
                name=_require_str(preset_data, "name"),
                family=_require_family(preset_data, "family"),
                severity=_require_severity(preset_data, "severity"),
                description=_require_str(preset_data, "description"),
                base_weight=cast(float, _optional_float(preset_data, "base_weight", 1.0)),
                highpass_hz=_optional_float(preset_data, "highpass_hz", None),
                lowpass_hz=_optional_float(preset_data, "lowpass_hz", None),
                pre_gain_db=cast(float, _optional_float(preset_data, "pre_gain_db", 0.0)),
                post_gain_db=cast(float, _optional_float(preset_data, "post_gain_db", 0.0)),
                bitcrusher_bits=_optional_int(preset_data, "bitcrusher_bits", None),
                bitcrusher_mix=cast(
                    float,
                    _optional_float(preset_data, "bitcrusher_mix", 1.0),
                ),
                soft_clip=_optional_bool(preset_data, "soft_clip", False),
                codec_name=_optional_str(preset_data, "codec_name"),
                container_extension=_optional_str(preset_data, "container_extension") or "wav",
                encode_sample_rate_hz=_optional_int(preset_data, "encode_sample_rate_hz", None),
                encode_bitrate=_optional_str(preset_data, "encode_bitrate"),
                ffmpeg_options=tuple(_optional_str_list(preset_data, "ffmpeg_options")),
                eq_bands=tuple(eq_bands),
                tags=tuple(_optional_str_list(preset_data, "tags")),
            )
        )

    return CodecBankPlan(
        notes=tuple(_optional_str_list(data, "notes")),
        probe=probe,
        severity_profiles=severity_profiles,
        presets=tuple(presets),
    )


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Codec bank field '{key}' is missing or invalid.")
    return value


def _optional_str(data: dict[str, object], key: str) -> str | None:
    if key not in data:
        return None
    return _require_str(data, key)


def _optional_str_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Codec bank field '{key}' must be a list of strings.")
    return cast(list[str], list(value))


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"Codec bank field '{key}' must be an integer.")


def _optional_int(
    data: dict[str, object],
    key: str,
    default: int | None,
) -> int | None:
    if key not in data:
        return default
    return _require_int(data, key)


def _require_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Codec bank field '{key}' must be numeric.")


def _optional_float(
    data: dict[str, object],
    key: str,
    default: float | None,
) -> float | None:
    if key not in data:
        return default
    return _require_float(data, key)


def _optional_bool(data: dict[str, object], key: str, default: bool) -> bool:
    if key not in data:
        return default
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Codec bank field '{key}' must be a boolean.")
    return value


def _require_literal(
    data: dict[str, object],
    key: str,
    allowed_values: tuple[str, ...],
) -> str:
    value = _require_str(data, key)
    if value not in allowed_values:
        raise ValueError(f"Codec bank field '{key}' must be one of: {', '.join(allowed_values)}.")
    return value


def _require_family(data: dict[str, object], key: str) -> CodecFamily:
    return cast(CodecFamily, _require_literal(data, key, ALLOWED_CODEC_FAMILIES))


def _require_severity(data: dict[str, object], key: str) -> CodecSeverity:
    return cast(CodecSeverity, _require_literal(data, key, ALLOWED_CODEC_SEVERITIES))


__all__ = ["load_codec_bank_plan"]
