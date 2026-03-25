"""Slice derivation helpers for verification error analysis."""

from __future__ import annotations

import math
from typing import Any

_PAIR_FIELDS = {"dataset", "source_dataset", "channel", "device", "language", "split", "role"}
_DURATION_BUCKETS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 1.0, "lt_1s"),
    (1.0, 2.0, "1_to_2s"),
    (2.0, 4.0, "2_to_4s"),
    (4.0, 8.0, "4_to_8s"),
    (8.0, None, "8_plus_s"),
)


def derive_slice_value(
    field_name: str,
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if field_name == "noise_slice":
        return _derive_noise_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "reverb_slice":
        return _derive_reverb_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "channel_slice":
        return _derive_channel_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "distance_slice":
        return _derive_distance_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "silence_slice":
        return _derive_silence_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "duration_bucket":
        return _derive_duration_bucket(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "role_pair":
        left_role = _coerce_label(None if left_metadata is None else left_metadata.get("role"))
        right_role = _coerce_label(None if right_metadata is None else right_metadata.get("role"))
        return f"{left_role}->{right_role}"
    if field_name.startswith("left_"):
        return _coerce_label(
            None if left_metadata is None else left_metadata.get(field_name.removeprefix("left_"))
        )
    if field_name.startswith("right_"):
        return _coerce_label(
            None
            if right_metadata is None
            else right_metadata.get(field_name.removeprefix("right_"))
        )
    if field_name.startswith("pair_"):
        field_name = field_name.removeprefix("pair_")
    if field_name in _PAIR_FIELDS:
        return _derive_pair_field_value(
            field_name,
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        )

    left_value = _coerce_label(None if left_metadata is None else left_metadata.get(field_name))
    right_value = _coerce_label(None if right_metadata is None else right_metadata.get(field_name))
    if left_value == right_value:
        return left_value
    if left_value == "unknown":
        return right_value
    if right_value == "unknown":
        return left_value
    return f"{left_value}|{right_value}"


def _derive_duration_bucket(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str:
    duration_values = [
        _coerce_float_or_none(metadata.get("duration_seconds"))
        for metadata in (left_metadata, right_metadata)
        if metadata is not None
    ]
    filtered_values = [value for value in duration_values if value is not None]
    if not filtered_values:
        return "unknown"
    mean_duration = sum(filtered_values) / float(len(filtered_values))
    for start, stop, label in _DURATION_BUCKETS:
        if stop is None and mean_duration >= start:
            return label
        if stop is not None and start <= mean_duration < stop:
            return label
    return "unknown"


def _derive_pair_field_value(
    field_name: str,
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str:
    left_value = _coerce_label(None if left_metadata is None else left_metadata.get(field_name))
    right_value = _coerce_label(None if right_metadata is None else right_metadata.get(field_name))
    if left_value == "unknown" and right_value == "unknown":
        return "unknown"
    if left_value == right_value:
        return left_value
    if left_value == "unknown":
        return right_value
    if right_value == "unknown":
        return left_value
    return "mixed"


def _derive_noise_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "noise"
    ):
        return None
    category = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="corruption_category",
    )
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    return _join_slice_parts(category, severity)


def _derive_reverb_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "reverb"
    ):
        return None
    direct = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="direct_condition",
    )
    rt60 = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="rt60_bucket",
    )
    field = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="field",
    )
    return _join_slice_parts(field, direct, rt60)


def _derive_channel_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "codec"
    ):
        return None
    codec_family = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="codec_family",
    )
    suite_id = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_suite",
    )
    if codec_family != "channel" and "channel" not in suite_id:
        return None
    codec_name = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="codec_name",
    )
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    return _join_slice_parts(codec_family, codec_name, severity)


def _derive_distance_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "distance"
    ):
        return None
    field = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="distance_field",
    )
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    return _join_slice_parts(field, severity)


def _derive_silence_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "silence"
    ):
        return None
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    candidate = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_candidate_id",
    )
    return _join_slice_parts(severity, candidate)


def _coerce_pair_label(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
    field_name: str,
) -> str:
    return _merge_pair_labels(
        _coerce_label(None if left_metadata is None else left_metadata.get(field_name)),
        _coerce_label(None if right_metadata is None else right_metadata.get(field_name)),
    )


def _coerce_pair_nested_label(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
    nested_field_name: str,
) -> str:
    return _merge_pair_labels(
        _coerce_label(_lookup_nested_metadata(left_metadata, nested_field_name)),
        _coerce_label(_lookup_nested_metadata(right_metadata, nested_field_name)),
    )


def _lookup_nested_metadata(
    metadata: dict[str, Any] | None,
    nested_field_name: str,
) -> Any:
    if metadata is None:
        return None
    container = metadata.get("corruption_metadata")
    if isinstance(container, dict):
        return container.get(nested_field_name)
    return None


def _merge_pair_labels(left_value: str, right_value: str) -> str:
    if left_value == "unknown" and right_value == "unknown":
        return "unknown"
    if left_value == right_value:
        return left_value
    if left_value == "unknown":
        return right_value
    if right_value == "unknown":
        return left_value
    return "mixed"


def _join_slice_parts(*parts: str) -> str | None:
    normalized = [part for part in parts if part and part != "unknown"]
    if not normalized:
        return None
    return "/".join(normalized)


def _coerce_label(value: Any) -> str:
    if value is None:
        return "unknown"
    normalized = str(value).strip()
    return normalized if normalized else "unknown"


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


__all__ = ["derive_slice_value"]
