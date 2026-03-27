"""Shared slice derivation helpers for verification reports and threshold calibration."""

from __future__ import annotations

import math
from typing import Any

from .verification_data import build_trial_item_index, resolve_trial_side_identifier

VERIFICATION_SLICE_FIELD_TITLES: dict[str, str] = {
    "noise_slice": "Noise",
    "reverb_slice": "Reverb",
    "rt60_slice": "RT60",
    "codec_slice": "Codec",
    "channel_slice": "Channel",
    "distance_slice": "Distance",
    "duration_bucket": "Duration",
    "silence_ratio_bucket": "Silence Ratio",
    "silence_slice": "Silence",
}

VERIFICATION_SLICE_FIELD_DESCRIPTIONS: dict[str, str] = {
    "noise_slice": (
        "Corruption-aware breakdown for additive noise suites using category and severity."
    ),
    "reverb_slice": "Room and direct-to-reverb breakdown for reverberant suites.",
    "rt60_slice": "Explicit RT60-bucket breakdown extracted from reverberation metadata.",
    "codec_slice": (
        "Codec-family breakdown for telephony, compression, VoIP, and other codec suites."
    ),
    "channel_slice": "Channel-style codec breakdown for `dev_channel`-like evaluations.",
    "distance_slice": "Near/mid/far style breakdown for far-field distance stress suites.",
    "duration_bucket": "Duration buckets derived from left/right trial metadata.",
    "silence_ratio_bucket": (
        "Observed silence-ratio bucket derived from manifest or embedding metadata."
    ),
    "silence_slice": "Silence/pause robustness breakdown using suite severity or candidate id.",
}

DEFAULT_SLICE_FIELDS: tuple[str, ...] = (
    "dataset",
    "channel",
    "role_pair",
    "duration_bucket",
    "noise_slice",
    "reverb_slice",
    "rt60_slice",
    "codec_slice",
    "channel_slice",
    "distance_slice",
    "silence_ratio_bucket",
    "silence_slice",
)

_PAIR_FIELDS = {"dataset", "source_dataset", "channel", "device", "language", "split", "role"}
_DURATION_BUCKETS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 1.0, "lt_1s"),
    (1.0, 2.0, "1_to_2s"),
    (2.0, 4.0, "2_to_4s"),
    (4.0, 8.0, "4_to_8s"),
    (8.0, None, "8_plus_s"),
)
_SILENCE_RATIO_BUCKETS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 0.2, "lt_20pct"),
    (0.2, 0.5, "20_to_50pct"),
    (0.5, None, "50pct_plus"),
)


def group_verification_rows_by_slice(
    *,
    raw_score_rows: list[dict[str, Any]],
    normalized_rows: list[dict[str, float | int]],
    trial_rows: list[dict[str, Any]] | None,
    metadata_rows: list[dict[str, Any]] | None,
    slice_fields: tuple[str, ...],
) -> dict[tuple[str, str], list[dict[str, float | int]]]:
    if not metadata_rows or not slice_fields:
        return {}

    metadata_index = build_trial_item_index(metadata_rows)
    trial_lookup = build_trial_lookup(trial_rows)
    grouped_rows: dict[tuple[str, str], list[dict[str, float | int]]] = {}
    for index, (raw_row, normalized_row) in enumerate(
        zip(raw_score_rows, normalized_rows, strict=True)
    ):
        merged_trial_row = merge_trial_row(raw_row, trial_lookup, row_index=index)
        left_identifier = resolve_trial_side_identifier(merged_trial_row, "left")
        right_identifier = resolve_trial_side_identifier(merged_trial_row, "right")
        left_metadata = None if left_identifier is None else metadata_index.get(left_identifier)
        right_metadata = None if right_identifier is None else metadata_index.get(right_identifier)
        for field_name in slice_fields:
            field_value = derive_slice_value(
                field_name,
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            )
            if field_value is None:
                continue
            grouped_rows.setdefault((field_name, field_value), []).append(normalized_row)
    return grouped_rows


def build_trial_lookup(
    trial_rows: list[dict[str, Any]] | None,
) -> dict[tuple[str, str, int] | tuple[str, int], dict[str, Any]]:
    if not trial_rows:
        return {}
    lookup: dict[tuple[str, str, int] | tuple[str, int], dict[str, Any]] = {}
    for index, row in enumerate(trial_rows):
        left_identifier = resolve_trial_side_identifier(row, "left")
        right_identifier = resolve_trial_side_identifier(row, "right")
        label = int(row.get("label", -1))
        if left_identifier and right_identifier and label in {0, 1}:
            lookup[(left_identifier, right_identifier, label)] = row
        lookup[(f"index:{index}", label)] = row
    return lookup


def merge_trial_row(
    raw_score_row: dict[str, Any],
    trial_lookup: dict[tuple[str, str, int] | tuple[str, int], dict[str, Any]],
    *,
    row_index: int,
) -> dict[str, Any]:
    if not trial_lookup:
        return raw_score_row
    left_identifier = resolve_trial_side_identifier(raw_score_row, "left")
    right_identifier = resolve_trial_side_identifier(raw_score_row, "right")
    label = int(raw_score_row.get("label", -1))
    matched_row = None
    if left_identifier and right_identifier and label in {0, 1}:
        matched_row = trial_lookup.get((left_identifier, right_identifier, label))
    if matched_row is None:
        matched_row = trial_lookup.get((f"index:{row_index}", label))
    if matched_row is None:
        return raw_score_row
    return {**matched_row, **raw_score_row}


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
    if field_name == "rt60_slice":
        return _derive_rt60_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "codec_slice":
        return _derive_codec_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "channel_slice":
        return _derive_channel_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "distance_slice":
        return _derive_distance_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "silence_ratio_bucket":
        return _derive_silence_ratio_bucket(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        )
    if field_name == "silence_slice":
        return _derive_silence_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "duration_bucket":
        duration_values = [
            coerce_float_or_none(metadata.get("duration_seconds"))
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
    if field_name == "role_pair":
        left_role = coerce_label(None if left_metadata is None else left_metadata.get("role"))
        right_role = coerce_label(None if right_metadata is None else right_metadata.get("role"))
        return f"{left_role}->{right_role}"
    if field_name.startswith("left_"):
        return coerce_label(
            None if left_metadata is None else left_metadata.get(field_name.removeprefix("left_"))
        )
    if field_name.startswith("right_"):
        return coerce_label(
            None
            if right_metadata is None
            else right_metadata.get(field_name.removeprefix("right_"))
        )
    if field_name.startswith("pair_"):
        field_name = field_name.removeprefix("pair_")
    if field_name in _PAIR_FIELDS:
        left_value = coerce_label(None if left_metadata is None else left_metadata.get(field_name))
        right_value = coerce_label(
            None if right_metadata is None else right_metadata.get(field_name)
        )
        if left_value == "unknown" and right_value == "unknown":
            return "unknown"
        if left_value == right_value:
            return left_value
        if left_value == "unknown":
            return right_value
        if right_value == "unknown":
            return left_value
        return "mixed"

    left_value = coerce_label(None if left_metadata is None else left_metadata.get(field_name))
    right_value = coerce_label(None if right_metadata is None else right_metadata.get(field_name))
    if left_value == right_value:
        return left_value
    if left_value == "unknown":
        return right_value
    if right_value == "unknown":
        return left_value
    return f"{left_value}|{right_value}"


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


def _derive_rt60_slice(
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
    rt60 = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="rt60_bucket",
    )
    return None if rt60 == "unknown" else rt60


def _derive_codec_slice(
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


def _derive_channel_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    codec_slice = _derive_codec_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if codec_slice is None:
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
    return codec_slice


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


def _derive_silence_ratio_bucket(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    precomputed = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="silence_ratio_bucket",
    )
    if precomputed != "unknown":
        return precomputed

    silence_ratios = [
        coerce_float_or_none(None if metadata is None else metadata.get("silence_ratio"))
        for metadata in (left_metadata, right_metadata)
    ]
    filtered_values = [value for value in silence_ratios if value is not None]
    if not filtered_values:
        return None
    mean_ratio = sum(filtered_values) / float(len(filtered_values))
    for start, stop, label in _SILENCE_RATIO_BUCKETS:
        if stop is None and mean_ratio >= start:
            return label
        if stop is not None and start <= mean_ratio < stop:
            return label
    return None


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
        coerce_label(None if left_metadata is None else left_metadata.get(field_name)),
        coerce_label(None if right_metadata is None else right_metadata.get(field_name)),
    )


def _coerce_pair_nested_label(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
    nested_field_name: str,
) -> str:
    return _merge_pair_labels(
        coerce_label(_lookup_nested_metadata(left_metadata, nested_field_name)),
        coerce_label(_lookup_nested_metadata(right_metadata, nested_field_name)),
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


def coerce_label(value: Any) -> str:
    if value is None:
        return "unknown"
    normalized = str(value).strip()
    return normalized if normalized else "unknown"


def coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


__all__ = [
    "DEFAULT_SLICE_FIELDS",
    "VERIFICATION_SLICE_FIELD_DESCRIPTIONS",
    "VERIFICATION_SLICE_FIELD_TITLES",
    "build_trial_lookup",
    "coerce_float_or_none",
    "coerce_label",
    "derive_slice_value",
    "group_verification_rows_by_slice",
    "merge_trial_row",
]
