"""Support helpers for verification error analysis builders."""

from __future__ import annotations

import math
from typing import Any

from ..verification_data import resolve_trial_side_identifier


def normalize_score_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(score_rows, start=1):
        try:
            label = int(row["label"])
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid verification score row at index {index}: {row!r}") from exc
        if label not in {0, 1}:
            raise ValueError(f"Verification labels must be 0 or 1, got {label!r} at row {index}.")
        if not math.isfinite(score):
            raise ValueError(f"Verification score must be finite, got {score!r} at row {index}.")
        normalized.append({"raw_row": row, "label": label, "score": score})
    return normalized


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


def classify_error(*, label: int, score: float, decision_threshold: float) -> str | None:
    if label == 0 and score >= decision_threshold:
        return "false_accept"
    if label == 1 and score < decision_threshold:
        return "false_reject"
    return None


def resolve_speaker_id(
    *,
    merged_row: dict[str, Any],
    metadata: dict[str, Any] | None,
    side: str,
) -> str | None:
    candidates = [
        merged_row.get(f"{side}_speaker_id"),
        None if metadata is None else metadata.get("speaker_id"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip()
        if normalized:
            return normalized
    return None


def resolve_positive_speaker_id(
    *,
    label: int,
    left_speaker_id: str | None,
    right_speaker_id: str | None,
) -> str | None:
    if label != 1:
        return None
    if left_speaker_id is not None and right_speaker_id is not None:
        if left_speaker_id == right_speaker_id:
            return left_speaker_id
        return None
    return left_speaker_id or right_speaker_id


def resolve_speaker_pair(
    left_speaker_id: str | None,
    right_speaker_id: str | None,
) -> tuple[str, str] | None:
    if left_speaker_id is None or right_speaker_id is None:
        return None
    if left_speaker_id == right_speaker_id:
        return None
    speaker_a, speaker_b = sorted((left_speaker_id, right_speaker_id))
    return speaker_a, speaker_b


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


__all__ = [
    "build_trial_lookup",
    "classify_error",
    "merge_trial_row",
    "normalize_score_rows",
    "resolve_positive_speaker_id",
    "resolve_speaker_id",
    "resolve_speaker_pair",
    "safe_rate",
]
