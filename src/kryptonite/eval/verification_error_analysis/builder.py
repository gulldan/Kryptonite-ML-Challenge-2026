"""Core builder for verification error-analysis reports."""

from __future__ import annotations

import math
from typing import Any

from ..verification_data import build_trial_item_index, resolve_trial_side_identifier
from .aggregates import (
    build_domain_failures,
    build_priority_findings,
    build_speaker_confusions,
    build_speaker_failures,
)
from .models import (
    VerificationErrorAnalysisReport,
    VerificationErrorAnalysisSummary,
    VerificationErrorExample,
)
from .slices import derive_slice_value
from .support import (
    build_trial_lookup,
    classify_error,
    merge_trial_row,
    normalize_score_rows,
    resolve_positive_speaker_id,
    resolve_speaker_id,
    resolve_speaker_pair,
    safe_rate,
)


def build_verification_error_analysis(
    score_rows: list[dict[str, Any]],
    *,
    decision_threshold: float,
    threshold_source: str,
    trial_rows: list[dict[str, Any]] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    slice_fields: tuple[str, ...] = (),
    max_examples_per_error: int = 10,
    max_domain_failures: int = 20,
    max_speaker_confusions: int = 10,
    max_speaker_failures: int = 10,
    max_priority_findings: int = 5,
) -> VerificationErrorAnalysisReport:
    if max_examples_per_error <= 0:
        raise ValueError("max_examples_per_error must be positive.")
    if not math.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite.")

    metadata_index = {} if not metadata_rows else build_trial_item_index(metadata_rows)
    trial_lookup = build_trial_lookup(trial_rows)
    score_records = normalize_score_rows(score_rows)

    total_by_slice: dict[tuple[str, str], dict[str, int]] = {}
    error_by_slice: dict[tuple[str, str], dict[str, Any]] = {}
    negative_trials_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    false_accepts_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    positive_trials_by_speaker: dict[str, dict[str, Any]] = {}
    false_rejects_by_speaker: dict[str, dict[str, Any]] = {}
    false_accept_examples: list[VerificationErrorExample] = []
    false_reject_examples: list[VerificationErrorExample] = []
    false_accept_count = 0
    false_reject_count = 0

    for row_index, record in enumerate(score_records):
        raw_row = record["raw_row"]
        merged_row = merge_trial_row(raw_row, trial_lookup, row_index=row_index)
        left_identifier = resolve_trial_side_identifier(merged_row, "left")
        right_identifier = resolve_trial_side_identifier(merged_row, "right")
        left_metadata = None if left_identifier is None else metadata_index.get(left_identifier)
        right_metadata = None if right_identifier is None else metadata_index.get(right_identifier)
        field_values = {
            field_name: derive_slice_value(
                field_name,
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            )
            for field_name in slice_fields
        }
        _accumulate_trial_slices(total_by_slice, field_values)

        left_speaker_id = resolve_speaker_id(
            merged_row=merged_row,
            metadata=left_metadata,
            side="left",
        )
        right_speaker_id = resolve_speaker_id(
            merged_row=merged_row,
            metadata=right_metadata,
            side="right",
        )

        _accumulate_negative_pair(
            negative_trials_by_pair,
            label=record["label"],
            left_speaker_id=left_speaker_id,
            right_speaker_id=right_speaker_id,
        )
        speaker_id = resolve_positive_speaker_id(
            label=record["label"],
            left_speaker_id=left_speaker_id,
            right_speaker_id=right_speaker_id,
        )
        _accumulate_positive_speaker(
            positive_trials_by_speaker,
            speaker_id=speaker_id,
        )

        error_type = classify_error(
            label=record["label"],
            score=record["score"],
            decision_threshold=decision_threshold,
        )
        if error_type is None:
            continue

        margin = (
            record["score"] - decision_threshold
            if error_type == "false_accept"
            else decision_threshold - record["score"]
        )
        example = _build_error_example(
            error_type=error_type,
            score=record["score"],
            label=record["label"],
            margin=margin,
            left_id=left_identifier,
            right_id=right_identifier,
            left_speaker_id=left_speaker_id,
            right_speaker_id=right_speaker_id,
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        )

        if error_type == "false_accept":
            false_accept_count += 1
            false_accept_examples.append(example)
            _accumulate_false_accept_pair(
                false_accepts_by_pair,
                score=record["score"],
                left_speaker_id=left_speaker_id,
                right_speaker_id=right_speaker_id,
            )
        else:
            false_reject_count += 1
            false_reject_examples.append(example)
            _accumulate_false_reject_speaker(
                false_rejects_by_speaker,
                score=record["score"],
                speaker_id=speaker_id,
            )

        _accumulate_error_slices(
            error_by_slice,
            field_values=field_values,
            error_type=error_type,
            margin=margin,
            score=record["score"],
        )

    positive_count = sum(1 for row in score_records if row["label"] == 1)
    negative_count = len(score_records) - positive_count
    total_error_count = false_accept_count + false_reject_count
    summary = VerificationErrorAnalysisSummary(
        threshold_source=threshold_source,
        decision_threshold=round(decision_threshold, 6),
        trial_count=len(score_records),
        positive_count=positive_count,
        negative_count=negative_count,
        false_accept_count=false_accept_count,
        false_reject_count=false_reject_count,
        total_error_count=total_error_count,
        false_accept_rate=safe_rate(false_accept_count, negative_count),
        false_reject_rate=safe_rate(false_reject_count, positive_count),
        total_error_rate=safe_rate(total_error_count, len(score_records)),
    )

    domain_failures = build_domain_failures(
        total_by_slice=total_by_slice,
        error_by_slice=error_by_slice,
        limit=max_domain_failures,
    )
    speaker_confusions = build_speaker_confusions(
        negative_trials_by_pair=negative_trials_by_pair,
        false_accepts_by_pair=false_accepts_by_pair,
        limit=max_speaker_confusions,
    )
    speaker_failures = build_speaker_failures(
        positive_trials_by_speaker=positive_trials_by_speaker,
        false_rejects_by_speaker=false_rejects_by_speaker,
        limit=max_speaker_failures,
    )
    priority_findings = build_priority_findings(
        domain_failures=domain_failures,
        speaker_confusions=speaker_confusions,
        speaker_failures=speaker_failures,
        limit=max_priority_findings,
    )

    return VerificationErrorAnalysisReport(
        summary=summary,
        priority_findings=tuple(priority_findings),
        hard_false_accepts=tuple(
            sorted(
                false_accept_examples,
                key=lambda item: (
                    -item.margin,
                    -item.score,
                    item.left_id or "",
                    item.right_id or "",
                ),
            )[:max_examples_per_error]
        ),
        hard_false_rejects=tuple(
            sorted(
                false_reject_examples,
                key=lambda item: (
                    -item.margin,
                    item.score,
                    item.left_id or "",
                    item.right_id or "",
                ),
            )[:max_examples_per_error]
        ),
        domain_failures=tuple(domain_failures),
        speaker_confusions=tuple(speaker_confusions),
        speaker_failures=tuple(speaker_failures),
    )


def _accumulate_trial_slices(
    total_by_slice: dict[tuple[str, str], dict[str, int]],
    field_values: dict[str, str | None],
) -> None:
    for field_name, field_value in field_values.items():
        if field_value is None:
            continue
        bucket = total_by_slice.setdefault((field_name, field_value), {"trial_count": 0})
        bucket["trial_count"] += 1


def _accumulate_negative_pair(
    negative_trials_by_pair: dict[tuple[str, str], dict[str, Any]],
    *,
    label: int,
    left_speaker_id: str | None,
    right_speaker_id: str | None,
) -> None:
    if label != 0:
        return
    pair_key = resolve_speaker_pair(left_speaker_id, right_speaker_id)
    if pair_key is None:
        return
    bucket = negative_trials_by_pair.setdefault(pair_key, {"trial_count": 0})
    bucket["trial_count"] += 1


def _accumulate_positive_speaker(
    positive_trials_by_speaker: dict[str, dict[str, Any]],
    *,
    speaker_id: str | None,
) -> None:
    if speaker_id is None:
        return
    bucket = positive_trials_by_speaker.setdefault(speaker_id, {"trial_count": 0})
    bucket["trial_count"] += 1


def _accumulate_false_accept_pair(
    false_accepts_by_pair: dict[tuple[str, str], dict[str, Any]],
    *,
    score: float,
    left_speaker_id: str | None,
    right_speaker_id: str | None,
) -> None:
    pair_key = resolve_speaker_pair(left_speaker_id, right_speaker_id)
    if pair_key is None:
        return
    bucket = false_accepts_by_pair.setdefault(pair_key, {"count": 0, "scores": []})
    bucket["count"] += 1
    bucket["scores"].append(score)


def _accumulate_false_reject_speaker(
    false_rejects_by_speaker: dict[str, dict[str, Any]],
    *,
    score: float,
    speaker_id: str | None,
) -> None:
    if speaker_id is None:
        return
    bucket = false_rejects_by_speaker.setdefault(speaker_id, {"count": 0, "scores": []})
    bucket["count"] += 1
    bucket["scores"].append(score)


def _accumulate_error_slices(
    error_by_slice: dict[tuple[str, str], dict[str, Any]],
    *,
    field_values: dict[str, str | None],
    error_type: str,
    margin: float,
    score: float,
) -> None:
    for field_name, field_value in field_values.items():
        if field_value is None:
            continue
        bucket = error_by_slice.setdefault(
            (field_name, field_value),
            {
                "error_count": 0.0,
                "false_accept_count": 0.0,
                "false_reject_count": 0.0,
                "margins": [],
                "scores": [],
            },
        )
        bucket["error_count"] += 1.0
        bucket[f"{error_type}_count"] += 1.0
        bucket["margins"].append(margin)
        bucket["scores"].append(score)


def _build_error_example(
    *,
    error_type: str,
    score: float,
    label: int,
    margin: float,
    left_id: str | None,
    right_id: str | None,
    left_speaker_id: str | None,
    right_speaker_id: str | None,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> VerificationErrorExample:
    return VerificationErrorExample(
        error_type=error_type,
        score=round(score, 6),
        label=label,
        margin=round(margin, 6),
        left_id=left_id,
        right_id=right_id,
        left_speaker_id=left_speaker_id,
        right_speaker_id=right_speaker_id,
        dataset=derive_slice_value(
            "dataset",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        channel=derive_slice_value(
            "channel",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        role_pair=derive_slice_value(
            "role_pair",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        duration_bucket=derive_slice_value(
            "duration_bucket",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        noise_slice=derive_slice_value(
            "noise_slice",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        reverb_slice=derive_slice_value(
            "reverb_slice",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        channel_slice=derive_slice_value(
            "channel_slice",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        distance_slice=derive_slice_value(
            "distance_slice",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
        silence_slice=derive_slice_value(
            "silence_slice",
            left_metadata=left_metadata,
            right_metadata=right_metadata,
        ),
    )


__all__ = ["build_verification_error_analysis"]
