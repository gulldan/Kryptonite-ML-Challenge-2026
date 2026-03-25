"""Aggregation helpers for verification error analysis."""

from __future__ import annotations

from typing import Any

from .models import (
    VerificationDomainFailure,
    VerificationPriorityFinding,
    VerificationSpeakerConfusion,
    VerificationSpeakerFailure,
)
from .support import safe_rate


def build_domain_failures(
    *,
    total_by_slice: dict[tuple[str, str], dict[str, int]],
    error_by_slice: dict[tuple[str, str], dict[str, Any]],
    limit: int,
) -> list[VerificationDomainFailure]:
    rows: list[VerificationDomainFailure] = []
    for key, error_bucket in error_by_slice.items():
        total_bucket = total_by_slice.get(key)
        if total_bucket is None:
            continue
        field_name, field_value = key
        trial_count = int(total_bucket["trial_count"])
        error_count = int(error_bucket["error_count"])
        margins = [float(value) for value in error_bucket["margins"]]
        scores = [float(value) for value in error_bucket["scores"]]
        rows.append(
            VerificationDomainFailure(
                field_name=field_name,
                field_value=field_value,
                trial_count=trial_count,
                error_count=error_count,
                false_accept_count=int(error_bucket["false_accept_count"]),
                false_reject_count=int(error_bucket["false_reject_count"]),
                error_rate=safe_rate(error_count, trial_count),
                mean_error_margin=round(sum(margins) / len(margins), 6),
                mean_error_score=round(sum(scores) / len(scores), 6),
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -item.error_rate,
            -item.error_count,
            -item.mean_error_margin,
            -item.trial_count,
            item.field_name,
            item.field_value,
        ),
    )[:limit]


def build_speaker_confusions(
    *,
    negative_trials_by_pair: dict[tuple[str, str], dict[str, Any]],
    false_accepts_by_pair: dict[tuple[str, str], dict[str, Any]],
    limit: int,
) -> list[VerificationSpeakerConfusion]:
    rows: list[VerificationSpeakerConfusion] = []
    for pair_key, error_bucket in false_accepts_by_pair.items():
        total_bucket = negative_trials_by_pair.get(pair_key)
        if total_bucket is None:
            continue
        speaker_a, speaker_b = pair_key
        scores = [float(value) for value in error_bucket["scores"]]
        error_count = int(error_bucket["count"])
        trial_count = int(total_bucket["trial_count"])
        rows.append(
            VerificationSpeakerConfusion(
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                trial_count=trial_count,
                false_accept_count=error_count,
                false_accept_rate=safe_rate(error_count, trial_count),
                mean_false_accept_score=round(sum(scores) / len(scores), 6),
                max_false_accept_score=round(max(scores), 6),
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -item.false_accept_rate,
            -item.false_accept_count,
            -item.max_false_accept_score,
            item.speaker_a,
            item.speaker_b,
        ),
    )[:limit]


def build_speaker_failures(
    *,
    positive_trials_by_speaker: dict[str, dict[str, Any]],
    false_rejects_by_speaker: dict[str, dict[str, Any]],
    limit: int,
) -> list[VerificationSpeakerFailure]:
    rows: list[VerificationSpeakerFailure] = []
    for speaker_id, error_bucket in false_rejects_by_speaker.items():
        total_bucket = positive_trials_by_speaker.get(speaker_id)
        if total_bucket is None:
            continue
        scores = [float(value) for value in error_bucket["scores"]]
        error_count = int(error_bucket["count"])
        trial_count = int(total_bucket["trial_count"])
        rows.append(
            VerificationSpeakerFailure(
                speaker_id=speaker_id,
                positive_trial_count=trial_count,
                false_reject_count=error_count,
                false_reject_rate=safe_rate(error_count, trial_count),
                mean_false_reject_score=round(sum(scores) / len(scores), 6),
                min_false_reject_score=round(min(scores), 6),
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -item.false_reject_rate,
            -item.false_reject_count,
            item.min_false_reject_score,
            item.speaker_id,
        ),
    )[:limit]


def build_priority_findings(
    *,
    domain_failures: list[VerificationDomainFailure],
    speaker_confusions: list[VerificationSpeakerConfusion],
    speaker_failures: list[VerificationSpeakerFailure],
    limit: int,
) -> list[VerificationPriorityFinding]:
    candidates: list[tuple[tuple[float, float, float], VerificationPriorityFinding]] = []
    for row in domain_failures[:6]:
        candidates.append(
            (
                (row.error_rate, float(row.error_count), row.mean_error_margin),
                VerificationPriorityFinding(
                    finding_type="domain_failure",
                    title=f"Slice {row.field_name}={row.field_value}",
                    evidence=(
                        f"FA `{row.false_accept_count}` / FR `{row.false_reject_count}`, "
                        f"mean error margin `{row.mean_error_margin}`"
                    ),
                    trial_count=row.trial_count,
                    error_count=row.error_count,
                    error_rate=row.error_rate,
                ),
            )
        )
    for row in speaker_confusions[:4]:
        candidates.append(
            (
                (row.false_accept_rate, float(row.false_accept_count), row.max_false_accept_score),
                VerificationPriorityFinding(
                    finding_type="speaker_confusion",
                    title=f"Speaker confusion {row.speaker_a} vs {row.speaker_b}",
                    evidence=(
                        f"recurrent false accepts with mean score `{row.mean_false_accept_score}` "
                        f"and max score `{row.max_false_accept_score}`"
                    ),
                    trial_count=row.trial_count,
                    error_count=row.false_accept_count,
                    error_rate=row.false_accept_rate,
                ),
            )
        )
    for row in speaker_failures[:4]:
        candidates.append(
            (
                (row.false_reject_rate, float(row.false_reject_count), -row.min_false_reject_score),
                VerificationPriorityFinding(
                    finding_type="speaker_failure",
                    title=f"Speaker fragility {row.speaker_id}",
                    evidence=(
                        f"recurrent false rejects with mean score `{row.mean_false_reject_score}` "
                        f"and min score `{row.min_false_reject_score}`"
                    ),
                    trial_count=row.positive_trial_count,
                    error_count=row.false_reject_count,
                    error_rate=row.false_reject_rate,
                ),
            )
        )
    return [
        finding
        for _, finding in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        )[:limit]
    ]


__all__ = [
    "build_domain_failures",
    "build_priority_findings",
    "build_speaker_confusions",
    "build_speaker_failures",
]
