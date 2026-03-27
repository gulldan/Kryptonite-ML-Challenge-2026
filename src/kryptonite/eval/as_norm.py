"""AS-norm score normalization for offline verification evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .score_normalization import (
    build_score_normalization_context,
    compute_identifier_cohort_statistics,
    resolve_trial_score_records,
)

DEFAULT_AS_NORM_TOP_K = 100
DEFAULT_AS_NORM_STD_EPSILON = 1e-6
VERIFICATION_AS_NORM_SCORES_JSONL_NAME = "verification_scores_as_norm.jsonl"
VERIFICATION_SCORE_NORMALIZATION_SUMMARY_JSON_NAME = "verification_score_normalization_summary.json"


@dataclass(frozen=True, slots=True)
class AdaptiveScoreNormalizationSummary:
    method: str
    trial_count: int
    cohort_size: int
    embedding_dim: int
    top_k: int
    effective_top_k: int
    unique_identifier_count: int
    unique_left_count: int
    unique_right_count: int
    floored_std_count: int
    mean_raw_score: float
    mean_normalized_score: float
    mean_score_shift: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveScoreNormalizationResult:
    score_rows: list[dict[str, Any]]
    summary: AdaptiveScoreNormalizationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_rows": [dict(row) for row in self.score_rows],
            "summary": self.summary.to_dict(),
        }


def apply_as_norm_to_verification_scores(
    score_rows: list[dict[str, Any]],
    *,
    embeddings_path: Path | str,
    metadata_path: Path | str,
    cohort_bank_root: Path | str,
    top_k: int = DEFAULT_AS_NORM_TOP_K,
    std_epsilon: float = DEFAULT_AS_NORM_STD_EPSILON,
    embeddings_key: str = "embeddings",
    ids_key: str | None = "point_ids",
    point_id_field: str = "atlas_point_id",
) -> AdaptiveScoreNormalizationResult:
    context = build_score_normalization_context(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        cohort_bank_root=cohort_bank_root,
        embeddings_key=embeddings_key,
        ids_key=ids_key,
        point_id_field=point_id_field,
    )
    trial_records = resolve_trial_score_records(score_rows)

    trial_identifiers: list[tuple[str, str]] = [
        (record.left_identifier, record.right_identifier) for record in trial_records
    ]
    raw_scores: list[float] = [record.raw_score for record in trial_records]
    unique_identifiers: list[str] = []
    unique_left_identifiers: set[str] = set()
    unique_right_identifiers: set[str] = set()
    seen_identifiers: set[str] = set()

    for record in trial_records:
        left_identifier = record.left_identifier
        right_identifier = record.right_identifier
        unique_left_identifiers.add(left_identifier)
        unique_right_identifiers.add(right_identifier)

        for identifier in (left_identifier, right_identifier):
            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)
            unique_identifiers.append(identifier)

    identifier_to_stats, stats_summary = compute_identifier_cohort_statistics(
        context,
        identifiers=tuple(unique_identifiers),
        top_k=top_k,
        std_epsilon=std_epsilon,
    )

    normalized_scores = np.empty((len(score_rows),), dtype=np.float64)
    for index, ((left_identifier, right_identifier), raw_score) in enumerate(
        zip(trial_identifiers, raw_scores, strict=True)
    ):
        left_stats = identifier_to_stats[left_identifier]
        right_stats = identifier_to_stats[right_identifier]
        normalized_scores[index] = 0.5 * (
            ((raw_score - left_stats.mean) / left_stats.std)
            + ((raw_score - right_stats.mean) / right_stats.std)
        )

    normalized_score_rows: list[dict[str, Any]] = []
    for raw_row, raw_score, normalized_score in zip(
        score_rows,
        raw_scores,
        normalized_scores,
        strict=True,
    ):
        normalized_score_rows.append(
            {
                **raw_row,
                "raw_score": round(raw_score, 8),
                "score": round(float(normalized_score), 8),
                "score_normalization": "as-norm",
            }
        )

    raw_score_array = np.asarray(raw_scores, dtype=np.float64)
    summary = AdaptiveScoreNormalizationSummary(
        method="as-norm",
        trial_count=len(score_rows),
        cohort_size=context.cohort_size,
        embedding_dim=context.embedding_dim,
        top_k=top_k,
        effective_top_k=stats_summary.effective_top_k,
        unique_identifier_count=len(unique_identifiers),
        unique_left_count=len(unique_left_identifiers),
        unique_right_count=len(unique_right_identifiers),
        floored_std_count=stats_summary.floored_std_count,
        mean_raw_score=round(float(raw_score_array.mean()), 6),
        mean_normalized_score=round(float(normalized_scores.mean()), 6),
        mean_score_shift=round(float((normalized_scores - raw_score_array).mean()), 6),
    )
    return AdaptiveScoreNormalizationResult(
        score_rows=normalized_score_rows,
        summary=summary,
    )


__all__ = [
    "AdaptiveScoreNormalizationResult",
    "AdaptiveScoreNormalizationSummary",
    "DEFAULT_AS_NORM_STD_EPSILON",
    "DEFAULT_AS_NORM_TOP_K",
    "VERIFICATION_AS_NORM_SCORES_JSONL_NAME",
    "VERIFICATION_SCORE_NORMALIZATION_SUMMARY_JSON_NAME",
    "apply_as_norm_to_verification_scores",
]
