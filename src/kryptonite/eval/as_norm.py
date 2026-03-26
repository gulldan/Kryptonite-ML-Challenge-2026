"""AS-norm score normalization for offline verification evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.models import cosine_score_matrix, l2_normalize_embeddings, rank_cosine_scores

from .cohort_bank import load_cohort_embedding_bank
from .embedding_atlas.io import align_metadata_rows, load_embedding_matrix, load_metadata_rows
from .verification_data import resolve_trial_side_identifier
from .verification_metrics import normalize_verification_score_rows

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
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if std_epsilon <= 0.0:
        raise ValueError("std_epsilon must be positive.")

    normalized_rows = normalize_verification_score_rows(score_rows)
    if not score_rows:
        raise ValueError("score_rows must not be empty.")

    evaluation_embeddings, point_ids = load_embedding_matrix(
        embeddings_path,
        embeddings_key=embeddings_key,
        ids_key=ids_key,
    )
    aligned_metadata_rows = align_metadata_rows(
        metadata_rows=load_metadata_rows(metadata_path),
        point_id_field=point_id_field,
        point_ids=point_ids,
        expected_count=int(evaluation_embeddings.shape[0]),
    )
    identifier_to_embedding = _build_embedding_lookup(
        metadata_rows=aligned_metadata_rows,
        embeddings=evaluation_embeddings,
    )
    cohort_bank = load_cohort_embedding_bank(cohort_bank_root)
    cohort_embeddings = l2_normalize_embeddings(
        cohort_bank.embeddings,
        field_name="cohort_embeddings",
    )

    trial_identifiers: list[tuple[str, str]] = []
    raw_scores: list[float] = []
    unique_identifiers: list[str] = []
    unique_left_identifiers: set[str] = set()
    unique_right_identifiers: set[str] = set()
    seen_identifiers: set[str] = set()

    for row_index, (raw_row, normalized_row) in enumerate(
        zip(score_rows, normalized_rows, strict=True),
        start=1,
    ):
        left_identifier = resolve_trial_side_identifier(raw_row, "left")
        right_identifier = resolve_trial_side_identifier(raw_row, "right")
        if left_identifier is None or right_identifier is None:
            raise ValueError(
                "AS-norm requires left/right identifiers for every score row; "
                f"row {row_index} is missing one or both sides."
            )

        trial_identifiers.append((left_identifier, right_identifier))
        raw_scores.append(float(normalized_row["score"]))
        unique_left_identifiers.add(left_identifier)
        unique_right_identifiers.add(right_identifier)

        for identifier in (left_identifier, right_identifier):
            if identifier in seen_identifiers:
                continue
            if identifier not in identifier_to_embedding:
                raise ValueError(
                    "AS-norm could not resolve an embedding for identifier "
                    f"{identifier!r} from {metadata_path}."
                )
            seen_identifiers.add(identifier)
            unique_identifiers.append(identifier)

    query_matrix = np.stack(
        [identifier_to_embedding[identifier] for identifier in unique_identifiers],
        axis=0,
    )
    means, stds, floored_std_count, effective_top_k = _compute_as_norm_statistics(
        query_embeddings=query_matrix,
        cohort_embeddings=cohort_embeddings,
        top_k=top_k,
        std_epsilon=std_epsilon,
    )
    identifier_to_stats = {
        identifier: (float(means[index]), float(stds[index]))
        for index, identifier in enumerate(unique_identifiers)
    }

    normalized_scores = np.empty((len(score_rows),), dtype=np.float64)
    for index, ((left_identifier, right_identifier), raw_score) in enumerate(
        zip(trial_identifiers, raw_scores, strict=True)
    ):
        left_mean, left_std = identifier_to_stats[left_identifier]
        right_mean, right_std = identifier_to_stats[right_identifier]
        normalized_scores[index] = 0.5 * (
            ((raw_score - left_mean) / left_std) + ((raw_score - right_mean) / right_std)
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
        cohort_size=int(cohort_embeddings.shape[0]),
        embedding_dim=int(cohort_embeddings.shape[1]),
        top_k=top_k,
        effective_top_k=effective_top_k,
        unique_identifier_count=len(unique_identifiers),
        unique_left_count=len(unique_left_identifiers),
        unique_right_count=len(unique_right_identifiers),
        floored_std_count=floored_std_count,
        mean_raw_score=round(float(raw_score_array.mean()), 6),
        mean_normalized_score=round(float(normalized_scores.mean()), 6),
        mean_score_shift=round(float((normalized_scores - raw_score_array).mean()), 6),
    )
    return AdaptiveScoreNormalizationResult(
        score_rows=normalized_score_rows,
        summary=summary,
    )


def _compute_as_norm_statistics(
    *,
    query_embeddings: np.ndarray,
    cohort_embeddings: np.ndarray,
    top_k: int,
    std_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    score_matrix = cosine_score_matrix(
        query_embeddings,
        cohort_embeddings,
        normalize=True,
    )
    effective_top_k = min(top_k, int(score_matrix.shape[1]))
    _, top_scores = rank_cosine_scores(score_matrix, top_k=effective_top_k)
    means = top_scores.mean(axis=1)
    stds = top_scores.std(axis=1, ddof=0)
    floored_mask = stds < std_epsilon
    if floored_mask.any():
        stds = np.where(floored_mask, std_epsilon, stds)
    return means, stds, int(floored_mask.sum()), effective_top_k


def _build_embedding_lookup(
    *,
    metadata_rows: list[dict[str, object]],
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    for index, row in enumerate(metadata_rows):
        for key in _metadata_lookup_keys(row):
            lookup.setdefault(key, embeddings[index])
    return lookup


def _metadata_lookup_keys(row: dict[str, object]) -> tuple[str, ...]:
    keys: list[str] = []
    for field_name in ("trial_item_id", "utterance_id", "audio_path"):
        value = row.get(field_name)
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        keys.append(normalized)
        if field_name == "audio_path":
            keys.append(Path(normalized).name)
    return tuple(dict.fromkeys(keys))


__all__ = [
    "AdaptiveScoreNormalizationResult",
    "AdaptiveScoreNormalizationSummary",
    "DEFAULT_AS_NORM_STD_EPSILON",
    "DEFAULT_AS_NORM_TOP_K",
    "VERIFICATION_AS_NORM_SCORES_JSONL_NAME",
    "VERIFICATION_SCORE_NORMALIZATION_SUMMARY_JSON_NAME",
    "apply_as_norm_to_verification_scores",
]
