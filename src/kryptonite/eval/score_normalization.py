"""Shared helpers for cohort-backed verification score normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.models import cosine_score_matrix, l2_normalize_embeddings

from .cohort_bank import load_cohort_embedding_bank
from .embedding_atlas.io import align_metadata_rows, load_embedding_matrix, load_metadata_rows
from .verification_data import resolve_trial_side_identifier
from .verification_metrics import normalize_verification_score_rows


@dataclass(frozen=True, slots=True)
class TrialScoreRecord:
    row_index: int
    label: int
    raw_score: float
    left_identifier: str
    right_identifier: str


@dataclass(frozen=True, slots=True)
class IdentifierCohortStatistics:
    mean: float
    std: float
    selected_count: int
    excluded_same_speaker_count: int


@dataclass(frozen=True, slots=True)
class CohortStatisticsComputationSummary:
    floored_std_count: int
    effective_top_k: int
    min_selected_count: int
    max_selected_count: int


@dataclass(frozen=True, slots=True)
class ScoreNormalizationContext:
    identifier_to_embedding: dict[str, np.ndarray]
    identifier_to_speaker_id: dict[str, str | None]
    cohort_embeddings: np.ndarray
    cohort_speaker_ids: tuple[str | None, ...]

    @property
    def cohort_size(self) -> int:
        return int(self.cohort_embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.cohort_embeddings.shape[1])


def build_score_normalization_context(
    *,
    embeddings_path: Path | str,
    metadata_path: Path | str,
    cohort_bank_root: Path | str,
    embeddings_key: str = "embeddings",
    ids_key: str | None = "point_ids",
    point_id_field: str = "atlas_point_id",
) -> ScoreNormalizationContext:
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
    identifier_to_embedding: dict[str, np.ndarray] = {}
    identifier_to_speaker_id: dict[str, str | None] = {}
    for index, row in enumerate(aligned_metadata_rows):
        speaker_id = _coerce_optional_string(row.get("speaker_id"))
        for key in metadata_lookup_keys(row):
            identifier_to_embedding.setdefault(key, evaluation_embeddings[index])
            identifier_to_speaker_id.setdefault(key, speaker_id)

    cohort_bank = load_cohort_embedding_bank(cohort_bank_root)
    cohort_embeddings = l2_normalize_embeddings(
        cohort_bank.embeddings,
        field_name="cohort_embeddings",
    )
    cohort_speaker_ids = tuple(
        _coerce_optional_string(row.get("speaker_id")) for row in cohort_bank.metadata_rows
    )
    return ScoreNormalizationContext(
        identifier_to_embedding=identifier_to_embedding,
        identifier_to_speaker_id=identifier_to_speaker_id,
        cohort_embeddings=cohort_embeddings,
        cohort_speaker_ids=cohort_speaker_ids,
    )


def resolve_trial_score_records(score_rows: list[dict[str, Any]]) -> list[TrialScoreRecord]:
    normalized_rows = normalize_verification_score_rows(score_rows)
    if not score_rows:
        raise ValueError("score_rows must not be empty.")

    records: list[TrialScoreRecord] = []
    for row_index, (raw_row, normalized_row) in enumerate(
        zip(score_rows, normalized_rows, strict=True),
        start=1,
    ):
        left_identifier = resolve_trial_side_identifier(raw_row, "left")
        right_identifier = resolve_trial_side_identifier(raw_row, "right")
        if left_identifier is None or right_identifier is None:
            raise ValueError(
                "Score normalization requires left/right identifiers for every score row; "
                f"row {row_index} is missing one or both sides."
            )
        records.append(
            TrialScoreRecord(
                row_index=row_index,
                label=int(normalized_row["label"]),
                raw_score=float(normalized_row["score"]),
                left_identifier=left_identifier,
                right_identifier=right_identifier,
            )
        )
    return records


def compute_identifier_cohort_statistics(
    context: ScoreNormalizationContext,
    *,
    identifiers: tuple[str, ...],
    top_k: int,
    std_epsilon: float,
    exclude_matching_speakers: bool = False,
) -> tuple[dict[str, IdentifierCohortStatistics], CohortStatisticsComputationSummary]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if std_epsilon <= 0.0:
        raise ValueError("std_epsilon must be positive.")
    if not identifiers:
        raise ValueError("identifiers must not be empty.")

    missing_identifiers = sorted(
        identifier
        for identifier in identifiers
        if identifier not in context.identifier_to_embedding
    )
    if missing_identifiers:
        preview = ", ".join(repr(identifier) for identifier in missing_identifiers[:3])
        raise ValueError(
            "Score normalization could not resolve an embedding for: "
            f"{preview}."
        )

    query_embeddings = np.stack(
        [context.identifier_to_embedding[identifier] for identifier in identifiers],
        axis=0,
    )
    score_matrix = cosine_score_matrix(
        query_embeddings,
        context.cohort_embeddings,
        normalize=True,
    )
    global_top_k = min(top_k, int(score_matrix.shape[1]))

    statistics: dict[str, IdentifierCohortStatistics] = {}
    floored_std_count = 0
    selected_counts: list[int] = []

    for row_index, identifier in enumerate(identifiers):
        scores = np.asarray(score_matrix[row_index], dtype=np.float64)
        excluded_same_speaker_count = 0
        if exclude_matching_speakers:
            speaker_id = context.identifier_to_speaker_id.get(identifier)
            if speaker_id is not None:
                mask = np.asarray(
                    [candidate == speaker_id for candidate in context.cohort_speaker_ids],
                    dtype=bool,
                )
                excluded_same_speaker_count = int(mask.sum())
                if excluded_same_speaker_count:
                    scores = scores[~mask]

        if scores.size == 0:
            raise ValueError(
                f"Score normalization removed every cohort candidate for identifier {identifier!r}."
            )

        selected_count = min(global_top_k, int(scores.shape[0]))
        if selected_count <= 0:
            raise ValueError("selected_count must stay positive after cohort filtering.")
        top_scores = np.partition(scores, -selected_count)[-selected_count:]
        mean = float(top_scores.mean())
        std = float(top_scores.std(ddof=0))
        if std < std_epsilon:
            std = std_epsilon
            floored_std_count += 1
        selected_counts.append(selected_count)
        statistics[identifier] = IdentifierCohortStatistics(
            mean=mean,
            std=std,
            selected_count=selected_count,
            excluded_same_speaker_count=excluded_same_speaker_count,
        )

    return statistics, CohortStatisticsComputationSummary(
        floored_std_count=floored_std_count,
        effective_top_k=global_top_k,
        min_selected_count=min(selected_counts),
        max_selected_count=max(selected_counts),
    )


def metadata_lookup_keys(row: dict[str, object]) -> tuple[str, ...]:
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


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "CohortStatisticsComputationSummary",
    "IdentifierCohortStatistics",
    "ScoreNormalizationContext",
    "TrialScoreRecord",
    "build_score_normalization_context",
    "compute_identifier_cohort_statistics",
    "metadata_lookup_keys",
    "resolve_trial_score_records",
]
