"""Reusable embedding-scoring service for thin HTTP adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np

from kryptonite.models import (
    average_normalized_embeddings,
    cosine_score_matrix,
    cosine_score_pairs,
    ensure_embedding_matrix,
    rank_cosine_scores,
)


class EnrollmentNotFoundError(LookupError):
    """Raised when a verify call references an unknown enrollment id."""


@dataclass(frozen=True, slots=True)
class EnrollmentRecord:
    enrollment_id: str
    sample_count: int
    embedding_dim: int
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enrollment_id": self.enrollment_id,
            "sample_count": self.sample_count,
            "embedding_dim": self.embedding_dim,
            "metadata": dict(self.metadata),
        }


class ScoringService:
    def __init__(
        self,
        *,
        initial_enrollments: Mapping[str, EnrollmentRecord] | None = None,
    ) -> None:
        self._lock = Lock()
        self._enrollments = {
            enrollment_id: _clone_enrollment_record(record)
            for enrollment_id, record in (initial_enrollments or {}).items()
        }

    def list_enrollments(self) -> dict[str, Any]:
        with self._lock:
            enrollments = [
                record.to_dict()
                for record in sorted(
                    self._enrollments.values(),
                    key=lambda candidate: candidate.enrollment_id,
                )
            ]
        return {
            "enrollment_count": len(enrollments),
            "enrollments": enrollments,
        }

    def score_pairwise(
        self,
        *,
        left: Any,
        right: Any,
        normalize: bool = True,
    ) -> dict[str, Any]:
        left_matrix = ensure_embedding_matrix(left, field_name="left")
        right_matrix = ensure_embedding_matrix(right, field_name="right")
        scores = cosine_score_pairs(left_matrix, right_matrix, normalize=normalize)
        return {
            "mode": "pairwise",
            "normalized": normalize,
            "embedding_dim": int(left_matrix.shape[1]),
            "score_count": int(scores.shape[0]),
            "scores": _round_scores(scores),
            "mean_score": _round_optional_float(float(scores.mean()) if scores.size else None),
        }

    def score_one_to_many(
        self,
        *,
        queries: Any,
        references: Any,
        normalize: bool = True,
        top_k: int | None = None,
        query_ids: list[str] | None = None,
        reference_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        query_matrix = ensure_embedding_matrix(queries, field_name="queries")
        reference_matrix = ensure_embedding_matrix(references, field_name="references")
        _validate_identifier_count(
            query_ids,
            expected=query_matrix.shape[0],
            field_name="query_ids",
        )
        _validate_identifier_count(
            reference_ids,
            expected=reference_matrix.shape[0],
            field_name="reference_ids",
        )

        scores = cosine_score_matrix(
            query_matrix,
            reference_matrix,
            normalize=normalize,
        )
        effective_top_k = reference_matrix.shape[0] if top_k is None else top_k
        ranked_indices, ranked_scores = rank_cosine_scores(scores, top_k=effective_top_k)
        resolved_reference_ids = reference_ids or [
            f"reference-{index:05d}" for index in range(reference_matrix.shape[0])
        ]
        resolved_query_ids = query_ids or [
            f"query-{index:05d}" for index in range(query_matrix.shape[0])
        ]

        return {
            "mode": "one_to_many",
            "normalized": normalize,
            "embedding_dim": int(query_matrix.shape[1]),
            "query_count": int(query_matrix.shape[0]),
            "reference_count": int(reference_matrix.shape[0]),
            "scores": [[round(float(score), 8) for score in row] for row in scores],
            "top_matches": [
                {
                    "query_id": resolved_query_ids[row_index],
                    "matches": [
                        {
                            "reference_id": resolved_reference_ids[int(column_index)],
                            "score": round(float(score), 8),
                        }
                        for column_index, score in zip(
                            ranked_indices[row_index],
                            ranked_scores[row_index],
                            strict=True,
                        )
                    ],
                }
                for row_index in range(scores.shape[0])
            ],
        }

    def enroll(
        self,
        *,
        enrollment_id: str,
        embeddings: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_enrollment_id = enrollment_id.strip()
        if not normalized_enrollment_id:
            raise ValueError("enrollment_id must not be empty.")

        matrix = ensure_embedding_matrix(embeddings, field_name="embeddings")
        pooled_embedding = average_normalized_embeddings(matrix, field_name="embeddings")
        record = EnrollmentRecord(
            enrollment_id=normalized_enrollment_id,
            sample_count=int(matrix.shape[0]),
            embedding_dim=int(matrix.shape[1]),
            embedding=pooled_embedding,
            metadata=_coerce_metadata(metadata),
        )
        with self._lock:
            replaced = normalized_enrollment_id in self._enrollments
            self._enrollments[normalized_enrollment_id] = record
        return {
            "enrollment_id": record.enrollment_id,
            "sample_count": record.sample_count,
            "embedding_dim": record.embedding_dim,
            "metadata": dict(record.metadata),
            "replaced": replaced,
        }

    def verify(
        self,
        *,
        enrollment_id: str,
        probes: Any,
        normalize: bool = True,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        normalized_enrollment_id = enrollment_id.strip()
        if not normalized_enrollment_id:
            raise ValueError("enrollment_id must not be empty.")

        with self._lock:
            record = self._enrollments.get(normalized_enrollment_id)
        if record is None:
            raise EnrollmentNotFoundError(f"Unknown enrollment_id: {normalized_enrollment_id!r}.")

        probe_matrix = ensure_embedding_matrix(probes, field_name="probes")
        enrollment_matrix = np.repeat(
            record.embedding.reshape(1, -1),
            probe_matrix.shape[0],
            axis=0,
        )
        scores = cosine_score_pairs(
            enrollment_matrix,
            probe_matrix,
            normalize=normalize,
        )
        decisions = None
        if threshold is not None:
            decisions = [bool(score >= threshold) for score in scores]

        return {
            "mode": "verify",
            "normalized": normalize,
            "enrollment_id": record.enrollment_id,
            "probe_count": int(probe_matrix.shape[0]),
            "embedding_dim": record.embedding_dim,
            "scores": _round_scores(scores),
            "decision_threshold": _round_optional_float(threshold),
            "decisions": decisions,
        }


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object when provided.")
    return dict(metadata)


def _clone_enrollment_record(record: EnrollmentRecord) -> EnrollmentRecord:
    return EnrollmentRecord(
        enrollment_id=record.enrollment_id,
        sample_count=record.sample_count,
        embedding_dim=record.embedding_dim,
        embedding=np.asarray(record.embedding, dtype=np.float64).copy(),
        metadata=dict(record.metadata),
    )


def _round_scores(scores: np.ndarray) -> list[float]:
    return [round(float(score), 8) for score in scores]


def _round_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 8)


def _validate_identifier_count(
    identifiers: list[str] | None,
    *,
    expected: int,
    field_name: str,
) -> None:
    if identifiers is None:
        return
    if len(identifiers) != expected:
        raise ValueError(f"{field_name} must contain exactly {expected} entries.")
    if any(not identifier.strip() for identifier in identifiers):
        raise ValueError(f"{field_name} must not contain empty identifiers.")


__all__ = [
    "EnrollmentNotFoundError",
    "EnrollmentRecord",
    "ScoringService",
]
