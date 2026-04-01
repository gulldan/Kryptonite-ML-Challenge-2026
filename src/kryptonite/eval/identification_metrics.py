"""Closed-set and open-set identification metrics.

Computes CMC (Cumulative Match Characteristic), FNIR, FPIR, FAR, FRR
from query-vs-gallery embedding score matrices.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from kryptonite.models.scoring import cosine_score_matrix, rank_cosine_scores


@dataclass(frozen=True, slots=True)
class CMCPoint:
    """One point on the Cumulative Match Characteristic curve."""

    rank: int
    identification_rate: float
    miss_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentificationOperatingPoint:
    """Identification metrics at a single decision threshold."""

    threshold: float
    fnir: float
    fpir: float
    far: float
    frr: float
    true_positive_count: int
    false_positive_count: int
    true_negative_count: int
    false_negative_count: int

    def to_dict(self) -> dict[str, Any]:
        threshold = self.threshold if math.isfinite(self.threshold) else None
        return {**asdict(self), "threshold": threshold}


@dataclass(frozen=True, slots=True)
class IdentificationMetricsSummary:
    """Aggregated identification evaluation results."""

    query_count: int
    gallery_count: int
    gallery_speaker_count: int
    rank_1_accuracy: float
    rank_5_accuracy: float
    rank_10_accuracy: float
    rank_20_accuracy: float
    cmc_curve: tuple[CMCPoint, ...]
    operating_points: tuple[IdentificationOperatingPoint, ...]
    fnir_at_fpir_0_01: float | None
    fnir_at_fpir_0_1: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "gallery_count": self.gallery_count,
            "gallery_speaker_count": self.gallery_speaker_count,
            "rank_1_accuracy": self.rank_1_accuracy,
            "rank_5_accuracy": self.rank_5_accuracy,
            "rank_10_accuracy": self.rank_10_accuracy,
            "rank_20_accuracy": self.rank_20_accuracy,
            "cmc_curve": [p.to_dict() for p in self.cmc_curve],
            "operating_points": [p.to_dict() for p in self.operating_points],
            "fnir_at_fpir_0_01": self.fnir_at_fpir_0_01,
            "fnir_at_fpir_0_1": self.fnir_at_fpir_0_1,
        }


def compute_identification_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_speaker_ids: list[str],
    gallery_speaker_ids: list[str],
    *,
    max_rank: int | None = None,
    num_thresholds: int = 200,
) -> IdentificationMetricsSummary:
    """Compute closed-set and open-set identification metrics.

    Parameters
    ----------
    query_embeddings:
        (N_query, D) embedding matrix for probe/test utterances.
    gallery_embeddings:
        (N_gallery, D) embedding matrix for enrolled/reference utterances.
    query_speaker_ids:
        Speaker identity label per query row.
    gallery_speaker_ids:
        Speaker identity label per gallery row.
    max_rank:
        Maximum rank for CMC curve. Defaults to gallery_count.
    num_thresholds:
        Number of threshold points for FNIR/FPIR operating curve.
    """
    n_query = query_embeddings.shape[0]
    n_gallery = gallery_embeddings.shape[0]
    if n_query == 0:
        raise ValueError("query_embeddings must contain at least one row.")
    if n_gallery == 0:
        raise ValueError("gallery_embeddings must contain at least one row.")
    if len(query_speaker_ids) != n_query:
        raise ValueError("query_speaker_ids length must match query_embeddings rows.")
    if len(gallery_speaker_ids) != n_gallery:
        raise ValueError("gallery_speaker_ids length must match gallery_embeddings rows.")

    gallery_speakers_unique = sorted(set(gallery_speaker_ids))
    effective_max_rank = min(max_rank or n_gallery, n_gallery)

    score_mat = cosine_score_matrix(query_embeddings, gallery_embeddings, normalize=True)
    sorted_indices, sorted_scores = rank_cosine_scores(score_mat, top_k=effective_max_rank)

    # --- CMC curve ---
    gallery_ids_arr = np.array(gallery_speaker_ids)
    query_ids_arr = np.array(query_speaker_ids)
    ranked_speaker_ids = gallery_ids_arr[sorted_indices]
    is_correct = ranked_speaker_ids == query_ids_arr[:, np.newaxis]
    first_correct_rank = np.full(n_query, effective_max_rank + 1, dtype=np.int64)
    for q_idx in range(n_query):
        correct_positions = np.where(is_correct[q_idx])[0]
        if correct_positions.size > 0:
            first_correct_rank[q_idx] = correct_positions[0] + 1

    cmc_points: list[CMCPoint] = []
    for rank in range(1, effective_max_rank + 1):
        rate = float(np.mean(first_correct_rank <= rank))
        cmc_points.append(
            CMCPoint(
                rank=rank,
                identification_rate=round(rate, 6),
                miss_rate=round(1.0 - rate, 6),
            )
        )

    rank_1_acc = cmc_points[0].identification_rate if cmc_points else 0.0
    rank_5_acc = (
        cmc_points[min(4, len(cmc_points) - 1)].identification_rate
        if len(cmc_points) >= 5
        else rank_1_acc
    )
    rank_10_acc = (
        cmc_points[min(9, len(cmc_points) - 1)].identification_rate
        if len(cmc_points) >= 10
        else rank_5_acc
    )
    rank_20_acc = (
        cmc_points[min(19, len(cmc_points) - 1)].identification_rate
        if len(cmc_points) >= 20
        else rank_10_acc
    )

    # --- FNIR / FPIR / FAR / FRR at thresholds ---
    # For each query:
    #   - top-1 score = max similarity to any gallery item
    #   - genuine = True if query speaker is in gallery
    #   - match_correct = True if top-1 gallery item has same speaker as query
    top1_scores = sorted_scores[:, 0]
    top1_gallery_speakers = gallery_ids_arr[sorted_indices[:, 0]]
    is_mate = np.array([qid in gallery_speakers_unique for qid in query_speaker_ids])
    top1_is_correct = top1_gallery_speakers == query_ids_arr

    score_min = float(np.min(score_mat))
    score_max = float(np.max(score_mat))
    thresholds = np.linspace(score_min, score_max, num_thresholds)

    n_mates = int(is_mate.sum())
    n_nonmates = n_query - n_mates

    operating_points: list[IdentificationOperatingPoint] = []
    for threshold in thresholds:
        above_threshold = top1_scores >= threshold

        # True positive: mate query, top-1 correct, score >= threshold
        tp = int(np.sum(is_mate & top1_is_correct & above_threshold))
        # False negative: mate query, either top-1 wrong or score < threshold
        fn = n_mates - tp
        # False positive: non-mate query, score >= threshold (wrongly accepted)
        fp = int(np.sum(~is_mate & above_threshold))
        # True negative: non-mate query, score < threshold (correctly rejected)
        tn = n_nonmates - fp

        fnir = fn / n_mates if n_mates > 0 else 0.0
        fpir = fp / n_nonmates if n_nonmates > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        operating_points.append(
            IdentificationOperatingPoint(
                threshold=round(float(threshold), 8),
                fnir=round(fnir, 6),
                fpir=round(fpir, 6),
                far=round(far, 6),
                frr=round(frr, 6),
                true_positive_count=tp,
                false_positive_count=fp,
                true_negative_count=tn,
                false_negative_count=fn,
            )
        )

    # FNIR@FPIR is only meaningful when non-mate queries exist
    if n_nonmates > 0:
        fnir_at_fpir_001 = _interpolate_fnir_at_fpir(operating_points, target_fpir=0.01)
        fnir_at_fpir_01 = _interpolate_fnir_at_fpir(operating_points, target_fpir=0.1)
    else:
        fnir_at_fpir_001 = None
        fnir_at_fpir_01 = None

    return IdentificationMetricsSummary(
        query_count=n_query,
        gallery_count=n_gallery,
        gallery_speaker_count=len(gallery_speakers_unique),
        rank_1_accuracy=rank_1_acc,
        rank_5_accuracy=rank_5_acc,
        rank_10_accuracy=rank_10_acc,
        rank_20_accuracy=rank_20_acc,
        cmc_curve=tuple(cmc_points),
        operating_points=tuple(operating_points),
        fnir_at_fpir_0_01=fnir_at_fpir_001,
        fnir_at_fpir_0_1=fnir_at_fpir_01,
    )


def _interpolate_fnir_at_fpir(
    points: list[IdentificationOperatingPoint],
    target_fpir: float,
) -> float | None:
    """Find FNIR at a given FPIR via linear interpolation."""
    if not points:
        return None
    sorted_pts = sorted(points, key=lambda p: p.fpir)
    if target_fpir <= sorted_pts[0].fpir:
        return sorted_pts[0].fnir
    if target_fpir >= sorted_pts[-1].fpir:
        return sorted_pts[-1].fnir
    for left, right in zip(sorted_pts, sorted_pts[1:], strict=True):
        if left.fpir <= target_fpir <= right.fpir:
            if math.isclose(left.fpir, right.fpir, abs_tol=1e-12):
                return left.fnir
            ratio = (target_fpir - left.fpir) / (right.fpir - left.fpir)
            return round(left.fnir + ratio * (right.fnir - left.fnir), 6)
    return None


__all__ = [
    "CMCPoint",
    "IdentificationMetricsSummary",
    "IdentificationOperatingPoint",
    "compute_identification_metrics",
]
