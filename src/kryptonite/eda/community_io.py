"""Evaluation and persistence helpers for community reranking workflows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl


def evaluate_labelled_topk(
    *,
    experiment_id: str,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    manifest: pl.DataFrame,
    query_only: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    labels = manifest["speaker_id"].cast(pl.Utf8).to_list()
    paths = manifest["filepath"].cast(pl.Utf8).to_list()
    if query_only and "is_query" in manifest.columns:
        query_indices = manifest.filter(pl.col("is_query"))["gallery_index"].to_numpy()
    elif "gallery_index" in manifest.columns:
        query_indices = manifest["gallery_index"].to_numpy()
    else:
        query_indices = np.arange(manifest.height)
    rows = []
    for query_index in query_indices:
        neighbours = top_indices[int(query_index)]
        scores = top_scores[int(query_index)]
        query_label = labels[int(query_index)]
        hits = [labels[int(index)] == query_label for index in neighbours]
        rows.append(
            {
                "experiment_id": experiment_id,
                "query_idx": int(query_index),
                "query_path": paths[int(query_index)],
                "speaker_id": query_label,
                "p10": float(np.mean(hits)),
                "n_correct_top10": int(sum(hits)),
                "top1_correct": bool(hits[0]),
                "top1_score": float(scores[0]),
                "top10_mean_score": float(np.mean(scores)),
            }
        )
    frame = pl.DataFrame(rows)
    summary = {
        "experiment_id": experiment_id,
        "query_count": frame.height,
        "gallery_count": manifest.height,
        "p10": _mean_float(frame["p10"]),
        "top1_accuracy": _mean_float(frame["top1_correct"]),
        "mean_top10_score": _mean_float(frame["top10_mean_score"]),
    }
    return frame, summary


def _mean_float(series: pl.Series) -> float:
    return float(cast(float | int, series.mean()))


def write_submission(
    *,
    manifest: pl.DataFrame,
    top_indices: np.ndarray,
    output_csv: Path,
) -> None:
    paths = manifest["filepath"].cast(pl.Utf8).to_list()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filepath", "neighbours"])
        for filepath, neighbours in zip(paths, top_indices, strict=True):
            writer.writerow([filepath, ",".join(str(int(index)) for index in neighbours)])


def write_cluster_assignments(
    *,
    manifest: pl.DataFrame,
    labels: np.ndarray,
    output_csv: Path,
) -> None:
    """Persist transductive cluster labels for audit and pseudo-label selection."""

    cluster_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    frame = manifest.select("filepath").with_columns(
        pl.Series("row_index", np.arange(manifest.height, dtype=np.int64)),
        pl.Series("cluster_id", labels.astype(np.int64, copy=False)),
        pl.Series("cluster_size", cluster_sizes[labels].astype(np.int64, copy=False)),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output_csv)
