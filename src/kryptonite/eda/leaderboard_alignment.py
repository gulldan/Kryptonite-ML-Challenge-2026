"""Rank-correlation helpers for local validation vs public leaderboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PUBLIC_LB_SCORES = {
    "B0_raw_center": 0.1024,
    "B2_raw_3crop": 0.1098,
    "B4_trim_3crop": 0.1150,
    "B7_trim_3crop_reciprocal_top50": 0.1206,
    "B8_trim_3crop_reciprocal_local_scaling": 0.1223,
}


def public_lb_for(experiment_id: str) -> float | None:
    return PUBLIC_LB_SCORES.get(experiment_id)


def public_status_for(experiment_id: str) -> str:
    return "submitted_public_lb_received" if experiment_id in PUBLIC_LB_SCORES else "not_submitted"


def write_leaderboard_alignment(
    *,
    validation_package_dir: Path,
    shifted_v2_dir: Path,
    output_dir: Path,
) -> None:
    runs = pl.read_csv(validation_package_dir / "runs_summary.csv")
    v2 = pl.read_csv(shifted_v2_dir / "dense_gallery_results.csv").select(
        "experiment_id", pl.col("p10").alias("dense_shifted_v2_p10")
    )
    all_runs = runs.join(v2, on="experiment_id", how="full", coalesce=True)
    rows = []
    for row in all_runs.iter_rows(named=True):
        public_lb = public_lb_for(str(row["experiment_id"]))
        if public_lb is None:
            continue
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "public_lb": public_lb,
                "smoke_val_p10": row["smoke_val_p10"],
                "dense_gallery_val_p10": row["dense_gallery_val_p10"],
                "dense_shifted_val_p10": row["dense_shifted_val_p10"],
                "dense_shifted_v2_p10": row["dense_shifted_v2_p10"],
            }
        )
    alignment = pl.DataFrame(rows).sort("public_lb")
    alignment.write_csv(output_dir / "leaderboard_alignment.csv")
    pl.DataFrame(_correlation_rows(alignment)).write_csv(output_dir / "rank_correlation.csv")


def _correlation_rows(alignment: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric in [
        "smoke_val_p10",
        "dense_gallery_val_p10",
        "dense_shifted_val_p10",
        "dense_shifted_v2_p10",
    ]:
        metric_alignment = alignment.filter(pl.col(metric).is_not_null())
        public = metric_alignment["public_lb"].to_numpy()
        local = metric_alignment[metric].to_numpy()
        rows.append(
            {
                "local_protocol": metric,
                "n_public_points": int(len(public)),
                "spearman": _spearman(local, public),
            }
        )
    return rows


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = _ranks(x)
    y_rank = _ranks(y)
    if len(x_rank) < 2:
        return float("nan")
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    return float(corr)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks
