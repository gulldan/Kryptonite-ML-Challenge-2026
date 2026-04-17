"""Plot writers for validation-cycle review packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def write_validation_cycle_plots(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    runs = pl.read_csv(output_dir / "runs_summary.csv")
    x = np.arange(runs.height)
    labels = runs["experiment_id"].to_list()
    plt.figure(figsize=(11, 5))
    plt.plot(x, runs["smoke_val_p10"].to_numpy(), marker="o", label="smoke")
    plt.plot(x, runs["dense_gallery_val_p10"].to_numpy(), marker="o", label="dense")
    plt.plot(x, runs["dense_shifted_val_p10"].to_numpy(), marker="o", label="shifted")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("P@10")
    plt.title("Run Ranking Across Local Protocols")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "run_ranking_smoke_dense_public.png", dpi=160)
    plt.close()

    _plot_bucket_delta(output_dir, plot_dir, runs, plt)
    _plot_hubness(output_dir, plot_dir, plt)
    _plot_runtime_score(plot_dir, runs, plt)


def _plot_bucket_delta(output_dir: Path, plot_dir: Path, runs: pl.DataFrame, plt: Any) -> None:
    bucket = pl.read_csv(output_dir / "bucket_metrics_by_run.csv")
    best = runs.sort("dense_shifted_val_p10", descending=True).row(0, named=True)["experiment_id"]
    b0 = _duration_bucket_frame(bucket, "B0_raw_center", "b0_p10")
    best_df = _duration_bucket_frame(bucket, str(best), "best_p10")
    delta = b0.join(best_df, on="bucket_value").with_columns(
        (pl.col("best_p10") - pl.col("b0_p10")).alias("delta")
    )
    plt.figure(figsize=(8, 4))
    plt.bar(delta["bucket_value"].to_list(), delta["delta"].to_numpy())
    plt.ylabel("P@10 delta")
    plt.title("Best vs Baseline by Duration Bucket, Shifted Val")
    plt.tight_layout()
    plt.savefig(plot_dir / "best_vs_baseline_bucket_delta.png", dpi=160)
    plt.close()


def _duration_bucket_frame(bucket: pl.DataFrame, experiment_id: str, alias: str) -> pl.DataFrame:
    return bucket.filter(
        (pl.col("protocol") == "dense_shifted_val")
        & (pl.col("experiment_id") == experiment_id)
        & (pl.col("bucket_name") == "duration_bucket")
    ).select("bucket_value", pl.col("p10").alias(alias))


def _plot_hubness(output_dir: Path, plot_dir: Path, plt: Any) -> None:
    hub = pl.read_csv(output_dir / "hubness_report.csv").filter(pl.col("status") == "computed")
    hub10 = hub.filter(pl.col("k") == 10)
    plt.figure(figsize=(9, 4))
    names = [f"{row['pool']}:{row['state']}" for row in hub10.iter_rows(named=True)]
    plt.bar(names, hub10["gini"].to_numpy())
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Gini@10")
    plt.title("Hubness Local vs Public")
    plt.tight_layout()
    plt.savefig(plot_dir / "hubness_local_vs_public.png", dpi=160)
    plt.close()


def _plot_runtime_score(plot_dir: Path, runs: pl.DataFrame, plt: Any) -> None:
    plt.figure(figsize=(7, 4))
    plt.scatter(runs["dense_shifted_runtime_total_s"], runs["dense_shifted_val_p10"])
    for row in runs.iter_rows(named=True):
        plt.text(
            row["dense_shifted_runtime_total_s"],
            row["dense_shifted_val_p10"],
            row["experiment_id"][:2],
        )
    plt.xlabel("Dense-shifted runtime, s")
    plt.ylabel("Dense-shifted P@10")
    plt.title("Runtime vs Score")
    plt.tight_layout()
    plt.savefig(plot_dir / "runtime_vs_score.png", dpi=160)
    plt.close()
