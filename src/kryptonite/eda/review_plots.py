"""Plot helpers for compact EDA review packages."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_review_plots(
    plot_dir: Path,
    *,
    file_stats: pl.DataFrame,
    local_query_eval: pl.DataFrame,
    public_eval: pl.DataFrame,
    speaker_stats: pl.DataFrame,
    audio_dir: Path,
    public_dir: Path,
) -> None:
    _plot_duration_hist(plot_dir / "duration_hist_train_local_public.png", file_stats)
    _plot_bucket_alignment(plot_dir / "bucket_alignment_local_vs_public.png", public_dir)
    _plot_local_p10_by_bucket(plot_dir / "local_p10_by_bucket.png", local_query_eval)
    _plot_same_diff(plot_dir / "local_same_vs_diff_score_distribution.png", audio_dir)
    _plot_public_local_scores(
        plot_dir / "public_vs_local_neighbor_score_distribution.png",
        local_query_eval,
        public_eval,
    )
    _plot_speaker_counts(plot_dir / "speaker_n_utts_distribution.png", speaker_stats)
    _plot_runtime(plot_dir / "runtime_breakdown.png", plot_dir.parent / "12_runtime_profile.csv")
    _plot_worst_gallery(
        plot_dir / "worst_queries_gallery.png", plot_dir.parent / "10_worst_queries.csv"
    )


def _plot_duration_hist(path: Path, file_stats: pl.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    for label, frame in [
        ("train", file_stats.filter(pl.col("split") == "train")),
        ("public", file_stats.filter(pl.col("split") == "test_public")),
    ]:
        values = frame["duration_s"].to_numpy()
        plt.hist(np.clip(values, 0, 60), bins=60, alpha=0.45, density=True, label=label)
    local_paths = pl.read_csv("artifacts/eda/participants_audio6/val_manifest.csv").select(
        "filepath"
    )
    local = local_paths.join(file_stats, on="filepath")
    plt.hist(
        np.clip(local["duration_s"].to_numpy(), 0, 60),
        bins=60,
        alpha=0.45,
        density=True,
        label="local_val",
    )
    _finish_plot(
        "Duration Distribution: Train vs Local vs Public",
        "duration_s clipped at 60",
        "density",
        path,
    )


def _plot_bucket_alignment(path: Path, public_dir: Path) -> None:
    frame = pl.read_csv(public_dir / "public_local_bucket_alignment.csv").filter(
        pl.col("bucket_name") == "domain_cluster"
    )
    labels = frame["bucket_value"].to_list()
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, frame["local_share"].to_numpy(), width=0.4, label="local_val")
    plt.bar(x + 0.2, frame["public_share"].to_numpy(), width=0.4, label="public")
    plt.xticks(x, labels, rotation=30, ha="right")
    _finish_plot("Domain Bucket Alignment: Local vs Public", "domain bucket", "share", path)


def _plot_local_p10_by_bucket(path: Path, local_query_eval: pl.DataFrame) -> None:
    summary = (
        local_query_eval.group_by("duration_bucket")
        .agg(pl.col("p10").mean().alias("mean_p10"))
        .sort("mean_p10")
    )
    plt.figure(figsize=(9, 5))
    plt.bar(summary["duration_bucket"].to_list(), summary["mean_p10"].to_numpy())
    plt.xticks(rotation=30, ha="right")
    _finish_plot("Local P@10 by Duration Bucket", "duration bucket", "mean P@10", path)


def _plot_same_diff(path: Path, audio_dir: Path) -> None:
    embeddings = np.load(audio_dir / "val_embeddings.npy")
    labels = pl.read_csv(audio_dir / "val_manifest.csv")["speaker_id"].to_numpy()
    normed = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(2026)
    same_scores = []
    for speaker in np.unique(labels):
        idx = np.flatnonzero(labels == speaker)
        scores = normed[idx] @ normed[idx].T
        same_scores.extend(scores[np.triu_indices(idx.size, k=1)].tolist())
    left = rng.integers(0, len(labels), size=250_000)
    right = rng.integers(0, len(labels), size=250_000)
    mask = labels[left] != labels[right]
    diff_scores = np.sum(normed[left[mask]] * normed[right[mask]], axis=1)
    plt.figure(figsize=(9, 5))
    plt.hist(diff_scores, bins=80, alpha=0.55, density=True, label="different speaker sample")
    plt.hist(np.asarray(same_scores), bins=80, alpha=0.55, density=True, label="same speaker")
    _finish_plot("Local Same vs Different Speaker Cosine", "cosine", "density", path)


def _plot_public_local_scores(
    path: Path,
    local_query_eval: pl.DataFrame,
    public_eval: pl.DataFrame,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(
        local_query_eval["top10_mean_score"].to_numpy(),
        bins=80,
        alpha=0.55,
        density=True,
        label="local top10 mean",
    )
    plt.hist(
        public_eval["top10_mean_score"].to_numpy(),
        bins=80,
        alpha=0.55,
        density=True,
        label="public top10 mean",
    )
    _finish_plot(
        "Neighbor Score Distribution: Local vs Public", "top10 mean cosine", "density", path
    )


def _plot_speaker_counts(path: Path, speaker_stats: pl.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(speaker_stats["n_utts"].to_numpy(), bins=50)
    _finish_plot("Train Speaker Utterance Counts", "n_utts", "speaker count", path)


def _plot_runtime(path: Path, runtime_csv: Path) -> None:
    frame = pl.read_csv(runtime_csv)
    labels = frame["split"].to_list()
    stages = ["audio_loading_s", "embedding_extraction_s", "search_s", "csv_write_s"]
    bottoms = np.zeros(frame.height)
    plt.figure(figsize=(9, 5))
    for stage in stages:
        values = frame[stage].fill_null(0).to_numpy()
        plt.bar(labels, values, bottom=bottoms, label=stage)
        bottoms += values
    _finish_plot("Runtime Breakdown", "split", "seconds", path)


def _plot_worst_gallery(path: Path, worst_csv: Path) -> None:
    worst = pl.read_csv(worst_csv).head(20)
    rows = []
    for row in worst.iter_rows(named=True):
        rows.append(
            [
                str(row["query_idx"]),
                str(row["speaker_id"])[:10],
                f"{float(row['duration_s']):.2f}",
                f"{float(row['p10']):.2f}",
                f"{float(row['top1_score']):.2f}",
                "peak" if bool(row.get("peak_limited_flag", False)) else "",
            ]
        )
    plt.figure(figsize=(12, 7))
    plt.axis("off")
    table = plt.table(
        cellText=rows,
        colLabels=["query", "speaker", "dur_s", "P@10", "top1", "flag"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    plt.title("Worst Local Queries: Top 20", pad=18)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _finish_plot(title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles and labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
