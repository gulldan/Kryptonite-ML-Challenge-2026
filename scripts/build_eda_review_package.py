from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from kryptonite.eda.review_defs import BUCKET_DEFS
from kryptonite.eda.review_plots import write_review_plots
from kryptonite.eda.review_summary import write_dataset_split_summary, write_review_summary


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "plots").mkdir(parents=True)

    audio_dir = Path(args.audio_artifact_dir)
    public_dir = Path(args.public_artifact_dir)
    file_stats = _with_review_columns(pl.read_parquet(audio_dir / "file_stats.parquet"))
    local_manifest = pl.read_csv(audio_dir / "val_manifest.csv")
    local_eval_raw = pl.read_parquet(audio_dir / "embedding_eval.parquet")
    local_query_eval = _build_local_query_eval(file_stats, local_manifest, local_eval_raw)
    public_eval = pl.read_csv(public_dir / "public_embedding_eval_unlabeled.csv")
    speaker_stats = _build_train_speaker_stats(file_stats)

    write_review_summary(
        output_dir / "00_summary.md",
        audio_dir=audio_dir,
        public_dir=public_dir,
        file_stats=file_stats,
        local_query_eval=local_query_eval,
        public_lb_score=args.public_lb_score,
    )
    write_dataset_split_summary(
        output_dir / "01_dataset_split_summary.json",
        audio_dir=audio_dir,
        file_stats=file_stats,
        local_manifest=local_manifest,
        public_lb_score=args.public_lb_score,
    )
    (output_dir / "02_bucket_defs.json").write_text(
        json.dumps(BUCKET_DEFS, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _build_bucket_alignment(file_stats, local_manifest).write_csv(
        output_dir / "03_bucket_alignment.csv"
    )
    speaker_stats.write_parquet(output_dir / "04_train_speaker_stats.parquet")
    _select_file_stats(file_stats, split="local_val", local_manifest=local_manifest).write_parquet(
        output_dir / "05_local_file_stats.parquet"
    )
    _select_file_stats(file_stats, split="public", local_manifest=None).write_parquet(
        output_dir / "06_public_file_stats.parquet"
    )
    local_query_eval.write_parquet(output_dir / "07_local_query_eval.parquet")
    _write_public_score_summary(public_dir, output_dir / "08_public_neighbor_score_summary.csv")
    _write_public_local_alignment(public_dir, output_dir / "09_public_local_alignment.csv")
    _write_worst_cases(local_query_eval, audio_dir, output_dir)
    _write_runtime_profile(audio_dir, public_dir, output_dir / "12_runtime_profile.csv")
    write_review_plots(
        output_dir / "plots",
        file_stats=file_stats,
        local_query_eval=local_query_eval,
        public_eval=public_eval,
        speaker_stats=speaker_stats,
        audio_dir=audio_dir,
        public_dir=public_dir,
    )
    zip_path = Path(args.zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    _zip_dir(output_dir, zip_path)
    print(f"Wrote review package: {zip_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-artifact-dir", default="artifacts/eda/participants_audio6")
    parser.add_argument(
        "--public-artifact-dir", default="artifacts/eda/participants_public_baseline"
    )
    parser.add_argument("--output-dir", default="artifacts/eda/eda_review_package")
    parser.add_argument("--zip-path", default="artifacts/eda/eda_review_package.zip")
    parser.add_argument("--public-lb-score", type=float, default=0.0779)
    return parser.parse_args()


def _with_review_columns(frame: pl.DataFrame) -> pl.DataFrame:
    domain = pl.read_csv("artifacts/eda/participants_audio6/domain_clusters.csv").select(
        "filepath", "domain_cluster"
    )
    return (
        frame.join(domain, on="filepath", how="left")
        .with_columns(
            _duration_bucket_expr(),
            _rms_bucket_expr(),
            _silence_bucket_expr("leading_silence_s", "leading_silence_bucket"),
            _silence_bucket_expr("trailing_silence_s", "trailing_silence_bucket"),
            (pl.col("peak_dbfs") >= -0.1).alias("peak_limited_flag"),
            (pl.col("clipping_frac") > 0.01).alias("hard_clipped_flag"),
            (pl.col("silence_ratio_40db") >= 0.5).alias("silence_heavy_flag"),
            ((pl.col("narrowband_proxy") >= 0.5) | (pl.col("rolloff95_hz") <= 3800.0)).alias(
                "narrowband_like"
            ),
            (pl.col("rms_dbfs") <= -40.0).alias("low_rms_flag"),
        )
        .with_columns(
            pl.when(pl.col("domain_cluster") == "clipped")
            .then(pl.lit("peak_limited"))
            .otherwise(pl.col("domain_cluster"))
            .alias("domain_cluster"),
            pl.col("band_energy_300_3400").alias("bandratio_300_3400"),
            pl.col("band_energy_3400_8000").alias("bandratio_3400_8000"),
            pl.col("silence_ratio_40db").alias("silence_ratio"),
        )
    )


def _duration_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("duration_s") < 2)
        .then(pl.lit("very_short"))
        .when(pl.col("duration_s") < 4)
        .then(pl.lit("short"))
        .when(pl.col("duration_s") < 6)
        .then(pl.lit("medium"))
        .when(pl.col("duration_s") < 10)
        .then(pl.lit("normal"))
        .when(pl.col("duration_s") < 20)
        .then(pl.lit("long"))
        .when(pl.col("duration_s") < 40)
        .then(pl.lit("very_long"))
        .otherwise(pl.lit("extra_long"))
        .alias("duration_bucket")
    )


def _rms_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("rms_dbfs") <= -40)
        .then(pl.lit("very_low"))
        .when(pl.col("rms_dbfs") <= -30)
        .then(pl.lit("low"))
        .when(pl.col("rms_dbfs") <= -20)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias("rms_bucket")
    )


def _silence_bucket_expr(column: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) < 0.2)
        .then(pl.lit("none"))
        .when(pl.col(column) < 1.0)
        .then(pl.lit("short"))
        .otherwise(pl.lit("long"))
        .alias(alias)
    )


def _build_train_speaker_stats(file_stats: pl.DataFrame) -> pl.DataFrame:
    return (
        file_stats.filter(pl.col("split") == "train")
        .group_by("speaker_id")
        .agg(
            pl.len().alias("n_utts"),
            pl.col("duration_s").sum().alias("total_duration_s"),
            pl.col("duration_s").mean().alias("mean_duration_s"),
            pl.col("duration_s").median().alias("median_duration_s"),
            (pl.col("duration_s") < 2).mean().alias("pct_short"),
            pl.col("silence_heavy_flag").mean().alias("pct_silence_heavy"),
            pl.col("peak_limited_flag").mean().alias("pct_peak_limited"),
            pl.col("hard_clipped_flag").mean().alias("pct_hard_clipped"),
            pl.col("narrowband_like").mean().alias("pct_narrowband_like"),
            pl.col("low_rms_flag").mean().alias("pct_low_rms"),
        )
        .sort("n_utts", descending=True)
    )


def _select_file_stats(
    file_stats: pl.DataFrame,
    *,
    split: str,
    local_manifest: pl.DataFrame | None,
) -> pl.DataFrame:
    columns = [
        "filepath",
        "duration_s",
        "rms_dbfs",
        "peak_dbfs",
        "clipping_frac",
        "leading_silence_s",
        "trailing_silence_s",
        "silence_ratio",
        "zcr",
        "spectral_centroid_hz",
        "rolloff95_hz",
        "spectral_flatness",
        "bandratio_300_3400",
        "bandratio_3400_8000",
        "narrowband_like",
        "peak_limited_flag",
        "hard_clipped_flag",
        "silence_heavy_flag",
        "duration_bucket",
        "leading_silence_bucket",
        "trailing_silence_bucket",
        "domain_cluster",
    ]
    if split == "local_val":
        assert local_manifest is not None
        return (
            local_manifest.select("filepath")
            .join(file_stats, on="filepath")
            .select(["speaker_id", *columns])
        )
    return file_stats.filter(pl.col("split") == "test_public").select(columns)


def _build_local_query_eval(
    file_stats: pl.DataFrame,
    local_manifest: pl.DataFrame,
    local_eval: pl.DataFrame,
) -> pl.DataFrame:
    pool_counts = local_manifest.group_by("speaker_id").len("speaker_n_utts_in_pool")
    top_scores = local_eval.get_column("top_scores").to_list()
    n_correct = [
        sum(speaker == row["speaker_id"] for speaker in row["top_speaker_ids"])
        for row in local_eval.iter_rows(named=True)
    ]
    score_features = pl.DataFrame(
        {
            "query_index": local_eval.get_column("query_index"),
            "top1_score": [float(values[0]) for values in top_scores],
            "top10_mean_score": [float(np.mean(values)) for values in top_scores],
            "top10_min_score": [float(values[-1]) for values in top_scores],
            "n_correct_top10": n_correct,
        }
    )
    return (
        local_eval.join(score_features, on="query_index")
        .join(pool_counts, on="speaker_id", how="left")
        .join(file_stats, on="filepath", how="left")
        .rename(
            {
                "query_index": "query_idx",
                "filepath": "query_path",
                "precision_at_10": "p10",
                "domain_cluster": "domain_bucket",
            }
        )
        .select(
            [
                "query_idx",
                "query_path",
                "speaker_id",
                "p10",
                "top1_correct",
                "first_correct_rank",
                "top1_score",
                "top10_mean_score",
                "top10_min_score",
                "n_correct_top10",
                "duration_s",
                "silence_ratio",
                "peak_limited_flag",
                "hard_clipped_flag",
                "narrowband_like",
                "duration_bucket",
                "leading_silence_bucket",
                "trailing_silence_bucket",
                "domain_bucket",
                "speaker_n_utts_in_pool",
            ]
        )
    )


def _build_bucket_alignment(file_stats: pl.DataFrame, local_manifest: pl.DataFrame) -> pl.DataFrame:
    train = file_stats.filter(pl.col("split") == "train")
    public = file_stats.filter(pl.col("split") == "test_public")
    local = local_manifest.select("filepath").join(file_stats, on="filepath", how="left")
    parts = []
    for split, frame in [("train", train), ("local_val", local), ("public", public)]:
        for name, column in [
            ("duration", "duration_bucket"),
            ("quality", "domain_cluster"),
            ("peak_limited_flag", "peak_limited_flag"),
            ("hard_clipped_flag", "hard_clipped_flag"),
            ("silence_heavy_flag", "silence_heavy_flag"),
            ("narrowband_like", "narrowband_like"),
            ("rms", "rms_bucket"),
            ("leading_silence", "leading_silence_bucket"),
            ("trailing_silence", "trailing_silence_bucket"),
        ]:
            parts.append(_bucket_counts(frame, split, name, column))
    return pl.concat(parts).sort(["bucket_name", "bucket_value", "split"])


def _bucket_counts(frame: pl.DataFrame, split: str, bucket_name: str, column: str) -> pl.DataFrame:
    return (
        frame.group_by(column)
        .len("count")
        .with_columns(
            pl.lit(split).alias("split"),
            pl.lit(bucket_name).alias("bucket_name"),
            pl.col(column).cast(pl.Utf8).alias("bucket_value"),
            (pl.col("count") / pl.col("count").sum()).alias("share"),
        )
        .select(["split", "bucket_name", "bucket_value", "count", "share"])
    )


def _write_public_score_summary(public_dir: Path, path: Path) -> None:
    rows = []
    for row in pl.read_csv(public_dir / "public_neighbor_score_summary.csv").iter_rows(named=True):
        for field in ["mean", "p10", "p50", "p90", "p99", "min", "max"]:
            rows.append({"metric": f"{row['metric']}_{field}", "value": row[field]})
    pl.DataFrame(rows).write_csv(path)


def _write_public_local_alignment(public_dir: Path, path: Path) -> None:
    frame = pl.read_csv(public_dir / "public_local_alignment.csv")
    frame.select(
        "metric",
        "local_value",
        "public_value",
        pl.col("public_div_local").alias("ratio"),
        pl.col("public_minus_local").alias("delta"),
        "note",
    ).write_csv(path)


def _write_worst_cases(local_query_eval: pl.DataFrame, audio_dir: Path, out_dir: Path) -> None:
    worst = local_query_eval.sort(["p10", "top1_correct", "first_correct_rank"]).head(200)
    worst.select(
        [
            "query_idx",
            "query_path",
            "speaker_id",
            "p10",
            "top1_correct",
            "first_correct_rank",
            "top1_score",
            "top10_mean_score",
            "duration_s",
            "silence_ratio",
            "peak_limited_flag",
            "hard_clipped_flag",
            "narrowband_like",
            "speaker_n_utts_in_pool",
        ]
    ).write_csv(out_dir / "10_worst_queries.csv")
    _write_worst_neighbors(audio_dir, worst, out_dir / "11_worst_query_neighbors.csv")


def _write_worst_neighbors(audio_dir: Path, worst: pl.DataFrame, path: Path) -> None:
    neighbours = pl.read_csv(audio_dir / "query_top10_neighbors.csv")
    manifest = pl.read_csv(audio_dir / "val_manifest.csv").with_row_index("neighbor_index")
    worst_ids = worst.select(pl.col("query_idx").alias("query_index"))
    (
        neighbours.join(worst_ids, on="query_index")
        .join(manifest.select(["neighbor_index", "filepath"]), on="neighbor_index", how="left")
        .rename(
            {
                "query_index": "query_idx",
                "filepath": "neighbor_path",
                "similarity": "score",
                "correct": "same_speaker",
            }
        )
        .select(["query_idx", "rank", "neighbor_index", "neighbor_path", "score", "same_speaker"])
        .rename({"neighbor_index": "neighbor_idx"})
        .write_csv(path)
    )


def _write_runtime_profile(audio_dir: Path, public_dir: Path, path: Path) -> None:
    local = _runtime_dict(audio_dir / "runtime_profile.csv")
    public_embed = _runtime_dict(public_dir / "public_embed_runtime.csv")
    public_search = _runtime_dict(public_dir / "public_search_runtime.csv")
    rows = [
        {
            "split": "local_val",
            "n_files": 5500,
            "provider": "CUDAExecutionProvider,CPUExecutionProvider",
            "audio_loading_s": local.get("audio_loading_s", 0.0),
            "embedding_extraction_s": local.get("embedding_extraction_s", 0.0),
            "search_s": local.get("retrieval_eval_s", 0.0),
            "csv_write_s": local.get("artifact_writing_s", 0.0),
            "total_s": local.get("total_wall", 0.0),
            "files_per_s": local.get("files_per_sec", 0.0),
            "batch_size": 512,
            "device": "cuda",
        },
        {
            "split": "public",
            "n_files": 134697,
            "provider": "CUDAExecutionProvider,CPUExecutionProvider",
            "audio_loading_s": public_embed.get("audio_loading_s", 0.0),
            "embedding_extraction_s": public_embed.get("embedding_extraction_s", 0.0),
            "search_s": public_search.get("search_wall_s", 0.0),
            "csv_write_s": None,
            "total_s": public_embed.get("total_embed_wall_s", 0.0)
            + public_search.get("search_wall_s", 0.0),
            "files_per_s": public_embed.get("files_per_sec", 0.0),
            "batch_size": 512,
            "device": "cuda",
        },
    ]
    pl.DataFrame(rows).write_csv(path)


def _runtime_dict(path: Path) -> dict[str, float]:
    return {row["stage"]: float(row["seconds"]) for row in pl.read_csv(path).iter_rows(named=True)}


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


if __name__ == "__main__":
    main()
