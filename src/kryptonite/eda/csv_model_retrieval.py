"""Model-probe and retrieval CSV export helpers."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any

import polars as pl


def inspect_onnx_metadata(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return [{"field": "status", "value": "baseline ONNX missing"}]
    rows = [
        {"field": "onnx_path", "value": str(path)},
        {"field": "onnx_size_bytes", "value": path.stat().st_size},
        {"field": "expected_sample_rate_hz_from_baseline_cli", "value": 16000},
        {"field": "expected_chunk_seconds_from_baseline_cli", "value": 6.0},
        {"field": "expected_input_channels", "value": "mono"},
    ]
    try:
        ort = importlib.import_module("onnxruntime")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for index, item in enumerate(session.get_inputs()):
            rows.append({"field": f"input_{index}_name", "value": item.name})
            rows.append({"field": f"input_{index}_shape", "value": json.dumps(item.shape)})
            rows.append({"field": f"input_{index}_type", "value": item.type})
        for index, item in enumerate(session.get_outputs()):
            rows.append({"field": f"output_{index}_name", "value": item.name})
            rows.append({"field": f"output_{index}_shape", "value": json.dumps(item.shape)})
            rows.append({"field": f"output_{index}_type", "value": item.type})
    except (ImportError, RuntimeError, ValueError) as exc:
        rows.append({"field": "onnxruntime_error", "value": str(exc)})
    return rows


def retrieval_breakdowns(
    embedding_eval: pl.DataFrame,
    file_stats: pl.DataFrame | None,
    domain_clusters: pl.DataFrame | None,
    out_root: Path,
) -> None:
    precision_col = next(
        (col for col in embedding_eval.columns if col.startswith("precision_at_")),
        None,
    )
    if precision_col is None:
        _write_status_csv(
            out_root / "retrieval_breakdown_by_bucket.csv", "precision column missing"
        )
        return

    joined = embedding_eval
    if file_stats is not None:
        keep = [
            col
            for col in ["filepath", "duration_s", "silence_ratio_40db", "narrowband_proxy"]
            if col in file_stats.columns
        ]
        if keep:
            joined = joined.join(file_stats.select(keep), on="filepath", how="left")
    if domain_clusters is not None and "domain_cluster" in domain_clusters.columns:
        joined = joined.join(
            domain_clusters.select(["filepath", "domain_cluster"]),
            on="filepath",
            how="left",
        )

    with_buckets = joined.with_columns(
        _bucket_duration_expr(),
        _bucket_silence_expr(),
        _bucket_narrowband_expr(),
    )
    rows = []
    for bucket_col in ("duration_bucket", "silence_bucket", "narrowband_bucket", "domain_cluster"):
        if bucket_col not in with_buckets.columns:
            continue
        summary = (
            with_buckets.group_by(bucket_col)
            .agg(
                pl.len().alias("query_count"),
                pl.col(precision_col).mean().alias("mean_precision"),
            )
            .rename({bucket_col: "bucket_value"})
            .with_columns(pl.lit(bucket_col).alias("bucket_name"))
        )
        rows.append(summary)
    if rows:
        pl.concat(rows, how="vertical").select(
            ["bucket_name", "bucket_value", "query_count", "mean_precision"]
        ).write_csv(out_root / "retrieval_breakdown_by_bucket.csv")
    else:
        _write_status_csv(out_root / "retrieval_breakdown_by_bucket.csv", "no bucket data")


def write_retrieval_empty_breakdowns(out_root: Path) -> None:
    pl.DataFrame(
        {
            "bucket_name": [],
            "bucket_value": [],
            "query_count": [],
            "mean_precision": [],
            "status": [],
        },
        schema={
            "bucket_name": pl.Utf8,
            "bucket_value": pl.Utf8,
            "query_count": pl.Int64,
            "mean_precision": pl.Float64,
            "status": pl.Utf8,
        },
    ).write_csv(out_root / "retrieval_breakdown_by_bucket.csv")


def write_dataframe_csv_flat(dataframe: pl.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataframe.columns)
        writer.writeheader()
        for row in dataframe.to_dicts():
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    )
                    for key, value in row.items()
                }
            )


def _bucket_duration_expr() -> pl.Expr:
    return (
        pl.when(pl.col("duration_s").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("duration_s") < 1)
        .then(pl.lit("0-1s"))
        .when(pl.col("duration_s") < 2)
        .then(pl.lit("1-2s"))
        .when(pl.col("duration_s") < 4)
        .then(pl.lit("2-4s"))
        .when(pl.col("duration_s") < 6)
        .then(pl.lit("4-6s"))
        .when(pl.col("duration_s") < 10)
        .then(pl.lit("6-10s"))
        .when(pl.col("duration_s") < 20)
        .then(pl.lit("10-20s"))
        .when(pl.col("duration_s") < 40)
        .then(pl.lit("20-40s"))
        .otherwise(pl.lit("40s+"))
        .alias("duration_bucket")
    )


def _bucket_silence_expr() -> pl.Expr:
    return (
        pl.when(pl.col("silence_ratio_40db").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("silence_ratio_40db") < 0.1)
        .then(pl.lit("0-10%"))
        .when(pl.col("silence_ratio_40db") < 0.3)
        .then(pl.lit("10-30%"))
        .when(pl.col("silence_ratio_40db") < 0.5)
        .then(pl.lit("30-50%"))
        .otherwise(pl.lit("50%+"))
        .alias("silence_bucket")
    )


def _bucket_narrowband_expr() -> pl.Expr:
    return (
        pl.when(pl.col("narrowband_proxy").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("narrowband_proxy") < 0.25)
        .then(pl.lit("low"))
        .when(pl.col("narrowband_proxy") < 0.5)
        .then(pl.lit("medium"))
        .when(pl.col("narrowband_proxy") < 0.75)
        .then(pl.lit("high"))
        .otherwise(pl.lit("very_high"))
        .alias("narrowband_bucket")
    )


def _write_status_csv(path: Path, message: str) -> None:
    pl.DataFrame({"status": ["missing_source"], "message": [message]}).write_csv(path)
