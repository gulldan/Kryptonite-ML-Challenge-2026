"""CSV-first EDA export pack generation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any

import polars as pl

from kryptonite.eda.csv_constants import NUMERIC_AUDIO_METRICS
from kryptonite.eda.csv_inventory import build_inventory
from kryptonite.eda.csv_model_retrieval import (
    inspect_onnx_metadata,
    retrieval_breakdowns,
    write_dataframe_csv_flat,
    write_retrieval_empty_breakdowns,
)


def export_eda_csv_pack(
    *,
    artifact_dir: Path | str,
    output_dir: Path | str,
    dataset_root: Path | str,
    baseline_onnx: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Export a CSV pack from offline EDA artifacts."""

    artifact_root = Path(artifact_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(baseline_onnx) if baseline_onnx is not None else None

    file_stats = _read_parquet_or_none(artifact_root / "file_stats.parquet")
    speaker_stats = _read_parquet_or_none(artifact_root / "speaker_stats.parquet")
    domain_clusters = _read_csv_or_none(artifact_root / "domain_clusters.csv")
    dataset_summary = _read_json_or_none(artifact_root / "dataset_summary.json")
    val_split_stats = _read_json_or_none(artifact_root / "val_split_stats.json")

    written: list[dict[str, Any]] = []
    written.extend(_write_manifest_tables(out_root, file_stats, speaker_stats))
    written.extend(_write_domain_tables(out_root, domain_clusters))
    written.extend(_write_summary_tables(out_root, dataset_summary, val_split_stats))
    written.extend(_write_audio_analysis_tables(out_root, file_stats, domain_clusters))
    written.extend(_write_validation_tables(out_root, val_split_stats))
    written.extend(_write_model_probe_tables(out_root, baseline_path))
    written.extend(_write_retrieval_tables(artifact_root, out_root, file_stats, domain_clusters))
    written.extend(_write_submission_runtime_experiment_tables(artifact_root, out_root))

    inventory = build_inventory(
        output_dir=out_root,
        file_stats=file_stats,
        speaker_stats=speaker_stats,
        domain_clusters=domain_clusters,
        baseline_onnx=baseline_path,
        artifact_root=artifact_root,
    )
    inventory.write_csv(out_root / "eda_data_inventory.csv")
    written.append(
        _artifact_row(out_root / "eda_data_inventory.csv", "EDA requested data inventory")
    )
    _write_artifact_index(out_root / "csv_pack_files.csv", written)
    return written


def _write_manifest_tables(
    out_root: Path,
    file_stats: pl.DataFrame | None,
    speaker_stats: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    if file_stats is not None:
        path = out_root / "file_stats.csv"
        file_stats.write_csv(path)
        written.append(_artifact_row(path, "per-file manifest/audio statistics"))
        if "duration_s" in file_stats.columns:
            _write_top_files(out_root, file_stats, ascending=True)
            _write_top_files(out_root, file_stats, ascending=False)
            written.append(_artifact_row(out_root / "top_short_files.csv", "shortest files"))
            written.append(_artifact_row(out_root / "top_long_files.csv", "longest files"))
        duration_hist = _duration_histogram(file_stats)
        if duration_hist is not None:
            duration_hist.write_csv(out_root / "duration_histogram.csv")
            written.append(_artifact_row(out_root / "duration_histogram.csv", "duration histogram"))
    if speaker_stats is not None:
        path = out_root / "speaker_stats.csv"
        speaker_stats.write_csv(path)
        written.append(_artifact_row(path, "per-speaker statistics"))
        speaker_stats.sort(["n_utts", "speaker_id"], descending=[True, False]).head(200).write_csv(
            out_root / "top_speakers_by_file_count.csv"
        )
        speaker_stats.group_by("n_utts").len().sort("n_utts").write_csv(
            out_root / "speakers_by_file_count.csv"
        )
        _speaker_cumulative(speaker_stats).write_csv(out_root / "speaker_cumulative_coverage.csv")
        written.append(_artifact_row(out_root / "top_speakers_by_file_count.csv", "top speakers"))
        written.append(
            _artifact_row(out_root / "speakers_by_file_count.csv", "speaker count histogram")
        )
        written.append(
            _artifact_row(out_root / "speaker_cumulative_coverage.csv", "cumulative coverage")
        )
    return written


def _write_domain_tables(
    out_root: Path,
    domain_clusters: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    if domain_clusters is None:
        _write_status_csv(out_root / "domain_status.csv", "domain_clusters.csv missing")
        return [_artifact_row(out_root / "domain_status.csv", "domain clustering status")]
    domain_clusters.write_csv(out_root / "domain_clusters.csv")
    summary_aggs = [pl.len().alias("file_count")]
    for metric in NUMERIC_AUDIO_METRICS:
        if metric in domain_clusters.columns:
            summary_aggs.append(pl.col(metric).mean().alias(f"mean_{metric}"))
    domain_clusters.group_by(["split", "domain_cluster"]).agg(summary_aggs).sort(
        ["split", "file_count"], descending=[False, True]
    ).write_csv(out_root / "domain_cluster_summary.csv")
    domain_clusters.group_by(["split", "domain_cluster"], maintain_order=True).head(10).write_csv(
        out_root / "domain_cluster_examples.csv"
    )
    return [
        _artifact_row(out_root / "domain_clusters.csv", "per-file domain bucket labels"),
        _artifact_row(out_root / "domain_cluster_summary.csv", "domain bucket summary"),
        _artifact_row(out_root / "domain_cluster_examples.csv", "domain bucket examples"),
    ]


def _write_summary_tables(
    out_root: Path,
    dataset_summary: dict[str, Any] | None,
    val_split_stats: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    if dataset_summary is not None:
        _flatten_json_rows(dataset_summary).write_csv(out_root / "dataset_summary.csv")
        written.append(_artifact_row(out_root / "dataset_summary.csv", "flattened dataset summary"))
    if val_split_stats is not None:
        rows = [
            {"key": key, "value": value}
            for key, value in val_split_stats.items()
            if key != "val_speakers"
        ]
        pl.DataFrame(rows).write_csv(out_root / "val_split_stats.csv")
        pl.DataFrame({"speaker_id": val_split_stats.get("val_speakers", [])}).write_csv(
            out_root / "val_speakers.csv"
        )
        written.append(_artifact_row(out_root / "val_split_stats.csv", "validation split summary"))
        written.append(_artifact_row(out_root / "val_speakers.csv", "validation speakers"))
    return written


def _write_audio_analysis_tables(
    out_root: Path,
    file_stats: pl.DataFrame | None,
    domain_clusters: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    if file_stats is None:
        return []
    written: list[dict[str, Any]] = []
    _numeric_summary_by_split(file_stats, NUMERIC_AUDIO_METRICS).write_csv(
        out_root / "audio_quality_summary_by_split.csv"
    )
    _audio_outliers(file_stats).write_csv(out_root / "audio_outliers.csv")
    _qualitative_audit_queue(file_stats, domain_clusters).write_csv(
        out_root / "qualitative_audit_queue.csv"
    )
    written.append(
        _artifact_row(out_root / "audio_quality_summary_by_split.csv", "audio quality summary")
    )
    written.append(_artifact_row(out_root / "audio_outliers.csv", "audio quality outliers"))
    written.append(_artifact_row(out_root / "qualitative_audit_queue.csv", "manual audit queue"))
    return written


def _write_validation_tables(
    out_root: Path,
    val_split_stats: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = [
        {
            "check_name": "speaker_disjoint",
            "status": "planned",
            "value": "true",
            "notes": "Use val_speakers.csv for a speaker-disjoint local validation pool.",
        },
        {
            "check_name": "val_speaker_min_11_utts",
            "status": "ready" if val_split_stats is not None else "missing_source",
            "value": None
            if val_split_stats is None
            else val_split_stats.get("eligible_val_speaker_count"),
            "notes": "P@10 needs speakers with at least 11 utterances.",
        },
    ]
    pl.DataFrame(rows).write_csv(out_root / "validation_design_checks.csv")
    return [_artifact_row(out_root / "validation_design_checks.csv", "validation design checks")]


def _write_model_probe_tables(out_root: Path, baseline_onnx: Path | None) -> list[dict[str, Any]]:
    metadata = inspect_onnx_metadata(baseline_onnx)
    pl.DataFrame(metadata).write_csv(out_root / "model_probe_metadata.csv")
    probe_rows = [
        {
            "probe_config": "raw_6s_l2",
            "crop_seconds": 6,
            "trim_silence": False,
            "gain_db": 0,
            "n_crops": 1,
            "embedding_norm": "l2",
            "status": "pending_embeddings",
        },
        {
            "probe_config": "trim_6s_l2",
            "crop_seconds": 6,
            "trim_silence": True,
            "gain_db": 0,
            "n_crops": 1,
            "embedding_norm": "l2",
            "status": "pending_embeddings",
        },
        {
            "probe_config": "multicrop_6s_l2",
            "crop_seconds": 6,
            "trim_silence": True,
            "gain_db": 0,
            "n_crops": 5,
            "embedding_norm": "l2",
            "status": "pending_embeddings",
        },
    ]
    pl.DataFrame(probe_rows).write_csv(out_root / "model_probe_experiment_grid.csv")
    return [
        _artifact_row(out_root / "model_probe_metadata.csv", "ONNX model metadata"),
        _artifact_row(out_root / "model_probe_experiment_grid.csv", "model probe grid"),
    ]


def _write_retrieval_tables(
    artifact_root: Path,
    out_root: Path,
    file_stats: pl.DataFrame | None,
    domain_clusters: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    embedding_eval = _read_parquet_or_none(artifact_root / "embedding_eval.parquet")
    written: list[dict[str, Any]] = []
    if embedding_eval is None:
        _write_status_csv(
            out_root / "retrieval_status.csv",
            "embedding_eval.parquet missing; run scripts/run_eda_retrieval_eval.py",
        )
        written.append(_artifact_row(out_root / "retrieval_status.csv", "retrieval status"))
        write_retrieval_empty_breakdowns(out_root)
        return written
    write_dataframe_csv_flat(embedding_eval, out_root / "embedding_eval.csv")
    written.append(_artifact_row(out_root / "embedding_eval.csv", "per-query retrieval metrics"))
    _write_optional_copy(artifact_root / "worst_queries.csv", out_root / "worst_queries.csv")
    _write_optional_copy(
        artifact_root / "confused_speaker_pairs.csv",
        out_root / "confused_speaker_pairs.csv",
    )
    written.append(_artifact_row(out_root / "worst_queries.csv", "worst retrieval queries"))
    written.append(_artifact_row(out_root / "confused_speaker_pairs.csv", "confused speaker pairs"))
    retrieval_breakdowns(embedding_eval, file_stats, domain_clusters, out_root)
    written.append(_artifact_row(out_root / "retrieval_breakdown_by_bucket.csv", "P@K buckets"))
    return written


def _write_submission_runtime_experiment_tables(
    artifact_root: Path,
    out_root: Path,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    runtime_profile = artifact_root / "runtime_profile.csv"
    if runtime_profile.is_file():
        shutil.copyfile(runtime_profile, out_root / "runtime_profile.csv")
    else:
        _write_status_csv(out_root / "runtime_profile.csv", "runtime profile missing")
    written.append(_artifact_row(out_root / "runtime_profile.csv", "runtime profile"))

    validation_report = _read_json_or_none(artifact_root / "submission_validation_report.json")
    if validation_report is not None:
        _flatten_json_rows(validation_report).write_csv(
            out_root / "submission_validation_report.csv"
        )
    else:
        _write_status_csv(
            out_root / "submission_validation_report.csv",
            "submission validation report missing",
        )
    written.append(
        _artifact_row(out_root / "submission_validation_report.csv", "submission validation")
    )

    experiment_log = artifact_root / "experiment_log.csv"
    if experiment_log.is_file():
        shutil.copyfile(experiment_log, out_root / "experiment_log.csv")
    else:
        _write_status_csv(out_root / "experiment_log.csv", "experiment log missing")
    _observation_to_decision_template().write_csv(out_root / "observation_to_decision.csv")
    written.append(_artifact_row(out_root / "experiment_log.csv", "experiment tracker"))
    written.append(
        _artifact_row(out_root / "observation_to_decision.csv", "observation decision log")
    )
    return written


def _duration_histogram(file_stats: pl.DataFrame) -> pl.DataFrame | None:
    if "duration_s" not in file_stats.columns or _non_null_count(file_stats, "duration_s") == 0:
        return None
    return _bucket_counts(file_stats, _bucket_duration_expr(), "duration_bucket")


def _write_top_files(out_root: Path, file_stats: pl.DataFrame, *, ascending: bool) -> None:
    if "duration_s" not in file_stats.columns:
        return
    name = "top_short_files.csv" if ascending else "top_long_files.csv"
    file_stats.drop_nulls("duration_s").sort("duration_s", descending=not ascending).head(
        200
    ).write_csv(out_root / name)


def _speaker_cumulative(speaker_stats: pl.DataFrame) -> pl.DataFrame:
    metric = (
        "total_duration_s" if _non_null_count(speaker_stats, "total_duration_s") > 0 else "n_utts"
    )
    sorted_frame = speaker_stats.sort(metric, descending=True).with_row_index("rank", offset=1)
    total = sorted_frame.get_column(metric).drop_nulls().sum()
    if total is None or float(total) <= 0.0:
        return sorted_frame.select(["rank", "speaker_id", "n_utts"]).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("cumulative_fraction")
        )
    columns = ["rank", "speaker_id", "n_utts", "cumulative_fraction"]
    if metric != "n_utts":
        columns.insert(3, metric)
    return sorted_frame.with_columns(
        (pl.col(metric).cum_sum() / float(total)).alias("cumulative_fraction")
    ).select(columns)


def _numeric_summary_by_split(frame: pl.DataFrame, metrics: Iterable[str]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in frame.get_column("split").unique().sort().to_list():
        split_frame = frame.filter(pl.col("split") == split)
        for metric in metrics:
            if metric not in split_frame.columns:
                continue
            rows.append({"split": split, "metric": metric, **_series_stats(split_frame, metric)})
    return pl.DataFrame(rows)


def _audio_outliers(file_stats: pl.DataFrame) -> pl.DataFrame:
    selectors: list[pl.DataFrame] = []
    rules = [
        ("short_duration", "duration_s", False),
        ("long_duration", "duration_s", True),
        ("high_silence", "silence_ratio_40db", True),
        ("high_clipping", "clipping_frac", True),
        ("low_volume", "rms_dbfs", False),
        ("narrowband_proxy", "narrowband_proxy", True),
    ]
    for rule_name, column, descending in rules:
        if column not in file_stats.columns or _non_null_count(file_stats, column) == 0:
            continue
        selectors.append(
            file_stats.drop_nulls(column)
            .sort(column, descending=descending)
            .head(50)
            .with_columns(pl.lit(rule_name).alias("outlier_rule"))
        )
    if not selectors:
        return pl.DataFrame({"status": ["no audio metrics available"]})
    return pl.concat(selectors, how="diagonal").unique(["filepath", "outlier_rule"])


def _qualitative_audit_queue(
    file_stats: pl.DataFrame,
    domain_clusters: pl.DataFrame | None,
) -> pl.DataFrame:
    base = file_stats
    if domain_clusters is not None and "domain_cluster" in domain_clusters.columns:
        base = base.join(
            domain_clusters.select(["filepath", "domain_cluster"]),
            on="filepath",
            how="left",
        )
    examples = []
    if "domain_cluster" in base.columns:
        examples.append(
            base.group_by(["split", "domain_cluster"], maintain_order=True)
            .head(5)
            .with_columns(pl.lit("domain_stratified").alias("audit_reason"))
        )
    outliers = _audio_outliers(base)
    if "filepath" in outliers.columns:
        examples.append(outliers.with_columns(pl.lit("outlier").alias("audit_reason")))
    if not examples:
        examples.append(
            base.head(100).with_columns(pl.lit("manifest_sample").alias("audit_reason"))
        )
    return (
        pl.concat(examples, how="diagonal")
        .unique("filepath")
        .head(300)
        .with_columns(
            pl.lit("").alias("manual_tag"),
            pl.lit("").alias("manual_comment"),
            pl.lit("").alias("reviewer"),
        )
    )


def _observation_to_decision_template() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "observation": "Fill after reviewing duration/audio/domain/retrieval CSVs.",
                "evidence_csv": "dataset_summary.csv",
                "decision": "",
                "result": "",
                "status": "todo",
            },
            {
                "observation": "Short utterance prevalence drives crop/filter policy.",
                "evidence_csv": "duration_histogram.csv,top_short_files.csv",
                "decision": "",
                "result": "",
                "status": "todo",
            },
            {
                "observation": (
                    "Narrowband/silence/clipping buckets drive preprocessing and augmentation."
                ),
                "evidence_csv": "domain_cluster_summary.csv,audio_outliers.csv",
                "decision": "",
                "result": "",
                "status": "todo",
            },
            {
                "observation": "Local pseudo-val is optimistic relative to public LB.",
                "evidence_csv": "embedding_eval.csv,runtime_profile.csv,experiment_log.csv",
                "decision": "Use public LB as external baseline; use local P@10 for ablations.",
                "result": "Fill with public LB metric after leaderboard submission is known.",
                "status": "ready",
            },
        ]
    )


def _bucket_counts(frame: pl.DataFrame, expr: pl.Expr, column: str) -> pl.DataFrame:
    return frame.with_columns(expr).group_by(["split", column]).len().sort(["split", column])


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


def _series_stats(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    series = frame.get_column(column).drop_nulls()
    if series.is_empty():
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(series.len()),
        "mean": _float_or_none(series.mean()),
        "min": _float_or_none(series.min()),
        "p10": _float_or_none(series.quantile(0.10)),
        "p50": _float_or_none(series.quantile(0.50)),
        "p90": _float_or_none(series.quantile(0.90)),
        "p99": _float_or_none(series.quantile(0.99)),
        "max": _float_or_none(series.max()),
    }


def _flatten_json_rows(payload: dict[str, Any]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), nested)
        elif isinstance(value, list):
            rows.append({"key": prefix, "value": json.dumps(value, ensure_ascii=False)})
        else:
            rows.append({"key": prefix, "value": value})

    visit("", payload)
    return pl.DataFrame(rows)


def _write_status_csv(path: Path, message: str) -> None:
    pl.DataFrame({"status": ["missing_source"], "message": [message]}).write_csv(path)


def _write_artifact_index(path: Path, rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _write_optional_copy(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copyfile(source, destination)
    else:
        _write_status_csv(destination, f"{source.name} missing")


def _read_parquet_or_none(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.is_file() else None


def _read_csv_or_none(path: Path) -> pl.DataFrame | None:
    return pl.read_csv(path) if path.is_file() else None


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _artifact_row(path: Path, description: str) -> dict[str, Any]:
    return {
        "csv_file": path.name,
        "path": str(path),
        "description": description,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _non_null_count(frame: pl.DataFrame | None, column: str) -> int:
    if frame is None or column not in frame.columns:
        return 0
    return int(frame.get_column(column).drop_nulls().len())


def _float_or_none(value: object) -> float | None:
    return round(float(value), 8) if isinstance(value, Real) else None
