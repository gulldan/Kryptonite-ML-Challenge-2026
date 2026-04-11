"""Speaker-level EDA summaries and validation split simulation."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import polars as pl


def build_speaker_stats(
    file_stats: pl.DataFrame, *, short_threshold_s: float = 2.0
) -> pl.DataFrame:
    """Aggregate per-file stats into speaker-level diagnostics."""

    train_rows = file_stats.filter(pl.col("speaker_id").is_not_null())
    if train_rows.is_empty():
        return pl.DataFrame()
    frame = _with_metric_columns(train_rows)
    return (
        frame.with_columns(
            pl.when(pl.col("duration_s").is_not_null())
            .then(pl.col("duration_s") < short_threshold_s)
            .otherwise(None)
            .alias("_is_short"),
            (
                (pl.col("silence_ratio_40db") >= 0.5)
                | (pl.col("clipping_frac") >= 0.01)
                | (pl.col("rms_dbfs") <= -40.0)
            ).alias("_is_bad_audio"),
        )
        .group_by("speaker_id")
        .agg(
            pl.len().alias("n_utts"),
            pl.when(pl.col("duration_s").is_not_null().any())
            .then(pl.col("duration_s").sum())
            .otherwise(None)
            .alias("total_duration_s"),
            pl.col("duration_s").mean().alias("mean_duration_s"),
            pl.col("duration_s").std().alias("std_duration_s"),
            pl.col("duration_s").min().alias("min_duration_s"),
            pl.col("duration_s").max().alias("max_duration_s"),
            pl.col("_is_short").mean().alias("short_utt_frac"),
            pl.col("_is_bad_audio").mean().alias("bad_audio_frac"),
            pl.col("sample_rate_hz").mode().first().alias("mode_sample_rate_hz"),
            pl.col("num_channels").mode().first().alias("mode_num_channels"),
        )
        .sort(["n_utts", "total_duration_s"], descending=[True, True])
    )


def build_dataset_summary(file_stats: pl.DataFrame, speaker_stats: pl.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable dataset summary for reports and dashboards."""

    summary: dict[str, Any] = {
        "file_count": int(file_stats.height),
        "speaker_count": int(speaker_stats.height),
        "splits": {},
        "duration_quantiles_s": _quantiles(file_stats, "duration_s"),
        "sample_rate_counts": _value_counts(file_stats, "sample_rate_hz"),
        "channel_counts": _value_counts(file_stats, "num_channels"),
        "read_error_count": _read_error_count(file_stats),
    }
    duration_sum = _sum_or_none(file_stats, "duration_s")
    summary["total_duration_h"] = (
        round(duration_sum / 3600.0, 6) if duration_sum is not None else None
    )
    if not speaker_stats.is_empty():
        summary["files_per_speaker"] = _series_summary(speaker_stats, "n_utts")
        summary["duration_per_speaker_s"] = _series_summary(speaker_stats, "total_duration_s")
        summary["speakers_with_at_least_11_utts"] = int(
            speaker_stats.filter(pl.col("n_utts") >= 11).height
        )
    for split in file_stats.get_column("split").unique().to_list():
        split_frame = file_stats.filter(pl.col("split") == split)
        split_duration = _sum_or_none(split_frame, "duration_s")
        summary["splits"][str(split)] = {
            "file_count": int(split_frame.height),
            "total_duration_h": (
                round(split_duration / 3600.0, 6) if split_duration is not None else None
            ),
            "duration_quantiles_s": _quantiles(split_frame, "duration_s"),
            "sample_rate_counts": _value_counts(split_frame, "sample_rate_hz"),
            "channel_counts": _value_counts(split_frame, "num_channels"),
        }
    return summary


def simulate_speaker_disjoint_split(
    speaker_stats: pl.DataFrame,
    *,
    val_fraction: float = 0.1,
    min_val_utts: int = 11,
    seed: int = 2026,
) -> dict[str, Any]:
    """Simulate a speaker-disjoint validation split eligible for local P@10."""

    if speaker_stats.is_empty():
        return {
            "train_speaker_count": 0,
            "val_speaker_count": 0,
            "eligible_val_speaker_count": 0,
            "note": "No speaker statistics are available.",
        }
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be within (0.0, 1.0).")
    if min_val_utts <= 0:
        raise ValueError("min_val_utts must be positive.")

    eligible = speaker_stats.filter(pl.col("n_utts") >= min_val_utts)
    rng = np.random.default_rng(seed)
    eligible_speakers = np.array(eligible.get_column("speaker_id").to_list(), dtype=object)
    val_count = min(eligible_speakers.size, max(1, int(round(speaker_stats.height * val_fraction))))
    if eligible_speakers.size == 0:
        selected = np.array([], dtype=object)
    else:
        selected = rng.choice(eligible_speakers, size=val_count, replace=False)
    val_set = set(str(value) for value in selected.tolist())
    val_stats = speaker_stats.filter(pl.col("speaker_id").is_in(val_set))
    train_stats = speaker_stats.filter(~pl.col("speaker_id").is_in(val_set))
    train_duration_s = _sum_or_none(train_stats, "total_duration_s")
    val_duration_s = _sum_or_none(val_stats, "total_duration_s")
    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "min_val_utts": min_val_utts,
        "speaker_count": int(speaker_stats.height),
        "eligible_val_speaker_count": int(eligible.height),
        "train_speaker_count": int(train_stats.height),
        "val_speaker_count": int(val_stats.height),
        "train_file_count": int(_sum_or_zero(train_stats, "n_utts")),
        "val_file_count": int(_sum_or_zero(val_stats, "n_utts")),
        "train_duration_h": (
            round(train_duration_s / 3600.0, 6) if train_duration_s is not None else None
        ),
        "val_duration_h": round(val_duration_s / 3600.0, 6) if val_duration_s is not None else None,
        "val_speakers": sorted(val_set),
        "train_mean_duration_s": _mean_or_none(train_stats, "mean_duration_s"),
        "val_mean_duration_s": _mean_or_none(val_stats, "mean_duration_s"),
    }


def _with_metric_columns(frame: pl.DataFrame) -> pl.DataFrame:
    defaults: dict[str, float | int | None] = {
        "duration_s": None,
        "silence_ratio_40db": None,
        "clipping_frac": None,
        "rms_dbfs": None,
        "sample_rate_hz": None,
        "num_channels": None,
    }
    expressions = [
        pl.lit(default).alias(name)
        for name, default in defaults.items()
        if name not in frame.columns
    ]
    return frame.with_columns(expressions) if expressions else frame


def _quantiles(frame: pl.DataFrame, column: str) -> dict[str, float | None]:
    if column not in frame.columns:
        return {key: None for key in ("p10", "p50", "p90", "p99")}
    series = frame.get_column(column).drop_nulls()
    if series.is_empty():
        return {key: None for key in ("p10", "p50", "p90", "p99")}
    return {
        "p10": _round_real(series.quantile(0.10)),
        "p50": _round_real(series.quantile(0.50)),
        "p90": _round_real(series.quantile(0.90)),
        "p99": _round_real(series.quantile(0.99)),
    }


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame.get_column(column).drop_nulls().value_counts().rows(named=True)
    return {str(row[column]): int(row["count"]) for row in counts}


def _series_summary(frame: pl.DataFrame, column: str) -> dict[str, float | None]:
    if column not in frame.columns:
        return {"min": None, "p10": None, "p50": None, "p90": None, "p99": None, "max": None}
    series = frame.get_column(column).drop_nulls()
    if series.is_empty():
        return {"min": None, "p10": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "min": _round_real(series.min()),
        "p10": _round_real(series.quantile(0.10)),
        "p50": _round_real(series.quantile(0.50)),
        "p90": _round_real(series.quantile(0.90)),
        "p99": _round_real(series.quantile(0.99)),
        "max": _round_real(series.max()),
    }


def _read_error_count(frame: pl.DataFrame) -> int:
    if "error" not in frame.columns:
        return 0
    return int(frame.filter(pl.col("error").is_not_null()).height)


def _sum_or_none(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    series = frame.get_column(column).drop_nulls()
    return None if series.is_empty() else float(series.sum())


def _sum_or_zero(frame: pl.DataFrame, column: str) -> float:
    value = _sum_or_none(frame, column)
    return 0.0 if value is None else value


def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.is_empty():
        return None
    value = frame.get_column(column).drop_nulls().mean()
    return _round_real(value)


def _round_real(value: object) -> float | None:
    if not isinstance(value, Real):
        return None
    return round(float(value), 6)
