"""Rule-based audio-condition buckets for first-pass EDA."""

from __future__ import annotations

import polars as pl


def assign_domain_buckets(file_stats: pl.DataFrame) -> pl.DataFrame:
    """Assign fast diagnostic buckets from low-level audio statistics."""

    required = {"filepath", "split", "speaker_id"}
    missing = sorted(required.difference(file_stats.columns))
    if missing:
        raise ValueError(f"file_stats is missing required columns: {missing}")

    frame = _with_default_metric_columns(file_stats)
    labels = (
        pl.when(pl.col("error").is_not_null())
        .then(pl.lit("read_error"))
        .when(pl.col("duration_s").is_null() & pl.col("rms_dbfs").is_null())
        .then(pl.lit("not_profiled"))
        .when(pl.col("duration_s") < 2.0)
        .then(pl.lit("short"))
        .when((pl.col("clipping_frac") >= 0.01) | (pl.col("peak_dbfs") >= -0.1))
        .then(pl.lit("clipped"))
        .when(pl.col("silence_ratio_40db") >= 0.5)
        .then(pl.lit("silence_heavy"))
        .when(pl.col("rms_dbfs") <= -40.0)
        .then(pl.lit("low_volume"))
        .when((pl.col("narrowband_proxy") >= 0.5) | (pl.col("rolloff95_hz") <= 3800.0))
        .then(pl.lit("narrowband_like"))
        .when((pl.col("spectral_flatness") >= 0.35) & (pl.col("rms_dbfs") >= -35.0))
        .then(pl.lit("noise_like"))
        .otherwise(pl.lit("cleanish"))
    )
    return frame.with_columns(labels.alias("domain_cluster")).select(
        "split",
        "speaker_id",
        "filepath",
        "resolved_path",
        "domain_cluster",
        "duration_s",
        "rms_dbfs",
        "peak_dbfs",
        "silence_ratio_40db",
        "clipping_frac",
        "rolloff95_hz",
        "spectral_flatness",
        "narrowband_proxy",
    )


def _with_default_metric_columns(frame: pl.DataFrame) -> pl.DataFrame:
    defaults = {
        "resolved_path": None,
        "error": None,
        "duration_s": None,
        "clipping_frac": None,
        "peak_dbfs": None,
        "silence_ratio_40db": None,
        "rms_dbfs": None,
        "narrowband_proxy": None,
        "rolloff95_hz": None,
        "spectral_flatness": None,
    }
    expressions = []
    for name, default in defaults.items():
        if name not in frame.columns:
            expressions.append(pl.lit(default).alias(name))
    return frame.with_columns(expressions) if expressions else frame
