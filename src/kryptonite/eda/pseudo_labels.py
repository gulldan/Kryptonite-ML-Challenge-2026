"""Pseudo-label selection and manifest materialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

DEFAULT_PRIOR_FEATURE_WEIGHTS: dict[str, float] = {
    "duration_s": 1.0,
    "non_silent_ratio": 0.8,
    "leading_silence_s": 0.55,
    "trailing_silence_s": 0.45,
    "spectral_bandwidth_hz": 0.7,
    "band_energy_ratio_3_8k": 0.5,
}

_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionConfig:
    experiment_id: str
    dataset_name: str = "participants_g6_pseudo"
    label_prefix: str = "pseudo_g6_"
    public_audio_prefix: str = "datasets/Для участников"
    min_cluster_size: int = 8
    max_cluster_size: int = 80
    min_top1_score: float | None = None
    min_top1_margin: float | None = None
    max_indegree_quantile: float | None = None
    indegree_top_k: int = 10
    max_prior_distance_quantile: float | None = None
    diversity_floor_quantile: float | None = None
    max_rows_per_cluster: int | None = None
    prior_feature_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PRIOR_FEATURE_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty.")
        if self.min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be positive.")
        if self.max_cluster_size < self.min_cluster_size:
            raise ValueError("max_cluster_size must be >= min_cluster_size.")
        if self.min_top1_margin is not None and self.min_top1_margin < 0.0:
            raise ValueError("min_top1_margin must be non-negative.")
        if self.indegree_top_k <= 0:
            raise ValueError("indegree_top_k must be positive.")
        if self.max_rows_per_cluster is not None and self.max_rows_per_cluster <= 0:
            raise ValueError("max_rows_per_cluster must be positive when provided.")
        _validate_optional_quantile(
            self.max_indegree_quantile,
            name="max_indegree_quantile",
            allow_one=True,
        )
        _validate_optional_quantile(
            self.max_prior_distance_quantile,
            name="max_prior_distance_quantile",
            allow_one=True,
        )
        _validate_optional_quantile(
            self.diversity_floor_quantile,
            name="diversity_floor_quantile",
            allow_one=False,
        )
        if self.diversity_floor_quantile is not None and self.max_prior_distance_quantile is None:
            raise ValueError(
                "diversity_floor_quantile requires max_prior_distance_quantile to be set."
            )
        if not self.prior_feature_weights:
            raise ValueError("prior_feature_weights must not be empty.")
        if any(weight <= 0.0 for weight in self.prior_feature_weights.values()):
            raise ValueError("prior feature weights must be positive.")


def build_pseudo_label_manifests(
    *,
    clusters_csv: Path | str,
    public_manifest_csv: Path | str,
    output_dir: Path | str,
    selection: PseudoLabelSelectionConfig,
    base_train_manifest: Path | str | None = None,
    topk_scores_npy: Path | str | None = None,
    topk_indices_npy: Path | str | None = None,
    public_stats_path: Path | str | None = None,
    prior_reference_stats_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write pseudo and mixed manifests plus an audit summary."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    clusters = _load_frame(Path(clusters_csv))
    public_manifest = _ensure_public_manifest(_load_frame(Path(public_manifest_csv)))
    topk_scores = _load_array(topk_scores_npy, kind="scores")
    topk_indices = _load_array(topk_indices_npy, kind="indices")
    public_stats = _load_frame(Path(public_stats_path)) if public_stats_path else None
    prior_reference_stats = (
        _load_frame(Path(prior_reference_stats_path)) if prior_reference_stats_path else None
    )

    selected, selection_summary = select_pseudo_label_rows(
        clusters=clusters,
        public_manifest=public_manifest,
        selection=selection,
        topk_scores=topk_scores,
        topk_indices=topk_indices,
        public_stats=public_stats,
        prior_reference_stats=prior_reference_stats,
    )

    pseudo_path = output_root / f"{selection.experiment_id}_pseudo_manifest.jsonl"
    mixed_path = output_root / f"{selection.experiment_id}_mixed_train_manifest.jsonl"
    summary_path = output_root / f"{selection.experiment_id}_summary.json"
    selected_pool_path = output_root / f"{selection.experiment_id}_selected_pool.csv"

    selected.write_csv(selected_pool_path)
    _write_pseudo_manifest(
        selected=selected,
        pseudo_path=pseudo_path,
        dataset_name=selection.dataset_name,
        label_prefix=selection.label_prefix,
        public_audio_prefix=selection.public_audio_prefix,
    )
    _write_mixed_manifest(
        base_train_manifest=Path(base_train_manifest) if base_train_manifest else None,
        pseudo_path=pseudo_path,
        mixed_path=mixed_path,
    )

    summary = {
        "experiment_id": selection.experiment_id,
        "clusters_csv": str(Path(clusters_csv)),
        "public_manifest_csv": str(Path(public_manifest_csv)),
        "base_train_manifest": str(base_train_manifest or ""),
        "pseudo_manifest": str(pseudo_path),
        "mixed_train_manifest": str(mixed_path),
        "selected_pool_csv": str(selected_pool_path),
        "pseudo_row_count": int(selected.height),
        "pseudo_cluster_count": int(selected.select("cluster_id").n_unique()),
        "mixed_row_count": _count_nonempty_lines(mixed_path),
        **selection_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def select_pseudo_label_rows(
    *,
    clusters: pl.DataFrame,
    public_manifest: pl.DataFrame,
    selection: PseudoLabelSelectionConfig,
    topk_scores: np.ndarray | None = None,
    topk_indices: np.ndarray | None = None,
    public_stats: pl.DataFrame | None = None,
    prior_reference_stats: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Select public rows that should become pseudo-label training rows."""

    _require_columns(clusters, ["cluster_id", "cluster_size", "row_index"], context="clusters")
    manifest = public_manifest.select(["row_index", "filepath"])
    selected = (
        clusters.join(manifest, on="row_index", how="inner")
        .sort(["cluster_id", "row_index"])
        .with_columns(
            pl.col("cluster_id").cast(pl.Int64),
            pl.col("cluster_size").cast(pl.Int64),
            pl.col("row_index").cast(pl.Int64),
            pl.col("filepath").cast(pl.Utf8),
        )
    )

    stage_counts: list[dict[str, int]] = []
    thresholds: dict[str, float] = {}
    _append_stage(stage_counts, "initial", selected)

    selected = selected.filter(
        (pl.col("cluster_size") >= selection.min_cluster_size)
        & (pl.col("cluster_size") <= selection.max_cluster_size)
    )
    _append_stage(stage_counts, "cluster_size", selected)

    if topk_scores is not None:
        selected = _attach_topk_scores(
            selected,
            topk_scores=topk_scores,
            row_count=public_manifest.height,
        )

    if selection.min_top1_score is not None:
        if "top1_score" not in selected.columns:
            raise ValueError("min_top1_score requires --topk-scores-npy.")
        selected = selected.filter(pl.col("top1_score") >= selection.min_top1_score)
        thresholds["top1_score_min"] = float(selection.min_top1_score)
        _append_stage(stage_counts, "top1_score", selected)

    if selection.min_top1_margin is not None:
        if "top1_margin" not in selected.columns:
            raise ValueError("min_top1_margin requires --topk-scores-npy with >=2 columns.")
        selected = selected.filter(pl.col("top1_margin") >= selection.min_top1_margin)
        thresholds["top1_margin_min"] = float(selection.min_top1_margin)
        _append_stage(stage_counts, "top1_margin", selected)

    if topk_indices is not None:
        selected, indegree_threshold = _attach_indegree(
            selected,
            topk_indices=topk_indices,
            row_count=public_manifest.height,
            top_k=selection.indegree_top_k,
            max_quantile=selection.max_indegree_quantile,
        )
        if indegree_threshold is not None:
            thresholds[f"indegree_at_{selection.indegree_top_k}_max"] = indegree_threshold
            _append_stage(stage_counts, f"indegree_at_{selection.indegree_top_k}", selected)
    elif selection.max_indegree_quantile is not None:
        raise ValueError("max_indegree_quantile requires --topk-indices-npy.")

    if public_stats is not None:
        selected, prior_meta = _attach_prior_features(
            selected,
            public_stats=public_stats,
            reference_stats=(
                prior_reference_stats if prior_reference_stats is not None else public_stats
            ),
            prior_feature_weights=selection.prior_feature_weights,
            max_quantile=selection.max_prior_distance_quantile,
            diversity_floor_quantile=selection.diversity_floor_quantile,
        )
        thresholds.update(prior_meta["thresholds"])
        if prior_meta["filter_applied"]:
            _append_stage(stage_counts, "prior_distance", selected)
    elif selection.max_prior_distance_quantile is not None:
        raise ValueError("max_prior_distance_quantile requires --public-stats.")

    if selection.max_rows_per_cluster is not None:
        selected = _limit_rows_per_cluster(
            selected,
            max_rows_per_cluster=selection.max_rows_per_cluster,
        )
        thresholds["max_rows_per_cluster"] = float(selection.max_rows_per_cluster)
        _append_stage(stage_counts, "cluster_cap", selected)

    selected = selected.sort(["cluster_id", "row_index"])
    summary = {
        "selection_stage_counts": stage_counts,
        "selection_thresholds": thresholds,
        "min_cluster_size": selection.min_cluster_size,
        "max_cluster_size": selection.max_cluster_size,
        "final_row_count": int(selected.height),
        "final_cluster_count": int(selected.select("cluster_id").n_unique()),
    }
    if "acoustic_bucket" in selected.columns:
        summary["final_acoustic_bucket_count"] = int(selected.select("acoustic_bucket").n_unique())
    return selected, summary


def _validate_optional_quantile(
    value: float | None,
    *,
    name: str,
    allow_one: bool,
) -> None:
    if value is None:
        return
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if value <= 0.0 or not upper_ok:
        comparator = "<= 1.0" if allow_one else "< 1.0"
        raise ValueError(f"{name} must satisfy 0 < {name} {comparator}.")


def _append_stage(stage_counts: list[dict[str, int]], stage: str, frame: pl.DataFrame) -> None:
    stage_counts.append(
        {
            "stage": stage,
            "row_count": int(frame.height),
            "cluster_count": int(frame.select("cluster_id").n_unique()) if frame.height else 0,
        }
    )


def _load_frame(path: Path) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Table does not exist: {path}")
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def _ensure_public_manifest(frame: pl.DataFrame) -> pl.DataFrame:
    if "filepath" not in frame.columns:
        raise ValueError("public manifest must contain filepath column.")
    if "row_index" not in frame.columns:
        frame = frame.with_row_index("row_index")
    return frame.select(pl.col("row_index").cast(pl.Int64), pl.col("filepath").cast(pl.Utf8))


def _load_array(path: Path | str | None, *, kind: str) -> np.ndarray | None:
    if path is None:
        return None
    array = np.load(Path(path))
    if array.ndim != 2:
        raise ValueError(f"{kind} array must be 2D, got shape {array.shape}.")
    return array


def _attach_topk_scores(
    frame: pl.DataFrame,
    *,
    topk_scores: np.ndarray,
    row_count: int,
) -> pl.DataFrame:
    if topk_scores.shape[0] != row_count:
        raise ValueError(
            f"topk_scores rows {topk_scores.shape[0]} do not match public row count {row_count}."
        )
    if topk_scores.shape[1] < 2:
        raise ValueError("topk_scores must contain at least 2 columns to compute margin.")
    stats = pl.DataFrame(
        {
            "row_index": np.arange(row_count, dtype=np.int64),
            "top1_score": topk_scores[:, 0].astype(np.float32, copy=False),
            "top1_margin": (topk_scores[:, 0] - topk_scores[:, 1]).astype(np.float32, copy=False),
        }
    )
    return frame.join(stats, on="row_index", how="left")


def _attach_indegree(
    frame: pl.DataFrame,
    *,
    topk_indices: np.ndarray,
    row_count: int,
    top_k: int,
    max_quantile: float | None,
) -> tuple[pl.DataFrame, float | None]:
    if topk_indices.shape[0] != row_count:
        raise ValueError(
            f"topk_indices rows {topk_indices.shape[0]} do not match public row count {row_count}."
        )
    if topk_indices.shape[1] < top_k:
        raise ValueError(
            f"topk_indices has only {topk_indices.shape[1]} columns < requested {top_k}."
        )
    clipped = np.asarray(topk_indices[:, :top_k], dtype=np.int64)
    indegree = np.bincount(clipped.ravel(), minlength=row_count).astype(np.int64, copy=False)
    column_name = f"indegree_at_{top_k}"
    frame = frame.join(
        pl.DataFrame({"row_index": np.arange(row_count, dtype=np.int64), column_name: indegree}),
        on="row_index",
        how="left",
    )
    threshold: float | None = None
    if max_quantile is not None:
        threshold = float(np.quantile(indegree, max_quantile))
        frame = frame.filter(pl.col(column_name) <= threshold)
    return frame, threshold


def _attach_prior_features(
    frame: pl.DataFrame,
    *,
    public_stats: pl.DataFrame,
    reference_stats: pl.DataFrame,
    prior_feature_weights: dict[str, float],
    max_quantile: float | None,
    diversity_floor_quantile: float | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    feature_names = list(prior_feature_weights)
    public_resolved = _resolve_prior_features(public_stats, feature_names)
    reference_resolved = _resolve_prior_features(reference_stats, feature_names)
    join_key = "row_index" if "row_index" in public_resolved.columns else "filepath"
    if join_key not in frame.columns:
        raise ValueError(f"frame must contain join key {join_key!r} for prior features.")
    joined = frame.join(public_resolved, on=join_key, how="left")
    _ensure_no_nulls(joined, feature_names, context="public prior features")

    reference_values = reference_resolved.select(feature_names).drop_nulls().to_numpy()
    if reference_values.size == 0:
        raise ValueError("reference prior feature table is empty after null filtering.")
    centers = np.median(reference_values, axis=0)
    mad = np.median(np.abs(reference_values - centers[None, :]), axis=0) * 1.4826
    std = np.std(reference_values, axis=0, ddof=0)
    scales = np.maximum(np.maximum(mad, std), _EPSILON)
    weights = np.asarray(
        [prior_feature_weights[name] for name in feature_names],
        dtype=np.float64,
    )
    values = joined.select(feature_names).to_numpy()
    distances = np.sum(
        weights[None, :] * np.square((values - centers[None, :]) / scales[None, :]),
        axis=1,
    )
    joined = joined.with_columns(
        pl.Series("prior_distance", distances.astype(np.float32, copy=False))
    )
    joined = joined.with_columns(
        _acoustic_bucket_expr(joined, reference_resolved).alias("acoustic_bucket")
    )

    thresholds: dict[str, float] = {}
    filter_applied = False
    if max_quantile is not None:
        global_threshold = float(np.quantile(distances, max_quantile))
        thresholds["prior_distance_max"] = global_threshold
        joined = joined.with_columns(
            (pl.col("prior_distance") <= global_threshold).alias("_keep_prior_global")
        )
        if diversity_floor_quantile is not None:
            group_cols = ["cluster_id", "acoustic_bucket"]
            joined = joined.with_columns(
                pl.len().over(group_cols).alias("_bucket_group_size"),
                pl.col("prior_distance").rank(method="ordinal").over(group_cols).alias("_bucket_rank"),
            ).with_columns(
                (
                    pl.col("_bucket_rank")
                    <= (pl.col("_bucket_group_size").cast(pl.Float64) * diversity_floor_quantile)
                    .ceil()
                    .clip(lower_bound=1.0)
                ).alias("_keep_bucket_floor")
            )
            thresholds["diversity_floor_quantile"] = float(diversity_floor_quantile)
            joined = joined.filter(pl.col("_keep_prior_global") | pl.col("_keep_bucket_floor"))
        else:
            joined = joined.filter(pl.col("_keep_prior_global"))
        filter_applied = True
        joined = joined.drop(
            [
                name
                for name in [
                    "_keep_prior_global",
                    "_bucket_group_size",
                    "_bucket_rank",
                    "_keep_bucket_floor",
                ]
                if name in joined.columns
            ]
        )

    meta = {
        "filter_applied": filter_applied,
        "thresholds": thresholds,
        "prior_feature_weights": dict(prior_feature_weights),
        "prior_feature_centers": {
            name: float(value) for name, value in zip(feature_names, centers, strict=True)
        },
        "prior_feature_scales": {
            name: float(value) for name, value in zip(feature_names, scales, strict=True)
        },
    }
    return joined, meta


def _resolve_prior_features(frame: pl.DataFrame, feature_names: list[str]) -> pl.DataFrame:
    key_columns = [name for name in ("row_index", "filepath") if name in frame.columns]
    if not key_columns:
        raise ValueError("prior feature table must contain row_index or filepath.")
    exprs: list[pl.Expr] = [pl.col(column) for column in key_columns]
    for feature_name in feature_names:
        exprs.append(_prior_feature_expr(frame, feature_name).alias(feature_name))
    return frame.select(exprs).unique(key_columns, keep="first")


def _prior_feature_expr(frame: pl.DataFrame, feature_name: str) -> pl.Expr:
    columns = set(frame.columns)
    if feature_name == "duration_s":
        return _first_column(columns, "duration_s", "dur", "duration_sec", "duration_seconds")
    if feature_name == "non_silent_ratio":
        if "non_silent_ratio" in columns:
            return pl.col("non_silent_ratio").cast(pl.Float64)
        if "silence_ratio_40db" in columns:
            return (1.0 - pl.col("silence_ratio_40db").cast(pl.Float64)).cast(pl.Float64)
    if feature_name == "leading_silence_s":
        return _first_column(columns, "leading_silence_s", "leading_silence_sec")
    if feature_name == "trailing_silence_s":
        return _first_column(columns, "trailing_silence_s", "trailing_silence_sec")
    if feature_name == "spectral_bandwidth_hz":
        return _first_column(columns, "spectral_bandwidth_hz")
    if feature_name == "band_energy_ratio_3_8k":
        return _first_column(columns, "band_energy_ratio_3_8k", "band_energy_3400_8000")
    raise ValueError(f"Unsupported prior feature: {feature_name!r}")


def _first_column(columns: set[str], *names: str) -> pl.Expr:
    for name in names:
        if name in columns:
            return pl.col(name).cast(pl.Float64)
    raise ValueError(f"Missing required columns, expected one of: {names!r}")


def _ensure_no_nulls(frame: pl.DataFrame, columns: list[str], *, context: str) -> None:
    missing = frame.filter(pl.any_horizontal([pl.col(name).is_null() for name in columns]))
    if missing.height:
        examples = missing.select("row_index", "filepath").head(5).to_dicts()
        raise ValueError(
            f"{context} left null values for {missing.height} rows; examples={examples}"
        )


def _acoustic_bucket_expr(frame: pl.DataFrame, reference_stats: pl.DataFrame) -> pl.Expr:
    parts: list[pl.Expr] = []
    if "duration_s" in frame.columns:
        duration = reference_stats.get_column("duration_s").to_numpy()
        q1, q2 = np.quantile(duration, [1.0 / 3.0, 2.0 / 3.0])
        parts.append(
            pl.when(pl.col("duration_s") <= q1)
            .then(pl.lit("dur_short"))
            .when(pl.col("duration_s") <= q2)
            .then(pl.lit("dur_mid"))
            .otherwise(pl.lit("dur_long"))
        )
    if "spectral_bandwidth_hz" in frame.columns:
        bandwidth_threshold = float(
            np.median(reference_stats.get_column("spectral_bandwidth_hz").to_numpy())
        )
        parts.append(
            pl.when(pl.col("spectral_bandwidth_hz") <= bandwidth_threshold)
            .then(pl.lit("bw_low"))
            .otherwise(pl.lit("bw_high"))
        )
    if {"leading_silence_s", "trailing_silence_s"}.issubset(frame.columns):
        reference_edge_silence = (
            reference_stats.get_column("leading_silence_s").to_numpy()
            + reference_stats.get_column("trailing_silence_s").to_numpy()
        )
        edge_threshold = float(np.median(reference_edge_silence))
        parts.append(
            pl.when((pl.col("leading_silence_s") + pl.col("trailing_silence_s")) <= edge_threshold)
            .then(pl.lit("edge_sil_low"))
            .otherwise(pl.lit("edge_sil_high"))
        )
    if not parts:
        return pl.lit("all")
    return pl.concat_str(parts, separator="|")


def _limit_rows_per_cluster(frame: pl.DataFrame, *, max_rows_per_cluster: int) -> pl.DataFrame:
    sort_by: list[str] = []
    descending: list[bool] = []
    for name in ("top1_score", "top1_margin"):
        if name in frame.columns:
            sort_by.append(name)
            descending.append(True)
    indegree_columns = sorted(
        [name for name in frame.columns if name.startswith("indegree_at_")],
        reverse=True,
    )
    for name in ("prior_distance", *indegree_columns):
        if name in frame.columns:
            sort_by.append(name)
            descending.append(False)
    sort_by.append("row_index")
    descending.append(False)
    ranked = (
        frame.sort(sort_by, descending=descending)
        .with_row_index("_global_order")
        .with_columns(
            pl.col("_global_order").rank(method="ordinal").over("cluster_id").alias("_cluster_rank")
        )
        .filter(pl.col("_cluster_rank") <= max_rows_per_cluster)
        .drop(["_global_order", "_cluster_rank"])
    )
    return ranked


def _write_pseudo_manifest(
    *,
    selected: pl.DataFrame,
    pseudo_path: Path,
    dataset_name: str,
    label_prefix: str,
    public_audio_prefix: str,
) -> None:
    with pseudo_path.open("w", encoding="utf-8") as handle:
        for row in selected.iter_rows(named=True):
            cluster_id = int(row["cluster_id"])
            row_index = int(row["row_index"])
            payload = {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": dataset_name,
                "source_dataset": "test_public_pseudo_labels",
                "speaker_id": f"{label_prefix}{cluster_id:06d}",
                "utterance_id": f"{label_prefix}{cluster_id:06d}:{row_index:06d}",
                "split": "pseudo_train",
                "audio_path": f"{public_audio_prefix.rstrip('/')}/{row['filepath']}",
                "channel": "mono",
            }
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _write_mixed_manifest(
    *,
    base_train_manifest: Path | None,
    pseudo_path: Path,
    mixed_path: Path,
) -> None:
    with mixed_path.open("w", encoding="utf-8") as mixed:
        if base_train_manifest is not None:
            if not base_train_manifest.is_file():
                raise FileNotFoundError(
                    f"Base train manifest does not exist: {base_train_manifest}"
                )
            for line in base_train_manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    mixed.write(line.rstrip() + "\n")
        mixed.write(pseudo_path.read_text(encoding="utf-8"))


def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _require_columns(frame: pl.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")
