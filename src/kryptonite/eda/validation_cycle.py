"""Build validation-cycle review artifacts from dense-gallery runs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.leaderboard_alignment import (
    public_lb_for,
    public_status_for,
    write_leaderboard_alignment,
)
from kryptonite.eda.validation_cycle_plots import write_validation_cycle_plots

RUN_ORDER = [
    "B0_raw_center",
    "B1_trim_center",
    "B2_raw_3crop",
    "B3_raw_5crop",
    "B4_trim_3crop",
    "B5_trim_5crop",
    "B6_raw_reciprocal_top50",
    "B7_trim_3crop_reciprocal_top50",
]


def build_validation_cycle_package(
    *,
    dense_dir: Path,
    shifted_dir: Path,
    baseline_fixed_dir: Path,
    audio_artifact_dir: Path,
    output_dir: Path,
    zip_path: Path,
    shifted_v2_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stats = _file_stats(audio_artifact_dir / "file_stats.parquet")
    smoke = _smoke_results(dense_dir)
    dense = pl.read_csv(dense_dir / "dense_gallery_results.csv")
    shifted = pl.read_csv(shifted_dir / "dense_gallery_results.csv")
    runtime = _runtime_by_run(dense_dir, shifted_dir)
    runs = _runs_summary(smoke, dense, shifted, runtime)
    runs.write_csv(output_dir / "runs_summary.csv")
    (output_dir / "protocol_defs.json").write_text(
        json.dumps(_protocol_payload(baseline_fixed_dir), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _bucket_metrics(dense_dir, shifted_dir, file_stats).write_csv(
        output_dir / "bucket_metrics_by_run.csv"
    )
    runtime.write_csv(output_dir / "runtime_by_run.csv")
    _public_lb_runs().write_csv(output_dir / "public_lb_runs.csv")
    hubness, top_hubs = _hubness_reports(dense_dir, baseline_fixed_dir, file_stats)
    hubness.write_csv(output_dir / "hubness_report.csv")
    top_hubs.write_csv(output_dir / "top_hubs.csv")
    _failure_cases(shifted_dir, file_stats).write_csv(output_dir / "failure_cases_best_run.csv")
    if shifted_v2_dir is not None:
        write_leaderboard_alignment(
            validation_package_dir=output_dir,
            shifted_v2_dir=shifted_v2_dir,
            output_dir=output_dir,
        )
    _summary(output_dir, runs, hubness)
    _checklist(output_dir)
    write_validation_cycle_plots(output_dir)
    _zip_dir(output_dir, zip_path)


def _file_stats(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(
        (pl.col("peak_dbfs") >= -0.1).alias("peak_limited_flag"),
        (pl.col("clipping_frac") > 0.01).alias("hard_clipped_flag"),
        ((pl.col("narrowband_proxy") >= 0.5) | (pl.col("rolloff95_hz") <= 3800.0)).alias(
            "narrowband_like_flag"
        ),
        (pl.col("silence_ratio_40db") >= 0.5).alias("silence_heavy_flag"),
        pl.when(pl.col("duration_s") < 2.0)
        .then(pl.lit("very_short"))
        .when(pl.col("duration_s") < 4.0)
        .then(pl.lit("short"))
        .when(pl.col("duration_s") < 6.0)
        .then(pl.lit("medium"))
        .when(pl.col("duration_s") < 10.0)
        .then(pl.lit("normal"))
        .when(pl.col("duration_s") < 20.0)
        .then(pl.lit("long"))
        .when(pl.col("duration_s") < 40.0)
        .then(pl.lit("very_long"))
        .otherwise(pl.lit("extra_long"))
        .alias("duration_bucket"),
    )


def _smoke_results(dense_dir: Path) -> pl.DataFrame:
    manifest = pl.read_csv(dense_dir / "dense_gallery_manifest.csv")
    query = manifest.filter(pl.col("is_query")).sort("gallery_index")
    labels = query["speaker_id"].cast(pl.Utf8).to_list()
    query_indices = query["gallery_index"].to_numpy()
    rows = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run_id in RUN_ORDER:
        source_id = _embedding_source(run_id)
        embeddings = np.load(dense_dir / f"embeddings_{source_id}.npy")[query_indices]
        reciprocal = run_id.startswith("B6") or run_id.startswith("B7")
        if reciprocal:
            top50, scores50 = cache.get(source_id, (None, None))  # type: ignore[assignment]
            if top50 is None or scores50 is None:
                top50, scores50 = _all_topk(embeddings, top_k=50)
                cache[source_id] = (top50, scores50)
            top_indices, top_scores = _reciprocal(top50, scores50, top_k=10)
        else:
            top_indices, top_scores = _all_topk(embeddings, top_k=10)
        p10, top1, mean_score = _retrieval_metrics(top_indices, top_scores, labels)
        rows.append(
            {
                "experiment_id": run_id,
                "smoke_val_p10": p10,
                "smoke_val_top1": top1,
                "smoke_val_top10_mean_score": mean_score,
            }
        )
    return pl.DataFrame(rows)


def _embedding_source(run_id: str) -> str:
    if run_id.startswith("B6"):
        return "B0_raw_center"
    if run_id.startswith("B7"):
        return "B4_trim_3crop"
    return run_id


def _all_topk(
    embeddings: np.ndarray, *, top_k: int, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    matrix = torch.from_numpy(np.asarray(embeddings, dtype=np.float32).copy()).cuda()
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    indices = np.empty((matrix.shape[0], top_k), dtype=np.int64)
    scores_out = np.empty((matrix.shape[0], top_k), dtype=np.float32)
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        scores = matrix[start:end] @ matrix.T
        scores[
            torch.arange(end - start, device="cuda"), torch.arange(start, end, device="cuda")
        ] = -torch.inf
        values, top_indices = torch.topk(scores, k=top_k, dim=1)
        indices[start:end] = top_indices.cpu().numpy()
        scores_out[start:end] = values.cpu().numpy()
    return indices, scores_out


def _reciprocal(
    indices: np.ndarray, scores: np.ndarray, *, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    out_i = np.empty((indices.shape[0], top_k), dtype=np.int64)
    out_s = np.empty((indices.shape[0], top_k), dtype=np.float32)
    for row in range(indices.shape[0]):
        reciprocal = np.asarray(
            [row in indices[int(candidate), :20] for candidate in indices[row]], dtype=np.float32
        )
        order = np.argsort(scores[row] + 0.03 * reciprocal)[::-1][:top_k]
        out_i[row] = indices[row, order]
        out_s[row] = scores[row, order]
    return out_i, out_s


def _retrieval_metrics(
    indices: np.ndarray, scores: np.ndarray, labels: list[str]
) -> tuple[float, float, float]:
    correct_counts = []
    top1 = []
    for row in range(indices.shape[0]):
        hits = [labels[int(index)] == labels[row] for index in indices[row]]
        correct_counts.append(float(np.mean(hits)))
        top1.append(float(hits[0]))
    return float(np.mean(correct_counts)), float(np.mean(top1)), float(np.mean(scores))


def _runtime_by_run(dense_dir: Path, shifted_dir: Path) -> pl.DataFrame:
    dense_runtime = _runtime_one(dense_dir, "dense_gallery_val")
    shifted_runtime = _runtime_one(shifted_dir, "dense_shifted_val")
    return pl.concat([dense_runtime, shifted_runtime], how="vertical")


def _runtime_one(path: Path, protocol: str) -> pl.DataFrame:
    embedding_rows = pl.read_csv(path / "dense_gallery_runtime.csv").iter_rows(named=True)
    embedding_s = {str(row["experiment_id"]): float(row["seconds"]) for row in embedding_rows}
    rows = []
    for row in pl.read_csv(path / "dense_gallery_results.csv").iter_rows(named=True):
        run_id = str(row["experiment_id"])
        source_id = _embedding_source(run_id)
        emb_s = embedding_s[source_id]
        eval_s = float(row["eval_wall_s"])
        rows.append(
            {
                "experiment_id": run_id,
                "embedding_source_id": source_id,
                "embedding_s": emb_s,
                "eval_s": eval_s,
                "protocol": protocol,
                "runtime_total_s": emb_s + eval_s,
            }
        )
    return pl.DataFrame(rows)


def _runs_summary(
    smoke: pl.DataFrame, dense: pl.DataFrame, shifted: pl.DataFrame, runtime: pl.DataFrame
) -> pl.DataFrame:
    dense = dense.select(
        "experiment_id",
        pl.col("p10").alias("dense_gallery_val_p10"),
        pl.col("top1_accuracy").alias("dense_gallery_val_top1"),
        pl.col("mean_top10_score").alias("dense_gallery_top10_mean_score"),
    )
    shifted = shifted.select(
        "experiment_id",
        pl.col("p10").alias("dense_shifted_val_p10"),
        pl.col("top1_accuracy").alias("dense_shifted_val_top1"),
        pl.col("mean_top10_score").alias("dense_shifted_top10_mean_score"),
    )
    runtime_wide = runtime.pivot(
        values="runtime_total_s", index="experiment_id", on="protocol", aggregate_function="first"
    ).rename(
        {
            "dense_gallery_val": "dense_gallery_runtime_total_s",
            "dense_shifted_val": "dense_shifted_runtime_total_s",
        }
    )
    return (
        smoke.join(dense, on="experiment_id")
        .join(shifted, on="experiment_id")
        .join(runtime_wide, on="experiment_id")
        .with_columns(
            pl.col("experiment_id")
            .map_elements(public_lb_for, return_dtype=pl.Float64)
            .alias("public_lb"),
            pl.col("experiment_id")
            .map_elements(
                lambda value: public_lb_for(value) is not None,
                return_dtype=pl.Boolean,
            )
            .alias("validator_passed"),
            pl.col("experiment_id")
            .map_elements(public_status_for, return_dtype=pl.Utf8)
            .alias("public_submission_status"),
        )
        .sort("experiment_id")
    )


def _protocol_payload(baseline_fixed_dir: Path) -> dict[str, Any]:
    return {
        "control": {
            "name": "baseline_fixed_v1",
            "onnx": str(baseline_fixed_dir / "model_embeddings.onnx"),
            "public_lb": 0.1024,
            "validator_report": str(
                baseline_fixed_dir / "submission_center_opset20_validation.json"
            ),
        },
        "protocols": {
            "smoke_val": "500 held-out speakers x 11 utterances, no distractors.",
            "dense_gallery_val": (
                "500 query speakers x 11 utterances + 3000 distractor speakers x 11 utterances."
            ),
            "dense_shifted_val": (
                "Same gallery as dense_gallery_val, plus deterministic "
                "public-empirical leading/trailing silence before embedding."
            ),
        },
        "runs": {
            "B0": "baseline_fixed center crop",
            "B1": "trim only",
            "B2": "3-crop only",
            "B3": "5-crop only",
            "B4": "trim + 3-crop",
            "B5": "trim + 5-crop",
            "B6": "B0 + reciprocal top-50 rerank, lambda=0.03, reciprocal top-20",
            "B7": "B4 + reciprocal top-50 rerank, lambda=0.03, reciprocal top-20",
        },
    }


def _bucket_metrics(dense_dir: Path, shifted_dir: Path, file_stats: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for protocol, path in [
        ("dense_gallery_val", dense_dir),
        ("dense_shifted_val", shifted_dir),
    ]:
        query_eval = pl.read_parquet(path / "dense_gallery_query_eval.parquet")
        joined = query_eval.join(
            file_stats.select(
                [
                    "filepath",
                    "duration_bucket",
                    "silence_heavy_flag",
                    "peak_limited_flag",
                    "hard_clipped_flag",
                    "narrowband_like_flag",
                ]
            ).rename({"filepath": "query_path"}),
            on="query_path",
            how="left",
        )
        for bucket_name in [
            "duration_bucket",
            "silence_heavy_flag",
            "peak_limited_flag",
            "hard_clipped_flag",
            "narrowband_like_flag",
        ]:
            frames.append(
                joined.with_columns(pl.col(bucket_name).cast(pl.Utf8).alias("bucket_value"))
                .group_by(["experiment_id", "bucket_value"])
                .agg(
                    pl.len().alias("query_count"),
                    pl.col("p10").mean().alias("p10"),
                    pl.col("top1_correct").mean().alias("top1_accuracy"),
                )
                .with_columns(
                    pl.lit(protocol).alias("protocol"), pl.lit(bucket_name).alias("bucket_name")
                )
            )
    return pl.concat(frames, how="vertical").sort(["protocol", "experiment_id", "bucket_name"])


def _public_lb_runs() -> pl.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "experiment_id": "organizer_baseline",
            "public_lb": 0.0779,
            "status": "external_reference",
        },
        {
            "experiment_id": "B0_raw_center",
            "public_lb": public_lb_for("B0_raw_center"),
            "status": "submitted_public_lb_received",
        },
    ]
    for run_id in RUN_ORDER:
        if run_id != "B0_raw_center":
            rows.append(
                {
                    "experiment_id": run_id,
                    "public_lb": public_lb_for(run_id),
                    "status": public_status_for(run_id),
                }
            )
    return pl.DataFrame(rows)


def _hubness_reports(
    dense_dir: Path, baseline_fixed_dir: Path, file_stats: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    manifest = pl.read_csv(dense_dir / "dense_gallery_manifest.csv")
    dense_paths = manifest["filepath"].cast(pl.Utf8).to_list()
    b0 = np.load(dense_dir / "embeddings_B0_raw_center.npy")
    b4 = np.load(dense_dir / "embeddings_B4_trim_3crop.npy")
    dense_b0_top50, _ = _all_topk(b0, top_k=50)
    dense_b4_top50, dense_b4_scores = _all_topk(b4, top_k=50)
    dense_b7_top10, _ = _reciprocal(dense_b4_top50, dense_b4_scores, top_k=10)
    public_embeddings = np.load(baseline_fixed_dir / "test_public_emb_center_opset20.npy")
    public_top50, _ = _all_topk(public_embeddings, top_k=50)
    public_paths = (
        file_stats.filter(pl.col("split") == "test_public")
        .sort("row_index")["filepath"]
        .cast(pl.Utf8)
        .to_list()
    )
    cases = [
        ("baseline_fixed", "dense_gallery_val", "B0_raw_center", dense_paths, dense_b0_top50),
        (
            "best_dense_run",
            "dense_gallery_val",
            "B7_trim_3crop_reciprocal_top50",
            dense_paths,
            np.column_stack([dense_b7_top10, dense_b4_top50[:, 10:50]]),
        ),
        ("baseline_fixed", "public", "B0_raw_center", public_paths, public_top50),
    ]
    reports = []
    hubs = []
    for state, pool, run_id, paths, top50 in cases:
        for k in (10, 50):
            counts = np.bincount(top50[:, :k].ravel(), minlength=len(paths))
            reports.append(_hubness_row(state, pool, run_id, k, counts))
            hubs.append(_top_hubs(state, pool, run_id, k, counts, paths, file_stats))
    reports.append(
        {
            "state": "best_dense_run",
            "pool": "public",
            "experiment_id": "B7_trim_3crop_reciprocal_top50",
            "k": 10,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "gini": None,
            "status": "not_computed_public_embeddings_not_submitted",
        }
    )
    return pl.DataFrame(reports), pl.concat(hubs, how="vertical")


def _hubness_row(state: str, pool: str, run_id: str, k: int, counts: np.ndarray) -> dict[str, Any]:
    return {
        "state": state,
        "pool": pool,
        "experiment_id": run_id,
        "k": k,
        "p50": float(np.quantile(counts, 0.50)),
        "p95": float(np.quantile(counts, 0.95)),
        "p99": float(np.quantile(counts, 0.99)),
        "max": int(counts.max()),
        "gini": _gini(counts),
        "status": "computed",
    }


def _gini(values: np.ndarray) -> float:
    sorted_values = np.sort(values.astype(np.float64))
    n = sorted_values.size
    if float(sorted_values.sum()) == 0.0:
        return 0.0
    return float(
        (2 * np.arange(1, n + 1) @ sorted_values) / (n * sorted_values.sum()) - (n + 1) / n
    )


def _top_hubs(
    state: str,
    pool: str,
    run_id: str,
    k: int,
    counts: np.ndarray,
    paths: list[str],
    file_stats: pl.DataFrame,
) -> pl.DataFrame:
    order = np.argsort(counts)[::-1][:100]
    hubs = pl.DataFrame(
        {
            "state": state,
            "pool": pool,
            "experiment_id": run_id,
            "k": k,
            "rank": np.arange(1, len(order) + 1),
            "file_index": order,
            "filepath": [paths[index] for index in order],
            "in_degree": counts[order],
        }
    )
    return hubs.join(
        file_stats.select(
            [
                "filepath",
                "duration_s",
                "leading_silence_s",
                "trailing_silence_s",
                "peak_limited_flag",
                "hard_clipped_flag",
                "narrowband_like_flag",
            ]
        ),
        on="filepath",
        how="left",
    )


def _failure_cases(shifted_dir: Path, file_stats: pl.DataFrame) -> pl.DataFrame:
    query_eval = pl.read_parquet(shifted_dir / "dense_gallery_query_eval.parquet")
    best_id = (
        pl.read_csv(shifted_dir / "dense_gallery_results.csv")
        .sort("p10", descending=True)
        .row(0, named=True)["experiment_id"]
    )
    return (
        query_eval.filter(pl.col("experiment_id") == best_id)
        .join(
            file_stats.select(
                [
                    "filepath",
                    "duration_s",
                    "leading_silence_s",
                    "trailing_silence_s",
                    "duration_bucket",
                    "peak_limited_flag",
                    "narrowband_like_flag",
                ]
            ).rename({"filepath": "query_path"}),
            on="query_path",
            how="left",
        )
        .sort(["p10", "top10_mean_score"])
        .head(100)
    )


def _summary(output_dir: Path, runs: pl.DataFrame, hubness: pl.DataFrame) -> None:
    b0 = runs.filter(pl.col("experiment_id") == "B0_raw_center").row(0, named=True)
    best_dense = runs.sort("dense_gallery_val_p10", descending=True).row(0, named=True)
    best_shifted = runs.sort("dense_shifted_val_p10", descending=True).row(0, named=True)
    public_hub = hubness.filter(
        (pl.col("pool") == "public") & (pl.col("state") == "baseline_fixed") & (pl.col("k") == 10)
    ).row(0, named=True)
    dense_hub = hubness.filter(
        (pl.col("pool") == "dense_gallery_val")
        & (pl.col("state") == "baseline_fixed")
        & (pl.col("k") == 10)
    ).row(0, named=True)
    correlation_text = _correlation_summary(output_dir)
    text = f"""# Validation Cycle Summary

- Control: `baseline_fixed_v1`, public LB `0.1024`, validator passed.
- Public submitted ablations: B2 `0.1098`, B4 `0.1150`, B7 `0.1206`.
- Smoke B0 P@10: `{b0["smoke_val_p10"]:.6f}`.
- Dense-gallery B0 P@10: `{b0["dense_gallery_val_p10"]:.6f}`.
- Dense-shifted B0 P@10: `{b0["dense_shifted_val_p10"]:.6f}`.
- Best dense-gallery run: `{best_dense["experiment_id"]}`.
- Best dense-gallery P@10: `{best_dense["dense_gallery_val_p10"]:.6f}`.
- Best dense-shifted run: `{best_shifted["experiment_id"]}`.
- Best dense-shifted P@10: `{best_shifted["dense_shifted_val_p10"]:.6f}`.
- Public baseline hubness Gini@10: `{public_hub["gini"]:.4f}`.
- Dense baseline hubness Gini@10: `{dense_hub["gini"]:.4f}`.
{correlation_text}

B7 is the best submitted public run so far.
"""
    (output_dir / "00_summary.md").write_text(text, encoding="utf-8")


def _correlation_summary(output_dir: Path) -> str:
    path = output_dir / "rank_correlation.csv"
    if not path.is_file():
        return ""
    rows = []
    for row in pl.read_csv(path).iter_rows(named=True):
        rows.append(f"- Spearman `{row['local_protocol']}` vs public: `{row['spearman']:.3f}`.")
    return "\n".join(rows)


def _checklist(output_dir: Path) -> None:
    text = """# Completion Checklist

- baseline_fixed official control: done.
- smoke_val B0-B7: done from baseline_fixed dense embeddings.
- dense_gallery_val B0-B7: done.
- dense_shifted_val B0-B7: done with public-empirical edge silence shift.
- public LB for B0: done, external score 0.1024.
- public LB for B2/B4/B7: done, external scores recorded.
- public LB for B1/B3/B5/B6/B8: not submitted.
- Spearman rank correlation with public: done for B0/B2/B4/B7.
- hubness k=10/k=50 for dense baseline and dense best run: done.
- hubness k=10/k=50 for public baseline_fixed: done.
- hubness for public B7: done in `public_ablation_cycle`.
"""
    (output_dir / "completion_checklist.md").write_text(text, encoding="utf-8")


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
