"""Dense-gallery validation and hubness reports for speaker retrieval EDA."""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from kryptonite.eda.dense_audio import (
    SyntheticShiftProfile,
    apply_channel_condition,
    eval_crops,
    l2_normalize_rows,
    load_eval_waveform,
    sample_channel_condition,
)
from kryptonite.eda.manifest import load_train_manifest
from kryptonite.eda.rerank import density_zscore, reciprocal_local_rerank


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    trim: bool
    n_crops: int
    reciprocal_from: str | None = None
    local_scaling: bool = False


EMBED_CONFIGS = [
    ExperimentConfig("B0_raw_center", trim=False, n_crops=1),
    ExperimentConfig("B1_trim_center", trim=True, n_crops=1),
    ExperimentConfig("B2_raw_3crop", trim=False, n_crops=3),
    ExperimentConfig("B3_raw_5crop", trim=False, n_crops=5),
    ExperimentConfig("B4_trim_3crop", trim=True, n_crops=3),
    ExperimentConfig("B5_trim_5crop", trim=True, n_crops=5),
]
RERANK_CONFIGS = [
    ExperimentConfig(
        "B6_raw_reciprocal_top50", trim=False, n_crops=1, reciprocal_from="B0_raw_center"
    ),
    ExperimentConfig(
        "B7_trim_3crop_reciprocal_top50",
        trim=True,
        n_crops=3,
        reciprocal_from="B4_trim_3crop",
    ),
    ExperimentConfig(
        "B8_trim_3crop_reciprocal_local_scaling",
        trim=True,
        n_crops=3,
        reciprocal_from="B4_trim_3crop",
        local_scaling=True,
    ),
    ExperimentConfig(
        "B9_trim_5crop_reciprocal_top50",
        trim=True,
        n_crops=5,
        reciprocal_from="B5_trim_5crop",
    ),
    ExperimentConfig(
        "B10_trim_5crop_reciprocal_local_scaling",
        trim=True,
        n_crops=5,
        reciprocal_from="B5_trim_5crop",
        local_scaling=True,
    ),
]


def run_dense_gallery_package(
    *,
    dataset_root: Path,
    onnx_path: Path,
    audio_artifact_dir: Path,
    public_artifact_dir: Path,
    bucket_file_stats_path: Path,
    output_dir: Path,
    zip_path: Path,
    query_speakers: int,
    distractor_speakers: int,
    utts_per_speaker: int,
    seed: int,
    batch_size: int,
    query_batch_size: int,
    synthetic_shift: bool = False,
    shift_mode: str = "none",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if synthetic_shift and shift_mode == "none":
        shift_mode = "edge_silence"
    shift_profile = (
        SyntheticShiftProfile.from_file_stats(audio_artifact_dir / "file_stats.parquet", seed=seed)
        if shift_mode != "none"
        else None
    )
    manifest = _build_or_load_manifest(
        dataset_root=dataset_root,
        output_dir=output_dir,
        query_speakers=query_speakers,
        distractor_speakers=distractor_speakers,
        utts_per_speaker=utts_per_speaker,
        seed=seed,
    )
    runtimes = []
    for config in EMBED_CONFIGS:
        runtimes.append(
            _ensure_embeddings(
                config=config,
                manifest=manifest,
                output_dir=output_dir,
                onnx_path=onnx_path,
                batch_size=batch_size,
                shift_profile=shift_profile,
                shift_mode=shift_mode,
            )
        )
    eval_frames = []
    summaries = []
    cached_gallery_topk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for config in [*EMBED_CONFIGS, *RERANK_CONFIGS]:
        source_id = config.reciprocal_from or config.experiment_id
        embeddings = np.load(output_dir / f"embeddings_{source_id}.npy")
        gallery_topk = None
        density_z = None
        if config.reciprocal_from is not None:
            cached = cached_gallery_topk.get(source_id)
            if cached is None:
                cached = _gallery_topk(embeddings, top_k=50, batch_size=query_batch_size)
                cached_gallery_topk[source_id] = cached
            gallery_topk = cached[0]
            if config.local_scaling:
                density_z = density_zscore(cached[1], top_n=20)
        started = time.perf_counter()
        query_eval, summary = _evaluate_dense(
            experiment_id=config.experiment_id,
            embeddings=embeddings,
            manifest=manifest,
            search_k=50,
            top_k=10,
            query_batch_size=query_batch_size,
            gallery_topk=gallery_topk,
            density_z=density_z,
        )
        summary["eval_wall_s"] = round(time.perf_counter() - started, 6)
        eval_frames.append(query_eval)
        summaries.append(summary)
    query_eval_all = pl.concat(eval_frames, how="vertical")
    query_eval_all.write_parquet(output_dir / "dense_gallery_query_eval.parquet")
    pl.DataFrame(summaries).write_csv(output_dir / "dense_gallery_results.csv")
    _write_bucket_breakdown(
        query_eval_all,
        output_dir / "dense_gallery_bucket_breakdown.csv",
        file_stats_path=bucket_file_stats_path,
    )
    _write_runtime(runtimes, output_dir / "dense_gallery_runtime.csv")
    _write_hubness_reports(
        audio_artifact_dir=audio_artifact_dir,
        public_artifact_dir=public_artifact_dir,
        output_dir=output_dir,
    )
    _write_summary(output_dir, summaries, manifest, query_speakers, distractor_speakers)
    _write_plots(output_dir)
    _zip_dir(output_dir, zip_path)


def _build_or_load_manifest(
    *,
    dataset_root: Path,
    output_dir: Path,
    query_speakers: int,
    distractor_speakers: int,
    utts_per_speaker: int,
    seed: int,
) -> pl.DataFrame:
    path = output_dir / "dense_gallery_manifest.csv"
    if path.is_file():
        return pl.read_csv(path)
    train = load_train_manifest(dataset_root)
    eligible = (
        train.group_by("speaker_id")
        .len("n_utts")
        .filter(pl.col("n_utts") >= utts_per_speaker)
        .sort("speaker_id")
        .sample(n=query_speakers + distractor_speakers, seed=seed, shuffle=True)
        .with_row_index("speaker_order")
        .with_columns(
            pl.when(pl.col("speaker_order") < query_speakers)
            .then(pl.lit("query"))
            .otherwise(pl.lit("distractor"))
            .alias("speaker_role")
        )
    )
    subset = train.join(
        eligible.select(["speaker_id", "speaker_order", "speaker_role"]), on="speaker_id"
    )
    sampled = (
        subset.sample(fraction=1.0, seed=seed, shuffle=True)
        .group_by("speaker_id")
        .head(utts_per_speaker)
        .sort(["speaker_order", "filepath"])
        .with_row_index("gallery_index")
        .with_columns((pl.col("speaker_role") == "query").alias("is_query"))
    )
    sampled.write_csv(path)
    return sampled


def _ensure_embeddings(
    *,
    config: ExperimentConfig,
    manifest: pl.DataFrame,
    output_dir: Path,
    onnx_path: Path,
    batch_size: int,
    shift_profile: SyntheticShiftProfile | None,
    shift_mode: str,
) -> dict[str, Any]:
    path = output_dir / f"embeddings_{config.experiment_id}.npy"
    if path.is_file():
        return {"experiment_id": config.experiment_id, "stage": "embedding_cached", "seconds": 0.0}
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(
        str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    started = time.perf_counter()
    crop_samples = 16_000 * 6
    sums: np.ndarray | None = None
    counts = np.zeros(manifest.height, dtype=np.int32)
    batch: list[np.ndarray] = []
    owners: list[int] = []
    for row in manifest.iter_rows(named=True):
        index = int(row["gallery_index"])
        waveform = load_eval_waveform(
            Path(str(row["resolved_path"])),
            trim=config.trim,
            shift_profile=shift_profile,
            shift_mode=shift_mode,
            shift_key=index,
        )
        condition = (
            sample_channel_condition(shift_profile, shift_key=index)
            if shift_profile is not None and shift_mode == "v2"
            else None
        )
        for crop in eval_crops(waveform, crop_samples=crop_samples, n_crops=config.n_crops):
            if condition is not None:
                crop = apply_channel_condition(crop, condition)
            batch.append(crop)
            owners.append(index)
        if len(batch) >= batch_size:
            sums = _flush(session, input_name, batch, owners, sums, counts, manifest.height)
            batch, owners = [], []
    if batch:
        sums = _flush(session, input_name, batch, owners, sums, counts, manifest.height)
    if sums is None:
        raise RuntimeError("No embeddings were extracted.")
    embeddings = sums / np.maximum(counts[:, None], 1)
    np.save(path, embeddings.astype(np.float32))
    seconds = time.perf_counter() - started
    return {
        "experiment_id": config.experiment_id,
        "stage": "embedding",
        "seconds": round(seconds, 6),
        "providers": ",".join(session.get_providers()),
        "n_crops": config.n_crops,
        "trim": config.trim,
    }


def _flush(
    session: Any,
    input_name: str,
    batch: list[np.ndarray],
    owners: list[int],
    sums: np.ndarray | None,
    counts: np.ndarray,
    row_count: int,
) -> np.ndarray:
    outputs = session.run(None, {input_name: np.stack(batch).astype(np.float32, copy=False)})
    values = l2_normalize_rows(np.asarray(outputs[0], dtype=np.float32))
    if sums is None:
        sums = np.zeros((row_count, values.shape[1]), dtype=np.float32)
    for owner, embedding in zip(owners, values, strict=True):
        sums[owner] += embedding
        counts[owner] += 1
    return sums


def _evaluate_dense(
    *,
    experiment_id: str,
    embeddings: np.ndarray,
    manifest: pl.DataFrame,
    search_k: int,
    top_k: int,
    query_batch_size: int,
    gallery_topk: np.ndarray | None,
    density_z: np.ndarray | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    import torch

    labels = manifest["speaker_id"].cast(pl.Utf8).to_list()
    paths = manifest["filepath"].cast(pl.Utf8).to_list()
    query_indices = manifest.filter(pl.col("is_query"))["gallery_index"].to_numpy()
    matrix = torch.from_numpy(np.asarray(embeddings, dtype=np.float32).copy()).cuda()
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    rows = []
    for start in range(0, len(query_indices), query_batch_size):
        batch_query = query_indices[start : start + query_batch_size].copy()
        batch_tensor = torch.as_tensor(batch_query, dtype=torch.long, device="cuda")
        scores = matrix[batch_tensor] @ matrix.T
        scores[torch.arange(len(batch_query), device="cuda"), batch_tensor] = -torch.inf
        values, indices = torch.topk(scores, k=search_k, dim=1)
        idx_np = indices.cpu().numpy()
        val_np = values.cpu().numpy()
        if gallery_topk is not None:
            idx_np, val_np = reciprocal_local_rerank(
                query_indices=batch_query,
                indices=idx_np,
                scores=val_np,
                gallery_topk=gallery_topk,
                top_k=search_k,
                density_z=density_z,
            )
        rows.extend(
            _query_rows(
                experiment_id, batch_query, idx_np[:, :top_k], val_np[:, :top_k], labels, paths
            )
        )
    frame = pl.DataFrame(rows)
    summary = {
        "experiment_id": experiment_id,
        "query_count": frame.height,
        "gallery_count": len(labels),
        "p10": float(cast(float, frame["p10"].mean())),
        "top1_accuracy": float(cast(float, frame["top1_correct"].mean())),
        "mean_top10_score": float(cast(float, frame["top10_mean_score"].mean())),
    }
    return frame, summary


def _query_rows(
    experiment_id: str,
    query_indices: np.ndarray,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    labels: list[str],
    paths: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for row, query_index in enumerate(query_indices):
        query_label = labels[int(query_index)]
        hits = [labels[int(index)] == query_label for index in top_indices[row]]
        first = next((rank + 1 for rank, hit in enumerate(hits) if hit), 0)
        rows.append(
            {
                "experiment_id": experiment_id,
                "query_idx": int(query_index),
                "query_path": paths[int(query_index)],
                "speaker_id": query_label,
                "p10": float(np.mean(hits)),
                "n_correct_top10": int(sum(hits)),
                "top1_correct": bool(hits[0]),
                "first_correct_rank": first,
                "top1_score": float(top_scores[row, 0]),
                "top10_mean_score": float(np.mean(top_scores[row])),
                "top10_min_score": float(top_scores[row, -1]),
            }
        )
    return rows


def _gallery_topk(
    embeddings: np.ndarray, *, top_k: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    matrix = torch.from_numpy(np.asarray(embeddings, dtype=np.float32).copy()).cuda()
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    out = np.empty((matrix.shape[0], top_k), dtype=np.int64)
    out_scores = np.empty((matrix.shape[0], top_k), dtype=np.float32)
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        scores = matrix[start:end] @ matrix.T
        scores[
            torch.arange(end - start, device="cuda"), torch.arange(start, end, device="cuda")
        ] = -torch.inf
        values, indices = torch.topk(scores, k=top_k, dim=1)
        out[start:end] = indices.cpu().numpy()
        out_scores[start:end] = values.cpu().numpy()
    return out, out_scores


def _write_bucket_breakdown(
    query_eval: pl.DataFrame,
    path: Path,
    *,
    file_stats_path: Path,
) -> None:
    file_stats = pl.read_parquet(file_stats_path)
    joined = query_eval.join(
        file_stats.select(["filepath", "duration_bucket"]).rename({"filepath": "query_path"}),
        on="query_path",
        how="left",
    )
    (
        joined.group_by(["experiment_id", "duration_bucket"])
        .agg(pl.len().alias("query_count"), pl.col("p10").mean().alias("p10"))
        .sort(["experiment_id", "duration_bucket"])
        .write_csv(path)
    )


def _write_runtime(rows: list[dict[str, Any]], path: Path) -> None:
    pl.DataFrame(rows).write_csv(path)


def _write_hubness_reports(
    *, audio_artifact_dir: Path, public_artifact_dir: Path, output_dir: Path
) -> None:
    local_eval = pl.read_parquet(audio_artifact_dir / "embedding_eval.parquet")
    local_manifest = pl.read_csv(audio_artifact_dir / "val_manifest.csv")
    public_eval = pl.read_csv(public_artifact_dir / "public_embedding_eval_unlabeled.csv")
    public_manifest = pl.read_csv(public_artifact_dir / "test_public_manifest.csv")
    local_counts = np.bincount(
        np.concatenate(local_eval["top_indices"].to_list()), minlength=local_manifest.height
    )
    public_cols = [f"neighbor_{rank}_index" for rank in range(1, 11)]
    public_counts = np.bincount(
        public_eval.select(public_cols).to_numpy().ravel(), minlength=public_manifest.height
    )
    _write_hubness_one("local", local_counts, local_manifest["filepath"].to_list(), output_dir)
    _write_hubness_one("public", public_counts, public_manifest["filepath"].to_list(), output_dir)
    pl.DataFrame(
        [_hubness_summary("local", local_counts), _hubness_summary("public", public_counts)]
    ).write_csv(output_dir / "hubness_summary.csv")
    pl.concat(
        [
            _hubness_distribution("local", local_counts),
            _hubness_distribution("public", public_counts),
        ]
    ).write_csv(output_dir / "hubness_indegree_distribution.csv")


def _write_hubness_one(name: str, counts: np.ndarray, paths: list[str], output_dir: Path) -> None:
    order = np.argsort(counts)[::-1][:100]
    pl.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "filepath": [paths[i] for i in order],
            "in_degree": counts[order],
        }
    ).write_csv(output_dir / f"hubness_top_hubs_{name}.csv")


def _hubness_summary(name: str, counts: np.ndarray) -> dict[str, Any]:
    return {
        "pool": name,
        "p95": float(np.quantile(counts, 0.95)),
        "p99": float(np.quantile(counts, 0.99)),
        "max": int(counts.max()),
        "gini": _gini(counts),
    }


def _hubness_distribution(name: str, counts: np.ndarray) -> pl.DataFrame:
    degrees, degree_counts = np.unique(counts, return_counts=True)
    return pl.DataFrame(
        {
            "pool": name,
            "in_degree": degrees.astype(np.int64),
            "file_count": degree_counts.astype(np.int64),
        }
    )


def _gini(values: np.ndarray) -> float:
    sorted_values = np.sort(values.astype(np.float64))
    n = sorted_values.size
    return float(
        (2 * np.arange(1, n + 1) @ sorted_values) / (n * sorted_values.sum()) - (n + 1) / n
    )


def _write_summary(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    manifest: pl.DataFrame,
    query_speakers: int,
    distractor_speakers: int,
) -> None:
    best = max(summaries, key=lambda row: float(row["p10"]))
    b0 = next(row for row in summaries if row["experiment_id"] == "B0_raw_center")
    hubness = pl.read_csv(output_dir / "hubness_summary.csv")
    local_hub = hubness.filter(pl.col("pool") == "local").row(0, named=True)
    public_hub = hubness.filter(pl.col("pool") == "public").row(0, named=True)
    best_delta = float(best["p10"]) - float(b0["p10"])
    text = f"""# Dense Gallery And Hubness Summary

- Query speakers: `{query_speakers}`; distractor speakers: `{distractor_speakers}`.
- Queries: `{int(manifest["is_query"].sum())}`; gallery size: `{manifest.height}`.
- Dense raw baseline B0 P@10: `{b0["p10"]:.6f}`.
- Best dense-gallery config: `{best["experiment_id"]}` with P@10 `{best["p10"]:.6f}`.
- Best delta vs B0: `{best_delta:+.6f}`.
- B6/B7 use reciprocal top-50 rerank; embedding configs B0-B5 use baseline.onnx.
- Hubness Gini: local `{local_hub["gini"]:.4f}`, public `{public_hub["gini"]:.4f}`.
- Max top-10 in-degree: local `{local_hub["max"]}`, public `{public_hub["max"]}`.

Use `dense_gallery_results.csv` for config ranking, `dense_gallery_query_eval.parquet`
for per-query analysis, and `hubness_summary.csv`, `hubness_indegree_distribution.csv`,
plus top-hub files for hubness checks.
"""
    (output_dir / "00_summary.md").write_text(text, encoding="utf-8")


def _write_plots(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    results = pl.read_csv(output_dir / "dense_gallery_results.csv").sort("p10")
    plt.figure(figsize=(10, 5))
    plt.barh(results["experiment_id"].to_list(), results["p10"].to_numpy())
    plt.xlabel("P@10")
    plt.title("Dense Gallery B0-B7")
    plt.tight_layout()
    plt.savefig(plot_dir / "dense_gallery_p10.png", dpi=160)
    plt.close()
    hubs = pl.read_csv(output_dir / "hubness_summary.csv")
    plt.figure(figsize=(7, 4))
    plt.bar(hubs["pool"].to_list(), hubs["gini"].to_numpy())
    plt.ylabel("Gini")
    plt.title("Top-10 In-degree Hubness")
    plt.tight_layout()
    plt.savefig(plot_dir / "hubness_gini_public_vs_local.png", dpi=160)
    plt.close()
    distribution = pl.read_csv(output_dir / "hubness_indegree_distribution.csv")
    plt.figure(figsize=(8, 5))
    for pool in ("local", "public"):
        pool_dist = distribution.filter(pl.col("pool") == pool).sort("in_degree")
        plt.plot(
            pool_dist["in_degree"].to_numpy(),
            pool_dist["file_count"].to_numpy(),
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=pool,
        )
    plt.yscale("log")
    plt.xlabel("Top-10 in-degree")
    plt.ylabel("File count, log scale")
    plt.title("Hubness In-degree Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "hubness_indegree_distribution.png", dpi=160)
    plt.close()


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(".npy"):
                archive.write(path, path.relative_to(source_dir))
