"""Public inference-only ablations for baseline_fixed control."""

from __future__ import annotations

import csv
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.dense_audio import eval_crops, l2_normalize_rows, load_eval_waveform
from kryptonite.eda.leaderboard_alignment import public_lb_for, public_status_for
from kryptonite.eda.rerank import density_zscore, gini, reciprocal_local_rerank
from kryptonite.eda.submission import validate_submission


@dataclass(frozen=True, slots=True)
class PublicRun:
    experiment_id: str
    trim: bool
    n_crops: int
    source_id: str | None = None


EMBED_RUNS = [
    PublicRun("B2_raw_3crop", trim=False, n_crops=3),
    PublicRun("B4_trim_3crop", trim=True, n_crops=3),
]
SUBMISSION_RUNS = [
    *EMBED_RUNS,
    PublicRun("B7_trim_3crop_reciprocal_top50", trim=True, n_crops=3, source_id="B4_trim_3crop"),
    PublicRun(
        "B8_trim_3crop_reciprocal_local_scaling",
        trim=True,
        n_crops=3,
        source_id="B4_trim_3crop",
    ),
]


def run_public_ablation_package(
    *,
    manifest_csv: Path,
    template_csv: Path,
    onnx_path: Path,
    file_stats_path: Path,
    output_dir: Path,
    zip_path: Path,
    batch_size: int,
    search_batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(manifest_csv)
    runtimes = []
    for run in EMBED_RUNS:
        runtimes.append(
            _ensure_embeddings(run, manifest, onnx_path, output_dir, batch_size=batch_size)
        )
    topk_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summaries = []
    for run in SUBMISSION_RUNS:
        source_id = run.source_id or run.experiment_id
        embeddings = np.load(output_dir / f"embeddings_{source_id}.npy")
        top50, scores50 = topk_cache.get(source_id, (None, None))  # type: ignore[assignment]
        if top50 is None or scores50 is None:
            started = time.perf_counter()
            top50, scores50 = _exact_topk(embeddings, top_k=50, batch_size=search_batch_size)
            topk_cache[source_id] = (top50, scores50)
            search_s = time.perf_counter() - started
        else:
            search_s = 0.0
        started = time.perf_counter()
        top_indices, top_scores = _submission_topk(run, top50, scores50)
        rerank_s = time.perf_counter() - started
        started = time.perf_counter()
        submission_path = output_dir / f"submission_{run.experiment_id}.csv"
        _write_submission(manifest, top_indices, submission_path)
        validation = validate_submission(template_csv=template_csv, submission_csv=submission_path)
        validation_path = output_dir / f"submission_{run.experiment_id}_validation.json"
        validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        csv_s = time.perf_counter() - started
        summaries.append(
            _summary_row(run.experiment_id, top_scores, search_s, rerank_s, csv_s, validation)
        )
    pl.DataFrame(runtimes).write_csv(output_dir / "runtime_embedding.csv")
    pl.DataFrame(summaries).write_csv(output_dir / "public_runs_summary.csv")
    _write_hubness(output_dir, manifest, file_stats_path, topk_cache)
    _write_score_summary(output_dir)
    _write_readme(output_dir)
    _zip_dir(output_dir, zip_path)


def _ensure_embeddings(
    run: PublicRun,
    manifest: pl.DataFrame,
    onnx_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    output_path = output_dir / f"embeddings_{run.experiment_id}.npy"
    if output_path.is_file():
        return {
            "experiment_id": run.experiment_id,
            "stage": "embedding_cached",
            "seconds": 0.0,
            "n_crops": run.n_crops,
            "trim": run.trim,
        }
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
    for index, row in enumerate(manifest.iter_rows(named=True)):
        waveform = load_eval_waveform(Path(str(row["resolved_path"])), trim=run.trim)
        for crop in eval_crops(waveform, crop_samples=crop_samples, n_crops=run.n_crops):
            batch.append(crop)
            owners.append(index)
        if len(batch) >= batch_size:
            sums = _flush(session, input_name, batch, owners, sums, counts, manifest.height)
            batch, owners = [], []
    if batch:
        sums = _flush(session, input_name, batch, owners, sums, counts, manifest.height)
    if sums is None:
        raise RuntimeError("No public embeddings were extracted.")
    embeddings = l2_normalize_rows(sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    np.save(output_path, embeddings)
    return {
        "experiment_id": run.experiment_id,
        "stage": "embedding",
        "seconds": round(time.perf_counter() - started, 6),
        "providers": ",".join(session.get_providers()),
        "n_crops": run.n_crops,
        "trim": run.trim,
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
    values = session.run(None, {input_name: np.stack(batch).astype(np.float32, copy=False)})[0]
    values = l2_normalize_rows(np.asarray(values, dtype=np.float32))
    if sums is None:
        sums = np.zeros((row_count, values.shape[1]), dtype=np.float32)
    for owner, embedding in zip(owners, values, strict=True):
        sums[owner] += embedding
        counts[owner] += 1
    return sums


def _exact_topk(
    embeddings: np.ndarray, *, top_k: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    matrix = torch.from_numpy(np.asarray(embeddings, dtype=np.float32).copy()).cuda()
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    indices = np.empty((matrix.shape[0], top_k), dtype=np.int64)
    scores = np.empty((matrix.shape[0], top_k), dtype=np.float32)
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        sims = matrix[start:end] @ matrix.T
        sims[
            torch.arange(end - start, device="cuda"), torch.arange(start, end, device="cuda")
        ] = -torch.inf
        values, top_indices = torch.topk(sims, k=top_k, dim=1)
        indices[start:end] = top_indices.cpu().numpy()
        scores[start:end] = values.cpu().numpy()
    return indices, scores


def _submission_topk(
    run: PublicRun, top50: np.ndarray, scores50: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if run.source_id is None:
        return top50[:, :10], scores50[:, :10]
    density_z = density_zscore(scores50, top_n=20) if "local_scaling" in run.experiment_id else None
    rows = np.arange(top50.shape[0], dtype=np.int64)
    return reciprocal_local_rerank(
        query_indices=rows,
        indices=top50,
        scores=scores50,
        gallery_topk=top50,
        top_k=10,
        density_z=density_z,
    )


def _write_submission(manifest: pl.DataFrame, top_indices: np.ndarray, path: Path) -> None:
    paths = manifest["filepath"].cast(pl.Utf8).to_list()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filepath", "neighbours"])
        for filepath, neighbours in zip(paths, top_indices, strict=True):
            writer.writerow([filepath, ",".join(str(int(index)) for index in neighbours)])


def _summary_row(
    experiment_id: str,
    scores: np.ndarray,
    search_s: float,
    rerank_s: float,
    csv_s: float,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "public_lb": public_lb_for(experiment_id),
        "public_submission_status": public_status_for(experiment_id),
        "validator_passed": bool(validation["passed"]),
        "search_s": round(search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "csv_validation_s": round(csv_s, 6),
        "top1_score_mean": float(scores[:, 0].mean()),
        "top10_mean_score_mean": float(scores[:, :10].mean()),
        "top10_min_score_mean": float(scores[:, 9].mean()),
    }


def _write_hubness(
    output_dir: Path,
    manifest: pl.DataFrame,
    file_stats_path: Path,
    topk_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    file_stats = pl.read_parquet(file_stats_path).with_columns(
        (pl.col("peak_dbfs") >= -0.1).alias("peak_limited_flag"),
        (pl.col("clipping_frac") > 0.01).alias("hard_clipped_flag"),
        ((pl.col("narrowband_proxy") >= 0.5) | (pl.col("rolloff95_hz") <= 3800.0)).alias(
            "narrowband_like_flag"
        ),
    )
    paths = manifest["filepath"].cast(pl.Utf8).to_list()
    rows = []
    hubs = []
    for run in SUBMISSION_RUNS:
        source_id = run.source_id or run.experiment_id
        top50, scores50 = topk_cache[source_id]
        if run.source_id is not None:
            density_z = (
                density_zscore(scores50, top_n=20) if "local_scaling" in run.experiment_id else None
            )
            top50, _ = reciprocal_local_rerank(
                query_indices=np.arange(top50.shape[0], dtype=np.int64),
                indices=top50,
                scores=scores50,
                gallery_topk=top50,
                top_k=50,
                density_z=density_z,
            )
        for k in (10, 50):
            counts = np.bincount(top50[:, :k].ravel(), minlength=len(paths))
            rows.append(_hubness_row(run.experiment_id, k, counts))
            if run.experiment_id in {
                "B7_trim_3crop_reciprocal_top50",
                "B8_trim_3crop_reciprocal_local_scaling",
            }:
                hubs.append(_top_hubs(run.experiment_id, k, counts, paths, file_stats))
    pl.DataFrame(rows).write_csv(output_dir / "public_hubness_report.csv")
    top_hubs = pl.concat(hubs, how="vertical")
    top_hubs.write_csv(output_dir / "public_top_hubs.csv")
    top_hubs.filter(pl.col("experiment_id") == "B7_trim_3crop_reciprocal_top50").write_csv(
        output_dir / "public_b7_top_hubs.csv"
    )


def _hubness_row(experiment_id: str, k: int, counts: np.ndarray) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "pool": "public",
        "k": k,
        "p50": float(np.quantile(counts, 0.50)),
        "p95": float(np.quantile(counts, 0.95)),
        "p99": float(np.quantile(counts, 0.99)),
        "max": int(counts.max()),
        "gini": gini(counts),
    }


def _top_hubs(
    experiment_id: str,
    k: int,
    counts: np.ndarray,
    paths: list[str],
    file_stats: pl.DataFrame,
) -> pl.DataFrame:
    order = np.argsort(counts)[::-1][:100]
    hubs = pl.DataFrame(
        {
            "experiment_id": experiment_id,
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


def _write_score_summary(output_dir: Path) -> None:
    summary = pl.read_csv(output_dir / "public_runs_summary.csv")
    summary.select(
        "experiment_id", "top1_score_mean", "top10_mean_score_mean", "top10_min_score_mean"
    ).write_csv(output_dir / "public_score_summary.csv")


def _write_readme(output_dir: Path) -> None:
    text = """# Public Ablation Package

Submissions for B2/B4/B7/B8 are generated and validator-checked. Public LB scores
were supplied externally after leaderboard upload and are recorded in
`public_runs_summary.csv`.
"""
    (output_dir / "00_summary.md").write_text(text, encoding="utf-8")


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(".npy"):
                archive.write(path, path.relative_to(source_dir))
