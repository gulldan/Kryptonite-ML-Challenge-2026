"""Run baseline.onnx on test_public and export public retrieval diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.data.audio_io import read_audio_file, resample_waveform
from kryptonite.eda import load_test_manifest
from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    manifest = load_test_manifest(dataset_root, name=Path(args.test_csv).stem)
    manifest.write_csv(output_dir / "test_public_manifest.csv")

    if args.stage in {"all", "embed"}:
        _run_embed_stage(
            manifest=manifest,
            output_dir=output_dir,
            onnx_path=Path(args.onnx_path),
            sample_rate_hz=args.sample_rate,
            crop_seconds=args.crop_seconds,
            batch_size=args.batch_size,
            device=args.device,
        )
    if args.stage in {"all", "search"}:
        _run_search_stage(
            manifest=manifest,
            output_dir=output_dir,
            top_k=args.top_k,
            query_batch_size=args.query_batch_size,
            device=args.search_device,
            write_long_neighbors=args.write_long_neighbors,
        )
    if args.stage in {"all", "compare"}:
        _run_compare_stage(
            public_dir=output_dir,
            local_artifact_dir=Path(args.local_artifact_dir),
            public_lb_score=args.public_lb_score,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="datasets/Для участников")
    parser.add_argument("--test-csv", default="test_public.csv")
    parser.add_argument("--onnx-path", default="datasets/Для участников/baseline.onnx")
    parser.add_argument("--output-dir", default="artifacts/eda/participants_public_baseline")
    parser.add_argument("--local-artifact-dir", default="artifacts/eda/participants_audio6")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "auto"), default="auto")
    parser.add_argument("--search-device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--stage", choices=("all", "embed", "search", "compare"), default="all")
    parser.add_argument("--public-lb-score", type=float, default=None)
    parser.add_argument(
        "--write-long-neighbors", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.query_batch_size <= 0:
        parser.error("--query-batch-size must be positive.")
    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    return args


def _run_embed_stage(
    *,
    manifest: pl.DataFrame,
    output_dir: Path,
    onnx_path: Path,
    sample_rate_hz: int,
    crop_seconds: float,
    batch_size: int,
    device: str,
) -> None:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    embedding_path = output_dir / "test_public_embeddings.npy"
    if embedding_path.exists():
        cached = np.load(embedding_path, mmap_mode="r")
        if cached.shape[0] == manifest.height:
            print(f"[PUBLIC-ONNX] embeddings already exist: {embedding_path}", flush=True)
            return
        raise ValueError(f"{embedding_path} rows={cached.shape[0]}, expected {manifest.height}.")

    started = time.perf_counter()
    session_started = time.perf_counter()
    session = ort.InferenceSession(str(onnx_path), providers=_providers(device, ort))
    session_seconds = time.perf_counter() - session_started
    providers = session.get_providers()
    print(f"[PUBLIC-ONNX] providers: {providers}", flush=True)

    timings = {
        "onnx_session_init_s": session_seconds,
        "audio_loading_s": 0.0,
        "embedding_extraction_s": 0.0,
    }
    embeddings, audio_seconds = _extract_embeddings(
        session=session,
        manifest=manifest,
        sample_rate_hz=sample_rate_hz,
        crop_seconds=crop_seconds,
        batch_size=batch_size,
        timings=timings,
    )
    np.save(embedding_path, embeddings)
    _write_embedding_norms(embeddings, manifest, output_dir / "public_embedding_norms.csv")
    _write_embedding_summary(
        embeddings,
        output_dir / "public_embedding_summary.csv",
        providers=providers,
        sample_rate_hz=sample_rate_hz,
        crop_seconds=crop_seconds,
    )
    timings["total_embed_wall_s"] = time.perf_counter() - started
    timings["files_per_sec"] = manifest.height / timings["total_embed_wall_s"]
    timings["real_time_factor"] = timings["total_embed_wall_s"] / audio_seconds
    _write_runtime_rows(output_dir / "public_embed_runtime.csv", timings)


def _providers(device: str, ort: Any) -> list[str]:
    available = set(ort.get_available_providers())
    if device == "auto" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _extract_embeddings(
    *,
    session: Any,
    manifest: pl.DataFrame,
    sample_rate_hz: int,
    crop_seconds: float,
    batch_size: int,
    timings: dict[str, float],
) -> tuple[np.ndarray, float]:
    input_name = session.get_inputs()[0].name
    crop_samples = int(sample_rate_hz * crop_seconds)
    batches: list[np.ndarray] = []
    batch_audio: list[np.ndarray] = []
    audio_seconds = 0.0
    for index, row in enumerate(manifest.iter_rows(named=True), start=1):
        load_started = time.perf_counter()
        waveform = _load_center_crop(
            Path(str(row["resolved_path"])),
            target_sample_rate_hz=sample_rate_hz,
            crop_samples=crop_samples,
        )
        timings["audio_loading_s"] += time.perf_counter() - load_started
        audio_seconds += waveform.size / sample_rate_hz
        batch_audio.append(waveform)
        if len(batch_audio) >= batch_size or index == manifest.height:
            input_batch = np.stack(batch_audio, axis=0).astype(np.float32, copy=False)
            infer_started = time.perf_counter()
            outputs = session.run(None, {input_name: input_batch})
            timings["embedding_extraction_s"] += time.perf_counter() - infer_started
            batches.append(np.asarray(outputs[0], dtype=np.float32))
            batch_audio = []
            print(f"[PUBLIC-ONNX] embedded {index}/{manifest.height}", flush=True)
    return np.concatenate(batches, axis=0), audio_seconds


def _load_center_crop(
    path: Path,
    *,
    target_sample_rate_hz: int,
    crop_samples: int,
) -> np.ndarray:
    waveform, info = read_audio_file(path)
    mono = np.asarray(waveform, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=0, dtype=np.float32)
    if info.sample_rate_hz != target_sample_rate_hz:
        mono = resample_waveform(
            mono,
            orig_freq=info.sample_rate_hz,
            new_freq=target_sample_rate_hz,
        )
    if mono.size >= crop_samples:
        start = max(0, (mono.size - crop_samples) // 2)
        return mono[start : start + crop_samples].astype(np.float32, copy=False)
    if mono.size == 0:
        return np.zeros(crop_samples, dtype=np.float32)
    repeat_factor = crop_samples // mono.size + 1
    return np.tile(mono, repeat_factor)[:crop_samples].astype(np.float32, copy=False)


def _run_search_stage(
    *,
    manifest: pl.DataFrame,
    output_dir: Path,
    top_k: int,
    query_batch_size: int,
    device: str,
    write_long_neighbors: bool,
) -> None:
    import torch

    embedding_path = output_dir / "test_public_embeddings.npy"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing {embedding_path}; run --stage embed first.")
    started = time.perf_counter()
    embeddings = np.load(embedding_path).astype(np.float32, copy=False)
    if embeddings.shape[0] != manifest.height:
        raise ValueError(f"Embedding rows={embeddings.shape[0]}, expected {manifest.height}.")

    torch_device = _torch_device(device, torch)
    matrix = torch.from_numpy(embeddings).to(torch_device)
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    all_indices = np.empty((matrix.shape[0], top_k), dtype=np.int64)
    all_scores = np.empty((matrix.shape[0], top_k), dtype=np.float32)
    for start in range(0, matrix.shape[0], query_batch_size):
        end = min(start + query_batch_size, matrix.shape[0])
        scores = matrix[start:end] @ matrix.T
        row_indices = torch.arange(end - start, device=torch_device)
        col_indices = torch.arange(start, end, device=torch_device)
        scores[row_indices, col_indices] = -torch.inf
        values, indices = torch.topk(scores, k=top_k, dim=1)
        all_indices[start:end] = indices.cpu().numpy()
        all_scores[start:end] = values.cpu().numpy()
        print(f"[PUBLIC-NN] searched {end}/{matrix.shape[0]}", flush=True)

    filepaths = manifest.get_column("filepath").cast(pl.Utf8).to_list()
    _write_submission(
        output_dir / "submission_baseline_public.csv",
        filepaths=filepaths,
        neighbour_indices=all_indices,
    )
    _write_public_eval_wide(
        output_dir / "public_embedding_eval_unlabeled.csv",
        filepaths=filepaths,
        neighbour_indices=all_indices,
        neighbour_scores=all_scores,
    )
    if write_long_neighbors:
        _write_public_neighbors_long(
            output_dir / "public_query_top10_neighbors.csv",
            filepaths=filepaths,
            neighbour_indices=all_indices,
            neighbour_scores=all_scores,
        )
    _write_score_summary(
        output_dir / "public_neighbor_score_summary.csv",
        neighbour_scores=all_scores,
    )
    report = validate_submission(
        template_csv=output_dir / "test_public_manifest.csv",
        submission_csv=output_dir / "submission_baseline_public.csv",
        k=top_k,
    )
    (output_dir / "submission_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_runtime_rows(
        output_dir / "public_search_runtime.csv",
        {
            "search_wall_s": time.perf_counter() - started,
            "query_count": float(manifest.height),
            "query_batch_size": float(query_batch_size),
            "top_k": float(top_k),
            "torch_cuda": float(torch_device.type == "cuda"),
        },
    )


def _torch_device(device: str, torch: Any) -> Any:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA search requested but torch.cuda.is_available() is false.")
    return torch.device(device)


def _write_submission(
    path: Path,
    *,
    filepaths: list[str],
    neighbour_indices: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "neighbours"])
        writer.writeheader()
        for filepath, indices in zip(filepaths, neighbour_indices, strict=True):
            writer.writerow(
                {
                    "filepath": filepath,
                    "neighbours": ",".join(str(int(index)) for index in indices),
                }
            )


def _write_public_eval_wide(
    path: Path,
    *,
    filepaths: list[str],
    neighbour_indices: np.ndarray,
    neighbour_scores: np.ndarray,
) -> None:
    fields = ["query_index", "filepath", "top1_score", "top10_min_score", "top10_mean_score"]
    fields.extend(f"neighbor_{rank}_index" for rank in range(1, neighbour_indices.shape[1] + 1))
    fields.extend(f"score_{rank}" for rank in range(1, neighbour_scores.shape[1] + 1))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_index, filepath in enumerate(filepaths):
            scores = neighbour_scores[query_index]
            row: dict[str, Any] = {
                "query_index": query_index,
                "filepath": filepath,
                "top1_score": float(scores[0]),
                "top10_min_score": float(scores[-1]),
                "top10_mean_score": float(np.mean(scores)),
            }
            for rank, value in enumerate(neighbour_indices[query_index], start=1):
                row[f"neighbor_{rank}_index"] = int(value)
            for rank, value in enumerate(scores, start=1):
                row[f"score_{rank}"] = float(value)
            writer.writerow(row)


def _write_public_neighbors_long(
    path: Path,
    *,
    filepaths: list[str],
    neighbour_indices: np.ndarray,
    neighbour_scores: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_index",
                "query_filepath",
                "rank",
                "neighbor_index",
                "neighbor_filepath",
                "similarity",
            ],
        )
        writer.writeheader()
        for query_index, filepath in enumerate(filepaths):
            for rank, (neighbor_index, score) in enumerate(
                zip(neighbour_indices[query_index], neighbour_scores[query_index], strict=True),
                start=1,
            ):
                writer.writerow(
                    {
                        "query_index": query_index,
                        "query_filepath": filepath,
                        "rank": rank,
                        "neighbor_index": int(neighbor_index),
                        "neighbor_filepath": filepaths[int(neighbor_index)],
                        "similarity": float(score),
                    }
                )


def _write_embedding_norms(embeddings: np.ndarray, manifest: pl.DataFrame, path: Path) -> None:
    norms = np.linalg.norm(embeddings, axis=1)
    manifest.select(["row_index", "filepath"]).with_columns(
        pl.Series("embedding_norm", norms)
    ).write_csv(path)


def _write_embedding_summary(
    embeddings: np.ndarray,
    path: Path,
    *,
    providers: list[str],
    sample_rate_hz: int,
    crop_seconds: float,
) -> None:
    norms = np.linalg.norm(embeddings, axis=1)
    rows = [
        {"metric": "query_count", "value": float(embeddings.shape[0])},
        {"metric": "embedding_dim", "value": float(embeddings.shape[1])},
        {"metric": "sample_rate_hz", "value": float(sample_rate_hz)},
        {"metric": "crop_seconds", "value": float(crop_seconds)},
        {"metric": "embedding_norm_mean", "value": float(np.mean(norms))},
        {"metric": "embedding_norm_p50", "value": float(np.quantile(norms, 0.50))},
        {"metric": "embedding_norm_p90", "value": float(np.quantile(norms, 0.90))},
        {"metric": "execution_providers", "value": ",".join(providers)},
    ]
    pl.DataFrame(rows).write_csv(path)


def _write_score_summary(path: Path, *, neighbour_scores: np.ndarray) -> None:
    metrics = {
        "top1_score": neighbour_scores[:, 0],
        "top10_min_score": neighbour_scores[:, -1],
        "top10_mean_score": neighbour_scores.mean(axis=1),
    }
    rows = []
    for metric, values in metrics.items():
        rows.append(
            {
                "metric": metric,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "p10": float(np.quantile(values, 0.10)),
                "p50": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
                "p99": float(np.quantile(values, 0.99)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    pl.DataFrame(rows).write_csv(path)


def _run_compare_stage(
    *,
    public_dir: Path,
    local_artifact_dir: Path,
    public_lb_score: float | None,
) -> None:
    local_eval = pl.read_parquet(local_artifact_dir / "embedding_eval.parquet")
    public_scores = pl.read_csv(public_dir / "public_neighbor_score_summary.csv")
    local_top_scores = _stack_list_column(local_eval.get_column("top_scores").to_list())
    local_summary = _score_summary_dict(local_top_scores)
    public_summary = {row["metric"]: row for row in public_scores.iter_rows(named=True)}
    rows = [
        _compare_row("pool_size", float(local_eval.height), _manifest_count(public_dir)),
        _compare_row(
            "top1_score_mean",
            local_summary["top1_score_mean"],
            float(public_summary["top1_score"]["mean"]),
        ),
        _compare_row(
            "top10_min_score_mean",
            local_summary["top10_min_score_mean"],
            float(public_summary["top10_min_score"]["mean"]),
        ),
        _compare_row(
            "top10_mean_score_mean",
            local_summary["top10_mean_score_mean"],
            float(public_summary["top10_mean_score"]["mean"]),
        ),
    ]
    precision_col = next(col for col in local_eval.columns if col.startswith("precision_at_"))
    rows.append(
        _compare_row(
            "labelled_metric",
            float(local_eval.get_column(precision_col).mean()),
            public_lb_score,
            note="public value is leaderboard score supplied by user, not locally computable",
        )
    )
    pl.DataFrame(rows).write_csv(public_dir / "public_local_alignment.csv")


def _stack_list_column(values: list[Any]) -> np.ndarray:
    return np.asarray([list(value) for value in values], dtype=np.float32)


def _score_summary_dict(top_scores: np.ndarray) -> dict[str, float]:
    return {
        "top1_score_mean": float(np.mean(top_scores[:, 0])),
        "top10_min_score_mean": float(np.mean(top_scores[:, -1])),
        "top10_mean_score_mean": float(np.mean(top_scores.mean(axis=1))),
    }


def _manifest_count(public_dir: Path) -> float:
    return float(pl.read_csv(public_dir / "test_public_manifest.csv").height)


def _compare_row(
    metric: str,
    local_value: float | None,
    public_value: float | None,
    *,
    note: str = "",
) -> dict[str, Any]:
    ratio = None
    delta = None
    if local_value is not None and public_value is not None:
        delta = public_value - local_value
        if local_value != 0:
            ratio = public_value / local_value
    return {
        "metric": metric,
        "local_value": local_value,
        "public_value": public_value,
        "public_minus_local": delta,
        "public_div_local": ratio,
        "note": note,
    }


def _write_runtime_rows(path: Path, timings: dict[str, float]) -> None:
    pl.DataFrame(
        [{"stage": key, "seconds": round(float(value), 6)} for key, value in timings.items()]
    ).write_csv(path)


if __name__ == "__main__":
    main()
