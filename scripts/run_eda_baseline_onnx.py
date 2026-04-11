"""Run a fast ONNX pseudo-baseline on a speaker-disjoint local validation subset."""

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
from kryptonite.eda import evaluate_retrieval_embeddings, load_train_manifest


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    dataset_root = Path(args.dataset_root)
    train_manifest = load_train_manifest(dataset_root)
    val_manifest = _build_val_manifest(
        train_manifest,
        max_speakers=args.max_speakers,
        utts_per_speaker=args.utts_per_speaker,
        seed=args.seed,
    )
    val_manifest.write_csv(output_dir / "val_manifest.csv")

    timings: dict[str, float] = {
        "embedding_cache_load_s": 0.0,
        "onnx_session_init_s": 0.0,
        "audio_loading_s": 0.0,
        "embedding_extraction_s": 0.0,
        "retrieval_eval_s": 0.0,
        "artifact_writing_s": 0.0,
    }
    execution_providers = ["cached_embeddings"]
    embedding_path = output_dir / "val_embeddings.npy"
    if embedding_path.exists():
        load_started = time.perf_counter()
        embeddings = np.load(embedding_path)
        timings["embedding_cache_load_s"] = time.perf_counter() - load_started
        if embeddings.shape[0] != val_manifest.height:
            raise ValueError(
                f"{embedding_path} has {embeddings.shape[0]} rows, expected {val_manifest.height}."
            )
        audio_seconds = float(val_manifest.height * args.crop_seconds)
        print(f"[EDA-ONNX] reused embeddings from {embedding_path}", flush=True)
    else:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        session_started = time.perf_counter()
        session = ort.InferenceSession(str(args.onnx_path), providers=_providers(args.device, ort))
        timings["onnx_session_init_s"] = time.perf_counter() - session_started
        execution_providers = session.get_providers()
        print(f"[EDA-ONNX] providers: {execution_providers}", flush=True)
        embeddings, audio_seconds = _extract_embeddings(
            session=session,
            manifest=val_manifest,
            dataset_root=dataset_root,
            sample_rate_hz=args.sample_rate,
            crop_seconds=args.crop_seconds,
            batch_size=args.batch_size,
            timings=timings,
        )
        np.save(embedding_path, embeddings)

    labels = val_manifest.get_column("speaker_id").cast(pl.Utf8).to_list()
    filepaths = val_manifest.get_column("filepath").cast(pl.Utf8).to_list()

    retrieval_started = time.perf_counter()
    result = evaluate_retrieval_embeddings(
        embeddings,
        labels=labels,
        filepaths=filepaths,
        top_k=args.top_k,
        normalize=True,
    )
    result.summary["onnx_execution_providers"] = execution_providers
    timings["retrieval_eval_s"] = time.perf_counter() - retrieval_started

    write_started = time.perf_counter()
    result.per_query.write_parquet(output_dir / "embedding_eval.parquet")
    _write_flat_csv(result.per_query, output_dir / "embedding_eval.csv")
    _write_flat_csv(result.worst_queries, output_dir / "worst_queries.csv")
    result.confused_speaker_pairs.write_csv(output_dir / "confused_speaker_pairs.csv")
    _write_query_neighbors(result.per_query, output_dir / "query_top10_neighbors.csv")
    _write_speaker_quality(result.per_query, output_dir / "speaker_retrieval_quality.csv")
    _write_speaker_cohesion(embeddings, labels, output_dir / "speaker_cohesion.csv")
    _write_embedding_norms(embeddings, val_manifest, output_dir / "embedding_norms.csv")
    _write_retrieval_summary(result.summary, output_dir / "retrieval_summary.json")
    timings["artifact_writing_s"] = time.perf_counter() - write_started
    _write_runtime_profile(
        output_dir / "runtime_profile.csv",
        timings=timings,
        total_wall_s=time.perf_counter() - run_started,
        file_count=val_manifest.height,
        audio_seconds=audio_seconds,
    )

    print(json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote ONNX pseudo-baseline artifacts to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="datasets/Для участников")
    parser.add_argument("--onnx-path", default="datasets/Для участников/baseline.onnx")
    parser.add_argument("--output-dir", default="artifacts/eda/participants_baseline")
    parser.add_argument("--max-speakers", type=int, default=500)
    parser.add_argument("--utts-per-speaker", type=int, default=11)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    args = parser.parse_args()
    if args.max_speakers <= 0:
        parser.error("--max-speakers must be positive.")
    if args.utts_per_speaker <= args.top_k:
        parser.error("--utts-per-speaker must be greater than --top-k for P@K.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    return args


def _build_val_manifest(
    train_manifest: pl.DataFrame,
    *,
    max_speakers: int,
    utts_per_speaker: int,
    seed: int,
) -> pl.DataFrame:
    eligible = (
        train_manifest.group_by("speaker_id")
        .agg(pl.len().alias("n_utts"))
        .filter(pl.col("n_utts") >= utts_per_speaker)
        .sort("speaker_id")
    )
    if eligible.is_empty():
        raise ValueError("No speakers have enough utterances for local P@10 validation.")
    speaker_count = min(max_speakers, eligible.height)
    speakers = (
        eligible.sample(n=speaker_count, seed=seed, shuffle=True)
        .get_column("speaker_id")
        .cast(pl.Utf8)
        .to_list()
    )
    selected_parts = []
    for speaker in speakers:
        rows = train_manifest.filter(pl.col("speaker_id") == speaker).sample(
            n=utts_per_speaker,
            seed=seed,
            shuffle=True,
        )
        selected_parts.append(rows)
    return pl.concat(selected_parts, how="vertical").sort(["speaker_id", "filepath"])


def _providers(device: str, ort: Any) -> list[str]:
    available = set(ort.get_available_providers())
    if device == "auto" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _extract_embeddings(
    *,
    session: Any,
    manifest: pl.DataFrame,
    dataset_root: Path,
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
            print(f"[EDA-ONNX] embedded {index}/{manifest.height}", flush=True)
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
    repeat_factor = crop_samples // max(1, mono.size) + 1
    if mono.size == 0:
        return np.zeros(crop_samples, dtype=np.float32)
    return np.tile(mono, repeat_factor)[:crop_samples].astype(np.float32, copy=False)


def _write_query_neighbors(per_query: pl.DataFrame, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for row in per_query.iter_rows(named=True):
        top_indices = row["top_indices"]
        top_scores = row["top_scores"]
        top_speakers = row["top_speaker_ids"]
        for rank, (index, score, speaker) in enumerate(
            zip(top_indices, top_scores, top_speakers, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "query_index": row["query_index"],
                    "query_filepath": row["filepath"],
                    "query_speaker_id": row["speaker_id"],
                    "rank": rank,
                    "neighbor_index": index,
                    "neighbor_speaker_id": speaker,
                    "similarity": score,
                    "correct": speaker == row["speaker_id"],
                }
            )
    pl.DataFrame(rows).write_csv(path)


def _write_flat_csv(dataframe: pl.DataFrame, path: Path) -> None:
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


def _write_speaker_quality(per_query: pl.DataFrame, path: Path) -> None:
    precision_col = next(col for col in per_query.columns if col.startswith("precision_at_"))
    per_query.group_by("speaker_id").agg(
        pl.len().alias("query_count"),
        pl.col(precision_col).mean().alias("mean_precision"),
        pl.col("top1_correct").mean().alias("top1_accuracy"),
        pl.col("first_correct_rank").mean().alias("mean_first_correct_rank"),
    ).sort("mean_precision").write_csv(path)


def _write_speaker_cohesion(embeddings: np.ndarray, labels: list[str], path: Path) -> None:
    normed = _l2_normalize(embeddings)
    label_array = np.asarray(labels, dtype=object)
    rows = []
    for speaker in sorted(set(labels)):
        indices = np.flatnonzero(label_array == speaker)
        speaker_embeddings = normed[indices]
        scores = speaker_embeddings @ speaker_embeddings.T
        pair_scores = scores[np.triu_indices(scores.shape[0], k=1)]
        rows.append(
            {
                "speaker_id": speaker,
                "n_utts": int(indices.size),
                "mean_pairwise_cosine": float(np.mean(pair_scores)),
                "std_pairwise_cosine": float(np.std(pair_scores)),
                "min_pairwise_cosine": float(np.min(pair_scores)),
                "p10_pairwise_cosine": float(np.quantile(pair_scores, 0.10)),
            }
        )
    pl.DataFrame(rows).sort("mean_pairwise_cosine").write_csv(path)


def _write_embedding_norms(
    embeddings: np.ndarray,
    manifest: pl.DataFrame,
    path: Path,
) -> None:
    norms = np.linalg.norm(embeddings, axis=1)
    manifest.select(["row_index", "speaker_id", "filepath"]).with_columns(
        pl.Series("embedding_norm", norms)
    ).write_csv(path)


def _write_retrieval_summary(summary: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_runtime_profile(
    path: Path,
    *,
    timings: dict[str, float],
    total_wall_s: float,
    file_count: int,
    audio_seconds: float,
) -> None:
    rows = [{"stage": stage, "seconds": round(seconds, 6)} for stage, seconds in timings.items()]
    rows.extend(
        [
            {"stage": "total_wall", "seconds": round(total_wall_s, 6)},
            {"stage": "files_per_sec", "seconds": round(file_count / total_wall_s, 6)},
            {"stage": "real_time_factor", "seconds": round(total_wall_s / audio_seconds, 6)},
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "seconds"])
        writer.writeheader()
        writer.writerows(rows)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


if __name__ == "__main__":
    main()
