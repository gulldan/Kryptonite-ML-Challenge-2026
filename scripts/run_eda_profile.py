"""Build offline EDA artifacts for participant challenge data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from kryptonite.eda import (
    assign_domain_buckets,
    build_dataset_summary,
    build_speaker_stats,
    compute_audio_stats_table,
    load_test_manifest,
    load_train_manifest,
    simulate_speaker_disjoint_split,
)


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [load_train_manifest(dataset_root)]
    if args.include_test_public:
        manifests.append(load_test_manifest(dataset_root))
    manifest = pl.concat(manifests, how="vertical")
    if args.max_files_per_split is not None:
        manifest = _limit_per_split(
            manifest,
            max_files_per_split=args.max_files_per_split,
            sample_strategy=args.sample_strategy,
            seed=args.seed,
        )

    if args.manifest_only:
        file_stats = _manifest_only_stats(manifest)
    elif args.batch_size is not None:
        file_stats = _compute_audio_stats_batched(
            manifest,
            output_dir=output_dir,
            analysis_seconds=args.analysis_seconds,
            workers=args.workers,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    else:
        file_stats = compute_audio_stats_table(
            manifest,
            analysis_seconds=args.analysis_seconds,
            workers=args.workers,
        )
    speaker_stats = build_speaker_stats(file_stats)
    summary = build_dataset_summary(file_stats, speaker_stats)
    split_stats = simulate_speaker_disjoint_split(
        speaker_stats,
        val_fraction=args.val_fraction,
        min_val_utts=args.min_val_utts,
        seed=args.seed,
    )
    domain_clusters = assign_domain_buckets(file_stats)

    file_stats.write_parquet(output_dir / "file_stats.parquet")
    speaker_stats.write_parquet(output_dir / "speaker_stats.parquet")
    domain_clusters.write_csv(output_dir / "domain_clusters.csv")
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "val_split_stats.json", split_stats)
    _write_experiment_log_template(output_dir / "experiment_log.csv")
    _write_observation_template(
        output_dir / "observation_to_decision.md",
        summary=summary,
        split_stats=split_stats,
    )
    print(f"Wrote EDA artifacts to {output_dir}")
    print(f"Rows profiled: {file_stats.height}")
    print(f"Speakers: {summary['speaker_count']}")
    print(f"Read errors: {summary['read_error_count']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="datasets/Для участников",
        help="Root containing train.csv, test_public.csv, train/, and test_public/.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eda/participants",
        help="Directory for Parquet/CSV/JSON/Markdown EDA artifacts.",
    )
    parser.add_argument(
        "--include-test-public",
        action="store_true",
        help="Also profile test_public.csv and test_public/ audio.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip waveform reads and write manifest-level artifacts only.",
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=None,
        help="Limit rows per split for smoke runs.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("head", "random"),
        default="head",
        help="How to apply --max-files-per-split.",
    )
    parser.add_argument(
        "--analysis-seconds",
        type=float,
        default=30.0,
        help="Maximum seconds decoded per file for low-level audio stats. Use <=0 for full file.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Audio profiling worker processes.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Profile audio in resumable Parquet batches of this many rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing batch Parquet files in --output-dir/batches.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-val-utts", type=int, default=11)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.max_files_per_split is not None and args.max_files_per_split <= 0:
        parser.error("--max-files-per-split must be positive when provided.")
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive when provided.")
    if args.analysis_seconds <= 0:
        args.analysis_seconds = None
    return args


def _limit_per_split(
    manifest: pl.DataFrame,
    *,
    max_files_per_split: int,
    sample_strategy: str,
    seed: int,
) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for split in manifest.get_column("split").unique().sort().to_list():
        split_frame = manifest.filter(pl.col("split") == split)
        limit = min(max_files_per_split, split_frame.height)
        if sample_strategy == "random":
            split_frame = split_frame.sample(n=limit, seed=seed, shuffle=True)
        else:
            split_frame = split_frame.head(limit)
        parts.append(split_frame)
    return pl.concat(parts, how="vertical").sort(["split", "row_index"])


def _compute_audio_stats_batched(
    manifest: pl.DataFrame,
    *,
    output_dir: Path,
    analysis_seconds: float | None,
    workers: int,
    batch_size: int,
    resume: bool,
) -> pl.DataFrame:
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: list[Path] = []
    total = manifest.height
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, total, batch_size)):
        batch_path = batches_dir / f"file_stats_batch_{batch_index:05d}.parquet"
        batch_paths.append(batch_path)
        if resume and batch_path.is_file():
            print(f"[EDA] reuse batch {batch_index} rows {start}:{min(start + batch_size, total)}")
            continue
        batch_manifest = manifest.slice(start, batch_size)
        batch_started = time.perf_counter()
        batch_stats = compute_audio_stats_table(
            batch_manifest,
            analysis_seconds=analysis_seconds,
            workers=workers,
        )
        batch_stats.write_parquet(batch_path)
        elapsed = time.perf_counter() - batch_started
        done = min(start + batch_size, total)
        rate = batch_stats.height / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[EDA] wrote batch {batch_index} rows {start}:{done} "
            f"elapsed={elapsed:.1f}s rate={rate:.1f} files/s total_done={done}/{total}",
            flush=True,
        )
    print(f"[EDA] combining {len(batch_paths)} batches", flush=True)
    frames = [pl.read_parquet(path) for path in batch_paths if path.is_file()]
    if not frames:
        raise RuntimeError("No audio profiling batches were produced.")
    file_stats = pl.concat(frames, how="diagonal").sort(["split", "row_index"])
    elapsed_total = time.perf_counter() - started
    rate_total = file_stats.height / elapsed_total if elapsed_total > 0.0 else 0.0
    print(
        f"[EDA] combined rows={file_stats.height} elapsed={elapsed_total:.1f}s "
        f"rate={rate_total:.1f} files/s",
        flush=True,
    )
    return file_stats


def _manifest_only_stats(manifest: pl.DataFrame) -> pl.DataFrame:
    return manifest.with_columns(
        pl.col("resolved_path").map_elements(lambda value: Path(value).is_file()).alias("exists"),
        pl.lit(None, dtype=pl.Utf8).alias("error"),
        pl.lit(None, dtype=pl.Utf8).alias("format"),
        pl.lit(None, dtype=pl.Utf8).alias("subtype"),
        pl.lit(None, dtype=pl.Int64).alias("sample_rate_hz"),
        pl.lit(None, dtype=pl.Int64).alias("num_channels"),
        pl.lit(None, dtype=pl.Int64).alias("frame_count"),
        pl.lit(None, dtype=pl.Float64).alias("duration_s"),
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_experiment_log_template(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        ",".join(
            [
                "experiment_id",
                "encoder_checkpoint",
                "input_sr",
                "crop_policy",
                "n_crops",
                "trim_silence",
                "vad",
                "normalize_audio",
                "embedding_norm",
                "similarity_metric",
                "augmentation_set",
                "loss",
                "sampler",
                "val_p_at_10",
                "top1",
                "runtime",
                "hypothesis",
                "result",
                "conclusion",
                "notes",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_observation_template(
    path: Path,
    *,
    summary: dict[str, object],
    split_stats: dict[str, object],
) -> None:
    duration = summary.get("duration_quantiles_s", {})
    file_count = summary.get("file_count")
    speaker_count = summary.get("speaker_count")
    total_hours = summary.get("total_duration_h")
    min_val_utts = split_stats.get("min_val_utts")
    eligible_val_count = split_stats.get("eligible_val_speaker_count")
    text = f"""# Observation To Decision

## Dataset

- Observation: train/test manifests were parsed from participant data.
- Evidence: files={file_count}, speakers={speaker_count}, total_hours={total_hours}.
- Decision: use this file as the running EDA log; replace placeholders after full profiling.

## Duration

- Observation: duration quantiles are `{duration}`.
- Decision: choose crop length and short-file filtering after reviewing the full duration histogram.

## Validation Split

- Observation: eligible val speakers with at least {min_val_utts} utterances = {eligible_val_count}.
- Decision: keep local validation speaker-disjoint and reserve only speakers that can support P@10.

## Audio Conditions

- Observation: review `domain_clusters.csv` for short, clipped, silence-heavy, low-volume,
  narrowband-like, noise-like, and cleanish buckets.
- Decision: map dominant bad buckets to preprocessing and augmentation hypotheses.

## Retrieval

- Observation: run `scripts/run_eda_retrieval_eval.py` after embeddings are available.
- Decision: compare P@10 by bucket before changing encoder, crop policy, or retrieval postprocess.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
