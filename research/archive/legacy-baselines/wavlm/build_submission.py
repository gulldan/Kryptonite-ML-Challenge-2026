#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    ensure_dir,
    get_git_sha,
    load_config,
    load_pretrained_components,
    maybe_log_mlflow,
    submissions_root,
    write_json,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build submission.csv for WavLM speaker-retrieval inference."
    )
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument(
        "--csv", default="", help="Path to test CSV. Defaults to config paths.test_csv."
    )
    parser.add_argument("--topk", type=int, default=10, help="Number of neighbours to write.")
    parser.add_argument(
        "--mode",
        default="",
        help="Evaluation mode. Defaults to config evaluation.primary_mode.",
    )
    parser.add_argument("--run-name", default="", help="Optional submission run name.")
    parser.add_argument("--save-embeddings", action="store_true", help="Also save embeddings.npy.")
    parser.add_argument("--data-root", default="", help="Override config paths.data_root.")
    parser.add_argument(
        "--experiment-root", default="", help="Override config paths.experiment_root."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device. Use auto, cpu, cuda, or a concrete torch device.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Override evaluation.batch_size.")
    parser.add_argument(
        "--retrieval-chunk-size",
        type=int,
        default=0,
        help="Override evaluation.retrieval_chunk_size.",
    )
    return parser.parse_args()


def load_duration_lookup_for_paths(config: dict, filepaths: list[str]) -> dict[str, float]:
    cache_path = config["paths"]["audio_header_cache"]
    file_set = set(filepaths)
    if cache_path and cache_path.exists():
        frame = pd.read_parquet(cache_path, columns=["filepath", "duration_sec"])
        frame = frame[frame["filepath"].isin(file_set)]
        lookup = dict(zip(frame["filepath"], frame["duration_sec"], strict=False))
    else:
        lookup = {}
    if len(lookup) == len(file_set):
        return lookup
    from common import compute_duration_sec

    data_root = config["paths"]["data_root"]
    for filepath in filepaths:
        if filepath not in lookup:
            lookup[filepath] = compute_duration_sec(data_root / filepath)
    return lookup


def build_manifest(test_df: pd.DataFrame, duration_lookup: dict[str, float]) -> pd.DataFrame:
    rows = []
    for row in test_df.itertuples(index=False):
        filepath = row.filepath
        rows.append(
            {
                "filepath": filepath,
                "path": filepath,
                "dur": float(duration_lookup[filepath]),
                "spk": filepath,
            }
        )
    return pd.DataFrame(rows)


def validate_submission(df: pd.DataFrame, expected_filepaths: list[str], topk: int) -> None:
    if list(df.columns) != ["filepath", "neighbours"]:
        raise ValueError(
            f"submission columns must be ['filepath', 'neighbours'], got {list(df.columns)}"
        )
    if len(df) != len(expected_filepaths):
        raise ValueError(f"submission row count {len(df)} != expected {len(expected_filepaths)}")
    if df["filepath"].tolist() != expected_filepaths:
        raise ValueError("submission filepath order does not match input CSV")
    n = len(df)
    for row_idx, value in enumerate(df["neighbours"].astype(str).tolist()):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != topk:
            raise ValueError(f"row {row_idx + 1} contains {len(parts)} neighbours, expected {topk}")
        if any(part == "" for part in parts):
            raise ValueError(f"row {row_idx + 1} contains empty neighbour ids")
        if any(not part.isdigit() for part in parts):
            raise ValueError(f"row {row_idx + 1} contains non-integer neighbour ids")
        numbers = [int(part) for part in parts]
        if len(set(numbers)) != topk:
            raise ValueError(f"row {row_idx + 1} contains duplicate neighbour ids")
        if any(number < 0 or number >= n for number in numbers):
            raise ValueError(f"row {row_idx + 1} contains out-of-range neighbour ids")
        if row_idx in numbers:
            raise ValueError(f"row {row_idx + 1} contains self index")


def main() -> None:
    args = parse_args()
    import torch
    from retrieval import extract_embeddings, topk_indices_from_embeddings

    config = load_config(args.config)
    if args.data_root:
        config["paths"]["data_root"] = Path(args.data_root).resolve()
    if args.experiment_root:
        config["paths"]["experiment_root"] = Path(args.experiment_root).resolve()
    if args.batch_size > 0:
        config["evaluation"]["batch_size"] = int(args.batch_size)
    if args.retrieval_chunk_size > 0:
        config["evaluation"]["retrieval_chunk_size"] = int(args.retrieval_chunk_size)

    csv_path = Path(args.csv).resolve() if args.csv else config["paths"]["test_csv"]
    test_df = pd.read_csv(csv_path)
    if list(test_df.columns) != ["filepath"]:
        raise ValueError(f"Expected only filepath column in test CSV, got {list(test_df.columns)}")

    mode = args.mode or str(config["evaluation"]["primary_mode"])
    run_name = args.run_name or f"submission_pretrained_{mode}_{csv_path.stem}"
    run_root = ensure_dir(submissions_root(config) / run_name)
    write_resolved_config(config, run_root / "config_resolved.yaml")

    filepaths = test_df["filepath"].tolist()
    duration_lookup = load_duration_lookup_for_paths(config, filepaths)
    manifest = build_manifest(test_df, duration_lookup)
    manifest_path = run_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    feature_extractor, model, source_ref = load_pretrained_components(config, device)

    start_time = time.perf_counter()
    print(
        f"[build_submission] run={run_name} mode={mode} rows={len(manifest)} "
        f"batch_size={int(config['evaluation']['batch_size'])}",
        flush=True,
    )
    embeddings, _ = extract_embeddings(
        manifest=manifest,
        feature_extractor=feature_extractor,
        model=model,
        data_root=config["paths"]["data_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        mode=mode,
        chunk_sec=float(config["evaluation"]["chunk_sec"]),
        max_load_len_sec=float(config["evaluation"]["max_load_len_sec"]),
        batch_size=int(config["evaluation"]["batch_size"]),
        device=device,
        progress_every_rows=int(config["evaluation"].get("progress_every_rows", 0)),
        progress_label=run_name,
    )
    embedding_seconds = time.perf_counter() - start_time
    print(
        f"[build_submission] extraction_done run={run_name} seconds={embedding_seconds:.2f}",
        flush=True,
    )

    search_start = time.perf_counter()
    print(
        f"[build_submission] search_start run={run_name} topk={int(args.topk)}",
        flush=True,
    )
    indices = topk_indices_from_embeddings(
        embeddings=embeddings,
        topk=int(args.topk),
        chunk_size=int(config["evaluation"]["retrieval_chunk_size"]),
        device=device,
    )
    search_seconds = time.perf_counter() - search_start

    submission_path = run_root / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "neighbours"])
        writer.writeheader()
        for filepath, row_indices in zip(filepaths, indices.tolist(), strict=True):
            writer.writerow(
                {
                    "filepath": filepath,
                    "neighbours": ",".join(str(int(idx)) for idx in row_indices),
                }
            )

    submission_df = pd.read_csv(submission_path)
    validate_submission(submission_df, expected_filepaths=filepaths, topk=int(args.topk))

    embeddings_path = run_root / "embeddings.npy"
    if args.save_embeddings:
        np.save(embeddings_path, embeddings)

    summary = {
        "source": "pretrained",
        "source_ref": source_ref,
        "input_csv": str(csv_path),
        "mode": mode,
        "topk": int(args.topk),
        "rows": int(len(filepaths)),
        "submission": str(submission_path),
        "manifest": str(manifest_path),
        "embedding_seconds": embedding_seconds,
        "search_seconds": search_seconds,
        "total_seconds": embedding_seconds + search_seconds,
        "device": str(device),
        "git_sha": get_git_sha(config["project_root"]),
        "saved_embeddings": str(embeddings_path) if args.save_embeddings else "",
    }
    summary_path = run_root / "run_summary.json"
    write_json(summary_path, summary)

    maybe_log_mlflow(
        config=config,
        run_name=run_name,
        params={
            "stage": "submission_build",
            "source": "pretrained",
            "source_ref": source_ref,
            "input_csv": str(csv_path),
            "mode": mode,
            "topk": int(args.topk),
            "git_sha": summary["git_sha"],
        },
        metrics={
            "embedding_seconds": embedding_seconds,
            "search_seconds": search_seconds,
            "total_seconds": embedding_seconds + search_seconds,
            "rows": float(len(filepaths)),
        },
        artifacts=[
            submission_path,
            summary_path,
            manifest_path,
            run_root / "config_resolved.yaml",
        ],
        tags={"stage": "submission_build"},
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(submission_path)


if __name__ == "__main__":
    main()
