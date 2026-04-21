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
    build_campp_embedding_model,
    compute_duration_sec,
    ensure_dir,
    get_git_sha,
    load_config,
    load_embedding_checkpoint,
    load_pretrained_embedding,
    maybe_log_mlflow,
    submissions_root,
    write_json,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build submission.csv for CAM++ retrieval inference."
    )
    parser.add_argument("--config", required=True, help="Path to CAM++ YAML config.")
    parser.add_argument("--checkpoint", default="", help="Optional finetuned checkpoint path.")
    parser.add_argument(
        "--csv", default="", help="Path to test CSV. Defaults to config paths.test_csv."
    )
    parser.add_argument("--topk", type=int, default=10, help="Number of neighbours to write.")
    parser.add_argument(
        "--mode", default="", help="Evaluation mode. Defaults to config evaluation.primary_mode."
    )
    parser.add_argument(
        "--best-mode-from", default="", help="Optional run_summary.json path with best_mode."
    )
    parser.add_argument("--run-name", default="", help="Optional submission run name.")
    parser.add_argument("--save-embeddings", action="store_true", help="Also save embeddings.npy.")
    return parser.parse_args()


def load_best_mode(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_mode = str(payload.get("best_mode") or "").strip()
    if not best_mode:
        raise ValueError(f"best_mode is missing in {path}")
    return best_mode


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
    csv_path = Path(args.csv).resolve() if args.csv else config["paths"]["test_csv"]
    test_df = pd.read_csv(csv_path)
    if list(test_df.columns) != ["filepath"]:
        raise ValueError(f"Expected only filepath column in test CSV, got {list(test_df.columns)}")

    if args.best_mode_from:
        mode = load_best_mode(Path(args.best_mode_from).resolve())
    else:
        mode = args.mode or str(config["evaluation"]["primary_mode"])

    source = "checkpoint" if args.checkpoint else "pretrained"
    run_name = args.run_name or (f"submission_{source}_{mode}_{csv_path.stem}")
    run_root = ensure_dir(submissions_root(config) / run_name)
    write_resolved_config(config, run_root / "config_resolved.yaml")

    filepaths = test_df["filepath"].tolist()
    duration_lookup = load_duration_lookup_for_paths(config, filepaths)
    manifest = build_manifest(test_df, duration_lookup)
    manifest_path = run_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_campp_embedding_model(config).to(device)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        load_embedding_checkpoint(checkpoint_path, model)
        source_ref = str(checkpoint_path)
    else:
        weight_path = load_pretrained_embedding(config, model)
        source_ref = str(weight_path)

    start_time = time.perf_counter()
    embeddings, _ = extract_embeddings(
        manifest=manifest,
        model=model,
        data_root=config["paths"]["data_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        n_mels=int(config["model"]["n_mels"]),
        mode=mode,
        eval_chunk_sec=float(config["training"]["eval_chunk_sec"]),
        segment_count=int(config["evaluation"]["segment_count"]),
        long_file_threshold_sec=float(config["evaluation"]["long_file_threshold_sec"]),
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        pad_mode=str(config["training"]["short_clip_pad_mode"]),
    )
    embedding_seconds = time.perf_counter() - start_time

    search_start = time.perf_counter()
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
                {"filepath": filepath, "neighbours": ",".join(str(int(idx)) for idx in row_indices)}
            )

    submission_df = pd.read_csv(submission_path)
    validate_submission(submission_df, expected_filepaths=filepaths, topk=int(args.topk))

    embeddings_path = run_root / "embeddings.npy"
    if args.save_embeddings:
        np.save(embeddings_path, embeddings)

    summary = {
        "source": source,
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
            "source": source,
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
        artifacts=[submission_path, summary_path, manifest_path, run_root / "config_resolved.yaml"],
        tags={"stage": "submission_build"},
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(submission_path)


if __name__ == "__main__":
    main()
