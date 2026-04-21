#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(__file__).resolve().parent / "ms42_release"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an MS42-style CAM++ class-aware retrieval submission."
    )
    parser.add_argument("--config", required=True, help="Path to CAM++ release YAML config.")
    parser.add_argument("--manifest-csv", default="", help="Input CSV with filepath column.")
    parser.add_argument("--template-csv", default="", help="Template CSV for validation.")
    parser.add_argument("--data-root", default="", help="Dataset root for relative filepaths.")
    parser.add_argument("--output-dir", default="", help="Output directory for run artifacts.")
    parser.add_argument("--run-id", default="", help="Run id used in artifact filenames.")
    parser.add_argument("--checkpoint-path", default="", help="CAM++ checkpoint path override.")
    parser.add_argument(
        "--encoder-backend",
        choices=("torch",),
        default="torch",
        help="Packaged backend. TensorRT is intentionally not vendored in this repo.",
    )
    parser.add_argument("--device", default="", help="Torch inference device.")
    parser.add_argument("--search-device", default="", help="Torch retrieval device.")
    parser.add_argument("--batch-size", type=int, default=0, help="Encoder batch override.")
    parser.add_argument("--frontend-workers", type=int, default=-1, help="Frontend workers.")
    parser.add_argument("--frontend-prefetch", type=int, default=-1, help="Frontend prefetch.")
    parser.add_argument("--search-batch-size", type=int, default=0, help="Search batch override.")
    parser.add_argument("--top-cache-k", type=int, default=0, help="Top-k cache width.")
    parser.add_argument("--output-top-k", type=int, default=10, help="Neighbours to write.")
    parser.add_argument(
        "--class-batch-size", type=int, default=0, help="Classifier batch override."
    )
    parser.add_argument("--class-top-k", type=int, default=0, help="Classifier top-k override.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return payload


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolved_str(value: str | Path) -> str:
    return str(resolve_path(value))


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))

    paths_cfg = config["paths"]
    run_cfg = config["run"]
    official_cfg = config["official_tail"]
    classifier_cfg = config["classifier"]
    graph_cfg = config["graph"]

    manifest_csv = resolve_path(args.manifest_csv or paths_cfg["test_csv"])
    template_csv = resolve_path(args.template_csv or args.manifest_csv or paths_cfg["test_csv"])
    data_root = resolve_path(args.data_root or paths_cfg["data_root"])
    output_dir = resolve_path(args.output_dir or paths_cfg["experiment_root"])
    run_id = args.run_id or str(run_cfg["run_id"])
    checkpoint_path = resolve_path(args.checkpoint_path or config["pretrained"]["weight_path"])
    device = args.device or str(run_cfg["device"])
    search_device = args.search_device or str(run_cfg["search_device"])

    batch_size = args.batch_size or int(official_cfg["batch_size"])
    frontend_workers = (
        args.frontend_workers
        if args.frontend_workers >= 0
        else int(official_cfg["frontend_workers"])
    )
    frontend_prefetch = (
        args.frontend_prefetch
        if args.frontend_prefetch >= 0
        else int(official_cfg["frontend_prefetch"])
    )
    search_batch_size = args.search_batch_size or int(official_cfg["search_batch_size"])
    output_top_k = int(args.output_top_k)
    manifest_row_count = _csv_row_count(manifest_csv)
    _validate_output_top_k(output_top_k, row_count=manifest_row_count)
    top_cache_k = _effective_top_cache_k(
        requested=args.top_cache_k or int(official_cfg["top_cache_k"]),
        output_top_k=output_top_k,
        row_count=manifest_row_count,
        graph_cfg=graph_cfg,
    )
    class_batch_size = args.class_batch_size or int(classifier_cfg["class_batch_size"])
    class_top_k = args.class_top_k or int(classifier_cfg["class_top_k"])

    base_run = f"{run_id}_base"
    classcache_id = f"{run_id}_classcache"
    base_out = output_dir / "base"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_out.mkdir(parents=True, exist_ok=True)

    commands = [
        (
            "official_tail",
            [
                sys.executable,
                str(RUNTIME_ROOT / "run_official_campp_tail.py"),
                "--checkpoint-path",
                str(checkpoint_path),
                "--manifest-csv",
                str(manifest_csv),
                "--template-csv",
                str(template_csv),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(base_out),
                "--experiment-id",
                base_run,
                "--encoder-backend",
                args.encoder_backend,
                "--device",
                device,
                "--search-device",
                search_device,
                "--batch-size",
                str(batch_size),
                "--frontend-workers",
                str(frontend_workers),
                "--frontend-prefetch",
                str(frontend_prefetch),
                "--search-batch-size",
                str(search_batch_size),
                "--top-cache-k",
                str(top_cache_k),
                "--output-top-k",
                str(output_top_k),
                "--mode",
                str(official_cfg["mode"]),
                "--eval-chunk-seconds",
                str(official_cfg["eval_chunk_seconds"]),
                "--segment-count",
                str(official_cfg["segment_count"]),
                "--long-file-threshold-seconds",
                str(official_cfg["long_file_threshold_seconds"]),
                "--pad-mode",
                str(official_cfg["pad_mode"]),
                "--skip-c4",
            ],
        ),
        (
            "classifier_cache",
            [
                sys.executable,
                str(RUNTIME_ROOT / "run_classifier_first_tail.py"),
                "--checkpoint-path",
                str(checkpoint_path),
                "--embeddings-path",
                str(base_out / f"embeddings_{base_run}.npy"),
                "--manifest-csv",
                str(manifest_csv),
                "--output-dir",
                str(output_dir),
                "--experiment-id",
                classcache_id,
                "--device",
                device,
                "--class-batch-size",
                str(class_batch_size),
                "--class-top-k",
                str(class_top_k),
                "--class-scale",
                str(classifier_cfg["class_scale"]),
                "--class-cache-only",
            ],
        ),
        (
            "class_aware_graph",
            [
                sys.executable,
                str(RUNTIME_ROOT / "run_class_aware_graph_tail.py"),
                "--indices-path",
                str(base_out / f"indices_{base_run}_top{top_cache_k}.npy"),
                "--scores-path",
                str(base_out / f"scores_{base_run}_top{top_cache_k}.npy"),
                "--class-indices-path",
                str(output_dir / f"class_indices_{classcache_id}_top{class_top_k}.npy"),
                "--class-probs-path",
                str(output_dir / f"class_probs_{classcache_id}_top{class_top_k}.npy"),
                "--manifest-csv",
                str(manifest_csv),
                "--template-csv",
                str(template_csv),
                "--output-dir",
                str(output_dir),
                "--experiment-id",
                run_id,
                "--output-top-k",
                str(output_top_k),
                "--class-overlap-top-k",
                str(graph_cfg["class_overlap_top_k"]),
                "--class-overlap-weight",
                str(graph_cfg["class_overlap_weight"]),
                "--same-top1-bonus",
                str(graph_cfg["same_top1_bonus"]),
                "--same-query-topk-bonus",
                str(graph_cfg["same_query_topk_bonus"]),
                "--edge-top",
                str(graph_cfg["edge_top"]),
                "--reciprocal-top",
                str(graph_cfg["reciprocal_top"]),
                "--rank-top",
                str(graph_cfg["rank_top"]),
                "--iterations",
                str(graph_cfg["iterations"]),
                "--label-min-size",
                str(graph_cfg["label_min_size"]),
                "--label-max-size",
                str(graph_cfg["label_max_size"]),
                "--label-min-candidates",
                str(graph_cfg["label_min_candidates"]),
                "--shared-top",
                str(graph_cfg["shared_top"]),
                "--shared-min-count",
                str(graph_cfg["shared_min_count"]),
                "--reciprocal-bonus",
                str(graph_cfg["reciprocal_bonus"]),
                "--density-penalty",
                str(graph_cfg["density_penalty"]),
            ],
        ),
    ]
    if bool(official_cfg.get("force_embeddings", True)):
        commands[0][1].append("--force-embeddings")
    if bool(classifier_cfg.get("force_classifier", True)):
        commands[1][1].append("--force-classifier")

    timings: dict[str, float] = {}
    for stage_name, command in commands:
        _print_command(stage_name, command)
        if args.dry_run:
            continue
        started = time.perf_counter()
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        timings[f"{stage_name}_seconds"] = round(time.perf_counter() - started, 3)

    if args.dry_run:
        return

    final_submission = output_dir / f"submission_{run_id}.csv"
    root_submission = PROJECT_ROOT / "submission.csv"
    shutil.copyfile(final_submission, root_submission)

    validation_command = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--template-csv",
        str(template_csv),
        "--submission-csv",
        str(root_submission),
        "--output-json",
        str(output_dir / "submission_validation.json"),
        "--k",
        str(output_top_k),
    ]
    _print_command("validate", validation_command)
    started = time.perf_counter()
    subprocess.run(validation_command, check=True, cwd=PROJECT_ROOT)
    timings["validation_seconds"] = round(time.perf_counter() - started, 3)

    checksum = hashlib.sha256(root_submission.read_bytes()).hexdigest()
    (output_dir / "submission.sha256").write_text(
        f"{checksum}  submission.csv\n",
        encoding="utf-8",
    )
    timings["total_seconds"] = round(sum(timings.values()), 3)
    _write_speed_summary(output_dir / "speed_summary.txt", timings)
    summary = {
        "run_id": run_id,
        "submission": str(root_submission),
        "output_submission": str(final_submission),
        "output_top_k": output_top_k,
        "sha256": checksum,
        "timings": timings,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def _print_command(stage_name: str, command: list[str]) -> None:
    print(f"[campp-ms42] {stage_name}: {' '.join(command)}", flush=True)


def _write_speed_summary(path: Path, timings: dict[str, float]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in timings.items()), encoding="utf-8")


def _csv_row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _validate_output_top_k(output_top_k: int, *, row_count: int) -> None:
    if output_top_k <= 0:
        raise ValueError("--output-top-k must be positive.")
    if output_top_k > row_count - 1:
        raise ValueError(
            f"--output-top-k={output_top_k} requires at least {output_top_k + 1} "
            f"manifest rows, got {row_count}."
        )


def _effective_top_cache_k(
    *,
    requested: int,
    output_top_k: int,
    row_count: int,
    graph_cfg: dict[str, Any],
) -> int:
    if requested <= 0:
        raise ValueError("--top-cache-k must be positive.")
    max_neighbours = row_count - 1
    required = max(
        output_top_k,
        int(graph_cfg["edge_top"]),
        int(graph_cfg["reciprocal_top"]),
        int(graph_cfg["rank_top"]),
        int(graph_cfg["shared_top"]),
    )
    return min(max(requested, required), max_neighbours)


if __name__ == "__main__":
    main()
