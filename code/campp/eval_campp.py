#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from common import (
    build_campp_embedding_model,
    ensure_dir,
    get_git_sha,
    load_config,
    load_embedding_checkpoint,
    load_pretrained_embedding,
    manifest_path_for_split,
    maybe_log_mlflow,
    prepared_root,
    runs_root,
    write_json,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained or finetuned CAM++ on speaker retrieval."
    )
    parser.add_argument("--config", required=True, help="Path to CAM++ YAML config.")
    parser.add_argument("--checkpoint", default="", help="Optional finetuned checkpoint path.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="Prepared split to evaluate.",
    )
    parser.add_argument("--manifest", default="", help="Optional manifest path. Overrides --split.")
    parser.add_argument("--modes", default="", help="Comma-separated evaluation modes.")
    parser.add_argument(
        "--best-mode-from",
        default="",
        help=(
            "Optional run_summary.json path. If set and --modes is empty, uses "
            "best_mode from that file."
        ),
    )
    parser.add_argument("--run-name", default="", help="Optional MLflow/display run name.")
    return parser.parse_args()


def load_best_mode(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_mode = str(payload.get("best_mode") or "").strip()
    if not best_mode:
        raise ValueError(f"best_mode is missing in {path}")
    return best_mode


def default_run_name(source: str, split: str, mode_count: int, run_stamp: str) -> str:
    if source == "pretrained":
        if split == "validation" and mode_count > 1:
            return "pretrained_val_modes"
        if split == "test":
            return "pretrained_test_bestmode"
        return f"baseline_pretrained_{split}_{run_stamp}"
    if split == "test":
        return "checkpoint_test_eval"
    return f"eval_checkpoint_{split}_{run_stamp}"


def main() -> None:
    args = parse_args()

    import torch
    from retrieval import extract_embeddings, retrieval_metrics_from_embeddings

    config = load_config(args.config)

    split_label = args.split
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else manifest_path_for_split(config, split_label)
    )
    split_summary_path = prepared_root(config) / "split_summary.json"
    manifest = pd.read_csv(manifest_path)
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    source = "checkpoint" if args.checkpoint else "pretrained"

    if args.modes:
        modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    elif args.best_mode_from:
        modes = [load_best_mode(Path(args.best_mode_from).resolve())]
    else:
        modes = list(config["evaluation"]["compare_modes"])

    run_name = args.run_name or default_run_name(source, split_label, len(modes), run_stamp)
    run_root = ensure_dir(runs_root(config) / run_name)
    write_resolved_config(config, run_root / "config_resolved.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_campp_embedding_model(config).to(device)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint_state = load_embedding_checkpoint(checkpoint_path, model)
        source_ref = str(checkpoint_path)
    else:
        weight_path = load_pretrained_embedding(config, model)
        checkpoint_state = {}
        source_ref = str(weight_path)

    rows: list[dict[str, object]] = []
    metrics_for_mlflow: dict[str, float] = {}
    best_mode = None
    best_p10 = float("-inf")

    for mode in modes:
        embeddings, labels = extract_embeddings(
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
        metrics = retrieval_metrics_from_embeddings(
            embeddings=embeddings,
            labels=labels,
            ks=config["evaluation"]["ks"],
            chunk_size=int(config["evaluation"]["retrieval_chunk_size"]),
            device=device,
        )
        p10 = metrics["precision@10"]
        if p10 > best_p10:
            best_p10 = p10
            best_mode = mode
        metrics_for_mlflow.update({f"{mode}.{key}": value for key, value in metrics.items()})
        rows.append({"mode": mode, **metrics})

    metrics_path = run_root / "metrics.csv"
    metric_columns = ["mode", *list(rows[0].keys())[1:]] if rows else ["mode"]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": source,
        "source_ref": source_ref,
        "split": split_label if not args.manifest else "custom",
        "manifest": str(manifest_path),
        "modes": modes,
        "best_mode": best_mode,
        "best_precision@10": best_p10,
        "rows": len(manifest),
        "unique_speakers": int(manifest["spk"].nunique()),
        "git_sha": get_git_sha(config["project_root"]),
        "checkpoint_metrics": checkpoint_state.get("metrics", {})
        if isinstance(checkpoint_state, dict)
        else {},
    }
    summary_path = run_root / "run_summary.json"
    write_json(summary_path, summary)

    maybe_log_mlflow(
        config=config,
        run_name=run_name,
        params={
            "stage": "baseline_pretrained" if source == "pretrained" else "eval_checkpoint",
            "source": source,
            "source_ref": source_ref,
            "split": summary["split"],
            "manifest": str(manifest_path),
            "modes": ",".join(modes),
            "git_sha": summary["git_sha"],
        },
        metrics=metrics_for_mlflow,
        artifacts=[
            metrics_path,
            summary_path,
            manifest_path,
            split_summary_path,
            run_root / "config_resolved.yaml",
        ],
        tags={"stage": "baseline_pretrained" if source == "pretrained" else "eval_checkpoint"},
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(metrics_path)


if __name__ == "__main__":
    main()
