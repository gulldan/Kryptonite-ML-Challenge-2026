#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from common import (
    ensure_dir,
    get_git_sha,
    load_config,
    load_pretrained_components,
    manifest_path_for_split,
    maybe_log_mlflow,
    prepared_root,
    runs_root,
    write_json,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WavLM speaker-retrieval baseline.")
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="Prepared split to evaluate.",
    )
    parser.add_argument("--manifest", default="", help="Optional manifest path. Overrides --split.")
    parser.add_argument("--modes", default="", help="Comma-separated evaluation modes.")
    parser.add_argument("--run-name", default="", help="Optional MLflow/display run name.")
    return parser.parse_args()


def default_run_name(split: str, mode_count: int, run_stamp: str) -> str:
    if split == "validation" and mode_count > 1:
        return "pretrained_validation_wavlm_modes"
    return f"pretrained_validation_wavlm_{split}_{run_stamp}"


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

    if args.modes:
        modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    else:
        modes = list(config["evaluation"]["compare_modes"])

    run_name = args.run_name or default_run_name(split_label, len(modes), run_stamp)
    run_root = ensure_dir(runs_root(config) / run_name)
    write_resolved_config(config, run_root / "config_resolved.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor, model, source_ref = load_pretrained_components(config, device)

    rows: list[dict[str, object]] = []
    metrics_for_mlflow: dict[str, float] = {}
    best_mode = None
    best_p10 = float("-inf")
    for mode in modes:
        embeddings, labels = extract_embeddings(
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
            progress_label=f"{run_name}:{mode}",
        )
        metrics = retrieval_metrics_from_embeddings(
            embeddings=embeddings,
            labels=labels,
            ks=config["evaluation"]["ks"],
            chunk_size=int(config["evaluation"]["retrieval_chunk_size"]),
            device=device,
        )
        precision_keys = sorted(
            (key for key in metrics if key.startswith("precision@")),
            key=lambda key: int(key.split("@", 1)[1]),
        )
        if not precision_keys:
            raise ValueError(f"No precision metrics were produced for mode={mode}")
        p10 = metrics["precision@10"] if "precision@10" in metrics else metrics[precision_keys[-1]]
        if p10 > best_p10:
            best_p10 = p10
            best_mode = mode
        rows.append({"mode": mode, **metrics})
        metrics_for_mlflow.update({f"{mode}.{key}": value for key, value in metrics.items()})

    metrics_path = run_root / "metrics.csv"
    metric_columns = ["mode", *list(rows[0].keys())[1:]] if rows else ["mode"]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": "pretrained",
        "source_ref": source_ref,
        "split": split_label if not args.manifest else "custom",
        "manifest": str(manifest_path),
        "modes": modes,
        "best_mode": best_mode,
        "best_precision@10": best_p10,
        "rows": len(manifest),
        "unique_speakers": int(manifest["spk"].nunique()),
        "git_sha": get_git_sha(config["project_root"]),
    }
    summary_path = run_root / "run_summary.json"
    write_json(summary_path, summary)

    maybe_log_mlflow(
        config=config,
        run_name=run_name,
        params={
            "stage": "baseline_pretrained",
            "source": "pretrained",
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
        tags={"stage": "baseline_pretrained"},
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(metrics_path)


if __name__ == "__main__":
    main()
