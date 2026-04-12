"""Fuse cached backbone embeddings and apply the current C4 retrieval tail."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.community import (
    LabelPropagationConfig,
    evaluate_labelled_topk,
    exact_topk,
    label_propagation_rerank,
    write_submission,
)
from kryptonite.eda.fusion import RankScoreFusionConfig, fuse_topk_rank_score
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    print(
        f"[fusion-c4] start experiment={args.experiment_id} rows={manifest.height} "
        f"output_dir={output_dir}",
        flush=True,
    )

    left_embeddings = _load_embeddings(Path(args.left_embeddings), "left")
    right_embeddings = _load_embeddings(Path(args.right_embeddings), "right")
    if left_embeddings.shape[0] != manifest.height or right_embeddings.shape[0] != manifest.height:
        raise ValueError(
            "Embedding row counts must match manifest rows: "
            f"left={left_embeddings.shape}, right={right_embeddings.shape}, "
            f"manifest={manifest.height}"
        )

    source_top_k = max(args.source_top_k, args.top_cache_k)
    started = time.perf_counter()
    left_indices, left_scores = exact_topk(
        left_embeddings,
        top_k=source_top_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    left_search_s = time.perf_counter() - started
    print(f"[fusion-c4] left exact_topk done seconds={left_search_s:.3f}", flush=True)

    started = time.perf_counter()
    right_indices, right_scores = exact_topk(
        right_embeddings,
        top_k=source_top_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    right_search_s = time.perf_counter() - started
    print(f"[fusion-c4] right exact_topk done seconds={right_search_s:.3f}", flush=True)

    fusion_config = RankScoreFusionConfig(
        experiment_id=args.experiment_id,
        left_name=args.left_name,
        right_name=args.right_name,
        left_weight=args.left_weight,
        right_weight=args.right_weight,
        source_top_k=source_top_k,
        output_top_k=args.top_cache_k,
        rank_weight=args.rank_weight,
        score_z_weight=args.score_z_weight,
    )
    fused_indices, fused_scores, fusion_meta = fuse_topk_rank_score(
        left_indices=left_indices,
        left_scores=left_scores,
        right_indices=right_indices,
        right_scores=right_scores,
        config=fusion_config,
    )
    np.save(output_dir / f"indices_{args.experiment_id}.npy", fused_indices)
    np.save(output_dir / f"scores_{args.experiment_id}.npy", fused_scores)
    print("[fusion-c4] fused top-k cache written", flush=True)

    label_config = LabelPropagationConfig(
        experiment_id=args.experiment_id,
        edge_top=args.edge_top,
        reciprocal_top=args.reciprocal_top,
        rank_top=args.rank_top,
        iterations=args.iterations,
        label_min_size=args.label_min_size,
        label_max_size=args.label_max_size,
        label_min_candidates=args.label_min_candidates,
        shared_top=args.shared_top,
        shared_min_count=args.shared_min_count,
        reciprocal_bonus=args.reciprocal_bonus,
        density_penalty=args.density_penalty,
    )
    started = time.perf_counter()
    top_indices, top_scores, label_meta = label_propagation_rerank(
        indices=fused_indices,
        scores=fused_scores,
        config=label_config,
        top_k=10,
    )
    rerank_s = time.perf_counter() - started
    print(f"[fusion-c4] label propagation done seconds={rerank_s:.3f}", flush=True)

    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "left_embeddings": args.left_embeddings,
        "right_embeddings": args.right_embeddings,
        "manifest_csv": args.manifest_csv,
        "left_search_s": round(left_search_s, 6),
        "right_search_s": round(right_search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "top1_score_mean": float(top_scores[:, 0].mean()),
        "top10_mean_score_mean": float(top_scores.mean()),
        "indegree_gini_10": gini(np.bincount(top_indices.ravel(), minlength=manifest.height)),
        "indegree_max_10": int(np.bincount(top_indices.ravel(), minlength=manifest.height).max()),
        **fusion_meta,
        **label_meta,
        "fusion_config": asdict(fusion_config),
        "label_config": asdict(label_config),
    }

    if args.template_csv:
        submission_path = output_dir / f"submission_{args.experiment_id}.csv"
        write_submission(manifest=manifest, top_indices=top_indices, output_csv=submission_path)
        validation = validate_submission(
            template_csv=Path(args.template_csv),
            submission_csv=submission_path,
        )
        validation_path = output_dir / f"submission_{args.experiment_id}_validation.json"
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.update(
            {
                "submission_path": str(submission_path),
                "validator_passed": bool(validation["passed"]),
            }
        )
    if _has_labels(manifest):
        query_eval, summary = evaluate_labelled_topk(
            experiment_id=args.experiment_id,
            top_indices=top_indices,
            top_scores=top_scores,
            manifest=manifest,
            query_only=True,
        )
        query_eval.write_parquet(output_dir / f"{args.experiment_id}_query_eval.parquet")
        rows.update(summary)

    (output_dir / f"{args.experiment_id}_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pl.DataFrame([{key: _csv_value(value) for key, value in rows.items()}]).write_csv(
        output_dir / f"{args.experiment_id}_summary.csv"
    )
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


def _load_embeddings(path: Path, label: str) -> np.ndarray:
    print(f"[fusion-c4] load {label} embeddings path={path}", flush=True)
    values = np.load(path)
    if values.ndim != 2:
        raise ValueError(f"{label} embeddings must be a 2D array")
    return np.asarray(values, dtype=np.float32)


def _has_labels(manifest: pl.DataFrame) -> bool:
    return "speaker_id" in manifest.columns and manifest["speaker_id"].drop_nulls().len() > 0


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-embeddings", required=True)
    parser.add_argument("--right-embeddings", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--left-name", default="eres2netv2")
    parser.add_argument("--right-name", default="campp")
    parser.add_argument("--left-weight", type=float, default=0.75)
    parser.add_argument("--right-weight", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--score-z-weight", type=float, default=0.15)
    parser.add_argument("--source-top-k", type=int, default=200)
    parser.add_argument("--top-cache-k", type=int, default=100)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--edge-top", type=int, default=10)
    parser.add_argument("--reciprocal-top", type=int, default=20)
    parser.add_argument("--rank-top", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--label-min-size", type=int, default=5)
    parser.add_argument("--label-max-size", type=int, default=120)
    parser.add_argument("--label-min-candidates", type=int, default=3)
    parser.add_argument("--shared-top", type=int, default=20)
    parser.add_argument("--shared-min-count", type=int, default=0)
    parser.add_argument("--reciprocal-bonus", type=float, default=0.03)
    parser.add_argument("--density-penalty", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    main()
