"""Apply class-aware score adjustment before the graph/community retrieval tail."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.classifier_first import (
    ClassAdjustedScoreConfig,
    class_adjusted_topk,
)
from kryptonite.eda.community import (
    LabelPropagationConfig,
    evaluate_labelled_topk,
    label_propagation_rerank,
    write_submission,
)
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pl.read_csv(args.manifest_csv)
    indices = np.load(args.indices_path)
    scores = np.load(args.scores_path)
    top_class_indices = np.load(args.class_indices_path)
    top_class_probs = np.load(args.class_probs_path)
    _validate_inputs(indices, scores, top_class_indices, top_class_probs, manifest.height)
    widths = _resolve_graph_widths(
        args=args,
        row_count=manifest.height,
        cache_width=indices.shape[1],
    )
    print(
        f"[class-aware-graph] start experiment={args.experiment_id} rows={manifest.height} "
        f"top_cache_k={indices.shape[1]} output_top_k={args.output_top_k} "
        f"output_dir={output_dir}",
        flush=True,
    )

    score_config = ClassAdjustedScoreConfig(
        class_overlap_top_k=args.class_overlap_top_k,
        class_overlap_weight=args.class_overlap_weight,
        same_top1_bonus=args.same_top1_bonus,
        same_query_topk_bonus=args.same_query_topk_bonus,
    )
    started = time.perf_counter()
    adjusted_indices, adjusted_scores, score_meta = class_adjusted_topk(
        indices=indices,
        scores=scores,
        top_class_indices=top_class_indices,
        top_class_probs=top_class_probs,
        config=score_config,
    )
    adjust_s = time.perf_counter() - started
    print(f"[class-aware-graph] score adjustment done seconds={adjust_s:.3f}", flush=True)

    label_config = LabelPropagationConfig(
        experiment_id=args.experiment_id,
        edge_top=widths["edge_top"],
        reciprocal_top=widths["reciprocal_top"],
        rank_top=widths["rank_top"],
        iterations=args.iterations,
        label_min_size=args.label_min_size,
        label_max_size=args.label_max_size,
        label_min_candidates=args.label_min_candidates,
        shared_top=widths["shared_top"],
        shared_min_count=args.shared_min_count,
        reciprocal_bonus=args.reciprocal_bonus,
        density_penalty=args.density_penalty,
    )
    started = time.perf_counter()
    top_indices, top_scores, label_meta = label_propagation_rerank(
        indices=adjusted_indices,
        scores=adjusted_scores,
        config=label_config,
        top_k=args.output_top_k,
    )
    rerank_s = time.perf_counter() - started
    print(f"[class-aware-graph] labelprop rerank done seconds={rerank_s:.3f}", flush=True)

    indegree = np.bincount(top_indices.ravel(), minlength=manifest.height)
    top10_limit = min(10, top_scores.shape[1])
    top10_scores = top_scores[:, :top10_limit]
    indegree10 = np.bincount(top_indices[:, :top10_limit].ravel(), minlength=manifest.height)
    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "manifest_csv": args.manifest_csv,
        "indices_path": args.indices_path,
        "scores_path": args.scores_path,
        "class_indices_path": args.class_indices_path,
        "class_probs_path": args.class_probs_path,
        "adjust_s": round(adjust_s, 6),
        "rerank_s": round(rerank_s, 6),
        "output_top_k": args.output_top_k,
        "top1_score_mean": float(top_scores[:, 0].mean()),
        "top10_mean_score_mean": float(top10_scores.mean()),
        "topk_mean_score_mean": float(top_scores.mean()),
        "indegree_gini_10": gini(indegree10),
        "indegree_max_10": int(indegree10.max()),
        "indegree_gini_k": gini(indegree),
        "indegree_max_k": int(indegree.max()),
        "class_adjusted_score_config": asdict(score_config),
        **score_meta,
        **label_meta,
    }

    if args.template_csv:
        submission_path = output_dir / f"submission_{args.experiment_id}.csv"
        write_submission(manifest=manifest, top_indices=top_indices, output_csv=submission_path)
        validation = validate_submission(
            template_csv=Path(args.template_csv),
            submission_csv=submission_path,
            k=args.output_top_k,
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


def _validate_inputs(
    indices: np.ndarray,
    scores: np.ndarray,
    top_class_indices: np.ndarray,
    top_class_probs: np.ndarray,
    manifest_rows: int,
) -> None:
    if indices.shape != scores.shape:
        raise ValueError(f"indices/scores shape mismatch: {indices.shape} != {scores.shape}")
    if indices.ndim != 2:
        raise ValueError("indices and scores must be 2D arrays")
    if indices.shape[0] != manifest_rows:
        raise ValueError(
            "top-k cache row count must match manifest rows: "
            f"cache={indices.shape[0]} manifest={manifest_rows}"
        )
    if top_class_indices.shape != top_class_probs.shape:
        raise ValueError("class indices/probs shape mismatch")
    if top_class_indices.shape[0] != manifest_rows:
        raise ValueError("class cache row count must match manifest rows")


def _resolve_graph_widths(
    *,
    args: argparse.Namespace,
    row_count: int,
    cache_width: int,
) -> dict[str, int]:
    max_neighbours = row_count - 1
    if args.output_top_k <= 0:
        raise ValueError("--output-top-k must be positive.")
    if args.output_top_k > max_neighbours:
        raise ValueError(
            f"--output-top-k={args.output_top_k} requires at least "
            f"{args.output_top_k + 1} manifest rows, got {row_count}."
        )
    raw_widths = {
        "edge_top": args.edge_top,
        "reciprocal_top": args.reciprocal_top,
        "rank_top": args.rank_top,
    }
    for name, value in raw_widths.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.shared_top < 0:
        raise ValueError("--shared-top must be non-negative.")
    if args.shared_min_count > 0 and args.shared_top == 0:
        raise ValueError("--shared-top must be positive when --shared-min-count is positive.")
    edge_top = min(args.edge_top, max_neighbours, cache_width)
    reciprocal_top = min(args.reciprocal_top, max_neighbours, cache_width)
    rank_top = min(max(args.rank_top, args.output_top_k), max_neighbours, cache_width)
    shared_top = min(args.shared_top, max_neighbours, cache_width)
    required_cache_width = max(args.output_top_k, edge_top, reciprocal_top, rank_top, shared_top)
    if cache_width < required_cache_width:
        raise ValueError(
            f"cached top-k width {cache_width} is smaller than required "
            f"{required_cache_width}; increase --top-cache-k in the producing stage."
        )
    return {
        "edge_top": edge_top,
        "reciprocal_top": reciprocal_top,
        "rank_top": rank_top,
        "shared_top": shared_top,
    }


def _has_labels(manifest: pl.DataFrame) -> bool:
    return "speaker_id" in manifest.columns and manifest["speaker_id"].drop_nulls().len() > 0


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices-path", required=True)
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--class-indices-path", required=True)
    parser.add_argument("--class-probs-path", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--class-overlap-top-k", type=int, default=3)
    parser.add_argument("--class-overlap-weight", type=float, default=0.08)
    parser.add_argument("--same-top1-bonus", type=float, default=0.02)
    parser.add_argument("--same-query-topk-bonus", type=float, default=0.01)
    parser.add_argument("--output-top-k", type=int, default=10)
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
