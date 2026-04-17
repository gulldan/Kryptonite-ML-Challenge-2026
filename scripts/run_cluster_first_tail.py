"""Apply cluster-first graph/community retrieval to cached embeddings or top-k caches."""

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
    ClusterFirstConfig,
    cluster_first_rerank,
    evaluate_labelled_topk,
    exact_topk,
    write_cluster_assignments,
    write_submission,
)
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    print(
        f"[cluster-first] start experiment={args.experiment_id} rows={manifest.height} "
        f"output_dir={output_dir}",
        flush=True,
    )

    started = time.perf_counter()
    indices, scores, search_meta = _load_or_build_top_cache(args, manifest, output_dir)
    search_s = time.perf_counter() - started
    print(f"[cluster-first] top-k cache ready seconds={search_s:.3f}", flush=True)

    config = ClusterFirstConfig(
        experiment_id=args.experiment_id,
        edge_top=args.edge_top,
        reciprocal_top=args.reciprocal_top,
        rank_top=args.rank_top,
        iterations=args.iterations,
        cluster_min_size=args.cluster_min_size,
        cluster_max_size=args.cluster_max_size,
        cluster_min_candidates=args.cluster_min_candidates,
        shared_top=args.shared_top,
        shared_min_count=args.shared_min_count,
        reciprocal_bonus=args.reciprocal_bonus,
        density_penalty=args.density_penalty,
        edge_score_quantile=args.edge_score_quantile,
        edge_min_score=args.edge_min_score,
        shared_weight=args.shared_weight,
        rank_weight=args.cluster_rank_weight,
        self_weight=args.self_weight,
        label_size_penalty=args.label_size_penalty,
        split_oversized=not args.no_split_oversized,
        split_edge_top=args.split_edge_top,
    )
    started = time.perf_counter()
    top_indices, top_scores, labels, cluster_meta = cluster_first_rerank(
        indices=indices,
        scores=scores,
        config=config,
        top_k=10,
    )
    rerank_s = time.perf_counter() - started
    print(f"[cluster-first] rerank done seconds={rerank_s:.3f}", flush=True)

    cluster_path = output_dir / f"clusters_{args.experiment_id}.csv"
    write_cluster_assignments(manifest=manifest, labels=labels, output_csv=cluster_path)
    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "manifest_csv": args.manifest_csv,
        "embeddings_path": args.embeddings_path,
        "indices_path": args.indices_path,
        "scores_path": args.scores_path,
        "search_s": round(search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "top1_score_mean": float(top_scores[:, 0].mean()),
        "top10_mean_score_mean": float(top_scores.mean()),
        "indegree_gini_10": gini(np.bincount(top_indices.ravel(), minlength=manifest.height)),
        "indegree_max_10": int(np.bincount(top_indices.ravel(), minlength=manifest.height).max()),
        "cluster_path": str(cluster_path),
        "cluster_first_config": asdict(config),
        **search_meta,
        **cluster_meta,
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


def _load_or_build_top_cache(
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices_path = Path(args.indices_path) if args.indices_path else None
    scores_path = Path(args.scores_path) if args.scores_path else None
    if (
        indices_path is not None
        and scores_path is not None
        and indices_path.is_file()
        and scores_path.is_file()
        and not args.force_search
    ):
        print(
            f"[cluster-first] load cached top-k indices={indices_path} scores={scores_path}",
            flush=True,
        )
        indices = np.load(indices_path)
        scores = np.load(scores_path)
        _validate_top_cache(indices, scores, manifest.height, args.top_cache_k)
        return (
            indices,
            scores,
            {
                "top_cache_source": "cached",
                "top_cache_k": int(indices.shape[1]),
                "loaded_indices_path": str(indices_path),
                "loaded_scores_path": str(scores_path),
            },
        )

    if not args.embeddings_path:
        raise ValueError("--embeddings-path is required when top-k cache files are unavailable")
    embeddings_path = Path(args.embeddings_path)
    print(f"[cluster-first] load embeddings path={embeddings_path}", flush=True)
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[0] != manifest.height:
        raise ValueError(
            "embeddings must be a 2D array with one row per manifest row: "
            f"embeddings={embeddings.shape} manifest_rows={manifest.height}"
        )
    indices, scores = exact_topk(
        embeddings,
        top_k=args.top_cache_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    if args.write_top_cache:
        written_indices = output_dir / f"indices_{args.experiment_id}_top{args.top_cache_k}.npy"
        written_scores = output_dir / f"scores_{args.experiment_id}_top{args.top_cache_k}.npy"
        np.save(written_indices, indices)
        np.save(written_scores, scores)
        cache_meta = {
            "written_indices_path": str(written_indices),
            "written_scores_path": str(written_scores),
        }
    else:
        cache_meta = {}
    return (
        indices,
        scores,
        {
            "top_cache_source": "embeddings",
            "top_cache_k": int(indices.shape[1]),
            **cache_meta,
        },
    )


def _validate_top_cache(
    indices: np.ndarray,
    scores: np.ndarray,
    manifest_rows: int,
    requested_top_k: int,
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
    if indices.shape[1] < requested_top_k:
        raise ValueError(
            f"cached top-k width {indices.shape[1]} is smaller than requested {requested_top_k}"
        )


def _has_labels(manifest: pl.DataFrame) -> bool:
    return "speaker_id" in manifest.columns and manifest["speaker_id"].drop_nulls().len() > 0


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-path", default="")
    parser.add_argument("--indices-path", default="")
    parser.add_argument("--scores-path", default="")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--top-cache-k", type=int, default=300)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--write-top-cache", action="store_true")
    parser.add_argument("--edge-top", type=int, default=50)
    parser.add_argument("--reciprocal-top", type=int, default=100)
    parser.add_argument("--rank-top", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--cluster-min-size", type=int, default=11)
    parser.add_argument("--cluster-max-size", type=int, default=220)
    parser.add_argument("--cluster-min-candidates", type=int, default=6)
    parser.add_argument("--shared-top", type=int, default=50)
    parser.add_argument("--shared-min-count", type=int, default=2)
    parser.add_argument("--reciprocal-bonus", type=float, default=0.03)
    parser.add_argument("--density-penalty", type=float, default=0.02)
    parser.add_argument("--edge-score-quantile", type=float, default=None)
    parser.add_argument("--edge-min-score", type=float, default=None)
    parser.add_argument("--shared-weight", type=float, default=0.04)
    parser.add_argument("--cluster-rank-weight", type=float, default=0.02)
    parser.add_argument("--self-weight", type=float, default=0.0)
    parser.add_argument("--label-size-penalty", type=float, default=0.0)
    parser.add_argument("--no-split-oversized", action="store_true")
    parser.add_argument("--split-edge-top", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
