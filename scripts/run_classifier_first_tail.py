"""Build class-aware retrieval from cached embeddings and a trained classifier head."""

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
    ClassFirstConfig,
    class_first_rerank,
    l2_normalize_rows,
)
from kryptonite.eda.community import (
    evaluate_labelled_topk,
    exact_topk,
    write_submission,
)
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission
from kryptonite.models.campp.losses import CosineClassifier


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pl.read_csv(args.manifest_csv)
    embeddings = np.load(args.embeddings_path).astype(np.float32, copy=False)
    if args.max_rows > 0:
        manifest = manifest.head(args.max_rows)
        embeddings = embeddings[: args.max_rows]
    if embeddings.ndim != 2 or embeddings.shape[0] != manifest.height:
        raise ValueError(
            "embeddings must be a 2D array with one row per manifest row: "
            f"embeddings={embeddings.shape} manifest_rows={manifest.height}"
        )
    embeddings = l2_normalize_rows(embeddings)
    print(
        f"[classifier-first] start experiment={args.experiment_id} rows={manifest.height} "
        f"embedding_dim={embeddings.shape[1]} output_dir={output_dir}",
        flush=True,
    )

    started = time.perf_counter()
    top_class_indices, top_class_probs, classifier_meta = _load_or_compute_class_topk(
        args=args,
        embeddings=embeddings,
        output_dir=output_dir,
    )
    class_s = time.perf_counter() - started
    print(f"[classifier-first] class top-k ready seconds={class_s:.3f}", flush=True)

    started = time.perf_counter()
    fallback_indices, fallback_scores, search_meta = _load_or_build_top_cache(
        args=args,
        embeddings=embeddings,
        output_dir=output_dir,
    )
    search_s = time.perf_counter() - started
    print(f"[classifier-first] embedding top-k ready seconds={search_s:.3f}", flush=True)

    config = ClassFirstConfig(
        top_k=args.output_top_k,
        min_class_candidates=args.min_class_candidates,
        class_fallback_k=args.class_fallback_k,
        max_class_candidates=args.max_class_candidates,
        embedding_weight=args.embedding_weight,
        class_overlap_weight=args.class_overlap_weight,
        same_top1_bonus=args.same_top1_bonus,
        fallback_rank_bonus=args.fallback_rank_bonus,
        bucket_backfill=not args.no_bucket_backfill,
    )
    started = time.perf_counter()
    top_indices, top_scores, rerank_meta = class_first_rerank(
        embeddings=embeddings,
        top_class_indices=top_class_indices,
        top_class_probs=top_class_probs,
        fallback_indices=fallback_indices,
        fallback_scores=fallback_scores,
        config=config,
    )
    rerank_s = time.perf_counter() - started
    print(f"[classifier-first] rerank done seconds={rerank_s:.3f}", flush=True)

    indegree = np.bincount(top_indices.ravel(), minlength=manifest.height)
    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "checkpoint_path": args.checkpoint_path,
        "embeddings_path": args.embeddings_path,
        "manifest_csv": args.manifest_csv,
        "class_s": round(class_s, 6),
        "search_s": round(search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "top1_score_mean": float(top_scores[:, 0].mean()),
        "top10_mean_score_mean": float(top_scores.mean()),
        "indegree_gini_10": gini(indegree),
        "indegree_max_10": int(indegree.max()),
        "class_first_config": asdict(config),
        **classifier_meta,
        **search_meta,
        **rerank_meta,
    }

    if args.template_csv and args.max_rows <= 0:
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


def _load_or_compute_class_topk(
    *,
    args: argparse.Namespace,
    embeddings: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices_path = output_dir / f"class_indices_{args.experiment_id}_top{args.class_top_k}.npy"
    probs_path = output_dir / f"class_probs_{args.experiment_id}_top{args.class_top_k}.npy"
    if indices_path.is_file() and probs_path.is_file() and not args.force_classifier:
        return (
            np.load(indices_path),
            np.load(probs_path),
            {
                "class_cache_source": "cached",
                "class_indices_path": str(indices_path),
                "class_probs_path": str(probs_path),
            },
        )

    import torch

    classifier, checkpoint_meta = _load_classifier_from_checkpoint(
        checkpoint_path=Path(args.checkpoint_path),
        embedding_dim=embeddings.shape[1],
        torch=torch,
    )
    device = _resolve_device(args.device, torch=torch)
    classifier = classifier.to(device)
    classifier.eval()
    top_indices = np.empty((embeddings.shape[0], args.class_top_k), dtype=np.int64)
    top_probs = np.empty((embeddings.shape[0], args.class_top_k), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], args.class_batch_size):
            end = min(start + args.class_batch_size, embeddings.shape[0])
            batch = torch.from_numpy(embeddings[start:end]).to(device=device, dtype=torch.float32)
            logits = classifier(batch) * args.class_scale
            probs = torch.softmax(logits, dim=1)
            values, indices = torch.topk(probs, k=args.class_top_k, dim=1)
            top_indices[start:end] = indices.cpu().numpy()
            top_probs[start:end] = values.cpu().numpy()
            if start == 0 or end == embeddings.shape[0] or end % (args.class_batch_size * 20) == 0:
                print(
                    f"[classifier-first] class rows={end}/{embeddings.shape[0]} "
                    f"pct={100.0 * end / embeddings.shape[0]:.1f}",
                    flush=True,
                )
    np.save(indices_path, top_indices)
    np.save(probs_path, top_probs)
    return (
        top_indices,
        top_probs,
        {
            "class_cache_source": "computed",
            "class_indices_path": str(indices_path),
            "class_probs_path": str(probs_path),
            "class_top_k": args.class_top_k,
            "class_scale": args.class_scale,
            **checkpoint_meta,
        },
    )


def _load_classifier_from_checkpoint(
    *,
    checkpoint_path: Path,
    embedding_dim: int,
    torch: Any,
) -> tuple[CosineClassifier, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain an object payload.")
    state = payload.get("classifier_state_dict")
    if not isinstance(state, dict) or "weight" not in state:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing `classifier_state_dict`.")
    baseline_config = payload.get("baseline_config")
    objective = baseline_config.get("objective", {}) if isinstance(baseline_config, dict) else {}
    classifier = CosineClassifier(
        embedding_dim,
        num_classes=int(state["weight"].shape[0]),
        num_blocks=int(objective.get("classifier_blocks", 0)),
        hidden_dim=int(objective.get("classifier_hidden_dim", 512)),
    )
    classifier.load_state_dict(state)
    speaker_to_index = payload.get("speaker_to_index")
    speaker_count = len(speaker_to_index) if isinstance(speaker_to_index, dict) else None
    return (
        classifier,
        {
            "classifier_class_count": int(state["weight"].shape[0]),
            "classifier_embedding_dim": int(embedding_dim),
            "speaker_to_index_count": speaker_count,
        },
    )


def _load_or_build_top_cache(
    *,
    args: argparse.Namespace,
    embeddings: np.ndarray,
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
        indices = np.load(indices_path)
        scores = np.load(scores_path)
        _validate_top_cache(indices, scores, embeddings.shape[0], args.top_cache_k)
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

    indices, scores = exact_topk(
        embeddings,
        top_k=args.top_cache_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    cache_meta: dict[str, Any] = {}
    if args.write_top_cache:
        written_indices = output_dir / f"indices_{args.experiment_id}_top{args.top_cache_k}.npy"
        written_scores = output_dir / f"scores_{args.experiment_id}_top{args.top_cache_k}.npy"
        np.save(written_indices, indices)
        np.save(written_scores, scores)
        cache_meta = {
            "written_indices_path": str(written_indices),
            "written_scores_path": str(written_scores),
        }
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
    row_count: int,
    requested_top_k: int,
) -> None:
    if indices.shape != scores.shape:
        raise ValueError(f"indices/scores shape mismatch: {indices.shape} != {scores.shape}")
    if indices.ndim != 2:
        raise ValueError("indices and scores must be 2D arrays")
    if indices.shape[0] != row_count:
        raise ValueError(f"top-k cache row count must match embeddings: {indices.shape[0]}")
    if indices.shape[1] < requested_top_k:
        raise ValueError(
            f"cached top-k width {indices.shape[1]} is smaller than requested {requested_top_k}"
        )


def _resolve_device(device: str, *, torch: Any) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _has_labels(manifest: pl.DataFrame) -> bool:
    return "speaker_id" in manifest.columns and manifest["speaker_id"].drop_nulls().len() > 0


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--indices-path", default="")
    parser.add_argument("--scores-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--class-batch-size", type=int, default=4096)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--class-top-k", type=int, default=5)
    parser.add_argument("--class-scale", type=float, default=32.0)
    parser.add_argument("--top-cache-k", type=int, default=100)
    parser.add_argument("--output-top-k", type=int, default=10)
    parser.add_argument("--min-class-candidates", type=int, default=6)
    parser.add_argument("--class-fallback-k", type=int, default=3)
    parser.add_argument("--max-class-candidates", type=int, default=600)
    parser.add_argument("--embedding-weight", type=float, default=0.50)
    parser.add_argument("--class-overlap-weight", type=float, default=0.35)
    parser.add_argument("--same-top1-bonus", type=float, default=0.15)
    parser.add_argument("--fallback-rank-bonus", type=float, default=0.03)
    parser.add_argument("--no-bucket-backfill", action="store_true")
    parser.add_argument("--write-top-cache", action="store_true")
    parser.add_argument("--force-classifier", action="store_true")
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
