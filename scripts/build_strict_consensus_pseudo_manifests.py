"""Build strict-consensus public pseudo-label manifests from multiple public rankings."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.community import ClusterFirstConfig, cluster_first_rerank


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pl.read_csv(args.public_manifest_csv)
    teacher_indices = np.load(args.teacher_indices)
    teacher_scores = np.load(args.teacher_scores)
    _validate_top_cache(teacher_indices, teacher_scores, manifest.height, "teacher")

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
    _, _, labels, cluster_meta = cluster_first_rerank(
        indices=teacher_indices,
        scores=teacher_scores,
        config=config,
        top_k=10,
    )
    cluster_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    margins = _teacher_margins(
        teacher_scores,
        best_rank=args.teacher_margin_best_rank,
        compare_rank=args.teacher_margin_compare_rank,
    )
    margin_threshold = _margin_threshold(
        margins=margins,
        quantile=args.teacher_margin_quantile,
        minimum=args.min_teacher_margin,
    )

    confirmation_counts, confirmation_meta = _confirmation_counts(
        labels=labels,
        manifest_height=manifest.height,
        indices_paths=args.confirmation_indices,
        submission_paths=args.confirmation_submission_csv,
        top_k=args.confirmation_top_k,
        min_same_cluster=args.min_confirmed_same_cluster,
    )
    row_preselected = (
        (cluster_sizes[labels] >= args.min_pseudo_cluster_size)
        & (cluster_sizes[labels] <= args.max_pseudo_cluster_size)
        & (margins >= margin_threshold)
        & (confirmation_counts >= args.min_confirming_models)
    )

    kept_clusters = _kept_clusters(
        labels=labels,
        row_preselected=row_preselected,
        cluster_sizes=cluster_sizes,
        min_cluster_rows=args.min_selected_rows_per_cluster,
        min_cluster_selected_share=args.min_cluster_selected_share,
    )
    selected = row_preselected & kept_clusters[labels]

    audit_path = output_dir / f"{args.experiment_id}_row_audit.csv"
    _write_row_audit(
        manifest=manifest,
        labels=labels,
        cluster_sizes=cluster_sizes,
        margins=margins,
        confirmation_counts=confirmation_counts,
        row_preselected=row_preselected,
        selected=selected,
        output_csv=audit_path,
    )
    pseudo_path = output_dir / f"{args.experiment_id}_pseudo_manifest.jsonl"
    mixed_path = output_dir / f"{args.experiment_id}_mixed_train_manifest.jsonl"
    _write_pseudo_manifest(
        manifest=manifest,
        labels=labels,
        selected=selected,
        label_prefix=args.label_prefix,
        dataset_name=args.dataset_name,
        public_audio_prefix=args.public_audio_prefix,
        output_path=pseudo_path,
    )
    _write_mixed_manifest(
        base_train_manifest=args.base_train_manifest,
        pseudo_path=pseudo_path,
        output_path=mixed_path,
    )

    selected_cluster_ids = np.unique(labels[selected])
    summary: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "public_manifest_csv": args.public_manifest_csv,
        "teacher_indices": args.teacher_indices,
        "teacher_scores": args.teacher_scores,
        "confirmation_meta": confirmation_meta,
        "cluster_first_config": asdict(config),
        "cluster_meta": cluster_meta,
        "min_pseudo_cluster_size": args.min_pseudo_cluster_size,
        "max_pseudo_cluster_size": args.max_pseudo_cluster_size,
        "teacher_margin_threshold": float(margin_threshold),
        "teacher_margin_quantile": args.teacher_margin_quantile,
        "min_teacher_margin": args.min_teacher_margin,
        "confirmation_top_k": args.confirmation_top_k,
        "min_confirmed_same_cluster": args.min_confirmed_same_cluster,
        "min_confirming_models": args.min_confirming_models,
        "min_selected_rows_per_cluster": args.min_selected_rows_per_cluster,
        "min_cluster_selected_share": args.min_cluster_selected_share,
        "preselected_row_count": int(row_preselected.sum()),
        "pseudo_row_count": int(selected.sum()),
        "pseudo_cluster_count": int(selected_cluster_ids.size),
        "mixed_train_manifest": str(mixed_path),
        "pseudo_manifest": str(pseudo_path),
        "row_audit_csv": str(audit_path),
        "base_train_manifest": args.base_train_manifest,
        "mixed_row_count": _count_lines(mixed_path),
    }
    summary_path = output_dir / f"{args.experiment_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _teacher_margins(
    scores: np.ndarray,
    *,
    best_rank: int,
    compare_rank: int,
) -> np.ndarray:
    if best_rank < 1 or compare_rank < 1:
        raise ValueError("teacher margin ranks are 1-based and must be positive")
    if compare_rank > scores.shape[1] or best_rank > scores.shape[1]:
        raise ValueError("teacher margin rank exceeds teacher score cache width")
    return scores[:, best_rank - 1] - scores[:, compare_rank - 1]


def _margin_threshold(
    *,
    margins: np.ndarray,
    quantile: float,
    minimum: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("--teacher-margin-quantile must be within [0, 1]")
    return max(float(np.quantile(margins, quantile)), float(minimum))


def _confirmation_counts(
    *,
    labels: np.ndarray,
    manifest_height: int,
    indices_paths: list[str],
    submission_paths: list[str],
    top_k: int,
    min_same_cluster: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    counts = np.zeros(manifest_height, dtype=np.int16)
    meta: list[dict[str, Any]] = []
    for path in indices_paths:
        indices = np.load(path)
        _validate_indices(indices, manifest_height, path)
        same_counts = _same_cluster_counts(labels=labels, indices=indices[:, :top_k])
        confirmed = same_counts >= min_same_cluster
        counts += confirmed.astype(np.int16)
        meta.append(
            {
                "kind": "indices",
                "path": path,
                "confirmed_share": float(confirmed.mean()),
                "same_cluster_count_p50": float(np.quantile(same_counts, 0.50)),
                "same_cluster_count_p95": float(np.quantile(same_counts, 0.95)),
            }
        )
    for path in submission_paths:
        indices = _read_submission_neighbours(Path(path), manifest_height)
        same_counts = _same_cluster_counts(labels=labels, indices=indices[:, :top_k])
        confirmed = same_counts >= min_same_cluster
        counts += confirmed.astype(np.int16)
        meta.append(
            {
                "kind": "submission_csv",
                "path": path,
                "confirmed_share": float(confirmed.mean()),
                "same_cluster_count_p50": float(np.quantile(same_counts, 0.50)),
                "same_cluster_count_p95": float(np.quantile(same_counts, 0.95)),
            }
        )
    if not meta:
        raise ValueError("at least one confirmation source is required")
    return counts, meta


def _same_cluster_counts(*, labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    query_labels = labels[:, None]
    candidate_labels = labels[indices]
    return (candidate_labels == query_labels).sum(axis=1).astype(np.int16, copy=False)


def _kept_clusters(
    *,
    labels: np.ndarray,
    row_preselected: np.ndarray,
    cluster_sizes: np.ndarray,
    min_cluster_rows: int,
    min_cluster_selected_share: float,
) -> np.ndarray:
    selected_counts = np.bincount(labels[row_preselected], minlength=cluster_sizes.shape[0])
    selected_share = selected_counts / np.maximum(cluster_sizes, 1)
    return (selected_counts >= min_cluster_rows) & (selected_share >= min_cluster_selected_share)


def _write_row_audit(
    *,
    manifest: pl.DataFrame,
    labels: np.ndarray,
    cluster_sizes: np.ndarray,
    margins: np.ndarray,
    confirmation_counts: np.ndarray,
    row_preselected: np.ndarray,
    selected: np.ndarray,
    output_csv: Path,
) -> None:
    frame = manifest.select("filepath").with_columns(
        pl.Series("row_index", np.arange(manifest.height, dtype=np.int64)),
        pl.Series("cluster_id", labels.astype(np.int64, copy=False)),
        pl.Series("cluster_size", cluster_sizes[labels].astype(np.int64, copy=False)),
        pl.Series("teacher_margin", margins.astype(np.float32, copy=False)),
        pl.Series("confirming_model_count", confirmation_counts.astype(np.int16, copy=False)),
        pl.Series("row_preselected", row_preselected),
        pl.Series("selected", selected),
    )
    frame.write_csv(output_csv)


def _write_pseudo_manifest(
    *,
    manifest: pl.DataFrame,
    labels: np.ndarray,
    selected: np.ndarray,
    label_prefix: str,
    dataset_name: str,
    public_audio_prefix: str,
    output_path: Path,
) -> None:
    filepaths = manifest["filepath"].cast(pl.Utf8).to_list()
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index in np.flatnonzero(selected):
            cluster_id = int(labels[row_index])
            payload = {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": dataset_name,
                "source_dataset": "test_public_strict_consensus_pseudo_labels",
                "speaker_id": f"{label_prefix}{cluster_id:06d}",
                "utterance_id": f"{label_prefix}{cluster_id:06d}:{row_index:06d}",
                "split": "pseudo_train",
                "audio_path": f"{public_audio_prefix.rstrip('/')}/{filepaths[row_index]}",
                "channel": "mono",
            }
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _write_mixed_manifest(
    *,
    base_train_manifest: str,
    pseudo_path: Path,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as mixed:
        if base_train_manifest:
            with Path(base_train_manifest).open(encoding="utf-8") as base:
                for line in base:
                    if line.strip():
                        mixed.write(line.rstrip() + "\n")
        with pseudo_path.open(encoding="utf-8") as pseudo:
            for line in pseudo:
                if line.strip():
                    mixed.write(line.rstrip() + "\n")


def _read_submission_neighbours(path: Path, row_count: int) -> np.ndarray:
    rows: list[list[int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "neighbours" not in (reader.fieldnames or []):
            raise ValueError(f"submission CSV lacks neighbours column: {path}")
        for row in reader:
            rows.append([int(value) for value in row["neighbours"].split(",") if value])
    values = np.asarray(rows, dtype=np.int64)
    _validate_indices(values, row_count, str(path))
    return values


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _validate_top_cache(
    indices: np.ndarray,
    scores: np.ndarray,
    manifest_height: int,
    name: str,
) -> None:
    if indices.shape != scores.shape:
        raise ValueError(f"{name} indices/scores shape mismatch: {indices.shape} != {scores.shape}")
    _validate_indices(indices, manifest_height, name)


def _validate_indices(indices: np.ndarray, manifest_height: int, name: str) -> None:
    if indices.ndim != 2 or indices.shape[0] != manifest_height:
        raise ValueError(
            f"{name} indices must be [manifest_rows, k], got {indices.shape}; "
            f"manifest_rows={manifest_height}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-indices", required=True)
    parser.add_argument("--teacher-scores", required=True)
    parser.add_argument("--confirmation-indices", action="append", default=[])
    parser.add_argument("--confirmation-submission-csv", action="append", default=[])
    parser.add_argument("--public-manifest-csv", required=True)
    parser.add_argument("--base-train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--label-prefix", default="pseudo_ms32_consensus_")
    parser.add_argument("--dataset-name", default="participants_ms32_strict_consensus_pseudo")
    parser.add_argument("--public-audio-prefix", default="datasets/Для участников")
    parser.add_argument("--edge-top", type=int, default=18)
    parser.add_argument("--reciprocal-top", type=int, default=50)
    parser.add_argument("--rank-top", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--cluster-min-size", type=int, default=5)
    parser.add_argument("--cluster-max-size", type=int, default=120)
    parser.add_argument("--cluster-min-candidates", type=int, default=3)
    parser.add_argument("--shared-top", type=int, default=50)
    parser.add_argument("--shared-min-count", type=int, default=4)
    parser.add_argument("--reciprocal-bonus", type=float, default=0.03)
    parser.add_argument("--density-penalty", type=float, default=0.02)
    parser.add_argument("--edge-score-quantile", type=float, default=None)
    parser.add_argument("--edge-min-score", type=float, default=None)
    parser.add_argument("--shared-weight", type=float, default=0.04)
    parser.add_argument("--cluster-rank-weight", type=float, default=0.02)
    parser.add_argument("--self-weight", type=float, default=0.0)
    parser.add_argument("--label-size-penalty", type=float, default=0.20)
    parser.add_argument("--no-split-oversized", action="store_true")
    parser.add_argument("--split-edge-top", type=int, default=8)
    parser.add_argument("--min-pseudo-cluster-size", type=int, default=8)
    parser.add_argument("--max-pseudo-cluster-size", type=int, default=60)
    parser.add_argument("--teacher-margin-best-rank", type=int, default=1)
    parser.add_argument("--teacher-margin-compare-rank", type=int, default=20)
    parser.add_argument("--teacher-margin-quantile", type=float, default=0.25)
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--confirmation-top-k", type=int, default=20)
    parser.add_argument("--min-confirmed-same-cluster", type=int, default=2)
    parser.add_argument("--min-confirming-models", type=int, default=2)
    parser.add_argument("--min-selected-rows-per-cluster", type=int, default=6)
    parser.add_argument("--min-cluster-selected-share", type=float, default=0.60)
    return parser.parse_args()


if __name__ == "__main__":
    main()
