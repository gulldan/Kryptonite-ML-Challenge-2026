"""Build a query-conditioned retrieval submission from existing tail outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.community import LabelPropagationConfig, write_submission
from kryptonite.eda.rerank import gini
from kryptonite.eda.rowwise_tail_router import (
    RowwiseTailRouterConfig,
    graph_tail_diagnostics,
    normalized_entropy,
    policy_overlap_counts,
    route_tail_policies,
)
from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    print(
        f"[rowwise-router] start experiment={args.experiment_id} rows={manifest.height} "
        f"output_dir={output_dir}",
        flush=True,
    )

    indices = np.load(args.indices_path)
    scores = np.load(args.scores_path)
    if indices.shape[0] != manifest.height:
        raise ValueError(
            f"indices row count {indices.shape[0]} must match manifest rows {manifest.height}"
        )
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
    diagnostics = graph_tail_diagnostics(
        indices=indices,
        scores=scores,
        config=label_config,
        top_k=args.k,
    )
    candidates = {
        name: _read_submission_neighbours(Path(path), row_count=manifest.height, k=args.k)
        for name, path in _parse_named_paths(args.candidate_csv).items()
    }
    router_config = RowwiseTailRouterConfig(
        default_policy=args.default_policy,
        full_policy=args.full_policy,
        exact_policy=args.exact_policy,
        weak_policy=args.weak_policy,
        reciprocal_policy=args.reciprocal_policy,
        soup_policy=args.soup_policy or None,
        classaware_policy=args.classaware_policy or None,
        low_margin=args.low_margin,
        high_margin=args.high_margin,
        low_reciprocal_support=args.low_reciprocal_support,
        high_reciprocal_support=args.high_reciprocal_support,
        suspicious_label_max_size=args.suspicious_label_max_size,
        extreme_label_max_size=args.extreme_label_max_size,
        min_same_label_candidates=args.min_same_label_candidates,
        strong_same_label_candidates=args.strong_same_label_candidates,
        min_consensus_overlap=args.min_consensus_overlap,
        strong_consensus_overlap=args.strong_consensus_overlap,
        low_class_entropy=args.low_class_entropy,
        high_class_entropy=args.high_class_entropy,
        short_duration_s=args.short_duration_s,
        soup_consensus_margin=args.soup_consensus_margin,
    )
    class_entropy = _load_class_entropy(args.class_probs_path, expected_rows=manifest.height)
    durations_s = _load_durations(args.file_stats_parquet, manifest)
    selected, selected_policy, details = route_tail_policies(
        candidates=candidates,
        diagnostics=diagnostics,
        config=router_config,
        class_entropy=class_entropy,
        durations_s=durations_s,
    )
    submission_path = output_dir / f"submission_{args.experiment_id}.csv"
    write_submission(manifest=manifest, top_indices=selected, output_csv=submission_path)
    validation: dict[str, Any] | None = None
    if args.template_csv:
        validation = validate_submission(
            template_csv=Path(args.template_csv),
            submission_csv=submission_path,
            k=args.k,
        )
        (output_dir / f"submission_{args.experiment_id}_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    row_diagnostics = _build_row_diagnostics(
        manifest=manifest,
        diagnostics=diagnostics,
        selected_policy=selected_policy,
        details=details,
        class_entropy=class_entropy,
        durations_s=durations_s,
    )
    row_diagnostics_path = output_dir / f"{args.experiment_id}_row_diagnostics.parquet"
    row_diagnostics.write_parquet(row_diagnostics_path)
    summary = _build_summary(
        args=args,
        selected=selected,
        selected_policy=selected_policy,
        candidates=candidates,
        validation=validation,
        router_config=router_config,
        label_config=label_config,
        diagnostics=diagnostics,
        submission_path=submission_path,
        row_diagnostics_path=row_diagnostics_path,
        elapsed_s=time.perf_counter() - started,
    )
    summary_path = output_dir / f"{args.experiment_id}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pl.DataFrame([{key: _csv_value(value) for key, value in summary.items()}]).write_csv(
        output_dir / f"{args.experiment_id}_summary.csv"
    )
    short_submission = output_dir / "submission.csv"
    short_submission.write_bytes(submission_path.read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def _parse_named_paths(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path candidate, got {value!r}")
        name, path = value.split("=", 1)
        if not name:
            raise ValueError(f"Candidate name is empty in {value!r}")
        out[name] = path
    return out


def _read_submission_neighbours(path: Path, *, row_count: int, k: int) -> np.ndarray:
    frame = pl.read_csv(path)
    if "filepath" not in frame.columns or "neighbours" not in frame.columns:
        raise ValueError(f"{path} must contain filepath and neighbours columns")
    if frame.height != row_count:
        raise ValueError(f"{path} has {frame.height} rows, expected {row_count}")
    neighbours = np.empty((frame.height, k), dtype=np.int64)
    for row_index, value in enumerate(
        frame.get_column("neighbours").cast(pl.Utf8).fill_null("").to_list()
    ):
        parts = [int(part.strip()) for part in value.split(",") if part.strip()]
        if len(parts) < k:
            raise ValueError(f"{path} row {row_index + 1} has fewer than {k} neighbours")
        neighbours[row_index] = np.asarray(parts[:k], dtype=np.int64)
    return neighbours


def _load_class_entropy(path: str, *, expected_rows: int) -> np.ndarray | None:
    if not path:
        return None
    probs = np.load(path)
    if probs.shape[0] != expected_rows:
        raise ValueError(f"class probs row count {probs.shape[0]} != {expected_rows}")
    return normalized_entropy(probs)


def _load_durations(path: str, manifest: pl.DataFrame) -> np.ndarray | None:
    if not path:
        return None
    stats = pl.read_parquet(path)
    if "filepath" not in stats.columns or "duration_s" not in stats.columns:
        raise ValueError(f"{path} must contain filepath and duration_s columns")
    joined = manifest.select("filepath").join(
        stats.select(["filepath", "duration_s"]),
        on="filepath",
        how="left",
    )
    return joined.get_column("duration_s").cast(pl.Float32).to_numpy()


def _build_row_diagnostics(
    *,
    manifest: pl.DataFrame,
    diagnostics: Any,
    selected_policy: np.ndarray,
    details: dict[str, np.ndarray],
    class_entropy: np.ndarray | None,
    durations_s: np.ndarray | None,
) -> pl.DataFrame:
    columns: dict[str, Any] = {
        "row_index": np.arange(manifest.height, dtype=np.int64),
        "filepath": manifest.get_column("filepath").to_list(),
        "selected_policy": selected_policy,
        "selected_reason": details["selected_reason"],
        "margin_top1_top10": diagnostics.margin_top1_top10,
        "reciprocal_support": diagnostics.reciprocal_support,
        "label": diagnostics.labels,
        "label_size": diagnostics.row_label_size,
        "same_label_candidates": diagnostics.same_label_candidates,
        "label_usable": diagnostics.label_usable,
        "label_confidence": diagnostics.label_confidence,
    }
    if class_entropy is not None:
        columns["class_entropy"] = class_entropy
    if durations_s is not None:
        columns["duration_s"] = durations_s
    columns.update(details)
    return pl.DataFrame(
        {
            name: values.tolist()
            if isinstance(values, np.ndarray) and values.dtype == object
            else values
            for name, values in columns.items()
        }
    )


def _build_summary(
    *,
    args: argparse.Namespace,
    selected: np.ndarray,
    selected_policy: np.ndarray,
    candidates: dict[str, np.ndarray],
    validation: dict[str, Any] | None,
    router_config: RowwiseTailRouterConfig,
    label_config: LabelPropagationConfig,
    diagnostics: Any,
    submission_path: Path,
    row_diagnostics_path: Path,
    elapsed_s: float,
) -> dict[str, Any]:
    indegree = np.bincount(selected.ravel(), minlength=selected.shape[0])
    usage = {
        str(name): int(count)
        for name, count in zip(*np.unique(selected_policy, return_counts=True), strict=True)
    }
    usage_share = {name: count / selected.shape[0] for name, count in usage.items()}
    overlaps = policy_overlap_counts(
        candidates={**candidates, "router": selected}, reference_policy="router"
    )
    overlap_summary: dict[str, dict[str, float]] = {}
    for name, values in overlaps.items():
        if name == "router":
            continue
        matrix = candidates[name]
        overlap_summary[name] = {
            "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.50)),
            "p10": float(np.quantile(values, 0.10)),
            "top1_equal_share": float((selected[:, 0] == matrix[:, 0]).mean()),
        }
    return {
        "experiment_id": args.experiment_id,
        "manifest_csv": args.manifest_csv,
        "template_csv": args.template_csv,
        "indices_path": args.indices_path,
        "scores_path": args.scores_path,
        "candidate_csv": _parse_named_paths(args.candidate_csv),
        "class_probs_path": args.class_probs_path,
        "file_stats_parquet": args.file_stats_parquet,
        "submission_path": str(submission_path),
        "submission_sha256": _sha256(submission_path),
        "row_diagnostics_path": str(row_diagnostics_path),
        "validator_passed": bool(validation["passed"]) if validation is not None else None,
        "validation": validation,
        "elapsed_s": round(elapsed_s, 6),
        "indegree_gini_10": float(gini(indegree)),
        "indegree_max_10": int(indegree.max()),
        "selected_policy_counts": usage,
        "selected_policy_share": usage_share,
        "overlap_vs_policy": overlap_summary,
        "graph_margin_p50": float(np.quantile(diagnostics.margin_top1_top10, 0.50)),
        "graph_margin_p10": float(np.quantile(diagnostics.margin_top1_top10, 0.10)),
        "graph_reciprocal_support_p50": float(np.quantile(diagnostics.reciprocal_support, 0.50)),
        "graph_label_usable_share": float(diagnostics.label_usable.mean()),
        "router_config": asdict(router_config),
        "label_config": asdict(label_config),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--indices-path", required=True)
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--candidate-csv", action="append", default=[], required=True)
    parser.add_argument("--class-probs-path", default="")
    parser.add_argument("--file-stats-parquet", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--default-policy", default="classaware_c4")
    parser.add_argument("--full-policy", default="full_c4")
    parser.add_argument("--exact-policy", default="exact")
    parser.add_argument("--weak-policy", default="weak_c4")
    parser.add_argument("--reciprocal-policy", default="reciprocal_only")
    parser.add_argument("--soup-policy", default="soup_c4")
    parser.add_argument("--classaware-policy", default="classaware_c4")
    parser.add_argument("--low-margin", type=float, default=0.015)
    parser.add_argument("--high-margin", type=float, default=0.045)
    parser.add_argument("--low-reciprocal-support", type=int, default=1)
    parser.add_argument("--high-reciprocal-support", type=int, default=3)
    parser.add_argument("--suspicious-label-max-size", type=int, default=95)
    parser.add_argument("--extreme-label-max-size", type=int, default=140)
    parser.add_argument("--min-same-label-candidates", type=int, default=3)
    parser.add_argument("--strong-same-label-candidates", type=int, default=6)
    parser.add_argument("--min-consensus-overlap", type=int, default=5)
    parser.add_argument("--strong-consensus-overlap", type=int, default=7)
    parser.add_argument("--low-class-entropy", type=float, default=0.55)
    parser.add_argument("--high-class-entropy", type=float, default=0.85)
    parser.add_argument("--short-duration-s", type=float, default=2.0)
    parser.add_argument("--soup-consensus-margin", type=float, default=0.02)
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
