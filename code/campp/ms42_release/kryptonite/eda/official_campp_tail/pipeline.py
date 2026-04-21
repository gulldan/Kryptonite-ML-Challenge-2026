"""Top-level orchestration for the official CAM++ tail pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.community import (
    LabelPropagationConfig,
    exact_topk,
    label_propagation_rerank,
    write_submission,
)
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission

from .config import OfficialCamPPTailConfig, parse_args, resolved_frontend_cache_mode
from .extraction import load_or_extract_embeddings


def run_official_campp_tail(config: OfficialCamPPTailConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(config.manifest_csv)
    widths = _resolve_retrieval_widths(config=config, row_count=manifest.height)
    print(
        f"[official-campp] start experiment={config.experiment_id} rows={manifest.height} "
        f"output_top_k={config.output_top_k} top_cache_k={widths['top_cache_k']} "
        f"output_dir={output_dir}",
        flush=True,
    )

    total_started = time.perf_counter()
    started = time.perf_counter()
    embeddings = load_or_extract_embeddings(config, manifest, output_dir)
    embedding_s = time.perf_counter() - started

    print("[official-campp] exact_topk start", flush=True)
    started = time.perf_counter()
    indices, scores = exact_topk(
        embeddings,
        top_k=widths["top_cache_k"],
        batch_size=config.search_batch_size,
        device=config.search_device,
    )
    search_s = time.perf_counter() - started
    if not config.skip_save_top_cache:
        np.save(
            output_dir / f"indices_{config.experiment_id}_top{widths['top_cache_k']}.npy",
            indices,
        )
        np.save(
            output_dir / f"scores_{config.experiment_id}_top{widths['top_cache_k']}.npy",
            scores,
        )

    rows = _build_summary_rows(
        config=config,
        manifest=manifest,
        embedding_s=embedding_s,
        search_s=search_s,
        scores=scores,
        indices=indices,
        widths=widths,
    )
    _write_exact_submission_if_requested(
        config=config,
        manifest=manifest,
        indices=indices,
        output_dir=output_dir,
        rows=rows,
    )
    if not config.skip_c4:
        _run_c4_rerank(
            config=config,
            manifest=manifest,
            indices=indices,
            scores=scores,
            output_dir=output_dir,
            rows=rows,
            widths=widths,
        )
    _finalize_timing_rows(rows=rows, total_started=total_started)
    _write_summary(rows=rows, output_dir=output_dir, experiment_id=config.experiment_id)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    rows = run_official_campp_tail(config)
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


def _build_summary_rows(
    *,
    config: OfficialCamPPTailConfig,
    manifest: pl.DataFrame,
    embedding_s: float,
    search_s: float,
    scores: np.ndarray,
    indices: np.ndarray,
    widths: dict[str, int],
) -> dict[str, Any]:
    top10_limit = min(10, indices.shape[1])
    exact_indegree10 = np.bincount(indices[:, :top10_limit].ravel(), minlength=manifest.height)
    rows: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "encoder_backend": config.encoder_backend,
        "checkpoint_path": config.checkpoint_path,
        "manifest_csv": config.manifest_csv,
        "data_root": config.data_root,
        "mode": config.mode,
        "sample_rate_hz": config.sample_rate_hz,
        "num_mel_bins": config.num_mel_bins,
        "eval_chunk_seconds": config.eval_chunk_seconds,
        "segment_count": config.segment_count,
        "long_file_threshold_seconds": config.long_file_threshold_seconds,
        "pad_mode": config.pad_mode,
        "frontend_cache_dir": config.frontend_cache_dir,
        "frontend_pack_dir": config.frontend_pack_dir,
        "frontend_cache_mode": resolved_frontend_cache_mode(config),
        "embedding_s": round(embedding_s, 6),
        "search_s": round(search_s, 6),
        "output_top_k": config.output_top_k,
        "top_cache_k": widths["top_cache_k"],
        "exact_top1_score_mean": float(scores[:, 0].mean()),
        "exact_top10_mean_score_mean": float(scores[:, :top10_limit].mean()),
        "exact_topk_mean_score_mean": float(scores[:, : config.output_top_k].mean()),
        "exact_indegree_gini_10": gini(exact_indegree10),
        "exact_indegree_max_10": int(exact_indegree10.max()),
    }
    if config.frontend_cache_stats is not None:
        rows["frontend_cache_stats"] = config.frontend_cache_stats
    if config.encoder_backend == "tensorrt":
        rows.update(
            {
                "tensorrt_config_path": config.tensorrt_config,
                "tensorrt_engine_path": config.resolved_tensorrt_engine_path
                or config.tensorrt_engine_path,
                "tensorrt_profile_ids": config.resolved_tensorrt_profile_ids,
            }
        )
    return rows


def _write_exact_submission_if_requested(
    *,
    config: OfficialCamPPTailConfig,
    manifest: pl.DataFrame,
    indices: np.ndarray,
    output_dir: Path,
    rows: dict[str, Any],
) -> None:
    if not config.template_csv:
        return
    exact_submission_path = output_dir / f"submission_{config.experiment_id}_exact.csv"
    started = time.perf_counter()
    write_submission(
        manifest=manifest,
        top_indices=indices[:, : config.output_top_k],
        output_csv=exact_submission_path,
    )
    exact_submit_write_s = time.perf_counter() - started
    started = time.perf_counter()
    exact_validation = validate_submission(
        template_csv=Path(config.template_csv),
        submission_csv=exact_submission_path,
        k=config.output_top_k,
    )
    exact_validation_s = time.perf_counter() - started
    (output_dir / f"submission_{config.experiment_id}_exact_validation.json").write_text(
        json.dumps(exact_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows.update(
        {
            "exact_submission_path": str(exact_submission_path),
            "exact_validator_passed": bool(exact_validation["passed"]),
            "exact_submit_write_s": round(exact_submit_write_s, 6),
            "exact_validation_s": round(exact_validation_s, 6),
            "exact_submit_generation_s": round(
                float(rows["embedding_s"]) + float(rows["search_s"]) + exact_submit_write_s,
                6,
            ),
        }
    )


def _run_c4_rerank(
    *,
    config: OfficialCamPPTailConfig,
    manifest: pl.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    output_dir: Path,
    rows: dict[str, Any],
    widths: dict[str, int],
) -> None:
    print("[official-campp] label_propagation_rerank start", flush=True)
    started = time.perf_counter()
    top_indices, top_scores, meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=LabelPropagationConfig(
            experiment_id=config.experiment_id,
            edge_top=widths["edge_top"],
            reciprocal_top=widths["reciprocal_top"],
            rank_top=widths["rank_top"],
            iterations=config.iterations,
            label_min_size=config.label_min_size,
            label_max_size=config.label_max_size,
            label_min_candidates=config.label_min_candidates,
            shared_top=widths["shared_top"],
            shared_min_count=config.shared_min_count,
            reciprocal_bonus=config.reciprocal_bonus,
            density_penalty=config.density_penalty,
        ),
        top_k=config.output_top_k,
    )
    top10_limit = min(10, top_scores.shape[1])
    top10_scores = top_scores[:, :top10_limit]
    indegree = np.bincount(top_indices.ravel(), minlength=manifest.height)
    indegree10 = np.bincount(top_indices[:, :top10_limit].ravel(), minlength=manifest.height)
    rows.update(
        {
            "c4_rerank_s": round(time.perf_counter() - started, 6),
            "c4_top1_score_mean": float(top_scores[:, 0].mean()),
            "c4_top10_mean_score_mean": float(top10_scores.mean()),
            "c4_topk_mean_score_mean": float(top_scores.mean()),
            "c4_indegree_gini_10": gini(indegree10),
            "c4_indegree_max_10": int(indegree10.max()),
            "c4_indegree_gini_k": gini(indegree),
            "c4_indegree_max_k": int(indegree.max()),
            **meta,
        }
    )
    if not config.template_csv:
        return
    c4_submission_path = output_dir / f"submission_{config.experiment_id}_c4.csv"
    started = time.perf_counter()
    write_submission(manifest=manifest, top_indices=top_indices, output_csv=c4_submission_path)
    c4_submit_write_s = time.perf_counter() - started
    started = time.perf_counter()
    c4_validation = validate_submission(
        template_csv=Path(config.template_csv),
        submission_csv=c4_submission_path,
        k=config.output_top_k,
    )
    c4_validation_s = time.perf_counter() - started
    (output_dir / f"submission_{config.experiment_id}_c4_validation.json").write_text(
        json.dumps(c4_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows.update(
        {
            "c4_submission_path": str(c4_submission_path),
            "c4_validator_passed": bool(c4_validation["passed"]),
            "c4_submit_write_s": round(c4_submit_write_s, 6),
            "c4_validation_s": round(c4_validation_s, 6),
            "c4_submit_generation_s": round(
                float(rows["embedding_s"])
                + float(rows["search_s"])
                + float(rows["c4_rerank_s"])
                + c4_submit_write_s,
                6,
            ),
        }
    )


def _finalize_timing_rows(*, rows: dict[str, Any], total_started: float) -> None:
    rows["wall_total_s"] = round(time.perf_counter() - total_started, 6)


def _resolve_retrieval_widths(
    *,
    config: OfficialCamPPTailConfig,
    row_count: int,
) -> dict[str, int]:
    max_neighbours = row_count - 1
    if config.output_top_k <= 0:
        raise ValueError("--output-top-k must be positive.")
    if config.output_top_k > max_neighbours:
        raise ValueError(
            f"--output-top-k={config.output_top_k} requires at least "
            f"{config.output_top_k + 1} manifest rows, got {row_count}."
        )
    raw_widths = {
        "top_cache_k": config.top_cache_k,
        "edge_top": config.edge_top,
        "reciprocal_top": config.reciprocal_top,
        "rank_top": config.rank_top,
    }
    for name, value in raw_widths.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if config.shared_top < 0:
        raise ValueError("--shared-top must be non-negative.")
    if config.shared_min_count > 0 and config.shared_top == 0:
        raise ValueError("--shared-top must be positive when --shared-min-count is positive.")
    edge_top = min(config.edge_top, max_neighbours)
    reciprocal_top = min(config.reciprocal_top, max_neighbours)
    rank_top = min(max(config.rank_top, config.output_top_k), max_neighbours)
    shared_top = min(config.shared_top, max_neighbours)
    top_cache_k = min(
        max(
            config.top_cache_k,
            config.output_top_k,
            edge_top,
            reciprocal_top,
            rank_top,
            shared_top,
        ),
        max_neighbours,
    )
    return {
        "top_cache_k": top_cache_k,
        "edge_top": edge_top,
        "reciprocal_top": reciprocal_top,
        "rank_top": rank_top,
        "shared_top": shared_top,
    }


def _write_summary(
    *,
    rows: dict[str, Any],
    output_dir: Path,
    experiment_id: str,
) -> None:
    (output_dir / f"{experiment_id}_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pl.DataFrame([{key: _csv_value(value) for key, value in rows.items()}]).write_csv(
        output_dir / f"{experiment_id}_summary.csv"
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


__all__ = ["main", "run_official_campp_tail"]
