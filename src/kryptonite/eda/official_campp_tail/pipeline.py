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
    print(
        f"[official-campp] start experiment={config.experiment_id} rows={manifest.height} "
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
        top_k=config.top_cache_k,
        batch_size=config.search_batch_size,
        device=config.search_device,
    )
    search_s = time.perf_counter() - started
    if not config.skip_save_top_cache:
        np.save(output_dir / f"indices_{config.experiment_id}_top{config.top_cache_k}.npy", indices)
        np.save(output_dir / f"scores_{config.experiment_id}_top{config.top_cache_k}.npy", scores)

    rows = _build_summary_rows(
        config=config,
        manifest=manifest,
        embedding_s=embedding_s,
        search_s=search_s,
        scores=scores,
        indices=indices,
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
) -> dict[str, Any]:
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
        "exact_top1_score_mean": float(scores[:, 0].mean()),
        "exact_top10_mean_score_mean": float(scores[:, :10].mean()),
        "exact_indegree_gini_10": gini(
            np.bincount(indices[:, :10].ravel(), minlength=manifest.height)
        ),
        "exact_indegree_max_10": int(
            np.bincount(indices[:, :10].ravel(), minlength=manifest.height).max()
        ),
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
        manifest=manifest, top_indices=indices[:, :10], output_csv=exact_submission_path
    )
    exact_submit_write_s = time.perf_counter() - started
    started = time.perf_counter()
    exact_validation = validate_submission(
        template_csv=Path(config.template_csv),
        submission_csv=exact_submission_path,
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
) -> None:
    print("[official-campp] label_propagation_rerank start", flush=True)
    started = time.perf_counter()
    top_indices, top_scores, meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=LabelPropagationConfig(
            experiment_id=config.experiment_id,
            edge_top=config.edge_top,
            reciprocal_top=config.reciprocal_top,
            rank_top=config.rank_top,
            iterations=config.iterations,
            label_min_size=config.label_min_size,
            label_max_size=config.label_max_size,
            label_min_candidates=config.label_min_candidates,
            shared_top=config.shared_top,
            shared_min_count=config.shared_min_count,
            reciprocal_bonus=config.reciprocal_bonus,
            density_penalty=config.density_penalty,
        ),
        top_k=10,
    )
    rows.update(
        {
            "c4_rerank_s": round(time.perf_counter() - started, 6),
            "c4_top1_score_mean": float(top_scores[:, 0].mean()),
            "c4_top10_mean_score_mean": float(top_scores.mean()),
            "c4_indegree_gini_10": gini(
                np.bincount(top_indices.ravel(), minlength=manifest.height)
            ),
            "c4_indegree_max_10": int(
                np.bincount(top_indices.ravel(), minlength=manifest.height).max()
            ),
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
