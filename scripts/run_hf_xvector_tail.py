"""Extract Hugging Face AudioXVector embeddings and apply the graph retrieval tail."""

from __future__ import annotations

import argparse
import json
import os
import time
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
from kryptonite.eda.dense_audio import eval_crops, l2_normalize_rows, load_eval_waveform
from kryptonite.eda.hf_xvector import extract_xvector_embeddings
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    if args.max_rows > 0:
        manifest = manifest.head(args.max_rows)
    print(
        f"[hf-xvector] start experiment={args.experiment_id} model_id={args.model_id} "
        f"rows={manifest.height} output_dir={output_dir}",
        flush=True,
    )

    started = time.perf_counter()
    embeddings = _load_or_extract_embeddings(args, manifest, output_dir)
    embedding_s = time.perf_counter() - started

    print("[hf-xvector] exact_topk start", flush=True)
    started = time.perf_counter()
    indices, scores = exact_topk(
        embeddings,
        top_k=args.top_cache_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    search_s = time.perf_counter() - started

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
    print("[hf-xvector] label_propagation_rerank start", flush=True)
    started = time.perf_counter()
    top_indices, top_scores, label_meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=label_config,
        top_k=10,
    )
    rerank_s = time.perf_counter() - started

    indegree = np.bincount(top_indices.ravel(), minlength=manifest.height)
    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "model_id": args.model_id,
        "revision": args.revision,
        "manifest_csv": args.manifest_csv,
        "embedding_s": round(embedding_s, 6),
        "search_s": round(search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "top1_score_mean": float(top_scores[:, 0].mean()),
        "top10_mean_score_mean": float(top_scores.mean()),
        "indegree_gini_10": gini(indegree),
        "indegree_max_10": int(indegree.max()),
        "pretrained_embedding_path": str(output_dir / f"embeddings_{args.experiment_id}.npy"),
        "label_config": _jsonable_label_config(label_config),
        **label_meta,
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
    (output_dir / f"{args.experiment_id}_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pl.DataFrame([{key: _csv_value(value) for key, value in rows.items()}]).write_csv(
        output_dir / f"{args.experiment_id}_summary.csv"
    )
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


def _load_or_extract_embeddings(
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    output_dir: Path,
) -> np.ndarray:
    output_path = output_dir / f"embeddings_{args.experiment_id}.npy"
    if output_path.is_file() and not args.force_embeddings:
        print(f"[hf-xvector] load cached embeddings path={output_path}", flush=True)
        return np.load(output_path)

    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    token = os.environ.get(args.hf_token_env) or None
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_id,
        revision=args.revision or None,
        token=token,
    )
    model = AutoModelForAudioXVector.from_pretrained(
        args.model_id,
        revision=args.revision or None,
        token=token,
    ).to(args.device)
    model.eval()

    crop_samples = int(round(args.crop_seconds * 16_000))
    sums: np.ndarray | None = None
    counts = np.zeros(manifest.height, dtype=np.int32)
    batch: list[np.ndarray] = []
    owners: list[int] = []
    started_at = time.perf_counter()
    log_every_rows = max(1, manifest.height // 20)
    with (
        torch.no_grad(),
        torch.amp.autocast(
            args.device,
            enabled=args.device == "cuda" and args.precision == "bf16",
        ),
    ):
        for index, row in enumerate(manifest.iter_rows(named=True)):
            row_index = int(row.get("gallery_index", index))
            waveform = load_eval_waveform(Path(str(row["resolved_path"])), trim=args.trim)
            for crop in eval_crops(waveform, crop_samples=crop_samples, n_crops=args.n_crops):
                batch.append(crop)
                owners.append(row_index)
            if len(batch) >= args.batch_size:
                sums = _flush_embeddings(
                    model=model,
                    feature_extractor=feature_extractor,
                    batch=batch,
                    owners=owners,
                    sums=sums,
                    counts=counts,
                    row_count=manifest.height,
                    args=args,
                    torch=torch,
                )
                batch, owners = [], []
            row_number = index + 1
            if row_number == 1 or row_number % log_every_rows == 0 or row_number == manifest.height:
                elapsed_s = max(time.perf_counter() - started_at, 1e-9)
                print(
                    f"[hf-xvector] extract rows={row_number}/{manifest.height} "
                    f"pct={100.0 * row_number / manifest.height:.1f} "
                    f"rows_per_s={row_number / elapsed_s:.2f} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )
        if batch:
            sums = _flush_embeddings(
                model=model,
                feature_extractor=feature_extractor,
                batch=batch,
                owners=owners,
                sums=sums,
                counts=counts,
                row_count=manifest.height,
                args=args,
                torch=torch,
            )
    if sums is None:
        raise RuntimeError("No embeddings were extracted.")
    embeddings = l2_normalize_rows(sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    np.save(output_path, embeddings)
    return embeddings


def _flush_embeddings(
    *,
    model: Any,
    feature_extractor: Any,
    batch: list[np.ndarray],
    owners: list[int],
    sums: np.ndarray | None,
    counts: np.ndarray,
    row_count: int,
    args: argparse.Namespace,
    torch: Any,
) -> np.ndarray:
    inputs = feature_extractor(
        batch,
        sampling_rate=16_000,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    outputs = model(**inputs)
    values = extract_xvector_embeddings(outputs).detach().float().cpu().numpy()
    values = l2_normalize_rows(values.astype(np.float32, copy=False))
    if sums is None:
        sums = np.zeros((row_count, values.shape[1]), dtype=np.float32)
    for owner, embedding in zip(owners, values, strict=True):
        sums[owner] += embedding
        counts[owner] += 1
    return sums


def _jsonable_label_config(config: LabelPropagationConfig) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "edge_top": config.edge_top,
        "reciprocal_top": config.reciprocal_top,
        "rank_top": config.rank_top,
        "iterations": config.iterations,
        "label_min_size": config.label_min_size,
        "label_max_size": config.label_max_size,
        "label_min_candidates": config.label_min_candidates,
        "shared_top": config.shared_top,
        "shared_min_count": config.shared_min_count,
        "reciprocal_bonus": config.reciprocal_bonus,
        "density_penalty": config.density_penalty,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--hf-token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--top-cache-k", type=int, default=100)
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--n-crops", type=int, default=3)
    parser.add_argument("--trim", dest="trim", action="store_true", default=True)
    parser.add_argument("--no-trim", dest="trim", action="store_false")
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
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
