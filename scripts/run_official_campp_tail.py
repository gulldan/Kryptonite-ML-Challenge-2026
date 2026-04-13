"""Run CAM++ with the official 3D-Speaker frontend and retrieval tails."""

from __future__ import annotations

import argparse
import json
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
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission
from kryptonite.features.campp_official import (
    even_waveform_segments,
    load_official_campp_waveform,
    official_campp_fbank,
)
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    print(
        f"[official-campp] start experiment={args.experiment_id} rows={manifest.height} "
        f"output_dir={output_dir}",
        flush=True,
    )
    started = time.perf_counter()
    embeddings = _load_or_extract_embeddings(args, manifest, output_dir)
    embedding_s = time.perf_counter() - started

    print("[official-campp] exact_topk start", flush=True)
    started = time.perf_counter()
    indices, scores = exact_topk(
        embeddings,
        top_k=args.top_cache_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    search_s = time.perf_counter() - started
    np.save(output_dir / f"indices_{args.experiment_id}_top{args.top_cache_k}.npy", indices)
    np.save(output_dir / f"scores_{args.experiment_id}_top{args.top_cache_k}.npy", scores)

    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "checkpoint_path": args.checkpoint_path,
        "manifest_csv": args.manifest_csv,
        "data_root": args.data_root,
        "mode": args.mode,
        "sample_rate_hz": args.sample_rate_hz,
        "num_mel_bins": args.num_mel_bins,
        "eval_chunk_seconds": args.eval_chunk_seconds,
        "segment_count": args.segment_count,
        "long_file_threshold_seconds": args.long_file_threshold_seconds,
        "pad_mode": args.pad_mode,
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
    if args.template_csv:
        exact_submission_path = output_dir / f"submission_{args.experiment_id}_exact.csv"
        write_submission(
            manifest=manifest,
            top_indices=indices[:, :10],
            output_csv=exact_submission_path,
        )
        exact_validation = validate_submission(
            template_csv=Path(args.template_csv),
            submission_csv=exact_submission_path,
        )
        (output_dir / f"submission_{args.experiment_id}_exact_validation.json").write_text(
            json.dumps(exact_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.update(
            {
                "exact_submission_path": str(exact_submission_path),
                "exact_validator_passed": bool(exact_validation["passed"]),
            }
        )

    if not args.skip_c4:
        print("[official-campp] label_propagation_rerank start", flush=True)
        config = LabelPropagationConfig(
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
        top_indices, top_scores, meta = label_propagation_rerank(
            indices=indices,
            scores=scores,
            config=config,
            top_k=10,
        )
        rerank_s = time.perf_counter() - started
        rows.update(
            {
                "c4_rerank_s": round(rerank_s, 6),
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
        if args.template_csv:
            c4_submission_path = output_dir / f"submission_{args.experiment_id}_c4.csv"
            write_submission(
                manifest=manifest, top_indices=top_indices, output_csv=c4_submission_path
            )
            c4_validation = validate_submission(
                template_csv=Path(args.template_csv),
                submission_csv=c4_submission_path,
            )
            (output_dir / f"submission_{args.experiment_id}_c4_validation.json").write_text(
                json.dumps(c4_validation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.update(
                {
                    "c4_submission_path": str(c4_submission_path),
                    "c4_validator_passed": bool(c4_validation["passed"]),
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
    if args.embeddings_path:
        embeddings_path = Path(args.embeddings_path)
        print(f"[official-campp] load provided embeddings path={embeddings_path}", flush=True)
        return np.load(embeddings_path)
    if output_path.is_file() and not args.force_embeddings:
        print(f"[official-campp] load cached embeddings path={output_path}", flush=True)
        return np.load(output_path)

    import torch

    _, _, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=args.checkpoint_path,
    )
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    sums: dict[int, list[np.ndarray]] = {}
    batch_features: list[Any] = []
    batch_owners: list[int] = []
    started_at = time.perf_counter()
    log_every_rows = max(1, manifest.height // 20)
    with torch.no_grad():
        for index, row in enumerate(manifest.iter_rows(named=True)):
            row_index = int(row.get("gallery_index", row.get("row_index", index)))
            waveform = load_official_campp_waveform(
                str(row["filepath"]),
                data_root=Path(args.data_root),
                sample_rate_hz=args.sample_rate_hz,
            )
            segments = _segments_for_mode(args, waveform)
            for segment in segments:
                batch_features.append(
                    official_campp_fbank(
                        segment,
                        sample_rate_hz=args.sample_rate_hz,
                        num_mel_bins=args.num_mel_bins,
                    )
                )
                batch_owners.append(row_index)
            if len(batch_features) >= args.batch_size:
                _flush_embeddings(model, batch_features, batch_owners, sums, device)
                batch_features, batch_owners = [], []
            row_number = index + 1
            if row_number == 1 or row_number % log_every_rows == 0 or row_number == manifest.height:
                elapsed_s = max(time.perf_counter() - started_at, 1e-9)
                print(
                    f"[official-campp] extract rows={row_number}/{manifest.height} "
                    f"pct={100.0 * row_number / manifest.height:.1f} "
                    f"rows_per_s={row_number / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )
        if batch_features:
            _flush_embeddings(model, batch_features, batch_owners, sums, device)
    embeddings = np.empty(
        (manifest.height, next(iter(sums.values()))[0].shape[0]), dtype=np.float32
    )
    for index in range(manifest.height):
        embeddings[index] = np.mean(np.stack(sums[index], axis=0), axis=0)
    np.save(output_path, embeddings)
    return embeddings


def _segments_for_mode(args: argparse.Namespace, waveform: Any) -> list[Any]:
    if args.mode == "full_file":
        return [waveform.flatten().float()]
    if args.mode == "single_crop":
        return even_waveform_segments(
            waveform,
            sample_rate_hz=args.sample_rate_hz,
            chunk_seconds=args.eval_chunk_seconds,
            segment_count=1,
            pad_mode=args.pad_mode,
        )
    if args.mode == "segment_mean":
        duration_s = float(waveform.numel()) / float(args.sample_rate_hz)
        segment_count = args.segment_count if duration_s > args.long_file_threshold_seconds else 1
        return even_waveform_segments(
            waveform,
            sample_rate_hz=args.sample_rate_hz,
            chunk_seconds=args.eval_chunk_seconds,
            segment_count=segment_count,
            pad_mode=args.pad_mode,
        )
    raise ValueError(f"Unsupported mode={args.mode!r}")


def _flush_embeddings(
    model: Any,
    batch_features: list[Any],
    batch_owners: list[int],
    sums: dict[int, list[np.ndarray]],
    device: Any,
) -> None:
    import torch

    max_frames = max(int(feature.shape[0]) for feature in batch_features)
    padded = []
    for feature in batch_features:
        if int(feature.shape[0]) < max_frames:
            feature = torch.nn.functional.pad(
                feature, (0, 0, 0, max_frames - int(feature.shape[0]))
            )
        padded.append(feature)
    batch = torch.stack(padded, dim=0).to(device)
    values = model(batch).detach().cpu().numpy()
    for owner, embedding in zip(batch_owners, values, strict=True):
        sums.setdefault(owner, []).append(embedding.astype(np.float32, copy=False))


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--data-root", default="datasets/Для участников")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--embeddings-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--top-cache-k", type=int, default=100)
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--num-mel-bins", type=int, default=80)
    parser.add_argument(
        "--mode", choices=("full_file", "single_crop", "segment_mean"), default="segment_mean"
    )
    parser.add_argument("--eval-chunk-seconds", type=float, default=6.0)
    parser.add_argument("--segment-count", type=int, default=3)
    parser.add_argument("--long-file-threshold-seconds", type=float, default=6.0)
    parser.add_argument("--pad-mode", choices=("repeat", "zero"), default="repeat")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--skip-c4", action="store_true")
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
