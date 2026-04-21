"""Extract checkpoint embeddings and apply the current C4 retrieval tail."""

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
    evaluate_labelled_topk,
    exact_topk,
    label_propagation_rerank,
    write_submission,
)
from kryptonite.eda.dense_audio import (
    SyntheticShiftProfile,
    apply_channel_condition,
    eval_crops,
    l2_normalize_rows,
    load_eval_waveform,
    sample_channel_condition,
)
from kryptonite.eda.submission import validate_submission
from kryptonite.features import FbankExtractionRequest, FbankExtractor
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.models.eres2netv2 import load_eres2netv2_encoder_from_checkpoint


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    print(
        f"[c4-tail] start experiment={args.experiment_id} model={args.model} "
        f"rows={manifest.height} output_dir={output_dir}",
        flush=True,
    )
    started = time.perf_counter()
    embeddings = _load_or_extract_embeddings(args, manifest, output_dir)
    embedding_s = time.perf_counter() - started

    print("[c4-tail] exact_topk start", flush=True)
    started = time.perf_counter()
    indices, scores = exact_topk(
        embeddings,
        top_k=args.top_cache_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    search_s = time.perf_counter() - started

    print("[c4-tail] label_propagation_rerank start", flush=True)
    config = LabelPropagationConfig(
        experiment_id=args.experiment_id,
        edge_top=args.edge_top,
        shared_min_count=args.shared_min_count,
    )
    started = time.perf_counter()
    top_indices, top_scores, meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=config,
        top_k=10,
    )
    rerank_s = time.perf_counter() - started

    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "model": args.model,
        "checkpoint_path": args.checkpoint_path,
        "manifest_csv": args.manifest_csv,
        "embedding_s": round(embedding_s, 6),
        "search_s": round(search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "top1_score_mean": float(top_scores[:, 0].mean()),
        "top10_mean_score_mean": float(top_scores.mean()),
        **meta,
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
    pl.DataFrame([rows]).write_csv(output_dir / f"{args.experiment_id}_summary.csv")
    print(json.dumps(rows, indent=2, sort_keys=True))


def _load_or_extract_embeddings(
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    output_dir: Path,
) -> np.ndarray:
    output_path = output_dir / f"embeddings_{args.experiment_id}.npy"
    if output_path.is_file() and not args.force_embeddings:
        print(f"[c4-tail] load cached embeddings path={output_path}", flush=True)
        return np.load(output_path)
    import torch

    model = _load_encoder(args.model, args.checkpoint_path, torch=torch).to(args.device)
    model.eval()
    extractor = FbankExtractor(FbankExtractionRequest())
    crop_samples = int(round(args.crop_seconds * 16_000))
    shift_profile = (
        SyntheticShiftProfile.from_file_stats(Path(args.file_stats_path), seed=args.seed)
        if args.shift_mode != "none"
        else None
    )
    sums: np.ndarray | None = None
    counts = np.zeros(manifest.height, dtype=np.int32)
    batch: list[Any] = []
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
            waveform = load_eval_waveform(
                Path(str(row["resolved_path"])),
                trim=args.trim,
                shift_profile=shift_profile,
                shift_mode=args.shift_mode,
                shift_key=row_index,
            )
            condition = (
                sample_channel_condition(shift_profile, shift_key=row_index)
                if shift_profile is not None and args.shift_mode == "v2"
                else None
            )
            for crop in eval_crops(waveform, crop_samples=crop_samples, n_crops=args.n_crops):
                if condition is not None:
                    crop = apply_channel_condition(crop, condition)
                batch.append(extractor.extract(crop, sample_rate_hz=16_000))
                owners.append(row_index)
            if len(batch) >= args.batch_size:
                sums = _flush_embeddings(model, batch, owners, sums, counts, manifest.height, args)
                batch, owners = [], []
            row_number = index + 1
            if row_number == 1 or row_number % log_every_rows == 0 or row_number == manifest.height:
                elapsed_s = max(time.perf_counter() - started_at, 1e-9)
                print(
                    f"[c4-tail] extract rows={row_number}/{manifest.height} "
                    f"pct={100.0 * row_number / manifest.height:.1f} "
                    f"rows_per_s={row_number / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )
        if batch:
            sums = _flush_embeddings(model, batch, owners, sums, counts, manifest.height, args)
    if sums is None:
        raise RuntimeError("No embeddings were extracted.")
    embeddings = l2_normalize_rows(sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    np.save(output_path, embeddings)
    return embeddings


def _flush_embeddings(
    model: Any,
    batch: list[Any],
    owners: list[int],
    sums: np.ndarray | None,
    counts: np.ndarray,
    row_count: int,
    args: argparse.Namespace,
) -> np.ndarray:
    import torch

    features = torch.stack(batch).to(device=args.device, dtype=torch.float32)
    values = model(features).detach().float().cpu().numpy()
    values = l2_normalize_rows(values.astype(np.float32, copy=False))
    if sums is None:
        sums = np.zeros((row_count, values.shape[1]), dtype=np.float32)
    for owner, embedding in zip(owners, values, strict=True):
        sums[owner] += embedding
        counts[owner] += 1
    return sums


def _load_encoder(model_name: str, checkpoint_path: str, *, torch: Any) -> Any:
    normalized = model_name.strip().lower()
    if normalized == "eres2netv2":
        _, _, model = load_eres2netv2_encoder_from_checkpoint(
            torch=torch,
            checkpoint_path=checkpoint_path,
        )
        return model
    if normalized == "campp":
        _, _, model = load_campp_encoder_from_checkpoint(
            torch=torch,
            checkpoint_path=checkpoint_path,
        )
        return model
    raise ValueError("model must be one of: eres2netv2, campp")


def _has_labels(manifest: pl.DataFrame) -> bool:
    return "speaker_id" in manifest.columns and manifest["speaker_id"].drop_nulls().len() > 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("eres2netv2", "campp"), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--file-stats-path",
        default="artifacts/eda/participants_audio6/file_stats.parquet",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--top-cache-k", type=int, default=100)
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--n-crops", type=int, default=3)
    parser.add_argument("--trim", dest="trim", action="store_true", default=True)
    parser.add_argument("--no-trim", dest="trim", action="store_false")
    parser.add_argument("--shift-mode", choices=("none", "edge_silence", "v2"), default="none")
    parser.add_argument("--edge-top", type=int, default=10)
    parser.add_argument("--shared-min-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
