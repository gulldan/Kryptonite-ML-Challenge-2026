"""Run CAM++ with the official 3D-Speaker frontend and retrieval tails."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.deployment import resolve_project_path
from kryptonite.eda.community import (
    LabelPropagationConfig,
    exact_topk,
    label_propagation_rerank,
    write_submission,
)
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission
from kryptonite.features.campp_official import (
    SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES,
    load_official_campp_frontend_feature_pack,
    load_or_compute_official_campp_features,
    stack_official_campp_feature_batch,
)
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.serve.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.serve.tensorrt_engine_config import load_tensorrt_fp16_config
from kryptonite.serve.tensorrt_engine_models import TensorRTFP16Profile
from kryptonite.serve.tensorrt_engine_runtime import _select_profile, _TensorRTEngineRunner

EncoderCallable = Callable[[Any], np.ndarray]


@dataclass(slots=True)
class FrontendCacheStats:
    disabled_rows: int = 0
    hit_rows: int = 0
    miss_rows: int = 0
    written_rows: int = 0

    def add(self, *, cache_active: bool, cache_hit: bool, cache_written: bool) -> None:
        if not cache_active:
            self.disabled_rows += 1
            return
        if cache_hit:
            self.hit_rows += 1
        else:
            self.miss_rows += 1
        if cache_written:
            self.written_rows += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "disabled_rows": self.disabled_rows,
            "hit_rows": self.hit_rows,
            "miss_rows": self.miss_rows,
            "written_rows": self.written_rows,
        }


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
    if not args.skip_save_top_cache:
        np.save(output_dir / f"indices_{args.experiment_id}_top{args.top_cache_k}.npy", indices)
        np.save(output_dir / f"scores_{args.experiment_id}_top{args.top_cache_k}.npy", scores)

    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "encoder_backend": args.encoder_backend,
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
        "frontend_cache_dir": args.frontend_cache_dir,
        "frontend_pack_dir": args.frontend_pack_dir,
        "frontend_cache_mode": _resolved_frontend_cache_mode(args),
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
    frontend_cache_stats = getattr(args, "frontend_cache_stats", None)
    if frontend_cache_stats is not None:
        rows["frontend_cache_stats"] = frontend_cache_stats
    if args.encoder_backend == "tensorrt":
        rows.update(
            {
                "tensorrt_config_path": args.tensorrt_config,
                "tensorrt_engine_path": getattr(
                    args,
                    "resolved_tensorrt_engine_path",
                    args.tensorrt_engine_path,
                ),
                "tensorrt_profile_ids": getattr(args, "resolved_tensorrt_profile_ids", []),
            }
        )
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

    encoder = _build_encoder(args)
    sums: dict[int, list[np.ndarray]] = {}
    batch_features: list[Any] = []
    batch_owners: list[int] = []
    frontend_cache_stats = FrontendCacheStats()
    started_at = time.perf_counter()
    log_every_rows = max(1, manifest.height // 20)
    if args.frontend_pack_dir:
        _extract_embeddings_from_frontend_pack(
            args,
            manifest,
            encoder=encoder,
            sums=sums,
            batch_features=batch_features,
            batch_owners=batch_owners,
            started_at=started_at,
            log_every_rows=log_every_rows,
        )
        args.frontend_cache_stats = {
            "disabled_rows": 0,
            "hit_rows": manifest.height,
            "miss_rows": 0,
            "written_rows": 0,
            "pack_rows": manifest.height,
        }
        embeddings = np.empty(
            (manifest.height, next(iter(sums.values()))[0].shape[0]), dtype=np.float32
        )
        for index in range(manifest.height):
            embeddings[index] = np.mean(np.stack(sums[index], axis=0), axis=0)
        _save_embeddings_if_requested(args, output_path, embeddings)
        return embeddings
    if args.frontend_workers > 0:
        _extract_embeddings_parallel_frontend(
            args,
            manifest,
            encoder=encoder,
            sums=sums,
            batch_features=batch_features,
            batch_owners=batch_owners,
            frontend_cache_stats=frontend_cache_stats,
            started_at=started_at,
            log_every_rows=log_every_rows,
        )
        args.frontend_cache_stats = frontend_cache_stats.to_dict()
        embeddings = np.empty(
            (manifest.height, next(iter(sums.values()))[0].shape[0]), dtype=np.float32
        )
        for index in range(manifest.height):
            embeddings[index] = np.mean(np.stack(sums[index], axis=0), axis=0)
        _save_embeddings_if_requested(args, output_path, embeddings)
        return embeddings

    for index, row in enumerate(manifest.iter_rows(named=True)):
        row_index, features, cache_hit, cache_written = _frontend_features_for_row(args, index, row)
        frontend_cache_stats.add(
            cache_active=_frontend_cache_active(args),
            cache_hit=cache_hit,
            cache_written=cache_written,
        )
        for feature in features:
            batch_features.append(feature)
            batch_owners.append(row_index)
        if len(batch_features) >= args.batch_size:
            _flush_ready_embedding_batches(
                encoder,
                batch_features,
                batch_owners,
                sums,
                batch_size=args.batch_size,
            )
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
        _flush_embeddings(encoder, batch_features, batch_owners, sums)
    args.frontend_cache_stats = frontend_cache_stats.to_dict()
    embeddings = np.empty(
        (manifest.height, next(iter(sums.values()))[0].shape[0]), dtype=np.float32
    )
    for index in range(manifest.height):
        embeddings[index] = np.mean(np.stack(sums[index], axis=0), axis=0)
    _save_embeddings_if_requested(args, output_path, embeddings)
    return embeddings


def _save_embeddings_if_requested(
    args: argparse.Namespace,
    output_path: Path,
    embeddings: np.ndarray,
) -> None:
    if args.skip_save_embeddings:
        return
    np.save(output_path, embeddings)


def _extract_embeddings_from_frontend_pack(
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    *,
    encoder: EncoderCallable,
    sums: dict[int, list[np.ndarray]],
    batch_features: list[Any],
    batch_owners: list[int],
    started_at: float,
    log_every_rows: int,
) -> None:
    pack = load_official_campp_frontend_feature_pack(
        Path(args.frontend_pack_dir),
        mmap_mode="c" if args.frontend_pack_fast_path else "r",
    )
    if pack.row_count < manifest.height:
        raise ValueError(
            f"Packed frontend cache has {pack.row_count} rows, but manifest has {manifest.height}."
        )
    if args.frontend_pack_fast_path:
        if _try_extract_embeddings_from_contiguous_frontend_pack(
            pack=pack,
            manifest=manifest,
            encoder=encoder,
            sums=sums,
            batch_size=args.batch_size,
            started_at=started_at,
            log_every_rows=log_every_rows,
        ):
            return
        print("[official-campp] pack-fast unavailable; falling back to row pack loop", flush=True)

    for index, row in enumerate(manifest.iter_rows(named=True)):
        row_index = int(row.get("gallery_index", row.get("row_index", index)))
        for feature in pack.features_for_row(index):
            batch_features.append(feature)
            batch_owners.append(row_index)
        if len(batch_features) >= args.batch_size:
            _flush_ready_embedding_batches(
                encoder,
                batch_features,
                batch_owners,
                sums,
                batch_size=args.batch_size,
            )
        row_number = index + 1
        if row_number == 1 or row_number % log_every_rows == 0 or row_number == manifest.height:
            elapsed_s = max(time.perf_counter() - started_at, 1e-9)
            print(
                f"[official-campp] pack rows={row_number}/{manifest.height} "
                f"pct={100.0 * row_number / manifest.height:.1f} "
                f"rows_per_s={row_number / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
    if batch_features:
        _flush_embeddings(encoder, batch_features, batch_owners, sums)


def _try_extract_embeddings_from_contiguous_frontend_pack(
    *,
    pack: Any,
    manifest: pl.DataFrame,
    encoder: EncoderCallable,
    sums: dict[int, list[np.ndarray]],
    batch_size: int,
    started_at: float,
    log_every_rows: int,
) -> bool:
    row_counts = np.asarray(pack.row_counts[: manifest.height], dtype=np.int64)
    row_offsets = np.asarray(pack.row_offsets[: manifest.height], dtype=np.int64)
    if np.any(row_counts <= 0):
        return False
    expected_offsets = np.empty_like(row_offsets)
    expected_offsets[0] = 0
    if manifest.height > 1:
        expected_offsets[1:] = np.cumsum(row_counts[:-1], dtype=np.int64)
    if not np.array_equal(row_offsets, expected_offsets):
        return False

    total_segments = int(row_counts.sum())
    if total_segments <= 0:
        return False
    row_owners = _manifest_row_owners(manifest)
    if row_owners.shape[0] != manifest.height:
        return False
    if int(row_owners.min(initial=0)) < 0 or int(row_owners.max(initial=0)) >= manifest.height:
        return False

    segment_owners = np.repeat(row_owners, row_counts)
    owner_counts = np.bincount(segment_owners, minlength=manifest.height).astype(np.float32)
    if np.any(owner_counts <= 0.0):
        return False

    embedding_sums: np.ndarray | None = None
    row_ends = row_offsets + row_counts
    next_log_row = 1
    for segment_start in range(0, total_segments, batch_size):
        segment_stop = min(segment_start + batch_size, total_segments)
        batch = _torch_batch_from_packed_features(pack.features[segment_start:segment_stop])
        values = encoder(batch).astype(np.float32, copy=False)
        if embedding_sums is None:
            embedding_sums = np.zeros((manifest.height, values.shape[1]), dtype=np.float32)
        _accumulate_sorted_owner_embeddings(
            embedding_sums,
            owners=segment_owners[segment_start:segment_stop],
            values=values,
        )

        completed_rows = int(np.searchsorted(row_ends, segment_stop, side="right"))
        if (
            completed_rows >= next_log_row
            or segment_stop == total_segments
            or completed_rows == manifest.height
        ):
            row_number = min(completed_rows, manifest.height)
            elapsed_s = max(time.perf_counter() - started_at, 1e-9)
            print(
                f"[official-campp] pack-fast rows={row_number}/{manifest.height} "
                f"pct={100.0 * row_number / manifest.height:.1f} "
                f"rows_per_s={row_number / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
            next_log_row = row_number + log_every_rows

    if embedding_sums is None:
        return False
    embeddings = embedding_sums / owner_counts[:, None]
    for row_index, embedding in enumerate(embeddings):
        sums.setdefault(row_index, []).append(embedding.astype(np.float32, copy=False))
    return True


def _manifest_row_owners(manifest: pl.DataFrame) -> np.ndarray:
    if "gallery_index" in manifest.columns:
        return np.asarray(manifest.get_column("gallery_index"), dtype=np.int64)
    if "row_index" in manifest.columns:
        return np.asarray(manifest.get_column("row_index"), dtype=np.int64)
    return np.arange(manifest.height, dtype=np.int64)


def _accumulate_sorted_owner_embeddings(
    embedding_sums: np.ndarray,
    *,
    owners: np.ndarray,
    values: np.ndarray,
) -> None:
    if owners.shape[0] != values.shape[0]:
        raise ValueError(f"Owner/value batch mismatch: {owners.shape[0]} != {values.shape[0]}.")
    if owners.shape[0] == 0:
        return
    boundary_indices = np.flatnonzero(owners[1:] != owners[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundary_indices))
    stops = np.concatenate((boundary_indices, np.array([owners.shape[0]], dtype=np.int64)))
    for start, stop in zip(starts, stops, strict=True):
        embedding_sums[int(owners[start])] += values[start:stop].sum(axis=0)


def _torch_batch_from_packed_features(features: np.ndarray) -> Any:
    import torch

    array = np.array(features, dtype=np.float32, copy=True, order="C")
    if array.ndim != 3:
        raise ValueError(f"Expected packed feature batch to be 3D, got shape {array.shape}.")
    return torch.from_numpy(array)


def _extract_embeddings_parallel_frontend(
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    *,
    encoder: EncoderCallable,
    sums: dict[int, list[np.ndarray]],
    batch_features: list[Any],
    batch_owners: list[int],
    frontend_cache_stats: FrontendCacheStats,
    started_at: float,
    log_every_rows: int,
) -> None:
    max_pending = args.frontend_prefetch or max(args.frontend_workers * 4, args.batch_size)
    iterator = enumerate(manifest.iter_rows(named=True))
    submitted = 0
    completed = 0
    if args.frontend_executor == "process":
        executor_context: Any = ProcessPoolExecutor(
            max_workers=args.frontend_workers,
            mp_context=mp.get_context("spawn"),
        )
    else:
        executor_context = ThreadPoolExecutor(max_workers=args.frontend_workers)
    with executor_context as executor:
        pending: dict[Future[tuple[int, list[Any], bool, bool]], None] = {}
        for _ in range(max_pending):
            if not _submit_next_frontend_task(args, executor, iterator, pending):
                break
            submitted += 1
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                row_index, features, cache_hit, cache_written = future.result()
                frontend_cache_stats.add(
                    cache_active=_frontend_cache_active(args),
                    cache_hit=cache_hit,
                    cache_written=cache_written,
                )
                for feature in features:
                    batch_features.append(feature)
                    batch_owners.append(row_index)
                if len(batch_features) >= args.batch_size:
                    _flush_ready_embedding_batches(
                        encoder,
                        batch_features,
                        batch_owners,
                        sums,
                        batch_size=args.batch_size,
                    )
                completed += 1
                if (
                    completed == 1
                    or completed % log_every_rows == 0
                    or completed == manifest.height
                ):
                    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
                    print(
                        f"[official-campp] extract rows={completed}/{manifest.height} "
                        f"pct={100.0 * completed / manifest.height:.1f} "
                        f"rows_per_s={completed / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
                        flush=True,
                    )
                if submitted < manifest.height and _submit_next_frontend_task(
                    args, executor, iterator, pending
                ):
                    submitted += 1
    if batch_features:
        _flush_embeddings(encoder, batch_features, batch_owners, sums)


def _submit_next_frontend_task(
    args: argparse.Namespace,
    executor: Any,
    iterator: Any,
    pending: dict[Future[tuple[int, list[Any], bool, bool]], None],
) -> bool:
    try:
        index, row = next(iterator)
    except StopIteration:
        return False
    pending[executor.submit(_frontend_features_for_row, args, index, row)] = None
    return True


def _frontend_features_for_row(
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
) -> tuple[int, list[Any], bool, bool]:
    row_index = int(row.get("gallery_index", row.get("row_index", index)))
    result = load_or_compute_official_campp_features(
        raw_path=str(row["filepath"]),
        data_root=Path(args.data_root),
        sample_rate_hz=args.sample_rate_hz,
        num_mel_bins=args.num_mel_bins,
        mode=args.mode,
        eval_chunk_seconds=args.eval_chunk_seconds,
        segment_count=args.segment_count,
        long_file_threshold_seconds=args.long_file_threshold_seconds,
        pad_mode=args.pad_mode,
        cache_root=_frontend_cache_root(args),
        cache_mode=_resolved_frontend_cache_mode(args),
    )
    return row_index, result.features, result.cache_hit, result.cache_written


def _build_encoder(args: argparse.Namespace) -> EncoderCallable:
    if args.encoder_backend == "torch":
        return _build_torch_encoder(args)
    if args.encoder_backend == "tensorrt":
        return _build_tensorrt_encoder(args)
    raise ValueError(f"Unsupported encoder backend={args.encoder_backend!r}")


def _build_torch_encoder(args: argparse.Namespace) -> EncoderCallable:
    import torch

    _, _, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=args.checkpoint_path,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = model.to(device)
    model.eval()

    def encode(batch: Any) -> np.ndarray:
        with torch.no_grad():
            return model(batch.to(device)).detach().cpu().numpy()

    return encode


def _build_tensorrt_encoder(args: argparse.Namespace) -> EncoderCallable:
    if not args.tensorrt_config:
        raise ValueError("--tensorrt-config is required when --encoder-backend=tensorrt.")

    import torch

    config = load_tensorrt_fp16_config(config_path=args.tensorrt_config)
    project_root = resolve_project_path(config.project_root, ".")
    metadata_path = resolve_project_path(
        str(project_root),
        config.artifacts.model_bundle_metadata_path,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = load_export_boundary_from_model_metadata(metadata)
    feature_dim = _require_static_axis_size(contract.input_tensor.axes[-1].size, "mel_bins")
    profiles = tuple(
        TensorRTFP16Profile(
            profile_id=profile.profile_id,
            min_shape=(profile.min_batch_size, profile.min_frame_count, feature_dim),
            opt_shape=(profile.opt_batch_size, profile.opt_frame_count, feature_dim),
            max_shape=(profile.max_batch_size, profile.max_frame_count, feature_dim),
        )
        for profile in config.build.profiles
    )
    if args.batch_size > max(profile.max_shape[0] for profile in profiles):
        raise ValueError(
            f"--batch-size={args.batch_size} exceeds the TensorRT max profile batch size "
            f"{max(profile.max_shape[0] for profile in profiles)}."
        )
    engine_path = (
        resolve_project_path(str(project_root), args.tensorrt_engine_path)
        if args.tensorrt_engine_path
        else resolve_project_path(str(project_root), config.artifacts.engine_output_path)
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("TensorRT encoder backend requires a CUDA device.")
    if device.index is not None:
        torch.cuda.set_device(device)
    runner = _TensorRTEngineRunner(
        engine_path=engine_path,
        input_name=contract.input_tensor.name,
        output_name=contract.output_tensor.name,
    )
    args.resolved_tensorrt_engine_path = str(engine_path)
    args.resolved_tensorrt_profile_ids = [profile.profile_id for profile in profiles]
    print(
        f"[official-campp] TensorRT encoder engine={engine_path} "
        f"profiles={[profile.profile_id for profile in profiles]}",
        flush=True,
    )

    def encode(batch: Any) -> np.ndarray:
        shape = (int(batch.shape[0]), int(batch.shape[1]), int(batch.shape[2]))
        if len(shape) != 3 or shape[-1] != feature_dim:
            raise ValueError(f"Unexpected TensorRT input shape {shape}; feature_dim={feature_dim}.")
        profile = _select_profile(profiles, shape=shape)
        profile_index = profiles.index(profile)
        with torch.inference_mode():
            output = runner.run(
                batch.to(device=device, dtype=torch.float32),
                profile_index=profile_index,
            )
        return output.detach().cpu().float().numpy()

    return encode


def _flush_ready_embedding_batches(
    encoder: EncoderCallable,
    batch_features: list[Any],
    batch_owners: list[int],
    sums: dict[int, list[np.ndarray]],
    *,
    batch_size: int,
) -> None:
    while len(batch_features) >= batch_size:
        _flush_embeddings(
            encoder,
            batch_features[:batch_size],
            batch_owners[:batch_size],
            sums,
        )
        del batch_features[:batch_size]
        del batch_owners[:batch_size]


def _flush_embeddings(
    encoder: EncoderCallable,
    batch_features: list[Any],
    batch_owners: list[int],
    sums: dict[int, list[np.ndarray]],
) -> None:
    batch = stack_official_campp_feature_batch(
        [np.asarray(feature, dtype=np.float32) for feature in batch_features]
    )
    values = encoder(batch)
    for owner, embedding in zip(batch_owners, values, strict=True):
        sums.setdefault(owner, []).append(embedding.astype(np.float32, copy=False))


def _frontend_cache_active(args: argparse.Namespace) -> bool:
    return bool(args.frontend_cache_dir) and _resolved_frontend_cache_mode(args) != "off"


def _frontend_cache_root(args: argparse.Namespace) -> Path | None:
    if not args.frontend_cache_dir:
        return None
    return Path(args.frontend_cache_dir)


def _resolved_frontend_cache_mode(args: argparse.Namespace) -> str:
    if not args.frontend_cache_dir:
        return "off"
    return str(args.frontend_cache_mode).lower()


def _require_static_axis_size(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be static for TensorRT CAM++ extraction.")
    return value


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
    parser.add_argument("--encoder-backend", choices=("torch", "tensorrt"), default="torch")
    parser.add_argument("--tensorrt-config", default="")
    parser.add_argument("--tensorrt-engine-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--frontend-workers",
        type=int,
        default=0,
        help="Parallel CPU workers for audio decode, segmenting, and official fbank extraction.",
    )
    parser.add_argument(
        "--frontend-executor",
        choices=("thread", "process"),
        default="thread",
        help="Parallel frontend executor. Process mode can use more CPU but has IPC overhead.",
    )
    parser.add_argument(
        "--frontend-prefetch",
        type=int,
        default=0,
        help="Maximum pending frontend rows. Defaults to max(frontend_workers*4, batch_size).",
    )
    parser.add_argument(
        "--frontend-cache-dir",
        default="",
        help=(
            "Optional persistent exact cache for official CAM++ Fbank segment arrays. "
            "Use an ignored artifacts path such as artifacts/cache/campp-official-public."
        ),
    )
    parser.add_argument(
        "--frontend-cache-mode",
        choices=sorted(SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES),
        default="readwrite",
        help=(
            "Frontend cache policy when --frontend-cache-dir is set. "
            "readwrite reuses hits and writes misses; refresh recomputes and overwrites."
        ),
    )
    parser.add_argument(
        "--frontend-pack-dir",
        default="",
        help=(
            "Optional packed frontend cache directory containing features.npy, row_offsets.npy, "
            "row_counts.npy, and metadata.json. Overrides per-row frontend workers/cache when set."
        ),
    )
    parser.add_argument(
        "--frontend-pack-fast-path",
        action="store_true",
        help="Experimental contiguous packed-cache batching path. Default keeps row pack loop.",
    )
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
    parser.add_argument(
        "--skip-save-embeddings",
        action="store_true",
        help="Do not persist embeddings_<experiment_id>.npy after extraction.",
    )
    parser.add_argument(
        "--skip-save-top-cache",
        action="store_true",
        help="Do not persist indices/scores top-k .npy caches after search.",
    )
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
