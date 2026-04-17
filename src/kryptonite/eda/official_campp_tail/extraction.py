"""Embedding extraction helpers for the official CAM++ tail pipeline."""

from __future__ import annotations

import multiprocessing as mp
import time
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

from kryptonite.features.campp_official import (
    load_official_campp_frontend_feature_pack,
    load_or_compute_official_campp_features,
    stack_official_campp_feature_batch,
)

from .config import (
    OfficialCamPPTailConfig,
    frontend_cache_active,
    frontend_cache_root,
    resolved_frontend_cache_mode,
)
from .encoder import EncoderCallable, build_encoder


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


def load_or_extract_embeddings(
    config: OfficialCamPPTailConfig,
    manifest: pl.DataFrame,
    output_dir: Any,
) -> np.ndarray:
    output_path = output_dir / f"embeddings_{config.experiment_id}.npy"
    if config.embeddings_path:
        embeddings_path = np.load(config.embeddings_path)
        print(
            f"[official-campp] load provided embeddings path={config.embeddings_path}", flush=True
        )
        return embeddings_path
    if output_path.is_file() and not config.force_embeddings:
        print(f"[official-campp] load cached embeddings path={output_path}", flush=True)
        return np.load(output_path)

    encoder = build_encoder(config)
    sums: dict[int, list[np.ndarray]] = {}
    batch_features: list[Any] = []
    batch_owners: list[int] = []
    frontend_stats = FrontendCacheStats()
    started_at = time.perf_counter()
    log_every_rows = max(1, manifest.height // 20)

    if config.frontend_pack_dir:
        _extract_embeddings_from_frontend_pack(
            config,
            manifest,
            encoder=encoder,
            sums=sums,
            batch_features=batch_features,
            batch_owners=batch_owners,
            started_at=started_at,
            log_every_rows=log_every_rows,
        )
        config.frontend_cache_stats = {
            "disabled_rows": 0,
            "hit_rows": manifest.height,
            "miss_rows": 0,
            "written_rows": 0,
            "pack_rows": manifest.height,
        }
        embeddings = _finalize_embeddings(manifest=manifest, sums=sums)
        _save_embeddings_if_requested(config, output_path, embeddings)
        return embeddings

    if config.frontend_workers > 0:
        _extract_embeddings_parallel_frontend(
            config,
            manifest,
            encoder=encoder,
            sums=sums,
            batch_features=batch_features,
            batch_owners=batch_owners,
            frontend_cache_stats=frontend_stats,
            started_at=started_at,
            log_every_rows=log_every_rows,
        )
        config.frontend_cache_stats = frontend_stats.to_dict()
        embeddings = _finalize_embeddings(manifest=manifest, sums=sums)
        _save_embeddings_if_requested(config, output_path, embeddings)
        return embeddings

    for index, row in enumerate(manifest.iter_rows(named=True)):
        row_index, features, cache_hit, cache_written = _frontend_features_for_row(
            config, index, row
        )
        frontend_stats.add(
            cache_active=frontend_cache_active(config),
            cache_hit=cache_hit,
            cache_written=cache_written,
        )
        for feature in features:
            batch_features.append(feature)
            batch_owners.append(row_index)
        if len(batch_features) >= config.batch_size:
            _flush_ready_embedding_batches(
                encoder,
                batch_features,
                batch_owners,
                sums,
                batch_size=config.batch_size,
            )
        row_number = index + 1
        if row_number == 1 or row_number % log_every_rows == 0 or row_number == manifest.height:
            _log_progress(
                prefix="extract",
                completed=row_number,
                total=manifest.height,
                started_at=started_at,
            )
    if batch_features:
        _flush_embeddings(encoder, batch_features, batch_owners, sums)
    config.frontend_cache_stats = frontend_stats.to_dict()
    embeddings = _finalize_embeddings(manifest=manifest, sums=sums)
    _save_embeddings_if_requested(config, output_path, embeddings)
    return embeddings


def _extract_embeddings_from_frontend_pack(
    config: OfficialCamPPTailConfig,
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
        Path(config.frontend_pack_dir),
        mmap_mode="c" if config.frontend_pack_fast_path else "r",
    )
    if pack.row_count < manifest.height:
        raise ValueError(
            f"Packed frontend cache has {pack.row_count} rows, but manifest has {manifest.height}."
        )
    if config.frontend_pack_fast_path and _try_extract_embeddings_from_contiguous_frontend_pack(
        pack=pack,
        manifest=manifest,
        encoder=encoder,
        sums=sums,
        batch_size=config.batch_size,
        started_at=started_at,
        log_every_rows=log_every_rows,
    ):
        return
    if config.frontend_pack_fast_path:
        print("[official-campp] pack-fast unavailable; falling back to row pack loop", flush=True)

    for index, row in enumerate(manifest.iter_rows(named=True)):
        row_index = int(row.get("gallery_index", row.get("row_index", index)))
        for feature in pack.features_for_row(index):
            batch_features.append(feature)
            batch_owners.append(row_index)
        if len(batch_features) >= config.batch_size:
            _flush_ready_embedding_batches(
                encoder,
                batch_features,
                batch_owners,
                sums,
                batch_size=config.batch_size,
            )
        row_number = index + 1
        if row_number == 1 or row_number % log_every_rows == 0 or row_number == manifest.height:
            _log_progress(
                prefix="pack",
                completed=row_number,
                total=manifest.height,
                started_at=started_at,
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
        values = encoder(
            _torch_batch_from_packed_features(pack.features[segment_start:segment_stop])
        )
        if embedding_sums is None:
            embedding_sums = np.zeros((manifest.height, values.shape[1]), dtype=np.float32)
        _accumulate_sorted_owner_embeddings(
            embedding_sums,
            owners=segment_owners[segment_start:segment_stop],
            values=values.astype(np.float32, copy=False),
        )

        completed_rows = int(np.searchsorted(row_ends, segment_stop, side="right"))
        if (
            completed_rows >= next_log_row
            or segment_stop == total_segments
            or completed_rows == manifest.height
        ):
            row_number = min(completed_rows, manifest.height)
            _log_progress(
                prefix="pack-fast",
                completed=row_number,
                total=manifest.height,
                started_at=started_at,
            )
            next_log_row = row_number + log_every_rows

    if embedding_sums is None:
        return False
    embeddings = embedding_sums / owner_counts[:, None]
    for row_index, embedding in enumerate(embeddings):
        sums.setdefault(row_index, []).append(embedding.astype(np.float32, copy=False))
    return True


def _extract_embeddings_parallel_frontend(
    config: OfficialCamPPTailConfig,
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
    max_pending = config.frontend_prefetch or max(config.frontend_workers * 4, config.batch_size)
    iterator = enumerate(manifest.iter_rows(named=True))
    submitted = 0
    completed = 0
    executor_factory: Any
    if config.frontend_executor == "process":
        executor_factory = ProcessPoolExecutor(
            max_workers=config.frontend_workers,
            mp_context=mp.get_context("spawn"),
        )
    else:
        executor_factory = ThreadPoolExecutor(max_workers=config.frontend_workers)

    with executor_factory as executor:
        pending: dict[Future[tuple[int, list[Any], bool, bool]], None] = {}
        for _ in range(max_pending):
            if not _submit_next_frontend_task(config, executor, iterator, pending):
                break
            submitted += 1
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                row_index, features, cache_hit, cache_written = future.result()
                frontend_cache_stats.add(
                    cache_active=frontend_cache_active(config),
                    cache_hit=cache_hit,
                    cache_written=cache_written,
                )
                for feature in features:
                    batch_features.append(feature)
                    batch_owners.append(row_index)
                if len(batch_features) >= config.batch_size:
                    _flush_ready_embedding_batches(
                        encoder,
                        batch_features,
                        batch_owners,
                        sums,
                        batch_size=config.batch_size,
                    )
                completed += 1
                if (
                    completed == 1
                    or completed % log_every_rows == 0
                    or completed == manifest.height
                ):
                    _log_progress(
                        prefix="extract",
                        completed=completed,
                        total=manifest.height,
                        started_at=started_at,
                    )
                if submitted < manifest.height and _submit_next_frontend_task(
                    config, executor, iterator, pending
                ):
                    submitted += 1
    if batch_features:
        _flush_embeddings(encoder, batch_features, batch_owners, sums)


def _submit_next_frontend_task(
    config: OfficialCamPPTailConfig,
    executor: Any,
    iterator: Any,
    pending: dict[Future[tuple[int, list[Any], bool, bool]], None],
) -> bool:
    try:
        index, row = next(iterator)
    except StopIteration:
        return False
    pending[executor.submit(_frontend_features_for_row, config, index, row)] = None
    return True


def _frontend_features_for_row(
    config: OfficialCamPPTailConfig,
    index: int,
    row: dict[str, Any],
) -> tuple[int, list[Any], bool, bool]:
    row_index = int(row.get("gallery_index", row.get("row_index", index)))
    result = load_or_compute_official_campp_features(
        raw_path=str(row["filepath"]),
        data_root=Path(config.data_root),
        sample_rate_hz=config.sample_rate_hz,
        num_mel_bins=config.num_mel_bins,
        mode=config.mode,
        eval_chunk_seconds=config.eval_chunk_seconds,
        segment_count=config.segment_count,
        long_file_threshold_seconds=config.long_file_threshold_seconds,
        pad_mode=config.pad_mode,
        cache_root=frontend_cache_root(config),
        cache_mode=resolved_frontend_cache_mode(config),
    )
    return row_index, result.features, result.cache_hit, result.cache_written


def _flush_ready_embedding_batches(
    encoder: EncoderCallable,
    batch_features: list[Any],
    batch_owners: list[int],
    sums: dict[int, list[np.ndarray]],
    *,
    batch_size: int,
) -> None:
    while len(batch_features) >= batch_size:
        _flush_embeddings(encoder, batch_features[:batch_size], batch_owners[:batch_size], sums)
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


def _finalize_embeddings(
    *,
    manifest: pl.DataFrame,
    sums: dict[int, list[np.ndarray]],
) -> np.ndarray:
    embeddings = np.empty(
        (manifest.height, next(iter(sums.values()))[0].shape[0]), dtype=np.float32
    )
    for index in range(manifest.height):
        embeddings[index] = np.mean(np.stack(sums[index], axis=0), axis=0)
    return embeddings


def _save_embeddings_if_requested(
    config: OfficialCamPPTailConfig,
    output_path: Any,
    embeddings: np.ndarray,
) -> None:
    if not config.skip_save_embeddings:
        np.save(output_path, embeddings)


def _log_progress(*, prefix: str, completed: int, total: int, started_at: float) -> None:
    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
    print(
        f"[official-campp] {prefix} rows={completed}/{total} "
        f"pct={100.0 * completed / total:.1f} "
        f"rows_per_s={completed / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


__all__ = [
    "FrontendCacheStats",
    "load_or_extract_embeddings",
    "_try_extract_embeddings_from_contiguous_frontend_pack",
]
