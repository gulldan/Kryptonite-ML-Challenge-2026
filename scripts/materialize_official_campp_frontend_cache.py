"""Materialize the exact official CAM++ frontend cache for manifest audio rows."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from kryptonite.features.campp_official import (
    SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES,
    build_official_campp_frontend_cache_key,
    compute_official_campp_features,
    resolve_official_campp_frontend_cache_path,
    write_official_campp_frontend_cache,
)


@dataclass(frozen=True, slots=True)
class CacheMaterializationRow:
    row_index: int
    cache_hit: bool
    cache_written: bool
    segment_count: int
    cache_path: str
    elapsed_s: float


@dataclass(slots=True)
class CacheMaterializationStats:
    rows: int = 0
    cache_hit_rows: int = 0
    cache_miss_rows: int = 0
    cache_written_rows: int = 0
    segments_computed: int = 0
    worker_sum_s: float = 0.0

    def add(self, row: CacheMaterializationRow) -> None:
        self.rows += 1
        if row.cache_hit:
            self.cache_hit_rows += 1
        else:
            self.cache_miss_rows += 1
        if row.cache_written:
            self.cache_written_rows += 1
        self.segments_computed += row.segment_count
        self.worker_sum_s += row.elapsed_s


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    if args.limit_rows > 0:
        manifest = manifest.head(args.limit_rows)
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[campp-cache] rows={manifest.height} workers={args.workers} "
        f"cache_mode={args.cache_mode} cache_dir={cache_root}",
        flush=True,
    )
    started = time.perf_counter()
    stats = _materialize(args=args, manifest=manifest, cache_root=cache_root)
    wall_s = time.perf_counter() - started
    payload = {
        "summary": {
            **asdict(stats),
            "wall_s": wall_s,
            "rows_per_s": manifest.height / wall_s if wall_s > 0.0 else 0.0,
            "effective_worker_parallelism": stats.worker_sum_s / wall_s if wall_s > 0.0 else 0.0,
        },
        "config": {
            "manifest_csv": args.manifest_csv,
            "data_root": args.data_root,
            "cache_dir": args.cache_dir,
            "cache_mode": args.cache_mode,
            "workers": args.workers,
            "prefetch": args.prefetch,
            "executor": args.executor,
            "limit_rows": args.limit_rows,
            "sample_rate_hz": args.sample_rate_hz,
            "num_mel_bins": args.num_mel_bins,
            "mode": args.mode,
            "eval_chunk_seconds": args.eval_chunk_seconds,
            "segment_count": args.segment_count,
            "long_file_threshold_seconds": args.long_file_threshold_seconds,
            "pad_mode": args.pad_mode,
        },
    }
    json_path = output_root / "frontend_cache_materialization.json"
    markdown_path = output_root / "frontend_cache_materialization.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"[campp-cache] wrote {json_path}", flush=True)


def _materialize(
    *,
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    cache_root: Path,
) -> CacheMaterializationStats:
    stats = CacheMaterializationStats()
    log_every = max(1, manifest.height // 20)
    max_pending = args.prefetch or max(args.workers * 4, 1)
    iterator = enumerate(manifest.iter_rows(named=True))
    completed = 0
    started = time.perf_counter()
    executor_class: Any = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    with executor_class(max_workers=args.workers) as executor:
        futures: dict[Future[CacheMaterializationRow], None] = {}
        for _ in range(max_pending):
            if not _submit_next(args, cache_root, iterator, executor, futures):
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                row = future.result()
                stats.add(row)
                completed += 1
                if completed == 1 or completed % log_every == 0 or completed == manifest.height:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"[campp-cache] rows={completed}/{manifest.height} "
                        f"hit={stats.cache_hit_rows} miss={stats.cache_miss_rows} "
                        f"written={stats.cache_written_rows} rows_per_s={completed / elapsed:.1f}",
                        flush=True,
                    )
                _submit_next(args, cache_root, iterator, executor, futures)
    return stats


def _submit_next(
    args: argparse.Namespace,
    cache_root: Path,
    iterator: Iterator[tuple[int, dict[str, Any]]],
    executor: Any,
    futures: dict[Future[CacheMaterializationRow], None],
) -> bool:
    try:
        index, row = next(iterator)
    except StopIteration:
        return False
    futures[executor.submit(_materialize_row, args, cache_root, index, row)] = None
    return True


def _materialize_row(
    args: argparse.Namespace,
    cache_root: Path,
    index: int,
    row: dict[str, Any],
) -> CacheMaterializationRow:
    started = time.perf_counter()
    raw_path = str(row["filepath"])
    cache_key = build_official_campp_frontend_cache_key(
        raw_path,
        data_root=Path(args.data_root),
        sample_rate_hz=args.sample_rate_hz,
        num_mel_bins=args.num_mel_bins,
        mode=args.mode,
        eval_chunk_seconds=args.eval_chunk_seconds,
        segment_count=args.segment_count,
        long_file_threshold_seconds=args.long_file_threshold_seconds,
        pad_mode=args.pad_mode,
    )
    cache_path = resolve_official_campp_frontend_cache_path(cache_root, cache_key)
    if args.cache_mode != "refresh" and cache_path.is_file():
        return CacheMaterializationRow(
            row_index=index,
            cache_hit=True,
            cache_written=False,
            segment_count=0,
            cache_path=str(cache_path),
            elapsed_s=time.perf_counter() - started,
        )
    if args.cache_mode == "readonly":
        return CacheMaterializationRow(
            row_index=index,
            cache_hit=False,
            cache_written=False,
            segment_count=0,
            cache_path=str(cache_path),
            elapsed_s=time.perf_counter() - started,
        )
    features = compute_official_campp_features(
        raw_path,
        data_root=Path(args.data_root),
        sample_rate_hz=args.sample_rate_hz,
        num_mel_bins=args.num_mel_bins,
        mode=args.mode,
        eval_chunk_seconds=args.eval_chunk_seconds,
        segment_count=args.segment_count,
        long_file_threshold_seconds=args.long_file_threshold_seconds,
        pad_mode=args.pad_mode,
    )
    write_official_campp_frontend_cache(cache_path, features)
    return CacheMaterializationRow(
        row_index=index,
        cache_hit=False,
        cache_written=True,
        segment_count=len(features),
        cache_path=str(cache_path),
        elapsed_s=time.perf_counter() - started,
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    config = payload["config"]
    lines = [
        "# Official CAM++ Frontend Cache Materialization",
        "",
        f"- Manifest: `{config['manifest_csv']}`",
        f"- Cache dir: `{config['cache_dir']}`",
        f"- Cache mode: `{config['cache_mode']}`",
        f"- Workers: `{config['workers']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Cache hits: `{summary['cache_hit_rows']}`",
        f"- Cache misses: `{summary['cache_miss_rows']}`",
        f"- Cache writes: `{summary['cache_written_rows']}`",
        f"- Wall seconds: `{summary['wall_s']:.6f}`",
        f"- Rows/s: `{summary['rows_per_s']:.3f}`",
        f"- Worker summed seconds: `{summary['worker_sum_s']:.6f}`",
        f"- Effective worker parallelism: `{summary['effective_worker_parallelism']:.3f}`",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--data-root", default="datasets/Для участников")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--cache-mode",
        choices=sorted(SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES - {"off"}),
        default="readwrite",
    )
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=256)
    parser.add_argument("--executor", choices=("thread", "process"), default="thread")
    parser.add_argument(
        "--mode",
        choices=("full_file", "single_crop", "segment_mean"),
        default="segment_mean",
    )
    parser.add_argument("--eval-chunk-seconds", type=float, default=6.0)
    parser.add_argument("--segment-count", type=int, default=3)
    parser.add_argument("--long-file-threshold-seconds", type=float, default=6.0)
    parser.add_argument("--pad-mode", choices=("repeat", "zero"), default="repeat")
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--num-mel-bins", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    main()
