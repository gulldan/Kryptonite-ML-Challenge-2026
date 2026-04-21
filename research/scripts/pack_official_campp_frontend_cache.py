"""Pack per-row official CAM++ frontend cache files into a contiguous feature array."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.features.campp_official import (
    build_official_campp_frontend_cache_key,
    resolve_official_campp_frontend_cache_path,
)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    row_index: int
    path: Path
    segment_count: int
    frame_count: int
    feature_dim: int


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    if args.limit_rows > 0:
        manifest = manifest.head(args.limit_rows)

    started = time.perf_counter()
    entries = _scan_entries(args=args, manifest=manifest)
    scan_s = time.perf_counter() - started
    total_segments = sum(entry.segment_count for entry in entries)
    if not entries:
        raise ValueError("Cannot pack an empty manifest.")
    frame_count = entries[0].frame_count
    feature_dim = entries[0].feature_dim

    print(
        f"[campp-pack] rows={len(entries)} segments={total_segments} "
        f"shape=({frame_count}, {feature_dim}) output={output_dir}",
        flush=True,
    )
    pack_started = time.perf_counter()
    features_path = output_dir / "features.npy"
    row_offsets = np.empty(len(entries), dtype=np.int64)
    row_counts = np.empty(len(entries), dtype=np.int32)
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_segments, frame_count, feature_dim),
    )
    offset = 0
    log_every = max(1, len(entries) // 20)
    for entry_index, entry in enumerate(entries):
        cached = np.load(entry.path, mmap_mode="r", allow_pickle=False)
        row_offsets[entry.row_index] = offset
        row_counts[entry.row_index] = entry.segment_count
        features[offset : offset + entry.segment_count] = cached
        offset += entry.segment_count
        if (
            entry_index == 0
            or (entry_index + 1) % log_every == 0
            or entry_index == len(entries) - 1
        ):
            elapsed = max(time.perf_counter() - pack_started, 1e-9)
            print(
                f"[campp-pack] rows={entry_index + 1}/{len(entries)} "
                f"segments={offset}/{total_segments} rows_per_s={(entry_index + 1) / elapsed:.1f}",
                flush=True,
            )
    features.flush()
    np.save(output_dir / "row_offsets.npy", row_offsets, allow_pickle=False)
    np.save(output_dir / "row_counts.npy", row_counts, allow_pickle=False)
    pack_s = time.perf_counter() - pack_started
    metadata = {
        "format_version": "kryptonite.campp.official.frontend.pack.v1",
        "manifest_csv": args.manifest_csv,
        "cache_dir": args.cache_dir,
        "row_count": len(entries),
        "total_segments": total_segments,
        "feature_shape": [frame_count, feature_dim],
        "feature_dtype": "float32",
        "scan_s": scan_s,
        "pack_s": pack_s,
        "wall_s": time.perf_counter() - started,
        "config": _frontend_config_payload(args),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "frontend_cache_pack.md").write_text(
        _render_markdown(metadata),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def _scan_entries(*, args: argparse.Namespace, manifest: pl.DataFrame) -> list[CacheEntry]:
    entries: list[CacheEntry] = []
    expected_shape: tuple[int, int] | None = None
    cache_root = Path(args.cache_dir)
    started = time.perf_counter()
    log_every = max(1, manifest.height // 20)
    for index, row in enumerate(manifest.iter_rows(named=True)):
        cache_key = build_official_campp_frontend_cache_key(
            str(row["filepath"]),
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
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing frontend cache for row {index}: {cache_path}")
        cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if cached.ndim != 3:
            raise ValueError(f"Expected 3D cache array at {cache_path}, got {cached.shape}.")
        shape = (int(cached.shape[1]), int(cached.shape[2]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"Feature shape mismatch at row {index}: {shape} != {expected_shape}.")
        entries.append(
            CacheEntry(
                row_index=index,
                path=cache_path,
                segment_count=int(cached.shape[0]),
                frame_count=shape[0],
                feature_dim=shape[1],
            )
        )
        if index == 0 or (index + 1) % log_every == 0 or index + 1 == manifest.height:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[campp-pack] scan rows={index + 1}/{manifest.height} "
                f"rows_per_s={(index + 1) / elapsed:.1f}",
                flush=True,
            )
    return entries


def _frontend_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_root": args.data_root,
        "sample_rate_hz": args.sample_rate_hz,
        "num_mel_bins": args.num_mel_bins,
        "mode": args.mode,
        "eval_chunk_seconds": args.eval_chunk_seconds,
        "segment_count": args.segment_count,
        "long_file_threshold_seconds": args.long_file_threshold_seconds,
        "pad_mode": args.pad_mode,
    }


def _render_markdown(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Official CAM++ Frontend Cache Pack",
            "",
            f"- Rows: `{metadata['row_count']}`",
            f"- Segments: `{metadata['total_segments']}`",
            f"- Feature shape: `{metadata['feature_shape']}`",
            f"- Scan seconds: `{metadata['scan_s']:.6f}`",
            f"- Pack seconds: `{metadata['pack_s']:.6f}`",
            f"- Wall seconds: `{metadata['wall_s']:.6f}`",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--data-root", default="datasets/Для участников")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-rows", type=int, default=0)
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
