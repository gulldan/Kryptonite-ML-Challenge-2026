"""Profile CAM++ public-inference pipeline stages on real audio rows."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.deployment import resolve_project_path
from kryptonite.eda.community import exact_topk
from kryptonite.features.campp_official import (
    SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES,
    build_official_campp_frontend_cache_key,
    load_official_campp_frontend_cache,
    load_official_campp_waveform,
    official_campp_fbank,
    official_campp_segments_for_mode,
    resolve_official_campp_frontend_cache_path,
    stack_official_campp_feature_batch,
    write_official_campp_frontend_cache,
)
from kryptonite.runtime.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.runtime.tensorrt_engine_config import load_tensorrt_fp16_config
from kryptonite.runtime.tensorrt_engine_models import TensorRTFP16Profile
from kryptonite.runtime.tensorrt_engine_runtime import _select_profile, _TensorRTEngineRunner


@dataclass(slots=True)
class FrontendTiming:
    cache_load_s: float = 0.0
    decode_s: float = 0.0
    segment_s: float = 0.0
    fbank_s: float = 0.0
    cache_write_s: float = 0.0
    rows: int = 0
    segments: int = 0
    cache_hit_rows: int = 0
    cache_miss_rows: int = 0
    cache_written_rows: int = 0

    @property
    def total_s(self) -> float:
        return (
            self.cache_load_s + self.decode_s + self.segment_s + self.fbank_s + self.cache_write_s
        )

    def add(self, other: FrontendTiming) -> None:
        self.cache_load_s += other.cache_load_s
        self.decode_s += other.decode_s
        self.segment_s += other.segment_s
        self.fbank_s += other.fbank_s
        self.cache_write_s += other.cache_write_s
        self.rows += other.rows
        self.segments += other.segments
        self.cache_hit_rows += other.cache_hit_rows
        self.cache_miss_rows += other.cache_miss_rows
        self.cache_written_rows += other.cache_written_rows


@dataclass(slots=True)
class EncoderTiming:
    padding_s: float = 0.0
    h2d_s: float = 0.0
    execute_s: float = 0.0
    d2h_s: float = 0.0
    aggregate_s: float = 0.0
    batches: int = 0
    segments: int = 0

    @property
    def total_s(self) -> float:
        return self.padding_s + self.h2d_s + self.execute_s + self.d2h_s + self.aggregate_s


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    if args.limit_rows > 0:
        manifest = manifest.head(args.limit_rows)

    print(
        f"[stage-profile] rows={manifest.height} batch_size={args.batch_size} "
        f"frontend_workers={args.frontend_workers}",
        flush=True,
    )
    torch = _import_torch()
    runner, profiles, feature_dim = _build_tensorrt_runner(args)
    frontend = FrontendTiming()
    encoder = EncoderTiming()
    sums: dict[int, list[np.ndarray]] = {}
    pending_features: list[np.ndarray] = []
    pending_owners: list[int] = []
    wall_started = time.perf_counter()
    frontend_wall_started = time.perf_counter()
    if args.frontend_workers > 0:
        _run_parallel_frontend(
            args=args,
            manifest=manifest,
            runner=runner,
            profiles=profiles,
            feature_dim=feature_dim,
            torch=torch,
            frontend=frontend,
            encoder=encoder,
            sums=sums,
            pending_features=pending_features,
            pending_owners=pending_owners,
        )
    else:
        for index, row in enumerate(manifest.iter_rows(named=True)):
            row_index, features, row_timing = _frontend_features_for_row(args, index, row)
            frontend.add(row_timing)
            pending_features.extend(features)
            pending_owners.extend([row_index] * len(features))
            _flush_ready_batches(
                args=args,
                runner=runner,
                profiles=profiles,
                feature_dim=feature_dim,
                torch=torch,
                frontend=frontend,
                encoder=encoder,
                sums=sums,
                pending_features=pending_features,
                pending_owners=pending_owners,
            )
    frontend_wall_s = time.perf_counter() - frontend_wall_started
    if pending_features:
        _flush_batch(
            args=args,
            runner=runner,
            profiles=profiles,
            feature_dim=feature_dim,
            torch=torch,
            encoder=encoder,
            sums=sums,
            pending_features=pending_features,
            pending_owners=pending_owners,
        )

    aggregation_started = time.perf_counter()
    embeddings = np.empty(
        (manifest.height, next(iter(sums.values()))[0].shape[0]), dtype=np.float32
    )
    for index in range(manifest.height):
        embeddings[index] = np.mean(np.stack(sums[index], axis=0), axis=0)
    encoder.aggregate_s += time.perf_counter() - aggregation_started

    search_started = time.perf_counter()
    top_k = min(args.top_k, manifest.height - 1)
    indices, scores = exact_topk(
        embeddings,
        top_k=top_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    search_s = time.perf_counter() - search_started
    wall_total_s = time.perf_counter() - wall_started

    np.save(output_root / "embeddings.npy", embeddings)
    np.save(output_root / f"indices_top{top_k}.npy", indices)
    np.save(output_root / f"scores_top{top_k}.npy", scores)
    payload = _build_report(
        args=args,
        manifest=manifest,
        frontend=frontend,
        encoder=encoder,
        frontend_wall_s=frontend_wall_s,
        search_s=search_s,
        wall_total_s=wall_total_s,
        top_k=top_k,
        scores=scores,
    )
    (output_root / "stage_profile.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "stage_profile.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"[stage-profile] wrote {output_root / 'stage_profile.json'}", flush=True)


def _run_parallel_frontend(
    *,
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    runner: _TensorRTEngineRunner,
    profiles: tuple[TensorRTFP16Profile, ...],
    feature_dim: int,
    torch: Any,
    frontend: FrontendTiming,
    encoder: EncoderTiming,
    sums: dict[int, list[np.ndarray]],
    pending_features: list[np.ndarray],
    pending_owners: list[int],
) -> None:
    max_pending = args.frontend_prefetch or max(args.frontend_workers * 4, args.batch_size)
    iterator = enumerate(manifest.iter_rows(named=True))
    completed = 0
    log_every = max(1, manifest.height // 10)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.frontend_workers) as executor:
        futures: dict[Future[tuple[int, list[np.ndarray], FrontendTiming]], None] = {}
        for _ in range(max_pending):
            if not _submit_next(args, executor, iterator, futures):
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                row_index, features, timing = future.result()
                frontend.add(timing)
                pending_features.extend(features)
                pending_owners.extend([row_index] * len(features))
                _flush_ready_batches(
                    args=args,
                    runner=runner,
                    profiles=profiles,
                    feature_dim=feature_dim,
                    torch=torch,
                    frontend=frontend,
                    encoder=encoder,
                    sums=sums,
                    pending_features=pending_features,
                    pending_owners=pending_owners,
                )
                completed += 1
                if completed == 1 or completed % log_every == 0 or completed == manifest.height:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"[stage-profile] frontend rows={completed}/{manifest.height} "
                        f"rows_per_s={completed / elapsed:.1f}",
                        flush=True,
                    )
                _submit_next(args, executor, iterator, futures)


def _submit_next(
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    iterator: Iterator[tuple[int, dict[str, Any]]],
    futures: dict[Future[tuple[int, list[np.ndarray], FrontendTiming]], None],
) -> bool:
    try:
        index, row = next(iterator)
    except StopIteration:
        return False
    futures[executor.submit(_frontend_features_for_row, args, index, row)] = None
    return True


def _frontend_features_for_row(
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
) -> tuple[int, list[np.ndarray], FrontendTiming]:
    timing = FrontendTiming(rows=1)
    row_index = int(row.get("gallery_index", row.get("row_index", index)))
    cache_path = _frontend_cache_path(args, str(row["filepath"]))
    cache_mode = _resolved_frontend_cache_mode(args)
    if cache_path is not None and cache_mode != "refresh" and cache_path.is_file():
        started = time.perf_counter()
        features = load_official_campp_frontend_cache(cache_path)
        timing.cache_load_s += time.perf_counter() - started
        timing.cache_hit_rows += 1
        timing.segments += len(features)
        return row_index, features, timing
    if cache_path is not None:
        timing.cache_miss_rows += 1

    started = time.perf_counter()
    waveform = load_official_campp_waveform(
        str(row["filepath"]),
        data_root=Path(args.data_root),
        sample_rate_hz=args.sample_rate_hz,
    )
    timing.decode_s += time.perf_counter() - started
    started = time.perf_counter()
    segments = _segments_for_mode(args, waveform)
    timing.segment_s += time.perf_counter() - started
    features = []
    started = time.perf_counter()
    for segment in segments:
        features.append(
            official_campp_fbank(
                segment,
                sample_rate_hz=args.sample_rate_hz,
                num_mel_bins=args.num_mel_bins,
            )
            .contiguous()
            .numpy()
        )
    timing.fbank_s += time.perf_counter() - started
    if cache_path is not None and cache_mode in {"readwrite", "refresh"}:
        started = time.perf_counter()
        write_official_campp_frontend_cache(cache_path, features)
        timing.cache_write_s += time.perf_counter() - started
        timing.cache_written_rows += 1
    timing.segments += len(features)
    return row_index, features, timing


def _segments_for_mode(args: argparse.Namespace, waveform: Any) -> list[Any]:
    return official_campp_segments_for_mode(
        waveform,
        mode=args.mode,
        sample_rate_hz=args.sample_rate_hz,
        eval_chunk_seconds=args.eval_chunk_seconds,
        segment_count=args.segment_count,
        long_file_threshold_seconds=args.long_file_threshold_seconds,
        pad_mode=args.pad_mode,
    )


def _flush_ready_batches(
    *,
    args: argparse.Namespace,
    runner: _TensorRTEngineRunner,
    profiles: tuple[TensorRTFP16Profile, ...],
    feature_dim: int,
    torch: Any,
    frontend: FrontendTiming,
    encoder: EncoderTiming,
    sums: dict[int, list[np.ndarray]],
    pending_features: list[np.ndarray],
    pending_owners: list[int],
) -> None:
    while len(pending_features) >= args.batch_size:
        _flush_batch(
            args=args,
            runner=runner,
            profiles=profiles,
            feature_dim=feature_dim,
            torch=torch,
            encoder=encoder,
            sums=sums,
            pending_features=pending_features[: args.batch_size],
            pending_owners=pending_owners[: args.batch_size],
        )
        del pending_features[: args.batch_size]
        del pending_owners[: args.batch_size]


def _flush_batch(
    *,
    args: argparse.Namespace,
    runner: _TensorRTEngineRunner,
    profiles: tuple[TensorRTFP16Profile, ...],
    feature_dim: int,
    torch: Any,
    encoder: EncoderTiming,
    sums: dict[int, list[np.ndarray]],
    pending_features: list[np.ndarray],
    pending_owners: list[int],
) -> None:
    started = time.perf_counter()
    cpu_batch = stack_official_campp_feature_batch(pending_features)
    encoder.padding_s += time.perf_counter() - started

    started = time.perf_counter()
    gpu_batch = cpu_batch.to(device=args.device, dtype=torch.float32, non_blocking=False)
    torch.cuda.synchronize()
    encoder.h2d_s += time.perf_counter() - started

    profile = _select_profile(
        profiles,
        shape=(int(gpu_batch.shape[0]), int(gpu_batch.shape[1]), feature_dim),
    )
    started = time.perf_counter()
    with torch.inference_mode():
        output = runner.run(gpu_batch, profile_index=profiles.index(profile))
    torch.cuda.synchronize()
    encoder.execute_s += time.perf_counter() - started

    started = time.perf_counter()
    values = output.detach().cpu().float().numpy()
    encoder.d2h_s += time.perf_counter() - started
    encoder.batches += 1
    encoder.segments += len(pending_features)
    for owner, embedding in zip(pending_owners, values, strict=True):
        sums.setdefault(owner, []).append(embedding.astype(np.float32, copy=False))


def _build_tensorrt_runner(
    args: argparse.Namespace,
) -> tuple[_TensorRTEngineRunner, tuple[TensorRTFP16Profile, ...], int]:
    config = load_tensorrt_fp16_config(config_path=args.tensorrt_config)
    project_root = resolve_project_path(config.project_root, ".")
    metadata_path = resolve_project_path(
        str(project_root), config.artifacts.model_bundle_metadata_path
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = load_export_boundary_from_model_metadata(metadata)
    feature_dim = _require_int(contract.input_tensor.axes[-1].size, "feature_dim")
    profiles = tuple(
        TensorRTFP16Profile(
            profile_id=profile.profile_id,
            min_shape=(profile.min_batch_size, profile.min_frame_count, feature_dim),
            opt_shape=(profile.opt_batch_size, profile.opt_frame_count, feature_dim),
            max_shape=(profile.max_batch_size, profile.max_frame_count, feature_dim),
        )
        for profile in config.build.profiles
    )
    engine_path = resolve_project_path(str(project_root), config.artifacts.engine_output_path)
    runner = _TensorRTEngineRunner(
        engine_path=engine_path,
        input_name=contract.input_tensor.name,
        output_name=contract.output_tensor.name,
    )
    return runner, profiles, feature_dim


def _build_report(
    *,
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    frontend: FrontendTiming,
    encoder: EncoderTiming,
    frontend_wall_s: float,
    search_s: float,
    wall_total_s: float,
    top_k: int,
    scores: np.ndarray,
) -> dict[str, Any]:
    stage_s = {
        "frontend_wall": frontend_wall_s,
        "frontend_cache_load_sum": frontend.cache_load_s,
        "frontend_decode_sum": frontend.decode_s,
        "frontend_segment_sum": frontend.segment_s,
        "frontend_fbank_sum": frontend.fbank_s,
        "frontend_cache_write_sum": frontend.cache_write_s,
        "encoder_padding": encoder.padding_s,
        "encoder_h2d": encoder.h2d_s,
        "encoder_execute": encoder.execute_s,
        "encoder_d2h": encoder.d2h_s,
        "embedding_aggregate": encoder.aggregate_s,
        "exact_topk": search_s,
    }
    wall_shares = {
        name: (value / wall_total_s if wall_total_s > 0 else 0.0)
        for name, value in stage_s.items()
        if not name.endswith("_sum")
    }
    return {
        "summary": {
            "rows": manifest.height,
            "segments": frontend.segments,
            "encoder_batches": encoder.batches,
            "wall_total_s": wall_total_s,
            "rows_per_s": manifest.height / wall_total_s,
            "segments_per_s": frontend.segments / wall_total_s,
            "top_k": top_k,
            "top1_score_mean": float(scores[:, 0].mean()),
            "topk_score_mean": float(scores.mean()),
        },
        "config": {
            "manifest_csv": args.manifest_csv,
            "data_root": args.data_root,
            "tensorrt_config": args.tensorrt_config,
            "batch_size": args.batch_size,
            "frontend_workers": args.frontend_workers,
            "frontend_prefetch": args.frontend_prefetch,
            "frontend_cache_dir": args.frontend_cache_dir,
            "frontend_cache_mode": _resolved_frontend_cache_mode(args),
            "mode": args.mode,
            "segment_count": args.segment_count,
            "eval_chunk_seconds": args.eval_chunk_seconds,
        },
        "stage_seconds": stage_s,
        "wall_stage_shares": wall_shares,
        "frontend_timing": asdict(frontend) | {"total_s": frontend.total_s},
        "encoder_timing": asdict(encoder) | {"total_s": encoder.total_s},
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# CAM++ Pipeline Stage Profile",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Segments: `{summary['segments']}`",
        f"- Wall total seconds: `{summary['wall_total_s']:.6f}`",
        f"- Rows/s: `{summary['rows_per_s']:.3f}`",
        "",
        "| Stage | Seconds | Wall share |",
        "| --- | ---: | ---: |",
    ]
    shares = payload["wall_stage_shares"]
    for name, seconds in payload["stage_seconds"].items():
        share = shares.get(name)
        share_text = "" if share is None else f"{share:.4f}"
        lines.append(f"| `{name}` | {seconds:.6f} | {share_text} |")
    return "\n".join(lines) + "\n"


def _frontend_cache_path(args: argparse.Namespace, raw_path: str) -> Path | None:
    cache_root = _frontend_cache_root(args)
    if cache_root is None or _resolved_frontend_cache_mode(args) == "off":
        return None
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
    return resolve_official_campp_frontend_cache_path(cache_root, cache_key)


def _frontend_cache_root(args: argparse.Namespace) -> Path | None:
    if not args.frontend_cache_dir:
        return None
    return Path(args.frontend_cache_dir)


def _resolved_frontend_cache_mode(args: argparse.Namespace) -> str:
    if not args.frontend_cache_dir:
        return "off"
    return str(args.frontend_cache_mode).lower()


def _require_int(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be static.")
    return value


def _import_torch() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage profiling with TensorRT requires CUDA.")
    return torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--data-root", default="datasets/Для участников")
    parser.add_argument("--tensorrt-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit-rows", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frontend-workers", type=int, default=16)
    parser.add_argument("--frontend-prefetch", type=int, default=256)
    parser.add_argument(
        "--frontend-cache-dir",
        default="",
        help="Optional persistent exact cache for official CAM++ Fbank segment arrays.",
    )
    parser.add_argument(
        "--frontend-cache-mode",
        choices=sorted(SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES),
        default="readwrite",
    )
    parser.add_argument("--search-batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--mode", choices=("full_file", "single_crop", "segment_mean"), default="segment_mean"
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
