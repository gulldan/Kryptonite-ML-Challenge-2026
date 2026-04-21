"""CLI configuration for the official CAM++ tail pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kryptonite.features.campp_official import (
    SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES,
)


@dataclass(slots=True)
class OfficialCamPPTailConfig:
    checkpoint_path: str
    manifest_csv: str
    output_dir: str
    experiment_id: str
    template_csv: str = ""
    data_root: str = "data/Для участников"
    embeddings_path: str = ""
    encoder_backend: str = "torch"
    tensorrt_config: str = ""
    tensorrt_engine_path: str = ""
    device: str = "cuda"
    search_device: str = "cuda"
    batch_size: int = 512
    frontend_workers: int = 0
    frontend_executor: str = "thread"
    frontend_prefetch: int = 0
    frontend_cache_dir: str = ""
    frontend_cache_mode: str = "readwrite"
    frontend_pack_dir: str = ""
    frontend_pack_fast_path: bool = False
    search_batch_size: int = 2048
    top_cache_k: int = 100
    output_top_k: int = 10
    sample_rate_hz: int = 16_000
    num_mel_bins: int = 80
    mode: str = "segment_mean"
    eval_chunk_seconds: float = 6.0
    segment_count: int = 3
    long_file_threshold_seconds: float = 6.0
    pad_mode: str = "repeat"
    force_embeddings: bool = False
    skip_save_embeddings: bool = False
    skip_save_top_cache: bool = False
    skip_c4: bool = False
    edge_top: int = 10
    reciprocal_top: int = 20
    rank_top: int = 100
    iterations: int = 5
    label_min_size: int = 5
    label_max_size: int = 120
    label_min_candidates: int = 3
    shared_top: int = 20
    shared_min_count: int = 0
    reciprocal_bonus: float = 0.03
    density_penalty: float = 0.02
    frontend_cache_stats: dict[str, int] | None = None
    resolved_tensorrt_engine_path: str = ""
    resolved_tensorrt_profile_ids: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAM++ with the official 3D-Speaker frontend and retrieval tails."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--data-root", default="data/Для участников")
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
            "Use an ignored data path such as data/campp_runs/ms42_release/cache/frontend."
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
    parser.add_argument("--output-top-k", type=int, default=10)
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--num-mel-bins", type=int, default=80)
    parser.add_argument(
        "--mode",
        choices=("full_file", "single_crop", "segment_mean"),
        default="segment_mean",
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> OfficialCamPPTailConfig:
    namespace = build_parser().parse_args(argv)
    return OfficialCamPPTailConfig(**vars(namespace))


def frontend_cache_active(config: OfficialCamPPTailConfig) -> bool:
    return bool(config.frontend_cache_dir) and resolved_frontend_cache_mode(config) != "off"


def frontend_cache_root(config: OfficialCamPPTailConfig) -> Path | None:
    if not config.frontend_cache_dir:
        return None
    return Path(config.frontend_cache_dir)


def resolved_frontend_cache_mode(config: OfficialCamPPTailConfig) -> str:
    if not config.frontend_cache_dir:
        return "off"
    return config.frontend_cache_mode.lower()


__all__ = [
    "OfficialCamPPTailConfig",
    "build_parser",
    "frontend_cache_active",
    "frontend_cache_root",
    "parse_args",
    "resolved_frontend_cache_mode",
]
