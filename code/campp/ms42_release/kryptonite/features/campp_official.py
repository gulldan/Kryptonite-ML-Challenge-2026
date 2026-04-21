"""Official 3D-Speaker CAM++ frontend helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

OFFICIAL_CAMPP_FRONTEND_CACHE_FORMAT_VERSION = "kryptonite.campp.official.frontend.cache.v1"
SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES = frozenset(
    {"off", "readonly", "readwrite", "refresh"}
)


@dataclass(frozen=True, slots=True)
class OfficialCAMPPFrontendCacheKey:
    cache_id: str
    relative_path: str
    resolved_audio_path: str


@dataclass(frozen=True, slots=True)
class OfficialCAMPPFrontendCacheResult:
    features: list[np.ndarray]
    cache_hit: bool
    cache_written: bool
    cache_path: str | None


@dataclass(frozen=True, slots=True)
class OfficialCAMPPFrontendFeaturePack:
    pack_dir: Path
    features: np.ndarray
    row_offsets: np.ndarray
    row_counts: np.ndarray
    metadata: dict[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.row_counts.shape[0])

    def features_for_row(self, row_index: int) -> list[np.ndarray]:
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError(f"row_index {row_index} is outside pack row count {self.row_count}.")
        offset = int(self.row_offsets[row_index])
        count = int(self.row_counts[row_index])
        if count <= 0:
            raise ValueError(f"Packed frontend cache row {row_index} has no segments.")
        return [
            np.asarray(self.features[offset + segment_index], dtype=np.float32)
            for segment_index in range(count)
        ]


def resolve_audio_path(raw_path: str, data_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return data_root / path


def load_official_campp_waveform(
    raw_path: str,
    *,
    data_root: Path,
    sample_rate_hz: int,
) -> Any:
    import soundfile as sf
    import torch
    import torchaudio

    samples, sample_rate = sf.read(
        resolve_audio_path(raw_path, data_root),
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(samples.T.copy())
    if int(sample_rate) != sample_rate_hz:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), sample_rate_hz)
    return waveform.mean(dim=0)


def repeat_or_pad_waveform(waveform: Any, *, target_samples: int, pad_mode: str = "repeat") -> Any:
    import torch

    values = waveform.flatten().float()
    if values.numel() >= target_samples:
        return values[:target_samples]
    if values.numel() == 0:
        return torch.zeros(target_samples, dtype=torch.float32)
    if pad_mode == "repeat":
        repeats = int(np.ceil(target_samples / values.numel()))
        return values.repeat(repeats)[:target_samples]
    return torch.nn.functional.pad(values, (0, target_samples - values.numel()))


def even_waveform_segments(
    waveform: Any,
    *,
    sample_rate_hz: int,
    chunk_seconds: float,
    segment_count: int,
    pad_mode: str = "repeat",
) -> list[Any]:
    values = waveform.flatten().float()
    chunk_samples = int(round(sample_rate_hz * chunk_seconds))
    if values.numel() <= chunk_samples:
        return [repeat_or_pad_waveform(values, target_samples=chunk_samples, pad_mode=pad_mode)]

    max_start = int(values.numel()) - chunk_samples
    if segment_count <= 1:
        starts = [max_start // 2]
    else:
        starts = np.linspace(0, max_start, num=segment_count, dtype=int).tolist()
    return [values[start : start + chunk_samples] for start in starts]


def official_campp_segments_for_mode(
    waveform: Any,
    *,
    mode: str,
    sample_rate_hz: int,
    eval_chunk_seconds: float,
    segment_count: int,
    long_file_threshold_seconds: float,
    pad_mode: str = "repeat",
) -> list[Any]:
    if mode == "full_file":
        return [waveform.flatten().float()]
    if mode == "single_crop":
        return even_waveform_segments(
            waveform,
            sample_rate_hz=sample_rate_hz,
            chunk_seconds=eval_chunk_seconds,
            segment_count=1,
            pad_mode=pad_mode,
        )
    if mode == "segment_mean":
        duration_s = float(waveform.numel()) / float(sample_rate_hz)
        resolved_segment_count = segment_count if duration_s > long_file_threshold_seconds else 1
        return even_waveform_segments(
            waveform,
            sample_rate_hz=sample_rate_hz,
            chunk_seconds=eval_chunk_seconds,
            segment_count=resolved_segment_count,
            pad_mode=pad_mode,
        )
    raise ValueError(f"Unsupported mode={mode!r}")


def official_campp_fbank(
    waveform: Any,
    *,
    sample_rate_hz: int,
    num_mel_bins: int,
) -> Any:
    import torchaudio.compliance.kaldi as kaldi

    features = kaldi.fbank(
        waveform.flatten().float().unsqueeze(0),
        num_mel_bins=int(num_mel_bins),
        sample_frequency=float(sample_rate_hz),
        dither=0.0,
    )
    return features - features.mean(dim=0, keepdim=True)


def load_or_compute_official_campp_features(
    raw_path: str,
    *,
    data_root: Path,
    sample_rate_hz: int,
    num_mel_bins: int,
    mode: str,
    eval_chunk_seconds: float,
    segment_count: int,
    long_file_threshold_seconds: float,
    pad_mode: str = "repeat",
    cache_root: Path | None = None,
    cache_mode: str = "off",
) -> OfficialCAMPPFrontendCacheResult:
    normalized_cache_mode = cache_mode.lower()
    if normalized_cache_mode not in SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES:
        raise ValueError(
            f"cache_mode must be one of {sorted(SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES)}"
        )

    cache_path: Path | None = None
    if cache_root is not None and normalized_cache_mode != "off":
        cache_key = build_official_campp_frontend_cache_key(
            raw_path,
            data_root=data_root,
            sample_rate_hz=sample_rate_hz,
            num_mel_bins=num_mel_bins,
            mode=mode,
            eval_chunk_seconds=eval_chunk_seconds,
            segment_count=segment_count,
            long_file_threshold_seconds=long_file_threshold_seconds,
            pad_mode=pad_mode,
        )
        cache_path = resolve_official_campp_frontend_cache_path(cache_root, cache_key)
        if normalized_cache_mode != "refresh" and cache_path.is_file():
            return OfficialCAMPPFrontendCacheResult(
                features=load_official_campp_frontend_cache(cache_path),
                cache_hit=True,
                cache_written=False,
                cache_path=str(cache_path),
            )

    features = compute_official_campp_features(
        raw_path,
        data_root=data_root,
        sample_rate_hz=sample_rate_hz,
        num_mel_bins=num_mel_bins,
        mode=mode,
        eval_chunk_seconds=eval_chunk_seconds,
        segment_count=segment_count,
        long_file_threshold_seconds=long_file_threshold_seconds,
        pad_mode=pad_mode,
    )
    cache_written = False
    if cache_path is not None and normalized_cache_mode in {"readwrite", "refresh"}:
        write_official_campp_frontend_cache(cache_path, features)
        cache_written = True
    return OfficialCAMPPFrontendCacheResult(
        features=features,
        cache_hit=False,
        cache_written=cache_written,
        cache_path=str(cache_path) if cache_path is not None else None,
    )


def compute_official_campp_features(
    raw_path: str,
    *,
    data_root: Path,
    sample_rate_hz: int,
    num_mel_bins: int,
    mode: str,
    eval_chunk_seconds: float,
    segment_count: int,
    long_file_threshold_seconds: float,
    pad_mode: str = "repeat",
) -> list[np.ndarray]:
    waveform = load_official_campp_waveform(
        raw_path,
        data_root=data_root,
        sample_rate_hz=sample_rate_hz,
    )
    segments = official_campp_segments_for_mode(
        waveform,
        mode=mode,
        sample_rate_hz=sample_rate_hz,
        eval_chunk_seconds=eval_chunk_seconds,
        segment_count=segment_count,
        long_file_threshold_seconds=long_file_threshold_seconds,
        pad_mode=pad_mode,
    )
    return [
        official_campp_fbank(
            segment,
            sample_rate_hz=sample_rate_hz,
            num_mel_bins=num_mel_bins,
        )
        .contiguous()
        .numpy()
        for segment in segments
    ]


def build_official_campp_frontend_cache_key(
    raw_path: str,
    *,
    data_root: Path,
    sample_rate_hz: int,
    num_mel_bins: int,
    mode: str,
    eval_chunk_seconds: float,
    segment_count: int,
    long_file_threshold_seconds: float,
    pad_mode: str = "repeat",
) -> OfficialCAMPPFrontendCacheKey:
    resolved_path = resolve_audio_path(raw_path, data_root).resolve()
    stat_result = resolved_path.stat()
    payload = {
        "format_version": OFFICIAL_CAMPP_FRONTEND_CACHE_FORMAT_VERSION,
        "resolved_audio_path": resolved_path.as_posix(),
        "file_size_bytes": stat_result.st_size,
        "file_mtime_ns": stat_result.st_mtime_ns,
        "sample_rate_hz": int(sample_rate_hz),
        "num_mel_bins": int(num_mel_bins),
        "mode": mode,
        "eval_chunk_seconds": float(eval_chunk_seconds),
        "segment_count": int(segment_count),
        "long_file_threshold_seconds": float(long_file_threshold_seconds),
        "pad_mode": pad_mode,
        "fbank": {
            "implementation": "torchaudio.compliance.kaldi.fbank",
            "dither": 0.0,
            "utterance_mean_normalization": True,
        },
    }
    cache_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    relative_path = (Path(cache_id[:2]) / cache_id[2:4] / f"{cache_id}.npy").as_posix()
    return OfficialCAMPPFrontendCacheKey(
        cache_id=cache_id,
        relative_path=relative_path,
        resolved_audio_path=resolved_path.as_posix(),
    )


def resolve_official_campp_frontend_cache_path(
    cache_root: Path,
    cache_key: OfficialCAMPPFrontendCacheKey,
) -> Path:
    return cache_root / cache_key.relative_path


def load_official_campp_frontend_cache(cache_path: Path) -> list[np.ndarray]:
    features = np.load(cache_path, allow_pickle=False)
    if features.ndim != 3:
        raise ValueError(f"Official CAM++ frontend cache must be 3D, got shape {features.shape}.")
    if features.dtype != np.float32:
        features = features.astype(np.float32, copy=False)
    return [np.ascontiguousarray(features[index]) for index in range(features.shape[0])]


def load_official_campp_frontend_feature_pack(
    pack_dir: Path,
    *,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
) -> OfficialCAMPPFrontendFeaturePack:
    metadata_path = pack_dir / "metadata.json"
    features_path = pack_dir / "features.npy"
    row_offsets_path = pack_dir / "row_offsets.npy"
    row_counts_path = pack_dir / "row_counts.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = np.load(features_path, mmap_mode=mmap_mode, allow_pickle=False)
    row_offsets = np.load(row_offsets_path, allow_pickle=False)
    row_counts = np.load(row_counts_path, allow_pickle=False)
    if features.ndim != 3:
        raise ValueError(f"Packed features must be 3D, got shape {features.shape}.")
    if row_offsets.ndim != 1 or row_counts.ndim != 1:
        raise ValueError("Packed row offsets/counts must be 1D arrays.")
    if row_offsets.shape != row_counts.shape:
        raise ValueError(
            f"Packed row offset/count shape mismatch: {row_offsets.shape} != {row_counts.shape}."
        )
    return OfficialCAMPPFrontendFeaturePack(
        pack_dir=pack_dir,
        features=features,
        row_offsets=row_offsets,
        row_counts=row_counts,
        metadata=metadata,
    )


def write_official_campp_frontend_cache(
    cache_path: Path,
    features: list[np.ndarray],
) -> None:
    if not features:
        raise ValueError("Cannot cache an empty official CAM++ feature list.")
    stacked = np.stack(
        [np.asarray(feature, dtype=np.float32) for feature in features],
        axis=0,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _allocate_temporary_cache_path(cache_path.parent)
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, stacked, allow_pickle=False)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def stack_official_campp_feature_batch(features: list[np.ndarray]) -> Any:
    if not features:
        raise ValueError("features must not be empty.")

    import torch

    max_frames = max(int(feature.shape[0]) for feature in features)
    feature_dim = int(features[0].shape[1])
    batch = np.zeros((len(features), max_frames, feature_dim), dtype=np.float32)
    for index, feature in enumerate(features):
        array = np.asarray(feature, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}.")
        if int(array.shape[1]) != feature_dim:
            raise ValueError(f"Feature dimension mismatch: {int(array.shape[1])} != {feature_dim}.")
        batch[index, : int(array.shape[0]), :] = array
    return torch.from_numpy(batch)


def _allocate_temporary_cache_path(directory: Path) -> Path:
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".tmp") as handle:
        return Path(handle.name)


__all__ = [
    "OFFICIAL_CAMPP_FRONTEND_CACHE_FORMAT_VERSION",
    "SUPPORTED_OFFICIAL_CAMPP_FRONTEND_CACHE_MODES",
    "OfficialCAMPPFrontendCacheKey",
    "OfficialCAMPPFrontendCacheResult",
    "OfficialCAMPPFrontendFeaturePack",
    "build_official_campp_frontend_cache_key",
    "compute_official_campp_features",
    "even_waveform_segments",
    "load_or_compute_official_campp_features",
    "load_official_campp_waveform",
    "load_official_campp_frontend_cache",
    "load_official_campp_frontend_feature_pack",
    "official_campp_fbank",
    "official_campp_segments_for_mode",
    "repeat_or_pad_waveform",
    "resolve_audio_path",
    "resolve_official_campp_frontend_cache_path",
    "stack_official_campp_feature_batch",
    "write_official_campp_frontend_cache",
]
