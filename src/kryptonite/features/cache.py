"""Disk-backed feature cache helpers for reproducible Fbank precompute flows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from kryptonite.config import FeatureCacheConfig
from kryptonite.data.audio_loader import LoadedAudio, LoadedManifestAudio
from kryptonite.deployment import resolve_project_path

from .fbank import FbankExtractionRequest, FbankExtractor

FEATURE_CACHE_FORMAT_VERSION = "kryptonite.features.fbank.cache.v1"
SUPPORTED_FEATURE_BENCHMARK_DEVICES = frozenset({"auto", "cpu", "cuda"})
SUPPORTED_FEATURE_CACHE_POLICIES = frozenset({"precompute_cpu", "optional", "runtime"})


@dataclass(frozen=True, slots=True)
class FeatureCacheSettings:
    namespace: str = "fbank-v1"
    train_policy: str = "precompute_cpu"
    dev_policy: str = "optional"
    infer_policy: str = "runtime"
    benchmark_device: str = "auto"
    benchmark_warmup_iterations: int = 1
    benchmark_iterations: int = 3

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise ValueError("namespace must not be empty")
        if self.train_policy.lower() not in SUPPORTED_FEATURE_CACHE_POLICIES:
            raise ValueError(
                f"train_policy must be one of {sorted(SUPPORTED_FEATURE_CACHE_POLICIES)}"
            )
        if self.dev_policy.lower() not in SUPPORTED_FEATURE_CACHE_POLICIES:
            raise ValueError(
                f"dev_policy must be one of {sorted(SUPPORTED_FEATURE_CACHE_POLICIES)}"
            )
        if self.infer_policy.lower() not in SUPPORTED_FEATURE_CACHE_POLICIES:
            raise ValueError(
                f"infer_policy must be one of {sorted(SUPPORTED_FEATURE_CACHE_POLICIES)}"
            )
        if self.benchmark_device.lower() not in SUPPORTED_FEATURE_BENCHMARK_DEVICES:
            raise ValueError(
                f"benchmark_device must be one of {sorted(SUPPORTED_FEATURE_BENCHMARK_DEVICES)}"
            )
        if self.benchmark_warmup_iterations < 0:
            raise ValueError("benchmark_warmup_iterations must be non-negative")
        if self.benchmark_iterations <= 0:
            raise ValueError("benchmark_iterations must be positive")

    @property
    def normalized_train_policy(self) -> str:
        return self.train_policy.lower()

    @property
    def normalized_dev_policy(self) -> str:
        return self.dev_policy.lower()

    @property
    def normalized_infer_policy(self) -> str:
        return self.infer_policy.lower()

    @property
    def normalized_benchmark_device(self) -> str:
        return self.benchmark_device.lower()

    @classmethod
    def from_config(cls, config: FeatureCacheConfig) -> FeatureCacheSettings:
        return cls(
            namespace=config.namespace,
            train_policy=config.train_policy,
            dev_policy=config.dev_policy,
            infer_policy=config.infer_policy,
            benchmark_device=config.benchmark_device,
            benchmark_warmup_iterations=config.benchmark_warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
        )


@dataclass(frozen=True, slots=True)
class FeatureCacheKey:
    cache_id: str
    relative_path: str
    loader_fingerprint: str
    feature_fingerprint: str
    source_audio_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "cache_id": self.cache_id,
            "relative_path": self.relative_path,
            "loader_fingerprint": self.loader_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "source_audio_path": self.source_audio_path,
        }


@dataclass(frozen=True, slots=True)
class FeatureCacheEntry:
    cache_key: str
    cache_path: str
    feature_shape: tuple[int, ...]
    feature_dtype: str
    cache_size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_key": self.cache_key,
            "cache_path": self.cache_path,
            "feature_shape": list(self.feature_shape),
            "feature_dtype": self.feature_dtype,
            "cache_size_bytes": self.cache_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FeatureCacheMaterializationRecord:
    manifest_path: str | None
    line_number: int | None
    audio_path: str
    speaker_id: str | None
    utterance_id: str | None
    input_duration_seconds: float
    cache_key: str
    cache_path: str
    cache_hit: bool
    feature_frame_count: int
    feature_dim: int
    feature_dtype: str
    cache_size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "line_number": self.line_number,
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "utterance_id": self.utterance_id,
            "input_duration_seconds": self.input_duration_seconds,
            "cache_key": self.cache_key,
            "cache_path": self.cache_path,
            "cache_hit": self.cache_hit,
            "feature_frame_count": self.feature_frame_count,
            "feature_dim": self.feature_dim,
            "feature_dtype": self.feature_dtype,
            "cache_size_bytes": self.cache_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FeatureCacheMaterializationSummary:
    row_count: int
    cache_hit_count: int
    cache_write_count: int
    unique_cache_entry_count: int
    total_input_duration_seconds: float
    total_feature_frames: int
    total_unique_cache_size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_write_count": self.cache_write_count,
            "unique_cache_entry_count": self.unique_cache_entry_count,
            "total_input_duration_seconds": self.total_input_duration_seconds,
            "total_feature_frames": self.total_feature_frames,
            "total_unique_cache_size_bytes": self.total_unique_cache_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FeatureCacheMaterializationReport:
    cache_root: str
    manifest_path: str
    request: FbankExtractionRequest
    records: list[FeatureCacheMaterializationRecord]
    summary: FeatureCacheMaterializationSummary

    def to_dict(self, *, include_records: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "cache_root": self.cache_root,
            "manifest_path": self.manifest_path,
            "request": _feature_request_payload(self.request),
            "summary": self.summary.to_dict(),
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


class FeatureCacheStore:
    def __init__(
        self,
        *,
        root: Path | str,
        settings: FeatureCacheSettings,
    ) -> None:
        base_root = Path(root)
        if not base_root.is_absolute():
            base_root = (Path.cwd() / base_root).resolve()
        self._base_root = base_root
        self._settings = settings

    @property
    def settings(self) -> FeatureCacheSettings:
        return self._settings

    @property
    def root(self) -> Path:
        return self._base_root / "features" / self._settings.namespace

    def build_key(
        self,
        *,
        loaded_audio: LoadedAudio,
        request: FbankExtractionRequest,
    ) -> FeatureCacheKey:
        loader_payload = _loader_payload(loaded_audio)
        feature_payload = _feature_request_payload(request)
        loader_fingerprint = _fingerprint_payload(loader_payload)
        feature_fingerprint = _fingerprint_payload(feature_payload)
        cache_id = _fingerprint_payload(
            {
                "format_version": FEATURE_CACHE_FORMAT_VERSION,
                "namespace": self._settings.namespace,
                "loader": loader_payload,
                "features": feature_payload,
            }
        )
        relative_path = (Path(cache_id[:2]) / cache_id[2:4] / f"{cache_id}.pt").as_posix()
        return FeatureCacheKey(
            cache_id=cache_id,
            relative_path=relative_path,
            loader_fingerprint=loader_fingerprint,
            feature_fingerprint=feature_fingerprint,
            source_audio_path=loaded_audio.configured_path,
        )

    def resolve_path(self, key: FeatureCacheKey) -> Path:
        return self.root / key.relative_path

    def exists(self, key: FeatureCacheKey) -> bool:
        return self.resolve_path(key).is_file()

    def inspect(self, key: FeatureCacheKey) -> FeatureCacheEntry:
        path = self.resolve_path(key)
        payload = _load_payload(path)
        tensor = _coerce_cached_tensor(payload)
        cache_key = payload.get("cache_key")
        if not isinstance(cache_key, str):
            raise ValueError(f"Cache payload at {path} is missing cache_key")
        return FeatureCacheEntry(
            cache_key=cache_key,
            cache_path=str(path),
            feature_shape=tuple(int(value) for value in tensor.shape),
            feature_dtype=str(tensor.dtype).replace("torch.", ""),
            cache_size_bytes=path.stat().st_size,
        )

    def load(
        self,
        key: FeatureCacheKey,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        path = self.resolve_path(key)
        payload = _load_payload(path)
        tensor = _coerce_cached_tensor(payload)
        if device is None:
            return tensor
        return tensor.to(device=_normalize_device(device))

    def write(
        self,
        key: FeatureCacheKey,
        *,
        features: torch.Tensor,
        loaded_audio: LoadedAudio,
        request: FbankExtractionRequest,
    ) -> FeatureCacheEntry:
        cache_path = self.resolve_path(key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FEATURE_CACHE_FORMAT_VERSION,
            "namespace": self._settings.namespace,
            "cache_key": key.cache_id,
            "loader_fingerprint": key.loader_fingerprint,
            "feature_fingerprint": key.feature_fingerprint,
            "source": _loader_payload(loaded_audio),
            "request": _feature_request_payload(request),
            "tensor": features.detach().to(device="cpu").contiguous(),
        }
        temporary_path = _allocate_temporary_path(cache_path.parent)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return self.inspect(key)


def resolve_feature_cache_root(
    *,
    project_root: Path | str,
    cache_root: Path | str,
) -> Path:
    return resolve_project_path(str(project_root), str(cache_root))


def materialize_feature_cache(
    samples: list[LoadedManifestAudio],
    *,
    store: FeatureCacheStore,
    request: FbankExtractionRequest,
    force: bool = False,
) -> FeatureCacheMaterializationReport:
    extractor = FbankExtractor(request=request)
    records: list[FeatureCacheMaterializationRecord] = []
    unique_paths: dict[str, int] = {}
    manifest_path = samples[0].manifest_path if samples else ""

    for sample in samples:
        key = store.build_key(loaded_audio=sample.audio, request=request)
        cache_hit = store.exists(key) and not force
        if cache_hit:
            entry = store.inspect(key)
        else:
            features = extractor.extract(
                sample.audio.waveform,
                sample_rate_hz=sample.audio.sample_rate_hz,
            )
            entry = store.write(
                key,
                features=features,
                loaded_audio=sample.audio,
                request=request,
            )

        unique_paths[entry.cache_path] = entry.cache_size_bytes
        records.append(
            FeatureCacheMaterializationRecord(
                manifest_path=sample.manifest_path,
                line_number=sample.line_number,
                audio_path=sample.row.audio_path,
                speaker_id=sample.row.speaker_id,
                utterance_id=sample.row.utterance_id,
                input_duration_seconds=sample.audio.duration_seconds,
                cache_key=key.cache_id,
                cache_path=entry.cache_path,
                cache_hit=cache_hit,
                feature_frame_count=(
                    int(entry.feature_shape[0]) if len(entry.feature_shape) >= 1 else 0
                ),
                feature_dim=(int(entry.feature_shape[-1]) if len(entry.feature_shape) >= 1 else 0),
                feature_dtype=entry.feature_dtype,
                cache_size_bytes=entry.cache_size_bytes,
            )
        )

    summary = FeatureCacheMaterializationSummary(
        row_count=len(records),
        cache_hit_count=sum(1 for record in records if record.cache_hit),
        cache_write_count=sum(1 for record in records if not record.cache_hit),
        unique_cache_entry_count=len(unique_paths),
        total_input_duration_seconds=round(
            sum(record.input_duration_seconds for record in records),
            6,
        ),
        total_feature_frames=sum(record.feature_frame_count for record in records),
        total_unique_cache_size_bytes=sum(unique_paths.values()),
    )
    return FeatureCacheMaterializationReport(
        cache_root=str(store.root),
        manifest_path=manifest_path or "",
        request=request,
        records=records,
        summary=summary,
    )


def _allocate_temporary_path(directory: Path) -> Path:
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".tmp") as handle:
        return Path(handle.name)


def _normalize_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Cache payload at {path} is invalid")
    if payload.get("format_version") != FEATURE_CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported cache payload format at {path}")
    return payload


def _coerce_cached_tensor(payload: dict[str, object]) -> torch.Tensor:
    tensor = payload.get("tensor")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Cache payload tensor is missing or invalid")
    return tensor


def _loader_payload(loaded_audio: LoadedAudio) -> dict[str, object]:
    resolved_path = Path(loaded_audio.resolved_path)
    stat_result = resolved_path.stat()
    return {
        "configured_path": loaded_audio.configured_path,
        "resolved_path": loaded_audio.resolved_path,
        "file_size_bytes": stat_result.st_size,
        "file_mtime_ns": stat_result.st_mtime_ns,
        "source_format": loaded_audio.source_format,
        "source_subtype": loaded_audio.source_subtype,
        "source_sample_rate_hz": loaded_audio.source_sample_rate_hz,
        "source_num_channels": loaded_audio.source_num_channels,
        "source_frame_count": loaded_audio.source_frame_count,
        "source_duration_seconds": loaded_audio.source_duration_seconds,
        "start_seconds": loaded_audio.start_seconds,
        "requested_duration_seconds": loaded_audio.requested_duration_seconds,
        "sample_rate_hz": loaded_audio.sample_rate_hz,
        "num_channels": loaded_audio.num_channels,
        "frame_count": loaded_audio.frame_count,
        "duration_seconds": loaded_audio.duration_seconds,
        "resampled": loaded_audio.resampled,
        "downmixed": loaded_audio.downmixed,
        "loudness_mode": loaded_audio.loudness_mode,
        "loudness_target_dbfs": loaded_audio.loudness_target_dbfs,
        "vad_mode": loaded_audio.vad_mode,
        "vad_backend": loaded_audio.vad_backend,
        "vad_provider": loaded_audio.vad_provider,
        "vad_min_output_duration_seconds": loaded_audio.vad_min_output_duration_seconds,
        "vad_min_retained_ratio": loaded_audio.vad_min_retained_ratio,
        "trim_reason": loaded_audio.trim_reason,
        "trim_start_seconds": loaded_audio.trim_start_seconds,
        "trim_end_seconds": loaded_audio.trim_end_seconds,
    }


def _feature_request_payload(request: FbankExtractionRequest) -> dict[str, object]:
    return {
        "frontend": request.frontend,
        "sample_rate_hz": request.sample_rate_hz,
        "num_mel_bins": request.num_mel_bins,
        "frame_length_ms": request.frame_length_ms,
        "frame_shift_ms": request.frame_shift_ms,
        "fft_size": request.fft_size,
        "window_type": request.window_type,
        "f_min_hz": request.f_min_hz,
        "f_max_hz": request.f_max_hz,
        "power": request.power,
        "log_offset": request.log_offset,
        "pad_end": request.pad_end,
        "cmvn_mode": request.cmvn_mode,
        "cmvn_window_frames": request.cmvn_window_frames,
        "output_dtype": request.output_dtype,
    }


def _fingerprint_payload(payload: dict[str, object]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = [
    "FEATURE_CACHE_FORMAT_VERSION",
    "SUPPORTED_FEATURE_BENCHMARK_DEVICES",
    "SUPPORTED_FEATURE_CACHE_POLICIES",
    "FeatureCacheEntry",
    "FeatureCacheKey",
    "FeatureCacheMaterializationRecord",
    "FeatureCacheMaterializationReport",
    "FeatureCacheMaterializationSummary",
    "FeatureCacheSettings",
    "FeatureCacheStore",
    "materialize_feature_cache",
    "resolve_feature_cache_root",
]
