"""Shared utterance chunking policy for train, eval, and demo stages."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from kryptonite.config import ChunkingConfig

SUPPORTED_CHUNKING_STAGES = frozenset({"train", "eval", "demo"})
SUPPORTED_CHUNK_POOLING_MODES = frozenset({"mean", "max"})
SUPPORTED_SHORT_UTTERANCE_POLICIES = frozenset({"full", "repeat_pad", "zero_pad"})


@dataclass(frozen=True, slots=True)
class UtteranceChunkingRequest:
    train_min_crop_seconds: float = 1.0
    train_max_crop_seconds: float = 4.0
    train_num_crops: int = 1
    train_short_utterance_policy: str = "repeat_pad"
    eval_max_full_utterance_seconds: float = 4.0
    eval_chunk_seconds: float = 4.0
    eval_chunk_overlap_seconds: float = 1.0
    eval_pooling: str = "mean"
    demo_max_full_utterance_seconds: float = 4.0
    demo_chunk_seconds: float = 4.0
    demo_chunk_overlap_seconds: float = 1.0
    demo_pooling: str = "mean"

    def __post_init__(self) -> None:
        if self.train_min_crop_seconds <= 0.0:
            raise ValueError("train_min_crop_seconds must be positive")
        if self.train_max_crop_seconds < self.train_min_crop_seconds:
            raise ValueError("train_max_crop_seconds must be >= train_min_crop_seconds")
        if self.train_num_crops <= 0:
            raise ValueError("train_num_crops must be positive")
        if self.train_short_utterance_policy.lower() not in SUPPORTED_SHORT_UTTERANCE_POLICIES:
            raise ValueError(
                "train_short_utterance_policy must be one of "
                f"{sorted(SUPPORTED_SHORT_UTTERANCE_POLICIES)}"
            )
        _validate_eval_like_stage(
            stage_name="eval",
            max_full_utterance_seconds=self.eval_max_full_utterance_seconds,
            chunk_seconds=self.eval_chunk_seconds,
            chunk_overlap_seconds=self.eval_chunk_overlap_seconds,
            pooling=self.eval_pooling,
        )
        _validate_eval_like_stage(
            stage_name="demo",
            max_full_utterance_seconds=self.demo_max_full_utterance_seconds,
            chunk_seconds=self.demo_chunk_seconds,
            chunk_overlap_seconds=self.demo_chunk_overlap_seconds,
            pooling=self.demo_pooling,
        )

    @property
    def normalized_train_short_utterance_policy(self) -> str:
        return self.train_short_utterance_policy.lower()

    @classmethod
    def from_config(cls, config: ChunkingConfig) -> UtteranceChunkingRequest:
        return cls(
            train_min_crop_seconds=config.train_min_crop_seconds,
            train_max_crop_seconds=config.train_max_crop_seconds,
            train_num_crops=config.train_num_crops,
            train_short_utterance_policy=config.train_short_utterance_policy,
            eval_max_full_utterance_seconds=config.eval_max_full_utterance_seconds,
            eval_chunk_seconds=config.eval_chunk_seconds,
            eval_chunk_overlap_seconds=config.eval_chunk_overlap_seconds,
            eval_pooling=config.eval_pooling,
            demo_max_full_utterance_seconds=config.demo_max_full_utterance_seconds,
            demo_chunk_seconds=config.demo_chunk_seconds,
            demo_chunk_overlap_seconds=config.demo_chunk_overlap_seconds,
            demo_pooling=config.demo_pooling,
        )

    def pooling_for_stage(self, stage: str) -> str:
        normalized_stage = _normalize_stage(stage)
        if normalized_stage == "train":
            return "mean"
        if normalized_stage == "eval":
            return self.eval_pooling.lower()
        return self.demo_pooling.lower()


@dataclass(frozen=True, slots=True)
class UtteranceChunk:
    index: int
    waveform: torch.Tensor
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    padded: bool
    padding_mode: str | None
    is_full_utterance: bool


@dataclass(frozen=True, slots=True)
class UtteranceChunkBatch:
    stage: str
    sample_rate_hz: int
    source_sample_count: int
    source_duration_seconds: float
    pooling_mode: str
    chunks: tuple[UtteranceChunk, ...]


def chunk_utterance(
    waveform: Any,
    *,
    sample_rate_hz: int,
    stage: str,
    request: UtteranceChunkingRequest | None = None,
    rng: random.Random | None = None,
) -> UtteranceChunkBatch:
    active_request = request or UtteranceChunkingRequest()
    normalized_stage = _normalize_stage(stage)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    waveform_tensor = _coerce_mono_waveform(waveform)
    source_sample_count = int(waveform_tensor.shape[-1])
    source_duration_seconds = round(float(source_sample_count) / float(sample_rate_hz), 6)

    if normalized_stage == "train":
        chunks = _build_train_chunks(
            waveform_tensor,
            sample_rate_hz=sample_rate_hz,
            request=active_request,
            rng=rng or random.Random(),
        )
    else:
        chunks = _build_eval_like_chunks(
            waveform_tensor,
            sample_rate_hz=sample_rate_hz,
            stage=normalized_stage,
            request=active_request,
        )

    return UtteranceChunkBatch(
        stage=normalized_stage,
        sample_rate_hz=sample_rate_hz,
        source_sample_count=source_sample_count,
        source_duration_seconds=source_duration_seconds,
        pooling_mode=active_request.pooling_for_stage(normalized_stage),
        chunks=tuple(chunks),
    )


def pool_chunk_tensors(
    chunk_tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    pooling_mode: str = "mean",
) -> torch.Tensor:
    if not chunk_tensors:
        raise ValueError("chunk_tensors must not be empty")

    normalized_pooling_mode = pooling_mode.lower()
    if normalized_pooling_mode not in SUPPORTED_CHUNK_POOLING_MODES:
        raise ValueError(f"pooling_mode must be one of {sorted(SUPPORTED_CHUNK_POOLING_MODES)}")

    first_shape = tuple(chunk_tensors[0].shape)
    first_dtype = chunk_tensors[0].dtype
    first_device = chunk_tensors[0].device
    for tensor in chunk_tensors[1:]:
        if tuple(tensor.shape) != first_shape:
            raise ValueError("All chunk tensors must share the same shape for pooling")
        if tensor.device != first_device:
            raise ValueError("All chunk tensors must live on the same device for pooling")

    stacked = torch.stack([tensor.to(dtype=torch.float32) for tensor in chunk_tensors], dim=0)
    if normalized_pooling_mode == "mean":
        pooled = stacked.mean(dim=0)
    else:
        pooled = stacked.max(dim=0).values
    return pooled.to(dtype=first_dtype, device=first_device)


def _build_train_chunks(
    waveform: torch.Tensor,
    *,
    sample_rate_hz: int,
    request: UtteranceChunkingRequest,
    rng: random.Random,
) -> list[UtteranceChunk]:
    total_samples = int(waveform.shape[-1])
    min_crop_samples = _seconds_to_sample_count(
        request.train_min_crop_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    max_crop_samples = _seconds_to_sample_count(
        request.train_max_crop_seconds,
        sample_rate_hz=sample_rate_hz,
    )

    if total_samples <= max_crop_samples:
        if total_samples < min_crop_samples:
            padded_waveform, padding_mode = _apply_short_utterance_policy(
                waveform,
                target_sample_count=min_crop_samples,
                policy=request.normalized_train_short_utterance_policy,
            )
            return [
                _make_chunk(
                    index=0,
                    waveform=padded_waveform,
                    start_sample=0,
                    end_sample=total_samples,
                    sample_rate_hz=sample_rate_hz,
                    padded=padding_mode is not None,
                    padding_mode=padding_mode,
                    is_full_utterance=True,
                )
            ]

        return [
            _make_chunk(
                index=0,
                waveform=waveform,
                start_sample=0,
                end_sample=total_samples,
                sample_rate_hz=sample_rate_hz,
                padded=False,
                padding_mode=None,
                is_full_utterance=True,
            )
        ]

    chunks: list[UtteranceChunk] = []
    for index in range(request.train_num_crops):
        target_samples = rng.randint(min_crop_samples, max_crop_samples)
        max_start = total_samples - target_samples
        start_sample = 0 if max_start <= 0 else rng.randint(0, max_start)
        end_sample = start_sample + target_samples
        chunks.append(
            _make_chunk(
                index=index,
                waveform=waveform[:, start_sample:end_sample],
                start_sample=start_sample,
                end_sample=end_sample,
                sample_rate_hz=sample_rate_hz,
                padded=False,
                padding_mode=None,
                is_full_utterance=False,
            )
        )
    return chunks


def _build_eval_like_chunks(
    waveform: torch.Tensor,
    *,
    sample_rate_hz: int,
    stage: str,
    request: UtteranceChunkingRequest,
) -> list[UtteranceChunk]:
    total_samples = int(waveform.shape[-1])
    if stage == "eval":
        max_full_utterance_seconds = request.eval_max_full_utterance_seconds
        chunk_seconds = request.eval_chunk_seconds
        chunk_overlap_seconds = request.eval_chunk_overlap_seconds
    else:
        max_full_utterance_seconds = request.demo_max_full_utterance_seconds
        chunk_seconds = request.demo_chunk_seconds
        chunk_overlap_seconds = request.demo_chunk_overlap_seconds

    max_full_utterance_samples = _seconds_to_sample_count(
        max_full_utterance_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    chunk_samples = _seconds_to_sample_count(chunk_seconds, sample_rate_hz=sample_rate_hz)
    chunk_overlap_samples = _seconds_to_sample_count(
        chunk_overlap_seconds,
        sample_rate_hz=sample_rate_hz,
    )

    if total_samples <= max_full_utterance_samples or total_samples <= chunk_samples:
        return [
            _make_chunk(
                index=0,
                waveform=waveform,
                start_sample=0,
                end_sample=total_samples,
                sample_rate_hz=sample_rate_hz,
                padded=False,
                padding_mode=None,
                is_full_utterance=True,
            )
        ]

    chunk_starts = _build_sliding_window_starts(
        total_samples=total_samples,
        chunk_samples=chunk_samples,
        chunk_overlap_samples=chunk_overlap_samples,
    )
    return [
        _make_chunk(
            index=index,
            waveform=waveform[:, start_sample : start_sample + chunk_samples],
            start_sample=start_sample,
            end_sample=min(total_samples, start_sample + chunk_samples),
            sample_rate_hz=sample_rate_hz,
            padded=False,
            padding_mode=None,
            is_full_utterance=False,
        )
        for index, start_sample in enumerate(chunk_starts)
    ]


def _apply_short_utterance_policy(
    waveform: torch.Tensor,
    *,
    target_sample_count: int,
    policy: str,
) -> tuple[torch.Tensor, str | None]:
    current_sample_count = int(waveform.shape[-1])
    if current_sample_count >= target_sample_count or policy == "full":
        return waveform, None

    padding_samples = target_sample_count - current_sample_count
    if policy == "zero_pad":
        padding = waveform.new_zeros((1, padding_samples))
        return torch.cat((waveform, padding), dim=-1), "zero_pad"

    repeated = waveform
    while int(repeated.shape[-1]) < target_sample_count:
        repeated = torch.cat((repeated, waveform), dim=-1)
    return repeated[:, :target_sample_count], "repeat_pad"


def _build_sliding_window_starts(
    *,
    total_samples: int,
    chunk_samples: int,
    chunk_overlap_samples: int,
) -> list[int]:
    if total_samples <= chunk_samples:
        return [0]

    hop_samples = chunk_samples - chunk_overlap_samples
    starts: list[int] = []
    current_start = 0
    while current_start + chunk_samples < total_samples:
        starts.append(current_start)
        current_start += hop_samples

    final_start = max(0, total_samples - chunk_samples)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _make_chunk(
    *,
    index: int,
    waveform: torch.Tensor,
    start_sample: int,
    end_sample: int,
    sample_rate_hz: int,
    padded: bool,
    padding_mode: str | None,
    is_full_utterance: bool,
) -> UtteranceChunk:
    sample_count = int(waveform.shape[-1])
    return UtteranceChunk(
        index=index,
        waveform=waveform,
        start_sample=start_sample,
        end_sample=end_sample,
        start_seconds=round(float(start_sample) / float(sample_rate_hz), 6),
        end_seconds=round(float(end_sample) / float(sample_rate_hz), 6),
        duration_seconds=round(float(sample_count) / float(sample_rate_hz), 6),
        padded=padded,
        padding_mode=padding_mode,
        is_full_utterance=is_full_utterance,
    )


def _coerce_mono_waveform(waveform: Any) -> torch.Tensor:
    tensor = torch.as_tensor(waveform)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or int(tensor.shape[0]) != 1:
        raise ValueError("Expected a mono waveform shaped as [samples] or [1, samples]")
    if tensor.numel() == 0:
        raise ValueError("waveform must not be empty")
    return tensor


def _normalize_stage(stage: str) -> str:
    normalized = stage.lower()
    if normalized not in SUPPORTED_CHUNKING_STAGES:
        raise ValueError(f"stage must be one of {sorted(SUPPORTED_CHUNKING_STAGES)}")
    return normalized


def _seconds_to_sample_count(seconds: float, *, sample_rate_hz: int) -> int:
    return max(1, round(seconds * sample_rate_hz))


def _validate_eval_like_stage(
    *,
    stage_name: str,
    max_full_utterance_seconds: float,
    chunk_seconds: float,
    chunk_overlap_seconds: float,
    pooling: str,
) -> None:
    if max_full_utterance_seconds <= 0.0:
        raise ValueError(f"{stage_name}_max_full_utterance_seconds must be positive")
    if chunk_seconds <= 0.0:
        raise ValueError(f"{stage_name}_chunk_seconds must be positive")
    if chunk_overlap_seconds < 0.0:
        raise ValueError(f"{stage_name}_chunk_overlap_seconds must be non-negative")
    if chunk_overlap_seconds >= chunk_seconds:
        raise ValueError(f"{stage_name}_chunk_overlap_seconds must be < {stage_name}_chunk_seconds")
    if pooling.lower() not in SUPPORTED_CHUNK_POOLING_MODES:
        raise ValueError(
            f"{stage_name}_pooling must be one of {sorted(SUPPORTED_CHUNK_POOLING_MODES)}"
        )


__all__ = [
    "SUPPORTED_CHUNKING_STAGES",
    "SUPPORTED_CHUNK_POOLING_MODES",
    "SUPPORTED_SHORT_UTTERANCE_POLICIES",
    "UtteranceChunk",
    "UtteranceChunkBatch",
    "UtteranceChunkingRequest",
    "chunk_utterance",
    "pool_chunk_tensors",
]
