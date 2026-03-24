from __future__ import annotations

import random

import pytest
import torch

from kryptonite.config import ChunkingConfig
from kryptonite.features import UtteranceChunkingRequest, chunk_utterance, pool_chunk_tensors


def test_train_chunking_keeps_medium_utterance_full_by_default() -> None:
    waveform = _waveform(duration_seconds=2.5)

    batch = chunk_utterance(waveform, sample_rate_hz=16_000, stage="train")

    assert batch.stage == "train"
    assert batch.pooling_mode == "mean"
    assert len(batch.chunks) == 1
    chunk = batch.chunks[0]
    assert chunk.is_full_utterance is True
    assert chunk.padded is False
    assert chunk.duration_seconds == pytest.approx(2.5, abs=1e-6)
    assert torch.equal(chunk.waveform, waveform)


def test_train_chunking_repeat_pads_short_utterance_to_min_crop() -> None:
    waveform = _waveform(duration_seconds=0.25)

    batch = chunk_utterance(waveform, sample_rate_hz=16_000, stage="train")

    assert len(batch.chunks) == 1
    chunk = batch.chunks[0]
    assert chunk.is_full_utterance is True
    assert chunk.padded is True
    assert chunk.padding_mode == "repeat_pad"
    assert chunk.duration_seconds == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(chunk.waveform[:, : waveform.shape[-1]], waveform)
    assert torch.equal(
        chunk.waveform[:, waveform.shape[-1] : waveform.shape[-1] * 2],
        waveform,
    )


def test_train_chunking_draws_deterministic_random_crops_for_long_utterance() -> None:
    waveform = _waveform(duration_seconds=8.0)
    request = UtteranceChunkingRequest(train_num_crops=2)

    first = chunk_utterance(
        waveform,
        sample_rate_hz=16_000,
        stage="train",
        request=request,
        rng=random.Random(7),
    )
    second = chunk_utterance(
        waveform,
        sample_rate_hz=16_000,
        stage="train",
        request=request,
        rng=random.Random(7),
    )

    first_ranges = [(chunk.start_sample, chunk.end_sample) for chunk in first.chunks]
    second_ranges = [(chunk.start_sample, chunk.end_sample) for chunk in second.chunks]

    assert first_ranges == second_ranges
    assert len(first.chunks) == 2
    assert all(chunk.is_full_utterance is False for chunk in first.chunks)
    assert all(1.0 <= chunk.duration_seconds <= 4.0 for chunk in first.chunks)


def test_eval_chunking_uses_full_utterance_for_medium_clip() -> None:
    waveform = _waveform(duration_seconds=3.5)

    batch = chunk_utterance(waveform, sample_rate_hz=16_000, stage="eval")

    assert batch.stage == "eval"
    assert batch.pooling_mode == "mean"
    assert len(batch.chunks) == 1
    assert batch.chunks[0].is_full_utterance is True
    assert batch.chunks[0].duration_seconds == pytest.approx(3.5, abs=1e-6)


def test_eval_chunking_uses_overlapping_windows_for_long_clip() -> None:
    waveform = _waveform(duration_seconds=9.0)

    batch = chunk_utterance(waveform, sample_rate_hz=16_000, stage="eval")

    assert len(batch.chunks) == 3
    assert [chunk.start_seconds for chunk in batch.chunks] == [0.0, 3.0, 5.0]
    assert [chunk.end_seconds for chunk in batch.chunks] == [4.0, 7.0, 9.0]
    assert all(chunk.duration_seconds == pytest.approx(4.0, abs=1e-6) for chunk in batch.chunks)
    assert all(chunk.is_full_utterance is False for chunk in batch.chunks)


def test_demo_chunking_uses_demo_pooling_configuration() -> None:
    waveform = _waveform(duration_seconds=5.0)
    request = UtteranceChunkingRequest(
        demo_max_full_utterance_seconds=2.0,
        demo_chunk_seconds=3.0,
        demo_chunk_overlap_seconds=1.0,
        demo_pooling="max",
    )

    batch = chunk_utterance(waveform, sample_rate_hz=16_000, stage="demo", request=request)

    assert batch.stage == "demo"
    assert batch.pooling_mode == "max"
    assert [chunk.start_seconds for chunk in batch.chunks] == [0.0, 2.0]


def test_chunking_request_from_config_uses_chunking_section() -> None:
    config = ChunkingConfig(
        train_num_crops=3,
        train_short_utterance_policy="zero_pad",
        eval_chunk_overlap_seconds=0.5,
        demo_pooling="max",
    )

    request = UtteranceChunkingRequest.from_config(config)

    assert request.train_num_crops == 3
    assert request.train_short_utterance_policy == "zero_pad"
    assert request.eval_chunk_overlap_seconds == 0.5
    assert request.demo_pooling == "max"


def test_pool_chunk_tensors_supports_mean_and_max() -> None:
    chunk_tensors = [
        torch.tensor([1.0, 3.0], dtype=torch.float32),
        torch.tensor([5.0, 2.0], dtype=torch.float32),
    ]

    assert torch.equal(
        pool_chunk_tensors(chunk_tensors, pooling_mode="mean"),
        torch.tensor([3.0, 2.5], dtype=torch.float32),
    )
    assert torch.equal(
        pool_chunk_tensors(chunk_tensors, pooling_mode="max"),
        torch.tensor([5.0, 3.0], dtype=torch.float32),
    )


def _waveform(*, duration_seconds: float, sample_rate_hz: int = 16_000) -> torch.Tensor:
    sample_count = round(duration_seconds * sample_rate_hz)
    return torch.arange(sample_count, dtype=torch.float32).reshape(1, -1)
