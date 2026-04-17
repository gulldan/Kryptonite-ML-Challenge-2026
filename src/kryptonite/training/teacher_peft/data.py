"""Raw-waveform datasets and collators for PEFT speaker runs."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.features import UtteranceChunkingRequest, chunk_utterance
from kryptonite.training.manifest_speaker_data import TrainingSampleRequest


class WaveformFeatureExtractor(Protocol):
    def __call__(
        self,
        waveforms: list[np.ndarray],
        *,
        sampling_rate: int,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class WaveformTrainingExample:
    waveform: torch.Tensor
    label: int
    speaker_id: str
    utterance_id: str | None


@dataclass(frozen=True, slots=True)
class WaveformTrainingBatch:
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str | None, ...]


class ManifestWaveformDataset(Dataset[WaveformTrainingExample]):
    def __init__(
        self,
        *,
        rows: list[ManifestRow],
        speaker_to_index: dict[str, int],
        project_root: Path | str,
        audio_request: AudioLoadRequest,
        chunking_request: UtteranceChunkingRequest,
        seed: int,
    ) -> None:
        self._rows = list(rows)
        self._speaker_to_index = dict(speaker_to_index)
        self._project_root = resolve_project_path(str(project_root), ".")
        self._audio_request = audio_request
        self._chunking_request = chunking_request
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: int | TrainingSampleRequest) -> WaveformTrainingExample:
        if isinstance(index, int):
            request = None
            row_index = index
        else:
            request = index
            row_index = request.row_index
        row = self._rows[row_index]
        loaded = load_manifest_audio(
            row,
            project_root=self._project_root,
            request=self._audio_request,
        )
        rng = random.Random(
            self._seed + (self._epoch * len(self._rows)) + row_index
            if request is None
            else request.request_seed
        )
        chunking_request = self._chunking_request
        if request is not None and request.crop_seconds is not None:
            chunking_request = UtteranceChunkingRequest(
                train_min_crop_seconds=request.crop_seconds,
                train_max_crop_seconds=request.crop_seconds,
                train_num_crops=self._chunking_request.train_num_crops,
                train_short_utterance_policy=self._chunking_request.train_short_utterance_policy,
                eval_max_full_utterance_seconds=self._chunking_request.eval_max_full_utterance_seconds,
                eval_chunk_seconds=self._chunking_request.eval_chunk_seconds,
                eval_chunk_overlap_seconds=self._chunking_request.eval_chunk_overlap_seconds,
                eval_pooling=self._chunking_request.eval_pooling,
                demo_max_full_utterance_seconds=self._chunking_request.demo_max_full_utterance_seconds,
                demo_chunk_seconds=self._chunking_request.demo_chunk_seconds,
                demo_chunk_overlap_seconds=self._chunking_request.demo_chunk_overlap_seconds,
                demo_pooling=self._chunking_request.demo_pooling,
            )
        chunk_batch = chunk_utterance(
            loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            stage="train",
            request=chunking_request,
            rng=rng,
        )
        if len(chunk_batch.chunks) != 1:
            raise ValueError(
                "Teacher PEFT training expects exactly one crop per utterance; "
                "set chunking.train_num_crops=1."
            )
        return WaveformTrainingExample(
            waveform=chunk_batch.chunks[0].waveform.to(dtype=torch.float32).contiguous(),
            label=self._speaker_to_index[row.speaker_id],
            speaker_id=row.speaker_id,
            utterance_id=row.utterance_id,
        )


def collate_waveform_examples(
    batch: list[WaveformTrainingExample],
    *,
    feature_extractor: WaveformFeatureExtractor,
    sample_rate_hz: int,
) -> WaveformTrainingBatch:
    if not batch:
        raise ValueError("Waveform training batch must not be empty")
    encoded = feature_extractor(
        [np.asarray(example.waveform, dtype=np.float32).reshape(-1) for example in batch],
        sampling_rate=sample_rate_hz,
        padding=True,
        return_tensors="pt",
    )
    model_inputs: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if key == "attention_mask":
            model_inputs[key] = value.to(dtype=torch.int32)
            continue
        model_inputs[key] = value.to(dtype=torch.float32)
    return WaveformTrainingBatch(
        model_inputs=model_inputs,
        labels=torch.tensor([example.label for example in batch], dtype=torch.long),
        speaker_ids=tuple(example.speaker_id for example in batch),
        utterance_ids=tuple(example.utterance_id for example in batch),
    )
