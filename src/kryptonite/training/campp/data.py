"""Manifest-backed dataset helpers for CAM++ baseline training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    features: torch.Tensor
    label: int
    speaker_id: str
    utterance_id: str | None


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    features: torch.Tensor
    labels: torch.Tensor
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str | None, ...]


def load_manifest_rows(
    manifest_path: Path | str,
    *,
    project_root: Path | str,
    limit: int | None = None,
) -> list[ManifestRow]:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifest_file = resolve_project_path(str(project_root_path), str(manifest_path))
    rows: list[ManifestRow] = []
    for line_number, raw_line in enumerate(manifest_file.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {manifest_file}:{line_number}")
        rows.append(
            ManifestRow.from_mapping(
                payload,
                manifest_path=str(manifest_file),
                line_number=line_number,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No manifest rows found in {manifest_file}")
    return rows


def build_speaker_index(rows: list[ManifestRow]) -> dict[str, int]:
    speakers = sorted({row.speaker_id for row in rows})
    if len(speakers) < 2:
        raise ValueError("CAM++ baseline training requires at least two speakers.")
    return {speaker_id: index for index, speaker_id in enumerate(speakers)}


class ManifestSpeakerDataset(Dataset[TrainingExample]):
    def __init__(
        self,
        *,
        rows: list[ManifestRow],
        speaker_to_index: dict[str, int],
        project_root: Path | str,
        audio_request: AudioLoadRequest,
        feature_request: FbankExtractionRequest,
        chunking_request: UtteranceChunkingRequest,
        seed: int,
    ) -> None:
        self._rows = list(rows)
        self._speaker_to_index = dict(speaker_to_index)
        self._project_root = resolve_project_path(str(project_root), ".")
        self._audio_request = audio_request
        self._feature_request = feature_request
        self._chunking_request = chunking_request
        self._seed = seed
        self._epoch = 0
        self._extractor: FbankExtractor | None = None

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: int) -> TrainingExample:
        row = self._rows[index]
        loaded = load_manifest_audio(
            row,
            project_root=self._project_root,
            request=self._audio_request,
        )
        rng = random.Random(self._seed + (self._epoch * len(self._rows)) + index)
        chunk_batch = chunk_utterance(
            loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            stage="train",
            request=self._chunking_request,
            rng=rng,
        )
        if len(chunk_batch.chunks) != 1:
            raise ValueError(
                "CAM++ baseline expects exactly one training chunk per utterance; "
                "set chunking.train_num_crops=1."
            )
        chunk = chunk_batch.chunks[0]
        features = self._get_extractor().extract(
            chunk.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
        )
        return TrainingExample(
            features=features,
            label=self._speaker_to_index[row.speaker_id],
            speaker_id=row.speaker_id,
            utterance_id=row.utterance_id,
        )

    def _get_extractor(self) -> FbankExtractor:
        if self._extractor is None:
            self._extractor = FbankExtractor(request=self._feature_request)
        return self._extractor


def collate_training_examples(batch: list[TrainingExample]) -> TrainingBatch:
    if not batch:
        raise ValueError("Training batch must not be empty")
    first_shape = tuple(batch[0].features.shape)
    for example in batch[1:]:
        if tuple(example.features.shape) != first_shape:
            raise ValueError(
                "CAM++ baseline training requires fixed-size crops; "
                "set chunking.train_min_crop_seconds == chunking.train_max_crop_seconds."
            )
    return TrainingBatch(
        features=torch.stack([example.features for example in batch], dim=0),
        labels=torch.tensor([example.label for example in batch], dtype=torch.long),
        speaker_ids=tuple(example.speaker_id for example in batch),
        utterance_ids=tuple(example.utterance_id for example in batch),
    )
