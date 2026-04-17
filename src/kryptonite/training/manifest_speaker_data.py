"""Manifest-backed dataset helpers for speaker-baseline training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

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

from .augmentation_scheduler import ScheduledAugmentation

if TYPE_CHECKING:
    from .augmentation_runtime import TrainingAugmentationRuntime


@dataclass(frozen=True, slots=True)
class TrainingSampleRequest:
    row_index: int
    request_seed: int
    crop_seconds: float | None = None
    clean_sample: bool = True
    recipe_stage: str = "steady"
    recipe_intensity: str = "clean"
    augmentations: tuple[ScheduledAugmentation, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingExample:
    features: torch.Tensor
    label: int
    speaker_id: str
    utterance_id: str | None
    source_dataset: str
    split: str | None
    sample_weight: float = 1.0
    pseudo_verified: bool = False
    clean_sample: bool = True
    crop_seconds: float | None = None
    recipe_stage: str | None = None
    recipe_intensity: str | None = None
    augmentation_trace: tuple[dict[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    features: torch.Tensor
    labels: torch.Tensor
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str | None, ...]
    source_datasets: tuple[str, ...]
    splits: tuple[str | None, ...]
    sample_weights: torch.Tensor
    pseudo_verified_mask: torch.Tensor
    clean_sample_mask: torch.Tensor
    crop_seconds: tuple[float | None, ...] = field(default_factory=tuple)
    recipe_stages: tuple[str | None, ...] = field(default_factory=tuple)
    recipe_intensities: tuple[str | None, ...] = field(default_factory=tuple)
    augmentation_traces: tuple[tuple[dict[str, object], ...], ...] = field(default_factory=tuple)


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
        raise ValueError("Speaker-baseline training requires at least two speakers.")
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
        augmentation_runtime: TrainingAugmentationRuntime | None = None,
    ) -> None:
        self._rows = list(rows)
        self._speaker_to_index = dict(speaker_to_index)
        self._project_root = resolve_project_path(str(project_root), ".")
        self._audio_request = audio_request
        self._feature_request = feature_request
        self._chunking_request = chunking_request
        self._seed = seed
        self._augmentation_runtime = augmentation_runtime
        self._epoch = 0
        self._extractor: FbankExtractor | None = None

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: int | TrainingSampleRequest) -> TrainingExample:
        request = index if isinstance(index, TrainingSampleRequest) else None
        row_index: int
        if request is None:
            assert isinstance(index, int)
            row_index = index
        else:
            row_index = request.row_index
        row = self._rows[row_index]
        loaded = load_manifest_audio(
            row,
            project_root=self._project_root,
            request=self._audio_request,
        )
        rng = random.Random(
            request.request_seed
            if request is not None
            else self._seed + (self._epoch * len(self._rows)) + row_index
        )
        waveform = loaded.audio.waveform
        sample_rate_hz = loaded.audio.sample_rate_hz
        augmentation_trace: tuple[dict[str, object], ...] = ()
        if request is not None and request.augmentations:
            if self._augmentation_runtime is None:
                raise ValueError(
                    "Received scheduled augmentations without an augmentation runtime."
                )
            waveform, augmentation_trace = self._augmentation_runtime.apply_augmentations(
                waveform,
                sample_rate_hz=sample_rate_hz,
                augmentations=request.augmentations,
                rng=rng,
            )
        chunking_request = self._chunking_request
        if request is not None and request.crop_seconds is not None:
            chunking_request = replace(
                self._chunking_request,
                train_min_crop_seconds=request.crop_seconds,
                train_max_crop_seconds=request.crop_seconds,
            )
        chunk_batch = chunk_utterance(
            waveform,
            sample_rate_hz=sample_rate_hz,
            stage="train",
            request=chunking_request,
            rng=rng,
        )
        if len(chunk_batch.chunks) != 1:
            raise ValueError(
                "Speaker-baseline training expects exactly one crop per utterance; "
                "set chunking.train_num_crops=1."
            )
        chunk = chunk_batch.chunks[0]
        features = self._get_extractor().extract(
            chunk.waveform,
            sample_rate_hz=sample_rate_hz,
        )
        return TrainingExample(
            features=features,
            label=self._speaker_to_index[row.speaker_id],
            speaker_id=row.speaker_id,
            utterance_id=row.utterance_id,
            source_dataset=row.source_dataset,
            split=row.split,
            sample_weight=_coerce_sample_weight(row.extra_fields.get("pseudo_sample_weight")),
            pseudo_verified=_coerce_bool(row.extra_fields.get("pseudo_verified")),
            clean_sample=True if request is None else request.clean_sample,
            crop_seconds=chunk.duration_seconds,
            recipe_stage=None if request is None else request.recipe_stage,
            recipe_intensity=None if request is None else request.recipe_intensity,
            augmentation_trace=augmentation_trace,
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
                "Speaker-baseline training requires fixed-size crops; set "
                "chunking.train_min_crop_seconds == chunking.train_max_crop_seconds."
            )
    return TrainingBatch(
        features=torch.stack([example.features for example in batch], dim=0),
        labels=torch.tensor([example.label for example in batch], dtype=torch.long),
        speaker_ids=tuple(example.speaker_id for example in batch),
        utterance_ids=tuple(example.utterance_id for example in batch),
        source_datasets=tuple(example.source_dataset for example in batch),
        splits=tuple(example.split for example in batch),
        sample_weights=torch.tensor(
            [example.sample_weight for example in batch],
            dtype=torch.float32,
        ),
        pseudo_verified_mask=torch.tensor(
            [example.pseudo_verified for example in batch],
            dtype=torch.bool,
        ),
        clean_sample_mask=torch.tensor(
            [example.clean_sample for example in batch],
            dtype=torch.bool,
        ),
        crop_seconds=tuple(example.crop_seconds for example in batch),
        recipe_stages=tuple(example.recipe_stage for example in batch),
        recipe_intensities=tuple(example.recipe_intensity for example in batch),
        augmentation_traces=tuple(example.augmentation_trace for example in batch),
    )


def _coerce_sample_weight(value: object) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"pseudo_sample_weight must be numeric, got {value!r}")
    weight = float(value)
    if weight < 0.0:
        raise ValueError("pseudo_sample_weight must be non-negative")
    return weight


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return bool(value)
