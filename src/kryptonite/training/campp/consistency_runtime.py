"""Runtime helpers for CAM++ clean/corrupted consistency training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
)
from kryptonite.training.manifest_speaker_data import TrainingSampleRequest
from kryptonite.training.optimization_runtime import TrainingOptimizationRuntime

from ..augmentation_runtime import TrainingAugmentationRuntime
from .stage2_sampler import Stage2BatchSampler


@dataclass(frozen=True, slots=True)
class ConsistencyExample:
    clean_features: torch.Tensor
    corrupted_features: torch.Tensor
    label: int
    speaker_id: str
    utterance_id: str | None
    pair_active: bool
    crop_seconds: float | None = None
    recipe_stage: str | None = None
    recipe_intensity: str | None = None
    corruption_trace: tuple[dict[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ConsistencyBatch:
    clean_features: torch.Tensor
    corrupted_features: torch.Tensor
    labels: torch.Tensor
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str | None, ...]
    pair_active_mask: torch.Tensor
    crop_seconds: tuple[float | None, ...] = field(default_factory=tuple)
    recipe_stages: tuple[str | None, ...] = field(default_factory=tuple)
    recipe_intensities: tuple[str | None, ...] = field(default_factory=tuple)
    corruption_traces: tuple[tuple[dict[str, object], ...], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ConsistencyEpochBreakdown:
    epoch: int
    mean_loss: float
    mean_clean_classification_loss: float
    mean_corrupted_classification_loss: float
    mean_embedding_loss: float
    mean_score_loss: float
    clean_accuracy: float
    corrupted_accuracy: float
    paired_examples: int
    paired_ratio: float
    learning_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "mean_loss": self.mean_loss,
            "mean_clean_classification_loss": self.mean_clean_classification_loss,
            "mean_corrupted_classification_loss": self.mean_corrupted_classification_loss,
            "mean_embedding_loss": self.mean_embedding_loss,
            "mean_score_loss": self.mean_score_loss,
            "clean_accuracy": self.clean_accuracy,
            "corrupted_accuracy": self.corrupted_accuracy,
            "paired_examples": self.paired_examples,
            "paired_ratio": self.paired_ratio,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True, slots=True)
class ConsistencyBatchMetrics:
    total_loss: float
    total_clean_classification_loss: float
    total_corrupted_classification_loss: float
    total_embedding_loss: float
    total_score_loss: float
    total_clean_correct: int
    total_corrupted_correct: int
    total_examples: int
    total_paired_examples: int


class ManifestConsistencyDataset(Dataset[ConsistencyExample]):
    """Manifest-backed dataset that yields aligned clean/corrupted feature pairs."""

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
        augmentation_runtime: TrainingAugmentationRuntime,
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
        self._augmentation_runtime = augmentation_runtime

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: int | TrainingSampleRequest) -> ConsistencyExample:
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
        base_seed = (
            request.request_seed
            if request is not None
            else self._seed + (self._epoch * len(self._rows)) + row_index
        )
        crop_rng = random.Random(base_seed)
        chunking_request = self._chunking_request
        if request is not None and request.crop_seconds is not None:
            chunking_request = replace(
                self._chunking_request,
                train_min_crop_seconds=request.crop_seconds,
                train_max_crop_seconds=request.crop_seconds,
            )
        chunk_batch = chunk_utterance(
            loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            stage="train",
            request=chunking_request,
            rng=crop_rng,
        )
        if len(chunk_batch.chunks) != 1:
            raise ValueError(
                "CAM++ consistency expects exactly one crop per utterance; "
                "set chunking.train_num_crops=1."
            )
        chunk = chunk_batch.chunks[0]
        clean_waveform = _to_numpy_waveform(chunk.waveform)
        clean_features = self._get_extractor().extract(
            clean_waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
        )

        pair_active = request is not None and bool(request.augmentations)
        corruption_trace: tuple[dict[str, object], ...] = ()
        corrupted_waveform = clean_waveform
        corrupted_sample_rate_hz = loaded.audio.sample_rate_hz
        if pair_active:
            assert request is not None
            corrupted_waveform, corrupted_sample_rate_hz, corruption_trace = (
                self._augmentation_runtime.apply_augmentations(
                    waveform=clean_waveform,
                    sample_rate_hz=loaded.audio.sample_rate_hz,
                    augmentations=request.augmentations,
                    rng=random.Random(base_seed + 1_000_003),
                )
            )
            corrupted_waveform = _match_waveform_length(
                corrupted_waveform,
                target_frames=clean_waveform.shape[-1],
            )
        corrupted_features = self._get_extractor().extract(
            corrupted_waveform,
            sample_rate_hz=corrupted_sample_rate_hz,
        )
        return ConsistencyExample(
            clean_features=clean_features,
            corrupted_features=corrupted_features,
            label=self._speaker_to_index[row.speaker_id],
            speaker_id=row.speaker_id,
            utterance_id=row.utterance_id,
            pair_active=pair_active,
            crop_seconds=chunk.duration_seconds,
            recipe_stage=None if request is None else request.recipe_stage,
            recipe_intensity=None if request is None else request.recipe_intensity,
            corruption_trace=corruption_trace,
        )

    def _get_extractor(self) -> FbankExtractor:
        if self._extractor is None:
            self._extractor = FbankExtractor(request=self._feature_request)
        return self._extractor


def collate_consistency_examples(batch: list[ConsistencyExample]) -> ConsistencyBatch:
    if not batch:
        raise ValueError("Consistency batch must not be empty.")
    first_clean_shape = tuple(batch[0].clean_features.shape)
    first_corrupted_shape = tuple(batch[0].corrupted_features.shape)
    for example in batch[1:]:
        if tuple(example.clean_features.shape) != first_clean_shape:
            raise ValueError(
                "CAM++ consistency requires fixed-size clean crops; set "
                "chunking.train_min_crop_seconds == chunking.train_max_crop_seconds."
            )
        if tuple(example.corrupted_features.shape) != first_corrupted_shape:
            raise ValueError(
                "CAM++ consistency requires fixed-size corrupted crops after augmentation."
            )
    return ConsistencyBatch(
        clean_features=torch.stack([example.clean_features for example in batch], dim=0),
        corrupted_features=torch.stack([example.corrupted_features for example in batch], dim=0),
        labels=torch.tensor([example.label for example in batch], dtype=torch.long),
        speaker_ids=tuple(example.speaker_id for example in batch),
        utterance_ids=tuple(example.utterance_id for example in batch),
        pair_active_mask=torch.tensor([example.pair_active for example in batch], dtype=torch.bool),
        crop_seconds=tuple(example.crop_seconds for example in batch),
        recipe_stages=tuple(example.recipe_stage for example in batch),
        recipe_intensities=tuple(example.recipe_intensity for example in batch),
        corruption_traces=tuple(example.corruption_trace for example in batch),
    )


def build_consistency_dataloader(
    *,
    rows: list[ManifestRow],
    speaker_to_index: dict[str, int],
    project: Any,
    chunking_request: UtteranceChunkingRequest,
    active_runtime: TrainingAugmentationRuntime,
    device: torch.device,
    hard_negative_fraction: float,
) -> tuple[ManifestConsistencyDataset, Stage2BatchSampler, DataLoader[ConsistencyBatch]]:
    project_root = resolve_project_path(project.paths.project_root, ".")
    audio_request = AudioLoadRequest.from_config(project.normalization, vad=project.vad)
    feature_request = FbankExtractionRequest.from_config(project.features)
    dataset = ManifestConsistencyDataset(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking_request=chunking_request,
        seed=project.runtime.seed,
        augmentation_runtime=active_runtime,
    )
    sampler = Stage2BatchSampler(
        rows=rows,
        batch_size=project.training.batch_size,
        seed=project.runtime.seed,
        chunking_request=chunking_request,
        hard_negative_fraction=hard_negative_fraction,
        augmentation_runtime=active_runtime,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": project.runtime.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_consistency_examples,
        "persistent_workers": False,
    }
    if project.runtime.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = cast(DataLoader[ConsistencyBatch], DataLoader(**loader_kwargs))
    return dataset, sampler, loader


def run_consistency_batches(
    *,
    model: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    training_runtime: TrainingOptimizationRuntime,
    loader: Any,
    device: torch.device,
    clean_classification_weight: float,
    corrupted_classification_weight: float,
    embedding_weight: float,
    score_weight: float,
) -> ConsistencyBatchMetrics:
    model.train()
    classifier.train()
    total_loss = 0.0
    total_clean_classification_loss = 0.0
    total_corrupted_classification_loss = 0.0
    total_embedding_loss = 0.0
    total_score_loss = 0.0
    total_clean_correct = 0
    total_corrupted_correct = 0
    total_examples = 0
    total_paired_examples = 0
    accumulation_steps = training_runtime.gradient_accumulation_steps
    pending_step = False

    training_runtime.zero_grad()
    for batch_index, batch in enumerate(loader, start=1):
        if not isinstance(batch, ConsistencyBatch):
            raise TypeError(f"Expected ConsistencyBatch instances, got {type(batch)!r}.")

        clean_features = batch.clean_features.to(device=device, dtype=torch.float32)
        corrupted_features = batch.corrupted_features.to(device=device, dtype=torch.float32)
        labels = batch.labels.to(device=device)
        active_mask = batch.pair_active_mask.to(device=device)

        with training_runtime.precision.autocast_context(device=device):
            clean_embeddings = model(clean_features)
            clean_logits = classifier(clean_embeddings)
            clean_classification_loss = criterion(clean_logits, labels)

            corrupted_embeddings = model(corrupted_features)
            if bool(active_mask.any().item()):
                corrupted_logits = classifier(corrupted_embeddings[active_mask])
                corrupted_labels = labels[active_mask]
                corrupted_classification_loss = criterion(corrupted_logits, corrupted_labels)
                embedding_loss = _embedding_consistency_loss(
                    clean_embeddings=clean_embeddings[active_mask],
                    corrupted_embeddings=corrupted_embeddings[active_mask],
                )
                score_loss = _score_consistency_loss(
                    clean_embeddings=clean_embeddings[active_mask],
                    corrupted_embeddings=corrupted_embeddings[active_mask],
                )
            else:
                corrupted_logits = None
                corrupted_classification_loss = clean_embeddings.new_zeros(())
                embedding_loss = clean_embeddings.new_zeros(())
                score_loss = clean_embeddings.new_zeros(())

            loss = (
                (clean_classification_loss * clean_classification_weight)
                + (corrupted_classification_loss * corrupted_classification_weight)
                + (embedding_loss * embedding_weight)
                + (score_loss * score_weight)
            )
            scaled_loss = loss / accumulation_steps

        training_runtime.backward(scaled_loss)
        pending_step = True
        if batch_index % accumulation_steps == 0:
            training_runtime.step_optimizer()
            training_runtime.zero_grad()
            pending_step = False

        batch_size = int(labels.shape[0])
        paired_examples = int(active_mask.sum().item())
        total_loss += float(loss.detach().item()) * batch_size
        total_clean_classification_loss += (
            float(clean_classification_loss.detach().item()) * batch_size
        )
        total_corrupted_classification_loss += (
            float(corrupted_classification_loss.detach().item()) * paired_examples
        )
        total_embedding_loss += float(embedding_loss.detach().item()) * paired_examples
        total_score_loss += float(score_loss.detach().item()) * paired_examples
        total_clean_correct += int((clean_logits.argmax(dim=1) == labels).sum().item())
        if corrupted_logits is not None:
            total_corrupted_correct += int(
                (corrupted_logits.argmax(dim=1) == labels[active_mask]).sum().item()
            )
        total_examples += batch_size
        total_paired_examples += paired_examples

    if pending_step:
        training_runtime.step_optimizer()
        training_runtime.zero_grad()

    return ConsistencyBatchMetrics(
        total_loss=total_loss,
        total_clean_classification_loss=total_clean_classification_loss,
        total_corrupted_classification_loss=total_corrupted_classification_loss,
        total_embedding_loss=total_embedding_loss,
        total_score_loss=total_score_loss,
        total_clean_correct=total_clean_correct,
        total_corrupted_correct=total_corrupted_correct,
        total_examples=total_examples,
        total_paired_examples=total_paired_examples,
    )


def _to_numpy_waveform(waveform: Any) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        array = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    return array


def _match_waveform_length(waveform: np.ndarray, *, target_frames: int) -> np.ndarray:
    if waveform.ndim == 1:
        if waveform.shape[0] >= target_frames:
            return waveform[:target_frames]
        return np.pad(waveform, (0, target_frames - waveform.shape[0]))
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        frames = waveform.shape[1]
        if frames >= target_frames:
            return waveform[:, :target_frames]
        return np.pad(waveform, ((0, 0), (0, target_frames - frames)))
    raise ValueError(
        "Consistency waveform crops must be mono tensors shaped [frames] or [1, frames], "
        f"got {tuple(waveform.shape)}."
    )


def _embedding_consistency_loss(
    *,
    clean_embeddings: torch.Tensor,
    corrupted_embeddings: torch.Tensor,
) -> torch.Tensor:
    if clean_embeddings.shape != corrupted_embeddings.shape:
        raise ValueError(
            "Clean and corrupted embedding tensors must match for consistency loss; "
            f"got {tuple(clean_embeddings.shape)} vs {tuple(corrupted_embeddings.shape)}."
        )
    clean_norm = torch_functional.normalize(clean_embeddings, dim=1)
    corrupted_norm = torch_functional.normalize(corrupted_embeddings, dim=1)
    return 1.0 - torch_functional.cosine_similarity(clean_norm, corrupted_norm, dim=1).mean()


def _score_consistency_loss(
    *,
    clean_embeddings: torch.Tensor,
    corrupted_embeddings: torch.Tensor,
) -> torch.Tensor:
    if clean_embeddings.shape[0] < 2:
        return clean_embeddings.new_zeros(())
    clean_norm = torch_functional.normalize(clean_embeddings, dim=1)
    corrupted_norm = torch_functional.normalize(corrupted_embeddings, dim=1)
    clean_scores = clean_norm @ clean_norm.transpose(0, 1)
    corrupted_scores = corrupted_norm @ corrupted_norm.transpose(0, 1)
    mask = torch.triu(torch.ones_like(clean_scores, dtype=torch.bool), diagonal=1)
    return torch_functional.mse_loss(clean_scores[mask], corrupted_scores[mask])


__all__ = [
    "ConsistencyBatch",
    "ConsistencyBatchMetrics",
    "ConsistencyEpochBreakdown",
    "ManifestConsistencyDataset",
    "build_consistency_dataloader",
    "collate_consistency_examples",
    "run_consistency_batches",
]
