"""Runtime helpers for CAM++ teacher-student distillation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

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
class DistillationExample:
    features: torch.Tensor
    waveform: torch.Tensor
    sample_rate_hz: int
    label: int
    speaker_id: str
    utterance_id: str | None
    clean_sample: bool = True
    crop_seconds: float | None = None
    recipe_stage: str | None = None
    recipe_intensity: str | None = None
    augmentation_trace: tuple[dict[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DistillationBatch:
    features: torch.Tensor
    waveforms: tuple[torch.Tensor, ...]
    sample_rate_hz: int
    labels: torch.Tensor
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str | None, ...]
    clean_sample_mask: torch.Tensor
    crop_seconds: tuple[float | None, ...] = field(default_factory=tuple)
    recipe_stages: tuple[str | None, ...] = field(default_factory=tuple)
    recipe_intensities: tuple[str | None, ...] = field(default_factory=tuple)
    augmentation_traces: tuple[tuple[dict[str, object], ...], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DistillationEpochBreakdown:
    epoch: int
    mean_loss: float
    mean_classification_loss: float
    mean_embedding_loss: float
    mean_score_loss: float
    accuracy: float
    learning_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "mean_loss": self.mean_loss,
            "mean_classification_loss": self.mean_classification_loss,
            "mean_embedding_loss": self.mean_embedding_loss,
            "mean_score_loss": self.mean_score_loss,
            "accuracy": self.accuracy,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True, slots=True)
class DistillationBatchMetrics:
    total_loss: float
    total_classification_loss: float
    total_embedding_loss: float
    total_score_loss: float
    total_correct: int
    total_examples: int


class ManifestDistillationDataset(Dataset[DistillationExample]):
    """Manifest-backed dataset that keeps the waveform crop used for student fbanks."""

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
        self._epoch = 0
        self._extractor: FbankExtractor | None = None
        self._augmentation_runtime = augmentation_runtime

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: int | TrainingSampleRequest) -> DistillationExample:
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
                raise ValueError("Augmentation request received without an augmentation runtime.")
            (
                waveform,
                sample_rate_hz,
                augmentation_trace,
            ) = self._augmentation_runtime.apply_augmentations(
                waveform=waveform,
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
                "CAM++ distillation expects exactly one crop per utterance; "
                "set chunking.train_num_crops=1."
            )
        chunk = chunk_batch.chunks[0]
        features = self._get_extractor().extract(
            chunk.waveform,
            sample_rate_hz=sample_rate_hz,
        )
        return DistillationExample(
            features=features,
            waveform=_flatten_mono_waveform(chunk.waveform),
            sample_rate_hz=sample_rate_hz,
            label=self._speaker_to_index[row.speaker_id],
            speaker_id=row.speaker_id,
            utterance_id=row.utterance_id,
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


def collate_distillation_examples(batch: list[DistillationExample]) -> DistillationBatch:
    if not batch:
        raise ValueError("Distillation batch must not be empty.")
    first_shape = tuple(batch[0].features.shape)
    sample_rate_hz = batch[0].sample_rate_hz
    for example in batch[1:]:
        if tuple(example.features.shape) != first_shape:
            raise ValueError(
                "CAM++ distillation requires fixed-size crops; set "
                "chunking.train_min_crop_seconds == chunking.train_max_crop_seconds."
            )
        if example.sample_rate_hz != sample_rate_hz:
            raise ValueError("Distillation batch mixes multiple sample rates.")
    return DistillationBatch(
        features=torch.stack([example.features for example in batch], dim=0),
        waveforms=tuple(example.waveform for example in batch),
        sample_rate_hz=sample_rate_hz,
        labels=torch.tensor([example.label for example in batch], dtype=torch.long),
        speaker_ids=tuple(example.speaker_id for example in batch),
        utterance_ids=tuple(example.utterance_id for example in batch),
        clean_sample_mask=torch.tensor(
            [example.clean_sample for example in batch],
            dtype=torch.bool,
        ),
        crop_seconds=tuple(example.crop_seconds for example in batch),
        recipe_stages=tuple(example.recipe_stage for example in batch),
        recipe_intensities=tuple(example.recipe_intensity for example in batch),
        augmentation_traces=tuple(example.augmentation_trace for example in batch),
    )


def build_distillation_dataloader(
    *,
    rows: list[ManifestRow],
    speaker_to_index: dict[str, int],
    project: Any,
    chunking_request: UtteranceChunkingRequest,
    active_runtime: TrainingAugmentationRuntime | None,
    device: torch.device,
    hard_negative_fraction: float,
) -> tuple[ManifestDistillationDataset, Stage2BatchSampler, DataLoader[DistillationBatch]]:
    project_root = resolve_project_path(project.paths.project_root, ".")
    audio_request = AudioLoadRequest.from_config(project.normalization, vad=project.vad)
    feature_request = FbankExtractionRequest.from_config(project.features)
    dataset = ManifestDistillationDataset(
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
        "collate_fn": collate_distillation_examples,
        "persistent_workers": False,
    }
    if project.runtime.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = cast(DataLoader[DistillationBatch], DataLoader(**loader_kwargs))
    return dataset, sampler, loader


def run_distillation_batches(
    *,
    model: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    teacher_encoder: nn.Module,
    teacher_feature_extractor: Any,
    training_runtime: TrainingOptimizationRuntime,
    loader: Any,
    device: torch.device,
    sample_rate_hz: int,
    classification_weight: float,
    embedding_weight: float,
    score_weight: float,
    teacher_batch_builder: Any,
) -> DistillationBatchMetrics:
    model.train()
    classifier.train()
    teacher_encoder.eval()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_embedding_loss = 0.0
    total_score_loss = 0.0
    total_correct = 0
    total_examples = 0
    accumulation_steps = training_runtime.gradient_accumulation_steps
    pending_step = False

    training_runtime.zero_grad()
    for batch_index, batch in enumerate(loader, start=1):
        if not isinstance(batch, DistillationBatch):
            raise TypeError(f"Expected DistillationBatch instances, got {type(batch)!r}.")

        features = batch.features.to(device=device, dtype=torch.float32)
        labels = batch.labels.to(device=device)
        with torch.no_grad():
            teacher_inputs = teacher_batch_builder(
                feature_extractor=teacher_feature_extractor,
                waveforms=batch.waveforms,
                sample_rate_hz=sample_rate_hz,
                device=device,
            )
            with training_runtime.precision.autocast_context(device=device):
                teacher_embeddings = teacher_encoder(**teacher_inputs)

        with training_runtime.precision.autocast_context(device=device):
            student_embeddings = model(features)
            logits = classifier(student_embeddings)
            classification_loss = criterion(logits, labels)
            embedding_loss = _embedding_alignment_loss(
                student_embeddings=student_embeddings,
                teacher_embeddings=teacher_embeddings,
            )
            score_loss = _score_matrix_loss(
                student_embeddings=student_embeddings,
                teacher_embeddings=teacher_embeddings,
            )
            loss = (
                (classification_loss * classification_weight)
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
        total_loss += float(loss.detach().item()) * batch_size
        total_classification_loss += float(classification_loss.detach().item()) * batch_size
        total_embedding_loss += float(embedding_loss.detach().item()) * batch_size
        total_score_loss += float(score_loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    if pending_step:
        training_runtime.step_optimizer()
        training_runtime.zero_grad()

    return DistillationBatchMetrics(
        total_loss=total_loss,
        total_classification_loss=total_classification_loss,
        total_embedding_loss=total_embedding_loss,
        total_score_loss=total_score_loss,
        total_correct=total_correct,
        total_examples=total_examples,
    )


def _flatten_mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.detach().to(dtype=torch.float32)
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        return waveform.squeeze(0).detach().to(dtype=torch.float32)
    raise ValueError(
        "Distillation waveform crops must be mono tensors shaped [frames] or [1, frames], "
        f"got {tuple(waveform.shape)}."
    )


def _embedding_alignment_loss(
    *,
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError(
            "Teacher and student embedding tensors must match for direct alignment; "
            f"got {tuple(student_embeddings.shape)} vs {tuple(teacher_embeddings.shape)}."
        )
    student_norm = torch_functional.normalize(student_embeddings, dim=1)
    teacher_norm = torch_functional.normalize(teacher_embeddings.detach(), dim=1)
    return 1.0 - torch_functional.cosine_similarity(student_norm, teacher_norm, dim=1).mean()


def _score_matrix_loss(
    *,
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    if student_embeddings.shape[0] < 2:
        return student_embeddings.new_zeros(())
    student_norm = torch_functional.normalize(student_embeddings, dim=1)
    teacher_norm = torch_functional.normalize(teacher_embeddings.detach(), dim=1)
    student_scores = student_norm @ student_norm.transpose(0, 1)
    teacher_scores = teacher_norm @ teacher_norm.transpose(0, 1)
    mask = torch.triu(
        torch.ones_like(student_scores, dtype=torch.bool),
        diagonal=1,
    )
    return torch_functional.mse_loss(student_scores[mask], teacher_scores[mask])


__all__ = [
    "DistillationBatch",
    "DistillationBatchMetrics",
    "DistillationEpochBreakdown",
    "ManifestDistillationDataset",
    "build_distillation_dataloader",
    "collate_distillation_examples",
    "run_distillation_batches",
]
