"""Shared helpers for warm-start CAM++ fine-tuning stages."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch import nn
from torch.utils.data import DataLoader

from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)

from ..augmentation_runtime import TrainingAugmentationRuntime
from ..manifest_speaker_data import (
    ManifestSpeakerDataset,
    TrainingBatch,
    collate_training_examples,
)
from ..speaker_baseline import EpochSummary
from .stage2_sampler import Stage2BatchSampler

logger = logging.getLogger(__name__)


def resolve_warm_start_checkpoint_path(
    *,
    checkpoint_path: str,
    project_root: Path,
    candidate_names: tuple[str, ...],
    source_label: str,
) -> Path:
    resolved = resolve_project_path(str(project_root), checkpoint_path)
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        for candidate_name in candidate_names:
            candidate = resolved / candidate_name
            if candidate.is_file():
                return candidate
        expected = ", ".join(str(resolved / candidate_name) for candidate_name in candidate_names)
        raise FileNotFoundError(
            f"{source_label} run directory does not contain a known checkpoint file. "
            f"Expected one of: {expected}."
        )
    raise FileNotFoundError(
        f"{source_label} checkpoint not found at {resolved}. Provide either a checkpoint file "
        "or a completed run directory."
    )


def load_warm_start_checkpoint(
    *,
    checkpoint_path: str,
    model: nn.Module,
    classifier: nn.Module,
    project_root: Path,
    candidate_names: tuple[str, ...],
    source_label: str,
) -> None:
    resolved = resolve_warm_start_checkpoint_path(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        candidate_names=candidate_names,
        source_label=source_label,
    )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "classifier_state_dict" in checkpoint:
        try:
            classifier.load_state_dict(checkpoint["classifier_state_dict"])
        except RuntimeError:
            logger.warning(
                "Classifier state from %s checkpoint is incompatible "
                "(speaker count mismatch?). Initialising classifier from scratch.",
                source_label,
            )
    logger.info("Loaded %s checkpoint from %s", source_label, resolved)


def mine_hard_negatives(
    *,
    model: nn.Module,
    rows: list[ManifestRow],
    project_root: Path,
    audio_request: AudioLoadRequest,
    feature_request: FbankExtractionRequest,
    base_chunking: Any,
    device: torch.device,
    top_k: int,
    max_rows: int | None,
    seed: int,
    epoch: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute per-speaker difficulty weights via centroid cosine similarity."""
    import random as _stdlib_random

    mining_rows = list(rows)
    if max_rows is not None and len(mining_rows) > max_rows:
        rng = _stdlib_random.Random(seed + epoch)
        mining_rows = rng.sample(mining_rows, max_rows)

    model.eval()
    extractor = FbankExtractor(request=feature_request)
    chunking_request = UtteranceChunkingRequest.from_config(base_chunking)

    speaker_embeddings: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for row in mining_rows:
            try:
                loaded = load_manifest_audio(row, project_root=project_root, request=audio_request)
                eval_chunks = chunk_utterance(
                    loaded.audio.waveform,
                    sample_rate_hz=loaded.audio.sample_rate_hz,
                    stage="eval",
                    request=chunking_request,
                )
                chunk_embs: list[torch.Tensor] = []
                for chunk in eval_chunks.chunks:
                    feat = extractor.extract(
                        chunk.waveform,
                        sample_rate_hz=loaded.audio.sample_rate_hz,
                    )
                    emb = model(feat.unsqueeze(0).to(device=device, dtype=torch.float32)).squeeze(0)
                    chunk_embs.append(emb.detach().cpu())
                pooled = pool_chunk_tensors(chunk_embs, pooling_mode=eval_chunks.pooling_mode)
                normed = torch_functional.normalize(pooled, dim=0).numpy()
                speaker_embeddings.setdefault(row.speaker_id, []).append(normed)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping row %s during mining: %s", row.audio_path, exc)

    model.train()

    if len(speaker_embeddings) < 2:
        return {}, {
            "epoch": epoch,
            "status": "skipped",
            "reason": "too_few_speakers",
            "speakers_mined": len(speaker_embeddings),
        }

    speaker_ids_ordered = sorted(speaker_embeddings.keys())
    centroids = np.stack(
        [np.mean(speaker_embeddings[speaker_id], axis=0) for speaker_id in speaker_ids_ordered],
        axis=0,
    )
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    centroids = centroids / norms

    sim_matrix = centroids @ centroids.T

    speaker_weights: dict[str, float] = {}
    for index, speaker_id in enumerate(speaker_ids_ordered):
        row_sims = sim_matrix[index].copy()
        row_sims[index] = -1.0
        k = min(top_k, len(speaker_ids_ordered) - 1)
        top_indices = np.argsort(row_sims)[::-1][:k]
        mean_top_k_sim = float(np.mean(row_sims[top_indices]))
        speaker_weights[speaker_id] = max(1.0, 1.0 + mean_top_k_sim * 3.0)

    all_weights = list(speaker_weights.values())
    mining_entry: dict[str, Any] = {
        "epoch": epoch,
        "status": "ok",
        "speakers_mined": len(speaker_ids_ordered),
        "rows_used": len(mining_rows),
        "mean_weight": round(float(np.mean(all_weights)), 4),
        "max_weight": round(float(max(all_weights)), 4),
    }
    return speaker_weights, mining_entry


def build_stage_finetune_dataloader(
    *,
    rows: list[ManifestRow],
    speaker_to_index: dict[str, int],
    project: Any,
    chunking_request: UtteranceChunkingRequest,
    active_runtime: TrainingAugmentationRuntime | None,
    device: torch.device,
    hard_negative_fraction: float,
) -> tuple[ManifestSpeakerDataset, Stage2BatchSampler, DataLoader[TrainingBatch]]:
    project_root = resolve_project_path(project.paths.project_root, ".")
    audio_request = AudioLoadRequest.from_config(project.normalization, vad=project.vad)
    feature_request = FbankExtractionRequest.from_config(project.features)

    dataset = ManifestSpeakerDataset(
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
        "collate_fn": collate_training_examples,
        "persistent_workers": False,
    }
    if project.runtime.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = cast(DataLoader[TrainingBatch], DataLoader(**loader_kwargs))
    return dataset, sampler, loader


def train_one_epoch(
    *,
    epoch: int,
    model: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[TrainingBatch],
    device: torch.device,
    grad_clip_norm: float | None,
    tracker_run: Any | None,
    extra_metrics: dict[str, float] | None = None,
) -> EpochSummary:
    model.train()
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        features = batch.features.to(device=device, dtype=torch.float32)
        labels = batch.labels.to(device=device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(features)
        logits = classifier(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=grad_clip_norm,
            )
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Training loader produced zero examples.")

    learning_rate = round(float(optimizer.param_groups[0]["lr"]), 8)
    summary = EpochSummary(
        epoch=epoch + 1,
        mean_loss=round(total_loss / total_examples, 6),
        accuracy=round(total_correct / total_examples, 6),
        learning_rate=learning_rate,
    )
    if tracker_run is not None:
        metrics = {
            "train_loss": summary.mean_loss,
            "train_accuracy": summary.accuracy,
            "learning_rate": summary.learning_rate,
        }
        if extra_metrics is not None:
            metrics.update(extra_metrics)
        tracker_run.log_metrics(metrics, step=summary.epoch)
    return summary


def build_fixed_crop_phases(
    *,
    enabled: bool,
    start_crop_seconds: float,
    end_crop_seconds: float,
    curriculum_epochs: int,
    base_chunking: Any,
) -> list[UtteranceChunkingRequest]:
    """Return one fixed-crop UtteranceChunkingRequest per curriculum phase."""
    eval_limit = base_chunking.eval_max_full_utterance_seconds
    if not enabled or curriculum_epochs <= 0:
        return [UtteranceChunkingRequest.from_config(base_chunking)]

    middle_crop_seconds = round((start_crop_seconds + end_crop_seconds) / 2.0, 6)
    return [
        UtteranceChunkingRequest(
            train_min_crop_seconds=crop_seconds,
            train_max_crop_seconds=crop_seconds,
            train_num_crops=1,
            eval_max_full_utterance_seconds=eval_limit,
        )
        for crop_seconds in (start_crop_seconds, middle_crop_seconds, end_crop_seconds)
    ]


def phase_for_epoch(
    epoch: int,
    *,
    curriculum_enabled: bool,
    curriculum_epochs: int,
    n_phases: int,
) -> int:
    if n_phases <= 1:
        return 0
    if not curriculum_enabled or curriculum_epochs <= 0:
        return 0
    return min(n_phases - 1, epoch // curriculum_epochs)


def build_cosine_lr_lambda(
    *,
    max_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    min_learning_rate: float,
) -> Any:
    min_ratio = min_learning_rate / learning_rate

    def _lambda(epoch_index: int) -> float:
        if warmup_epochs == 0 and epoch_index == 0:
            return 1.0
        current_epoch = epoch_index + 1
        if warmup_epochs > 0 and current_epoch <= warmup_epochs:
            return max(min_ratio, current_epoch / warmup_epochs)
        cosine_steps = max(1, max_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (current_epoch - warmup_epochs) / cosine_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + ((1.0 - min_ratio) * cosine)

    return _lambda


def margin_for_epoch(
    epoch: int,
    *,
    enabled: bool,
    start_margin: float,
    end_margin: float,
    ramp_epochs: int,
) -> float:
    if not enabled:
        return start_margin
    if ramp_epochs <= 1:
        return end_margin
    progress = min(1.0, max(0.0, epoch / (ramp_epochs - 1)))
    return start_margin + ((end_margin - start_margin) * progress)


__all__ = [
    "build_cosine_lr_lambda",
    "build_fixed_crop_phases",
    "build_stage_finetune_dataloader",
    "load_warm_start_checkpoint",
    "margin_for_epoch",
    "mine_hard_negatives",
    "phase_for_epoch",
    "resolve_warm_start_checkpoint_path",
    "train_one_epoch",
]
