"""Training and export helpers for Teacher PEFT runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from kryptonite.data import AudioLoadRequest, load_manifest_audio
from kryptonite.features import UtteranceChunkingRequest, chunk_utterance, pool_chunk_tensors
from kryptonite.training.production_dataloader import BalancedSpeakerBatchSampler
from kryptonite.training.speaker_baseline import (
    EMBEDDING_METADATA_JSONL_NAME,
    EMBEDDING_METADATA_PARQUET_NAME,
    EMBEDDINGS_FILE_NAME,
    EmbeddingExportSummary,
    EpochSummary,
    _load_manifest_metadata_lookup,
    _lookup_manifest_metadata_row,
)

from .model import TeacherPeftEncoder, build_feature_batch
from .progress import (
    emit_progress,
    format_cuda_memory,
    format_duration,
    format_eta,
    resolve_log_interval,
)

if TYPE_CHECKING:
    from .data import ManifestWaveformDataset, WaveformTrainingBatch


def train_teacher_epochs(
    *,
    model: TeacherPeftEncoder,
    classifier: nn.Module,
    criterion: nn.Module,
    training_runtime: Any,
    loader: DataLoader[WaveformTrainingBatch],
    dataset: ManifestWaveformDataset,
    sampler: BalancedSpeakerBatchSampler,
    feature_extractor: Any,
    sample_rate_hz: int,
    device: Any,
    max_epochs: int,
    tracker_run: Any | None,
) -> list[EpochSummary]:
    summaries: list[EpochSummary] = []
    total_batches = len(loader)
    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        emit_progress(
            f"[train] epoch {epoch + 1}/{max_epochs} start batches={total_batches} "
            f"batch_size={getattr(loader.batch_sampler, '_batch_size', 'unknown')} "
            f"accumulation={training_runtime.gradient_accumulation_steps} "
            f"lr={training_runtime.current_learning_rate():.8f}"
        )
        total_loss, total_correct, total_examples = run_teacher_batches(
            model=model,
            classifier=classifier,
            criterion=criterion,
            training_runtime=training_runtime,
            loader=loader,
            feature_extractor=feature_extractor,
            sample_rate_hz=sample_rate_hz,
            device=device,
            epoch_index=epoch + 1,
            max_epochs=max_epochs,
        )
        learning_rate = round(training_runtime.current_learning_rate(), 8)
        summary = EpochSummary(
            epoch=epoch + 1,
            mean_loss=round(total_loss / total_examples, 6),
            accuracy=round(total_correct / total_examples, 6),
            learning_rate=learning_rate,
        )
        summaries.append(summary)
        if tracker_run is not None:
            tracker_run.log_metrics(
                {
                    "train_loss": summary.mean_loss,
                    "train_accuracy": summary.accuracy,
                    "learning_rate": summary.learning_rate,
                },
                step=summary.epoch,
            )
        training_runtime.step_scheduler(mean_loss=summary.mean_loss)
        emit_progress(
            f"[train] epoch {summary.epoch}/{max_epochs} complete "
            f"loss={summary.mean_loss:.6f} accuracy={summary.accuracy:.6f} "
            f"lr={summary.learning_rate:.8f}"
        )
    return summaries


def run_teacher_batches(
    *,
    model: TeacherPeftEncoder,
    classifier: nn.Module,
    criterion: nn.Module,
    training_runtime: Any,
    loader: DataLoader[WaveformTrainingBatch],
    feature_extractor: Any,
    sample_rate_hz: int,
    device: Any,
    epoch_index: int,
    max_epochs: int,
) -> tuple[float, int, int]:
    del feature_extractor, sample_rate_hz
    model.train()
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    accumulation_steps = training_runtime.gradient_accumulation_steps
    pending_step = False
    total_batches = len(loader)
    started_at = time.monotonic()
    log_interval = resolve_log_interval(
        total_batches,
        target_updates=25,
        min_interval=10,
        max_interval=200,
    )

    training_runtime.zero_grad()
    for batch_index, batch in enumerate(loader, start=1):
        model_inputs = {
            key: value.to(device=device, non_blocking=True)
            for key, value in batch.model_inputs.items()
        }
        labels = batch.labels.to(device=device)
        with training_runtime.precision.autocast_context(device=device):
            embeddings = model(**model_inputs)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            scaled_loss = loss / accumulation_steps
        training_runtime.backward(scaled_loss)
        pending_step = True

        if batch_index % accumulation_steps == 0:
            training_runtime.step_optimizer()
            training_runtime.zero_grad()
            pending_step = False

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size
        if batch_index == 1 or batch_index == total_batches or batch_index % log_interval == 0:
            elapsed_seconds = time.monotonic() - started_at
            average_loss = total_loss / total_examples
            average_accuracy = total_correct / total_examples
            examples_per_second = total_examples / elapsed_seconds if elapsed_seconds > 0 else 0.0
            eta = format_eta(
                elapsed_seconds=elapsed_seconds,
                completed_items=batch_index,
                total_items=total_batches,
            )
            emit_progress(
                f"[train] epoch {epoch_index}/{max_epochs} batch {batch_index}/{total_batches} "
                f"loss={loss.detach().item():.6f} avg_loss={average_loss:.6f} "
                f"avg_acc={average_accuracy:.6f} lr={training_runtime.current_learning_rate():.8f} "
                f"ex_per_sec={examples_per_second:.2f} elapsed={format_duration(elapsed_seconds)} "
                f"eta={eta} mem={format_cuda_memory(device)}",
            )

    if pending_step:
        training_runtime.step_optimizer()
        training_runtime.zero_grad()

    if total_examples == 0:
        raise ValueError("Teacher PEFT loader produced zero examples.")
    return total_loss, total_correct, total_examples


def export_teacher_embeddings(
    *,
    output_root: Path,
    model: TeacherPeftEncoder,
    feature_extractor: Any,
    rows: list[Any],
    manifest_path: str,
    project_root: Path,
    audio_request: AudioLoadRequest,
    sample_rate_hz: int,
    chunking: Any,
    eval_batch_size: int,
    device: Any,
    embedding_source: str,
) -> tuple[EmbeddingExportSummary, list[dict[str, Any]]]:
    model.eval()
    manifest_metadata_lookup = _load_manifest_metadata_lookup(
        manifest_path=manifest_path,
        project_root=project_root,
    )
    metadata_rows: list[dict[str, Any]] = []
    embeddings: list[torch.Tensor] = []
    point_ids: list[str] = []
    total_rows = len(rows)
    started_at = time.monotonic()
    log_interval = resolve_log_interval(
        total_rows,
        target_updates=20,
        min_interval=25,
        max_interval=250,
    )

    with torch.no_grad():
        for index, row in enumerate(rows):
            loaded = load_manifest_audio(row, project_root=project_root, request=audio_request)
            eval_chunks = chunk_utterance(
                loaded.audio.waveform,
                sample_rate_hz=loaded.audio.sample_rate_hz,
                stage="eval",
                request=UtteranceChunkingRequest.from_config(chunking),
            )
            chunk_embeddings: list[torch.Tensor] = []
            for chunk_start in range(0, len(eval_chunks.chunks), eval_batch_size):
                chunk_waveforms = [
                    chunk.waveform.to(dtype=torch.float32)
                    for chunk in eval_chunks.chunks[chunk_start : chunk_start + eval_batch_size]
                ]
                model_inputs = build_feature_batch(
                    feature_extractor=feature_extractor,
                    waveforms=chunk_waveforms,
                    sample_rate_hz=sample_rate_hz,
                    device=device,
                )
                batch_embeddings = model(**model_inputs).detach().to(device="cpu")
                chunk_embeddings.extend(batch_embeddings)
            pooled = pool_chunk_tensors(chunk_embeddings, pooling_mode=eval_chunks.pooling_mode)
            normalized = torch.nn.functional.normalize(pooled, dim=0)
            trial_item_id = row.utterance_id or row.audio_path
            point_id = f"utt-{index:05d}"

            embeddings.append(normalized)
            point_ids.append(point_id)
            metadata_rows.append(
                {
                    **_lookup_manifest_metadata_row(
                        row=row,
                        trial_item_id=trial_item_id,
                        manifest_metadata_lookup=manifest_metadata_lookup,
                    ),
                    "atlas_point_id": point_id,
                    "trial_item_id": trial_item_id,
                    "speaker_id": row.speaker_id,
                    "utterance_id": row.utterance_id,
                    "audio_path": row.audio_path,
                    "split": row.split,
                    "role": row.role,
                    "channel": row.channel,
                    "dataset": row.dataset,
                    "source_dataset": row.source_dataset,
                    "duration_seconds": loaded.audio.duration_seconds,
                    "embedding_device": str(device),
                    "embedding_source": embedding_source,
                }
            )
            completed_rows = index + 1
            if (
                completed_rows == 1
                or completed_rows == total_rows
                or completed_rows % log_interval == 0
            ):
                elapsed_seconds = time.monotonic() - started_at
                utterances_per_second = (
                    completed_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0
                )
                emit_progress(
                    f"[embed] utterance {completed_rows}/{total_rows} "
                    f"utt_per_sec={utterances_per_second:.2f} "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    "eta="
                    f"{
                        format_eta(
                            elapsed_seconds=elapsed_seconds,
                            completed_items=completed_rows,
                            total_items=total_rows,
                        )
                    } "
                    f"mem={format_cuda_memory(device)}"
                )

    embeddings_matrix = torch.stack(embeddings, dim=0).to(dtype=torch.float32).numpy()
    npz_path = output_root / EMBEDDINGS_FILE_NAME
    jsonl_path = output_root / EMBEDDING_METADATA_JSONL_NAME
    parquet_path = output_root / EMBEDDING_METADATA_PARQUET_NAME
    np.savez(npz_path, embeddings=embeddings_matrix, point_ids=np.asarray(point_ids, dtype=str))
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    pl.DataFrame(metadata_rows).write_parquet(parquet_path)

    return (
        EmbeddingExportSummary(
            manifest_path=manifest_path,
            embedding_dim=int(embeddings_matrix.shape[1]),
            utterance_count=int(embeddings_matrix.shape[0]),
            speaker_count=len({row["speaker_id"] for row in metadata_rows}),
            embeddings_path=str(npz_path),
            metadata_jsonl_path=str(jsonl_path),
            metadata_parquet_path=str(parquet_path),
        ),
        metadata_rows,
    )
