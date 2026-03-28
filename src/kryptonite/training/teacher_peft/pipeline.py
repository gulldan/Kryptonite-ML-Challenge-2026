"""Runnable WavLM / w2v-BERT teacher branch with PEFT-only adaptation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from torch import nn
from torch.utils.data import DataLoader

from kryptonite.data import AudioLoadRequest, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    build_verification_evaluation_report,
    load_verification_score_rows,
    write_verification_evaluation_report,
)
from kryptonite.features import UtteranceChunkingRequest, chunk_utterance, pool_chunk_tensors
from kryptonite.models import ArcMarginLoss, CosineClassifier
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id
from kryptonite.training.manifest_speaker_data import build_speaker_index, load_manifest_rows
from kryptonite.training.optimization_runtime import (
    build_training_runtime,
    validate_training_precision,
)
from kryptonite.training.production_dataloader import BalancedSpeakerBatchSampler
from kryptonite.training.speaker_baseline import (
    EMBEDDING_METADATA_JSONL_NAME,
    EMBEDDING_METADATA_PARQUET_NAME,
    EMBEDDINGS_FILE_NAME,
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    EmbeddingExportSummary,
    SpeakerBaselineRunArtifacts,
    TrainingSummary,
    _load_manifest_metadata_lookup,
    _lookup_manifest_metadata_row,
    build_default_cohort_bank,
    build_fixed_train_chunking_request,
    load_or_generate_trials,
    prepare_demo_artifacts_if_needed,
    render_markdown_report,
    resolve_device,
    score_trials,
)

from .config import TeacherPeftConfig
from .data import ManifestWaveformDataset, WaveformTrainingBatch, collate_waveform_examples
from .model import (
    TeacherPeftEncoder,
    build_feature_batch,
    build_teacher_peft_backbone,
    count_trainable_parameters,
    load_teacher_feature_extractor,
    resolve_hidden_size,
    write_teacher_checkpoint,
)

REPORT_FILE_NAME = "teacher_peft_report.md"
TeacherPeftRunArtifacts = SpeakerBaselineRunArtifacts


def run_teacher_peft(
    config: TeacherPeftConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
    feature_extractor_factory: Callable[..., Any] = load_teacher_feature_extractor,
    backbone_factory: Callable[..., nn.Module] = build_teacher_peft_backbone,
) -> TeacherPeftRunArtifacts:
    prepare_demo_artifacts_if_needed(
        project=config.project,
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        enabled=config.data.generate_demo_artifacts_if_missing,
    )
    validate_training_precision(config.project.training.precision, baseline_name="Teacher PEFT")
    if config.project.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive for teacher PEFT runs.")

    seed_state = set_global_seed(
        config.project.runtime.seed,
        deterministic=config.project.reproducibility.deterministic,
        pythonhashseed=config.project.reproducibility.pythonhashseed,
    )
    del seed_state

    device = resolve_device(device_override or config.project.runtime.device)
    project_root = resolve_project_path(config.project.paths.project_root, ".")
    token = config.project.resolved_secrets.get("huggingface_hub_token")

    train_rows = load_manifest_rows(
        config.data.train_manifest,
        project_root=project_root,
        limit=config.data.max_train_rows,
    )
    dev_rows = load_manifest_rows(
        config.data.dev_manifest,
        project_root=project_root,
        limit=config.data.max_dev_rows,
    )
    speaker_to_index = build_speaker_index(train_rows)

    audio_request = AudioLoadRequest.from_config(
        config.project.normalization,
        vad=config.project.vad,
    )
    chunking_request = build_fixed_train_chunking_request(
        chunking=config.project.chunking,
        baseline_name="Teacher PEFT",
    )
    dataset = ManifestWaveformDataset(
        rows=train_rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        audio_request=audio_request,
        chunking_request=chunking_request,
        seed=config.project.runtime.seed,
    )
    sampler = BalancedSpeakerBatchSampler(
        rows=train_rows,
        batch_size=config.project.training.batch_size,
        seed=config.project.runtime.seed,
        chunking_request=chunking_request,
    )
    loader = cast(
        DataLoader[WaveformTrainingBatch],
        DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=config.project.runtime.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_waveform_examples,
            persistent_workers=False,
            prefetch_factor=2 if config.project.runtime.num_workers > 0 else None,
        ),
    )

    feature_extractor = feature_extractor_factory(
        model_config=config.model,
        token=token,
    )
    backbone = backbone_factory(
        model_config=config.model,
        adapter_config=config.adapter,
        token=token,
    )
    encoder = TeacherPeftEncoder(
        backbone=backbone.to(device),
        hidden_size=resolve_hidden_size(backbone),
        embedding_dim=config.model.embedding_dim,
        projection_dropout=config.model.projection_dropout,
    ).to(device)
    trainable_parameters, total_parameters = count_trainable_parameters(encoder)

    tracker_run = None
    if config.project.tracking.enabled:
        tracker = build_tracker(config=config.project)
        tracker_run = tracker.start_run(kind="teacher-peft", config=config.to_dict())
        run_id = tracker_run.run_id
    else:
        run_id = create_run_id()

    output_root = resolve_project_path(str(project_root), config.data.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    classifier = CosineClassifier(
        config.model.embedding_dim,
        num_classes=len(speaker_to_index),
        num_blocks=config.objective.classifier_blocks,
        hidden_dim=config.objective.classifier_hidden_dim,
    ).to(device)
    criterion = ArcMarginLoss(
        scale=config.objective.scale,
        margin=config.objective.margin,
        easy_margin=config.objective.easy_margin,
    )
    training_runtime = build_training_runtime(
        parameters=[*encoder.parameters(), *classifier.parameters()],
        optimization_config=config.optimization,
        precision=config.project.training.precision,
        device=device,
        max_epochs=config.project.training.max_epochs,
    )
    epoch_summaries = train_teacher_epochs(
        model=encoder,
        classifier=classifier,
        criterion=criterion,
        training_runtime=training_runtime,
        loader=loader,
        dataset=dataset,
        sampler=sampler,
        feature_extractor=feature_extractor,
        sample_rate_hz=config.project.normalization.target_sample_rate_hz,
        device=device,
        max_epochs=config.project.training.max_epochs,
        tracker_run=tracker_run,
    )

    checkpoint_dir = output_root / config.data.checkpoint_name
    checkpoint_files = write_teacher_checkpoint(
        checkpoint_dir=checkpoint_dir,
        encoder=encoder,
        classifier=classifier,
        feature_extractor=feature_extractor,
        model_config=config.model,
        adapter_config=config.adapter,
        baseline_config=config.to_dict(),
        speaker_to_index=speaker_to_index,
    )

    training_summary = TrainingSummary(
        device=str(device),
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        provenance_ruleset=config.provenance.ruleset,
        provenance_initialization=config.provenance.initialization,
        speaker_count=len(speaker_to_index),
        train_row_count=len(train_rows),
        dev_row_count=len(dev_rows),
        checkpoint_path=str(checkpoint_dir),
        epochs=tuple(epoch_summaries),
    )
    training_summary_path = output_root / TRAINING_SUMMARY_FILE_NAME
    training_summary_path.write_text(
        json.dumps(training_summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    embedding_summary, metadata_rows = export_teacher_embeddings(
        output_root=output_root,
        model=encoder,
        feature_extractor=feature_extractor,
        rows=dev_rows,
        manifest_path=config.data.dev_manifest,
        project_root=project_root,
        audio_request=audio_request,
        sample_rate_hz=config.project.normalization.target_sample_rate_hz,
        chunking=config.project.chunking,
        eval_batch_size=config.project.training.eval_batch_size,
        device=device,
        embedding_source=f"teacher_peft:{config.model.model_id}",
    )
    trials_path, trial_rows = load_or_generate_trials(
        output_root=output_root,
        configured_trials_manifest=config.data.trials_manifest,
        metadata_rows=metadata_rows,
        project_root=project_root,
    )
    cohort_bank = build_default_cohort_bank(
        output_root=output_root,
        embedding_summary=embedding_summary,
        train_manifest_path=config.data.train_manifest,
        trials_path=trials_path,
        project_root=project_root,
    )
    score_summary = score_trials(
        output_root=output_root,
        trials_path=trials_path,
        metadata_rows=metadata_rows,
        trial_rows=trial_rows,
    )
    score_summary_path = output_root / SCORE_SUMMARY_FILE_NAME
    score_summary_path.write_text(
        json.dumps(score_summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    verification_report = write_verification_evaluation_report(
        build_verification_evaluation_report(
            load_verification_score_rows(score_summary.scores_path),
            scores_path=score_summary.scores_path,
            trials_path=trials_path,
            metadata_path=embedding_summary.metadata_parquet_path,
            trial_rows=trial_rows,
            metadata_rows=metadata_rows,
        ),
        output_root=output_root,
    )

    reproducibility = build_reproducibility_snapshot(
        config=config.project,
        config_path=resolve_project_path(str(project_root), str(config_path)),
    )
    reproducibility_path = output_root / REPRODUCIBILITY_FILE_NAME
    reproducibility_path.write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report_path = output_root / REPORT_FILE_NAME
    report_path.write_text(
        render_markdown_report(
            title="Teacher PEFT Report",
            provenance=config.provenance,
            training_summary=training_summary,
            embedding_summary=embedding_summary,
            score_summary=score_summary,
            verification_report=verification_report,
            output_root=output_root,
            project_root=project_root,
        ),
        encoding="utf-8",
    )

    if tracker_run is not None:
        final_epoch = training_summary.epochs[-1]
        tracker_run.log_metrics(
            {
                "train_loss": final_epoch.mean_loss,
                "train_accuracy": final_epoch.accuracy,
                "score_gap": score_summary.score_gap or 0.0,
                "eer": verification_report.summary.metrics.eer,
                "min_dcf": verification_report.summary.metrics.min_dcf,
                "teacher_trainable_parameters": float(trainable_parameters),
                "teacher_total_parameters": float(total_parameters),
            },
            step=config.project.training.max_epochs,
        )
        artifact_paths = [
            training_summary_path,
            Path(embedding_summary.embeddings_path),
            Path(embedding_summary.metadata_jsonl_path),
            Path(embedding_summary.metadata_parquet_path),
            Path(cohort_bank.embeddings_path),
            Path(cohort_bank.metadata_jsonl_path),
            Path(cohort_bank.metadata_parquet_path),
            Path(cohort_bank.summary_path),
            Path(trials_path),
            Path(score_summary.scores_path),
            score_summary_path,
            Path(verification_report.report_json_path),
            Path(verification_report.report_markdown_path),
            Path(verification_report.slice_dashboard_path),
            reproducibility_path,
            report_path,
            *[path for path in checkpoint_files if path.is_file()],
        ]
        if verification_report.error_analysis_json_path is not None:
            artifact_paths.append(Path(verification_report.error_analysis_json_path))
        if verification_report.error_analysis_markdown_path is not None:
            artifact_paths.append(Path(verification_report.error_analysis_markdown_path))
        for artifact_path in artifact_paths:
            tracker_run.log_artifact(artifact_path)
        tracker_run.finish(
            summary={
                "output_root": str(output_root),
                "score_gap": score_summary.score_gap,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            }
        )

    return TeacherPeftRunArtifacts(
        output_root=str(output_root),
        checkpoint_path=str(checkpoint_dir),
        training_summary_path=str(training_summary_path),
        embeddings_path=embedding_summary.embeddings_path,
        embedding_metadata_jsonl_path=embedding_summary.metadata_jsonl_path,
        embedding_metadata_parquet_path=embedding_summary.metadata_parquet_path,
        trials_path=str(trials_path),
        scores_path=score_summary.scores_path,
        score_summary_path=str(score_summary_path),
        reproducibility_path=str(reproducibility_path),
        report_path=str(report_path),
        training_summary=training_summary,
        embedding_summary=embedding_summary,
        score_summary=score_summary,
        verification_report=verification_report,
        tracking_run_dir=None if tracker_run is None else str(tracker_run.run_dir),
    )


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
) -> list[Any]:
    from kryptonite.training.speaker_baseline import EpochSummary

    summaries: list[EpochSummary] = []
    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        total_loss, total_correct, total_examples = run_teacher_batches(
            model=model,
            classifier=classifier,
            criterion=criterion,
            training_runtime=training_runtime,
            loader=loader,
            feature_extractor=feature_extractor,
            sample_rate_hz=sample_rate_hz,
            device=device,
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
) -> tuple[float, int, int]:
    model.train()
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    accumulation_steps = training_runtime.gradient_accumulation_steps
    pending_step = False

    training_runtime.zero_grad()
    for batch_index, batch in enumerate(loader, start=1):
        model_inputs = build_feature_batch(
            feature_extractor=feature_extractor,
            waveforms=batch.waveforms,
            sample_rate_hz=sample_rate_hz,
            device=device,
        )
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
) -> tuple[Any, list[dict[str, Any]]]:
    import torch

    model.eval()
    manifest_metadata_lookup = _load_manifest_metadata_lookup(
        manifest_path=manifest_path,
        project_root=project_root,
    )
    metadata_rows: list[dict[str, Any]] = []
    embeddings: list[torch.Tensor] = []
    point_ids: list[str] = []

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
