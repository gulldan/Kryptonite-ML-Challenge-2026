"""CAM++ stage-2 heavy multi-condition training pipeline.

Stage-2 builds on a stage-1 pretrained checkpoint and applies:
- Heavy augmentation (corruption bank, multi-severity) from epoch 1 (no warmup/ramp)
- Hard negative mining: periodic inference pass to find confusable speaker pairs,
  then oversampling of those speakers during batch construction
- Short-utterance curriculum: starts with short crops and ramps up to full length
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch import nn
from torch.utils.data import DataLoader

from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    build_verification_evaluation_report,
    load_verification_score_rows,
    write_verification_evaluation_report,
)
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)
from kryptonite.models import ArcMarginLoss, CAMPPlusEncoder, CosineClassifier
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id

from ..augmentation_runtime import TrainingAugmentationRuntime
from ..manifest_speaker_data import (
    ManifestSpeakerDataset,
    TrainingBatch,
    build_speaker_index,
    collate_training_examples,
    load_manifest_rows,
)
from ..speaker_baseline import (
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    EpochSummary,
    SpeakerBaselineRunArtifacts,
    TrainingSummary,
    export_dev_embeddings,
    load_or_generate_trials,
    prepare_demo_artifacts_if_needed,
    render_markdown_report,
    resolve_device,
    score_trials,
    validate_fp32_only,
    write_checkpoint,
)
from .stage2_config import CAMPPlusStage2Config, Stage2UtteranceCurriculumConfig
from .stage2_sampler import Stage2BatchSampler

logger = logging.getLogger(__name__)

REPORT_FILE_NAME = "campp_stage2_report.md"
HARD_NEGATIVE_LOG_FILE_NAME = "hard_negative_mining_log.jsonl"

CAMPPlusStage2RunArtifacts = SpeakerBaselineRunArtifacts


def run_campp_stage2(
    config: CAMPPlusStage2Config,
    *,
    config_path: Path | str,
    device_override: str | None = None,
) -> CAMPPlusStage2RunArtifacts:
    prepare_demo_artifacts_if_needed(
        project=config.project,
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        enabled=config.data.generate_demo_artifacts_if_missing,
    )
    validate_fp32_only(config.project.training.precision, baseline_name="CAM++ stage-2")
    if config.project.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive for CAM++ stage-2 runs.")
    seed_state = set_global_seed(
        config.project.runtime.seed,
        deterministic=config.project.reproducibility.deterministic,
        pythonhashseed=config.project.reproducibility.pythonhashseed,
    )
    del seed_state

    device = resolve_device(device_override or config.project.runtime.device)
    project_root = resolve_project_path(config.project.paths.project_root, ".")
    max_epochs = config.project.training.max_epochs

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
    feature_request = FbankExtractionRequest.from_config(config.project.features)
    audio_request = AudioLoadRequest.from_config(
        config.project.normalization,
        vad=config.project.vad,
    )

    tracker_run = None
    if config.project.tracking.enabled:
        tracker = build_tracker(config=config.project)
        tracker_run = tracker.start_run(kind="campp-stage2", config=config.to_dict())
        run_id = tracker_run.run_id
    else:
        run_id = create_run_id()

    output_root = resolve_project_path(str(project_root), config.data.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    model = CAMPPlusEncoder(config.model).to(device)
    classifier = CosineClassifier(
        config.model.embedding_size,
        num_classes=len(speaker_to_index),
        num_blocks=config.objective.classifier_blocks,
        hidden_dim=config.objective.classifier_hidden_dim,
    ).to(device)
    _load_stage1_checkpoint(
        checkpoint_path=config.stage2.stage1_checkpoint,
        model=model,
        classifier=classifier,
        project_root=project_root,
    )

    criterion = ArcMarginLoss(
        scale=config.objective.scale,
        margin=config.objective.margin,
        easy_margin=config.objective.easy_margin,
    )
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr=config.optimization.learning_rate,
        momentum=config.optimization.momentum,
        nesterov=True,
        weight_decay=config.optimization.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_lr_lambda(config),
    )

    augmentation_runtime = TrainingAugmentationRuntime.from_project_config(
        project_root=project_root,
        scheduler_config=config.project.augmentation_scheduler,
        silence_config=config.project.silence_augmentation,
        total_epochs=max_epochs,
    )
    active_runtime = (
        augmentation_runtime if augmentation_runtime.has_effective_augmentation else None
    )

    chunking_phases = _build_chunking_phases(
        config.stage2.utterance_curriculum,
        max_epochs=max_epochs,
        base_chunking=config.project.chunking,
    )
    hard_negative_log: list[dict[str, Any]] = []
    epoch_summaries: list[EpochSummary] = []

    current_phase_index = -1
    dataset: ManifestSpeakerDataset | None = None
    sampler: Stage2BatchSampler | None = None
    loader: DataLoader[TrainingBatch] | None = None

    for epoch in range(max_epochs):
        phase_index = _phase_for_epoch(
            epoch,
            curriculum=config.stage2.utterance_curriculum,
            n_phases=len(chunking_phases),
        )
        if phase_index != current_phase_index:
            current_phase_index = phase_index
            chunking_request = chunking_phases[phase_index]
            dataset, sampler, loader = _build_stage2_dataloader(
                rows=train_rows,
                speaker_to_index=speaker_to_index,
                project=config.project,
                chunking_request=chunking_request,
                active_runtime=active_runtime,
                device=device,
                hard_negative_fraction=config.stage2.hard_negative.hard_negative_fraction,
            )
            logger.info(
                "Stage-2 phase %d: fixed crop %.2f s (epoch %d/%d)",
                phase_index,
                chunking_request.train_min_crop_seconds,
                epoch + 1,
                max_epochs,
            )

        hn_cfg = config.stage2.hard_negative
        if hn_cfg.enabled and sampler is not None and epoch % hn_cfg.mining_interval_epochs == 0:
            speaker_weights, mining_entry = _mine_hard_negatives(
                model=model,
                rows=train_rows,
                project_root=project_root,
                audio_request=audio_request,
                feature_request=feature_request,
                base_chunking=config.project.chunking,
                device=device,
                top_k=hn_cfg.top_k_per_speaker,
                max_rows=hn_cfg.max_train_rows_for_mining,
                seed=config.project.runtime.seed,
                epoch=epoch,
            )
            sampler.update_speaker_weights(speaker_weights)
            hard_negative_log.append(mining_entry)
            logger.info(
                "Hard-negative mining at epoch %d: %d speakers re-weighted "
                "(max_weight=%.3f, mean_weight=%.3f)",
                epoch + 1,
                mining_entry.get("speakers_mined", 0),
                mining_entry.get("max_weight", 1.0),
                mining_entry.get("mean_weight", 1.0),
            )

        assert dataset is not None and sampler is not None and loader is not None
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)

        summary = _train_one_epoch(
            epoch=epoch,
            model=model,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            loader=loader,
            device=device,
            grad_clip_norm=config.optimization.grad_clip_norm,
            tracker_run=tracker_run,
        )
        epoch_summaries.append(summary)
        lr_scheduler.step()

    checkpoint_path = output_root / config.data.checkpoint_name
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        classifier=classifier,
        model_config=asdict(config.model),
        baseline_config=config.to_dict(),
        speaker_to_index=speaker_to_index,
    )

    if hard_negative_log:
        hn_log_path = output_root / HARD_NEGATIVE_LOG_FILE_NAME
        hn_log_path.write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in hard_negative_log),
            encoding="utf-8",
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
        checkpoint_path=str(checkpoint_path),
        epochs=tuple(epoch_summaries),
    )
    training_summary_path = output_root / TRAINING_SUMMARY_FILE_NAME
    training_summary_path.write_text(
        json.dumps(training_summary.to_dict(), indent=2, sort_keys=True)
    )

    embedding_summary, metadata_rows = export_dev_embeddings(
        output_root=output_root,
        model=model,
        rows=dev_rows,
        manifest_path=config.data.dev_manifest,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking=config.project.chunking,
        device=device,
        embedding_source="campp_stage2",
    )
    trials_path, trial_rows = load_or_generate_trials(
        output_root=output_root,
        configured_trials_manifest=config.data.trials_manifest,
        metadata_rows=metadata_rows,
        project_root=project_root,
    )
    score_summary = score_trials(
        output_root=output_root,
        trials_path=trials_path,
        metadata_rows=metadata_rows,
        trial_rows=trial_rows,
    )
    score_summary_path = output_root / SCORE_SUMMARY_FILE_NAME
    score_summary_path.write_text(json.dumps(score_summary.to_dict(), indent=2, sort_keys=True))

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
    reproducibility_path.write_text(json.dumps(reproducibility, indent=2, sort_keys=True))

    report_path = output_root / REPORT_FILE_NAME
    report_path.write_text(
        render_markdown_report(
            title="CAM++ Stage-2 Report",
            provenance=config.provenance,
            training_summary=training_summary,
            embedding_summary=embedding_summary,
            score_summary=score_summary,
            verification_report=verification_report,
            output_root=output_root,
            project_root=project_root,
        )
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
            },
            step=max_epochs,
        )
        artifact_paths = [
            checkpoint_path,
            training_summary_path,
            Path(embedding_summary.embeddings_path),
            Path(embedding_summary.metadata_jsonl_path),
            Path(embedding_summary.metadata_parquet_path),
            Path(trials_path),
            Path(score_summary.scores_path),
            score_summary_path,
            Path(verification_report.report_json_path),
            Path(verification_report.report_markdown_path),
            Path(verification_report.roc_curve_path),
            Path(verification_report.det_curve_path),
            Path(verification_report.calibration_curve_path),
            Path(verification_report.histogram_path),
            Path(verification_report.slice_breakdown_path),
            reproducibility_path,
            report_path,
        ]
        if hard_negative_log:
            artifact_paths.append(output_root / HARD_NEGATIVE_LOG_FILE_NAME)
        if verification_report.error_analysis_json_path is not None:
            artifact_paths.append(Path(verification_report.error_analysis_json_path))
        if verification_report.error_analysis_markdown_path is not None:
            artifact_paths.append(Path(verification_report.error_analysis_markdown_path))
        for artifact_path in artifact_paths:
            tracker_run.log_artifact(artifact_path)
        tracker_run.finish(
            summary={
                "checkpoint_path": str(checkpoint_path),
                "score_gap": score_summary.score_gap,
                "trial_count": score_summary.trial_count,
                "eer": verification_report.summary.metrics.eer,
                "min_dcf": verification_report.summary.metrics.min_dcf,
            }
        )

    return CAMPPlusStage2RunArtifacts(
        output_root=str(output_root),
        checkpoint_path=str(checkpoint_path),
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
        tracking_run_dir=(None if tracker_run is None else str(tracker_run.run_dir)),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_stage1_checkpoint(
    *,
    checkpoint_path: str,
    model: nn.Module,
    classifier: nn.Module,
    project_root: Path,
) -> None:
    resolved = _resolve_stage1_checkpoint_path(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
    )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "classifier_state_dict" in checkpoint:
        try:
            classifier.load_state_dict(checkpoint["classifier_state_dict"])
        except RuntimeError:
            logger.warning(
                "Classifier state from stage-1 checkpoint is incompatible "
                "(speaker count mismatch?). Initialising classifier from scratch."
            )
    logger.info("Loaded stage-1 checkpoint from %s", resolved)


def _resolve_stage1_checkpoint_path(*, checkpoint_path: str, project_root: Path) -> Path:
    resolved = resolve_project_path(str(project_root), checkpoint_path)
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        for candidate_name in ("campp_stage1_encoder.pt", "campp_encoder.pt"):
            candidate = resolved / candidate_name
            if candidate.is_file():
                return candidate
        expected_stage1 = resolved / "campp_stage1_encoder.pt"
        expected_baseline = resolved / "campp_encoder.pt"
        raise FileNotFoundError(
            "Stage-1 run directory does not contain a known checkpoint file. "
            f"Expected one of: {expected_stage1}, {expected_baseline}."
        )
    raise FileNotFoundError(
        f"Stage-1 checkpoint not found at {resolved}. Provide either a checkpoint file "
        "or a completed stage-1 run directory."
    )


def _mine_hard_negatives(
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
    """Compute per-speaker difficulty weights via centroid cosine similarity.

    Returns:
        speaker_weights: speaker_id → sampling weight (≥ 1.0).  Higher weight
            means the speaker is more confusable and will be oversampled.
        mining_entry: metadata dict written to the hard-negative log.
    """
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
        [np.mean(speaker_embeddings[s], axis=0) for s in speaker_ids_ordered],
        axis=0,
    )
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    centroids = centroids / norms

    sim_matrix = centroids @ centroids.T

    speaker_weights: dict[str, float] = {}
    for i, speaker_id in enumerate(speaker_ids_ordered):
        row_sims = sim_matrix[i].copy()
        row_sims[i] = -1.0
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


def _build_stage2_dataloader(
    *,
    rows: list[ManifestRow],
    speaker_to_index: dict[str, int],
    project: Any,
    chunking_request: UtteranceChunkingRequest,
    active_runtime: TrainingAugmentationRuntime | None,
    device: torch.device,
    hard_negative_fraction: float,
) -> tuple[ManifestSpeakerDataset, Stage2BatchSampler, DataLoader[TrainingBatch]]:
    from typing import cast

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


def _train_one_epoch(
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
        tracker_run.log_metrics(
            {
                "train_loss": summary.mean_loss,
                "train_accuracy": summary.accuracy,
                "learning_rate": summary.learning_rate,
            },
            step=summary.epoch,
        )
    return summary


def _build_chunking_phases(
    curriculum: Stage2UtteranceCurriculumConfig,
    *,
    max_epochs: int,
    base_chunking: Any,
) -> list[UtteranceChunkingRequest]:
    """Return one UtteranceChunkingRequest per curriculum phase.

    With curriculum enabled and curriculum_epochs > 0, returns three phases:
    - Phase 0: short_crop_seconds (first third of training)
    - Phase 1: midpoint crop (middle third)
    - Phase 2: long_crop_seconds (final third)

    Without curriculum, returns a single phase that uses the base_chunking config.
    """
    eval_limit = base_chunking.eval_max_full_utterance_seconds
    if not curriculum.enabled or curriculum.curriculum_epochs <= 0:
        return [UtteranceChunkingRequest.from_config(base_chunking)]

    short = curriculum.short_crop_seconds
    long = curriculum.long_crop_seconds
    mid = round((short + long) / 2.0, 6)

    return [
        UtteranceChunkingRequest(
            train_min_crop_seconds=crop,
            train_max_crop_seconds=crop,
            train_num_crops=1,
            eval_max_full_utterance_seconds=eval_limit,
        )
        for crop in (short, mid, long)
    ]


def _phase_for_epoch(
    epoch: int,
    *,
    curriculum: Stage2UtteranceCurriculumConfig,
    n_phases: int,
) -> int:
    """Map an epoch index to its curriculum phase index (0-based)."""
    if n_phases <= 1:
        return 0
    if not curriculum.enabled or curriculum.curriculum_epochs <= 0:
        return 0
    return min(n_phases - 1, epoch // curriculum.curriculum_epochs)


def _build_lr_lambda(config: CAMPPlusStage2Config):
    max_epochs = config.project.training.max_epochs
    warmup_epochs = config.optimization.warmup_epochs
    max_lr = config.optimization.learning_rate
    min_lr = config.optimization.min_learning_rate
    min_ratio = min_lr / max_lr

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


__all__ = [
    "CAMPPlusStage2RunArtifacts",
    "run_campp_stage2",
]
