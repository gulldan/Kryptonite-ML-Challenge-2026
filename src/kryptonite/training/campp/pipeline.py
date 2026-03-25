"""End-to-end CAM++ baseline training, embedding export, and cosine scoring."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    build_verification_evaluation_report,
    load_verification_score_rows,
    write_verification_evaluation_report,
)
from kryptonite.features import (
    FbankExtractionRequest,
)
from kryptonite.models import ArcMarginLoss, CAMPPlusEncoder, CosineClassifier
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id

from ..manifest_speaker_data import (
    ManifestSpeakerDataset,
    build_speaker_index,
    collate_training_examples,
    load_manifest_rows,
)
from ..speaker_baseline import (
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    SpeakerBaselineRunArtifacts,
    TrainingSummary,
    build_fixed_train_chunking_request,
    export_dev_embeddings,
    load_or_generate_trials,
    prepare_demo_artifacts_if_needed,
    render_markdown_report,
    resolve_device,
    score_trials,
    train_epochs,
    validate_fp32_only,
    write_checkpoint,
)
from .config import CAMPPlusBaselineConfig

REPORT_FILE_NAME = "campp_baseline_report.md"
CAMPPlusRunArtifacts = SpeakerBaselineRunArtifacts


def run_campp_baseline(
    config: CAMPPlusBaselineConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
) -> CAMPPlusRunArtifacts:
    prepare_demo_artifacts_if_needed(
        project=config.project,
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        enabled=config.data.generate_demo_artifacts_if_missing,
    )
    validate_fp32_only(config.project.training.precision, baseline_name="CAM++")
    if config.project.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive for CAM++ baseline runs.")
    seed_state = set_global_seed(
        config.project.runtime.seed,
        deterministic=config.project.reproducibility.deterministic,
        pythonhashseed=config.project.reproducibility.pythonhashseed,
    )
    del seed_state

    device = resolve_device(device_override or config.project.runtime.device)
    project_root = resolve_project_path(config.project.paths.project_root, ".")

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
    feature_request = FbankExtractionRequest.from_config(config.project.features)
    chunking_request = build_fixed_train_chunking_request(
        chunking=config.project.chunking,
        baseline_name="CAM++",
    )

    train_dataset = ManifestSpeakerDataset(
        rows=train_rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking_request=chunking_request,
        seed=config.project.runtime.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.project.training.batch_size,
        shuffle=True,
        num_workers=config.project.runtime.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_training_examples,
        drop_last=False,
    )

    tracker_run = None
    if config.project.tracking.enabled:
        tracker = build_tracker(config=config.project)
        tracker_run = tracker.start_run(kind="campp-baseline", config=config.to_dict())
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_lr_lambda(config),
    )

    epoch_summaries = train_epochs(
        model=model,
        classifier=classifier,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=train_loader,
        dataset=train_dataset,
        device=device,
        max_epochs=config.project.training.max_epochs,
        grad_clip_norm=config.optimization.grad_clip_norm,
        tracker_run=tracker_run,
    )
    checkpoint_path = output_root / config.data.checkpoint_name
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        classifier=classifier,
        model_config=asdict(config.model),
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
        embedding_source="campp_baseline",
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
            title="CAM++ Baseline Report",
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
            step=config.project.training.max_epochs,
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

    return CAMPPlusRunArtifacts(
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


def _build_lr_lambda(config: CAMPPlusBaselineConfig):
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
