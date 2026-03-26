"""End-to-end ERes2NetV2 baseline training, embedding export, and cosine scoring."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    build_verification_evaluation_report,
    load_verification_score_rows,
    write_verification_evaluation_report,
)
from kryptonite.features import FbankExtractionRequest
from kryptonite.models import ArcMarginLoss, CosineClassifier, ERes2NetV2Encoder
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id

from ..manifest_speaker_data import (
    build_speaker_index,
    load_manifest_rows,
)
from ..optimization_runtime import build_training_runtime, validate_training_precision
from ..production_dataloader import build_production_train_dataloader
from ..speaker_baseline import (
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    SpeakerBaselineRunArtifacts,
    TrainingSummary,
    export_dev_embeddings,
    load_or_generate_trials,
    prepare_demo_artifacts_if_needed,
    render_markdown_report,
    resolve_device,
    score_trials,
    train_epochs,
    write_checkpoint,
)
from .config import ERes2NetV2BaselineConfig

REPORT_FILE_NAME = "eres2netv2_baseline_report.md"
ERes2NetV2RunArtifacts = SpeakerBaselineRunArtifacts


def run_eres2netv2_baseline(
    config: ERes2NetV2BaselineConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
) -> ERes2NetV2RunArtifacts:
    prepare_demo_artifacts_if_needed(
        project=config.project,
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        enabled=config.data.generate_demo_artifacts_if_missing,
    )
    validate_training_precision(config.project.training.precision, baseline_name="ERes2NetV2")
    if config.project.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive for ERes2NetV2 baseline runs.")

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

    feature_request = FbankExtractionRequest.from_config(config.project.features)
    train_dataset, train_sampler, train_loader = build_production_train_dataloader(
        rows=train_rows,
        speaker_to_index=speaker_to_index,
        project=config.project,
        total_epochs=config.project.training.max_epochs,
        pin_memory=device.type == "cuda",
    )

    tracker_run = None
    if config.project.tracking.enabled:
        tracker = build_tracker(config=config.project)
        tracker_run = tracker.start_run(kind="eres2netv2-baseline", config=config.to_dict())
        run_id = tracker_run.run_id
    else:
        run_id = create_run_id()

    output_root = resolve_project_path(str(project_root), config.data.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    model = ERes2NetV2Encoder(config.model).to(device)
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
    training_runtime = build_training_runtime(
        parameters=[*model.parameters(), *classifier.parameters()],
        optimization_config=config.optimization,
        precision=config.project.training.precision,
        device=device,
        max_epochs=config.project.training.max_epochs,
    )

    epoch_summaries = train_epochs(
        model=model,
        classifier=classifier,
        criterion=criterion,
        training_runtime=training_runtime,
        loader=train_loader,
        dataset=train_dataset,
        sampler=train_sampler,
        device=device,
        max_epochs=config.project.training.max_epochs,
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
        audio_request=AudioLoadRequest.from_config(
            config.project.normalization,
            vad=config.project.vad,
        ),
        feature_request=feature_request,
        chunking=config.project.chunking,
        device=device,
        embedding_source="eres2netv2_baseline",
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
            title="ERes2NetV2 Baseline Report",
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

    return ERes2NetV2RunArtifacts(
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
