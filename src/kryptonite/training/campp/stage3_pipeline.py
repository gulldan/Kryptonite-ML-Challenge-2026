"""CAM++ stage-3 large-margin fine-tuning pipeline."""

from __future__ import annotations

import json
import logging
import math
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
from kryptonite.models import ArcMarginLoss, CAMPPlusEncoder, CosineClassifier
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id

from ..augmentation_runtime import TrainingAugmentationRuntime
from ..manifest_speaker_data import build_speaker_index, load_manifest_rows
from ..optimization_runtime import build_training_runtime, validate_training_precision
from ..speaker_baseline import (
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    EpochSummary,
    SpeakerBaselineRunArtifacts,
    TrainingSummary,
    build_default_cohort_bank,
    export_dev_embeddings,
    load_or_generate_trials,
    prepare_demo_artifacts_if_needed,
    render_markdown_report,
    resolve_device,
    score_trials,
    write_checkpoint,
)
from .finetune_common import (
    build_fixed_crop_phases,
    build_stage_finetune_dataloader,
    load_warm_start_checkpoint,
    margin_for_epoch,
    mine_hard_negatives,
    phase_for_epoch,
    train_one_epoch,
)
from .stage3_config import CAMPPlusStage3Config

logger = logging.getLogger(__name__)

REPORT_FILE_NAME = "campp_stage3_report.md"
HARD_NEGATIVE_LOG_FILE_NAME = "hard_negative_mining_log.jsonl"
SCHEDULE_FILE_NAME = "stage3_schedule.json"

CAMPPlusStage3RunArtifacts = SpeakerBaselineRunArtifacts


def run_campp_stage3(
    config: CAMPPlusStage3Config,
    *,
    config_path: Path | str,
    device_override: str | None = None,
) -> CAMPPlusStage3RunArtifacts:
    prepare_demo_artifacts_if_needed(
        project=config.project,
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        enabled=config.data.generate_demo_artifacts_if_missing,
    )
    validate_training_precision(config.project.training.precision, baseline_name="CAM++ stage-3")
    if config.project.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive for CAM++ stage-3 runs.")
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
        tracker_run = tracker.start_run(kind="campp-stage3", config=config.to_dict())
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
    load_warm_start_checkpoint(
        checkpoint_path=config.stage3.stage2_checkpoint,
        model=model,
        classifier=classifier,
        project_root=project_root,
        candidate_names=(
            "campp_stage2_encoder.pt",
            "campp_stage1_encoder.pt",
            "campp_encoder.pt",
        ),
        source_label="Stage-2",
    )

    initial_margin = (
        config.stage3.margin_schedule.start_margin
        if config.stage3.margin_schedule.enabled
        else config.objective.margin
    )
    criterion = ArcMarginLoss(
        scale=config.objective.scale,
        margin=initial_margin,
        easy_margin=config.objective.easy_margin,
    )
    training_runtime = build_training_runtime(
        parameters=[*model.parameters(), *classifier.parameters()],
        optimization_config=config.optimization,
        precision=config.project.training.precision,
        device=device,
        max_epochs=max_epochs,
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

    chunking_phases = build_fixed_crop_phases(
        enabled=config.stage3.crop_curriculum.enabled,
        start_crop_seconds=config.stage3.crop_curriculum.start_crop_seconds,
        end_crop_seconds=config.stage3.crop_curriculum.end_crop_seconds,
        curriculum_epochs=config.stage3.crop_curriculum.curriculum_epochs,
        base_chunking=config.project.chunking,
    )
    hard_negative_log: list[dict[str, object]] = []
    epoch_summaries: list[EpochSummary] = []
    schedule_rows: list[dict[str, float | int]] = []

    current_phase_index = -1
    dataset = None
    sampler = None
    loader = None
    previous_margin: float | None = None

    for epoch in range(max_epochs):
        phase_index = phase_for_epoch(
            epoch,
            curriculum_enabled=config.stage3.crop_curriculum.enabled,
            curriculum_epochs=config.stage3.crop_curriculum.curriculum_epochs,
            n_phases=len(chunking_phases),
        )
        if phase_index != current_phase_index:
            current_phase_index = phase_index
            chunking_request = chunking_phases[phase_index]
            dataset, sampler, loader = build_stage_finetune_dataloader(
                rows=train_rows,
                speaker_to_index=speaker_to_index,
                project=config.project,
                chunking_request=chunking_request,
                active_runtime=active_runtime,
                device=device,
                hard_negative_fraction=config.stage3.hard_negative.hard_negative_fraction,
            )
            logger.info(
                "Stage-3 phase %d: fixed crop %.2f s (epoch %d/%d)",
                phase_index,
                chunking_request.train_min_crop_seconds,
                epoch + 1,
                max_epochs,
            )

        assert dataset is not None and sampler is not None and loader is not None
        current_margin = margin_for_epoch(
            epoch,
            enabled=config.stage3.margin_schedule.enabled,
            start_margin=(
                config.stage3.margin_schedule.start_margin
                if config.stage3.margin_schedule.enabled
                else config.objective.margin
            ),
            end_margin=(
                config.stage3.margin_schedule.end_margin
                if config.stage3.margin_schedule.enabled
                else config.objective.margin
            ),
            ramp_epochs=config.stage3.margin_schedule.ramp_epochs,
        )
        criterion.update(current_margin)
        if previous_margin is None or not math.isclose(
            previous_margin, current_margin, rel_tol=0.0, abs_tol=1e-9
        ):
            logger.info(
                "Stage-3 margin updated to %.4f at epoch %d/%d",
                current_margin,
                epoch + 1,
                max_epochs,
            )
            previous_margin = current_margin

        hn_cfg = config.stage3.hard_negative
        if hn_cfg.enabled and epoch % hn_cfg.mining_interval_epochs == 0:
            speaker_weights, mining_entry = mine_hard_negatives(
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

        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)

        current_crop_seconds = chunking_phases[current_phase_index].train_min_crop_seconds
        schedule_rows.append(
            {
                "epoch": epoch + 1,
                "phase_index": current_phase_index,
                "crop_seconds": round(float(current_crop_seconds), 6),
                "margin": round(float(current_margin), 6),
            }
        )
        summary = train_one_epoch(
            epoch=epoch,
            model=model,
            classifier=classifier,
            criterion=criterion,
            training_runtime=training_runtime,
            loader=loader,
            device=device,
            tracker_run=tracker_run,
            extra_metrics={
                "margin": round(float(current_margin), 6),
                "crop_seconds": round(float(current_crop_seconds), 6),
            },
        )
        epoch_summaries.append(summary)
        training_runtime.step_scheduler(mean_loss=summary.mean_loss)

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

    schedule_path = output_root / SCHEDULE_FILE_NAME
    schedule_path.write_text(
        json.dumps(
            {
                "stage2_checkpoint": config.stage3.stage2_checkpoint,
                "crop_curriculum": asdict(config.stage3.crop_curriculum),
                "margin_schedule": asdict(config.stage3.margin_schedule),
                "epochs": schedule_rows,
            },
            indent=2,
            sort_keys=True,
        ),
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
        embedding_source="campp_stage3",
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
            title="CAM++ Stage-3 Report",
            provenance=config.provenance,
            training_summary=training_summary,
            embedding_summary=embedding_summary,
            score_summary=score_summary,
            verification_report=verification_report,
            output_root=output_root,
            project_root=project_root,
        )
        + _render_stage3_appendix(config=config, schedule_rows=schedule_rows)
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
            schedule_path,
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

    return CAMPPlusStage3RunArtifacts(
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


def _render_stage3_appendix(
    *,
    config: CAMPPlusStage3Config,
    schedule_rows: list[dict[str, float | int]],
) -> str:
    lines = [
        "",
        "## Stage-3 Schedule",
        "",
        f"- Warm start: `{config.stage3.stage2_checkpoint}`",
        (
            "- Crop curriculum: "
            f"{config.stage3.crop_curriculum.start_crop_seconds:.2f}s -> "
            f"{config.stage3.crop_curriculum.end_crop_seconds:.2f}s"
        ),
        (
            "- Margin schedule: "
            f"{config.stage3.margin_schedule.start_margin:.3f} -> "
            f"{config.stage3.margin_schedule.end_margin:.3f}"
        ),
        "",
        "| Epoch | Phase | Crop (s) | Margin |",
        "|-------|-------|----------|--------|",
    ]
    for row in schedule_rows:
        lines.append(
            f"| {row['epoch']} | {row['phase_index']} | {row['crop_seconds']} | {row['margin']} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CAMPPlusStage3RunArtifacts",
    "run_campp_stage3",
]
