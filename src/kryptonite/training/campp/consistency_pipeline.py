"""Runnable CAM++ clean/corrupted consistency fine-tuning built on stage-3."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    build_verification_evaluation_report,
    load_verification_score_rows,
    write_verification_evaluation_report,
)
from kryptonite.features import FbankExtractionRequest
from kryptonite.models import ArcMarginLoss, CAMPPlusEncoder, CosineClassifier
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id
from kryptonite.training.manifest_speaker_data import build_speaker_index, load_manifest_rows
from kryptonite.training.optimization_runtime import (
    build_training_runtime,
    validate_training_precision,
)
from kryptonite.training.speaker_baseline import (
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    EmbeddingExportSummary,
    EpochSummary,
    ScoreSummary,
    TrainingSummary,
    build_default_cohort_bank,
    export_dev_embeddings,
    load_or_generate_trials,
    prepare_demo_artifacts_if_needed,
    relative_to_project,
    render_markdown_report,
    resolve_device,
    score_trials,
)

from ..augmentation_runtime import TrainingAugmentationRuntime
from .consistency_ablation import (
    WrittenConsistencyAblationReport,
    write_consistency_ablation_report,
)
from .consistency_config import CAMPPlusConsistencyConfig
from .consistency_runtime import (
    ConsistencyEpochBreakdown,
    build_consistency_dataloader,
    run_consistency_batches,
)
from .finetune_common import (
    build_fixed_crop_phases,
    load_warm_start_checkpoint,
    margin_for_epoch,
    mine_hard_negatives,
    phase_for_epoch,
)

REPORT_FILE_NAME = "campp_consistency_report.md"
CONSISTENCY_SUMMARY_FILE_NAME = "consistency_summary.json"
CONSISTENCY_SCHEDULE_FILE_NAME = "consistency_schedule.json"
BASELINE_COMPARISON_JSON_NAME = "baseline_comparison.json"
BASELINE_COMPARISON_MARKDOWN_NAME = "baseline_comparison.md"
HARD_NEGATIVE_LOG_FILE_NAME = "hard_negative_mining_log.jsonl"


@dataclass(frozen=True, slots=True)
class ConsistencyBaselineComparisonSummary:
    baseline_checkpoint_path: str
    consistency_checkpoint_path: str
    baseline_report_path: str
    consistency_report_path: str
    baseline_score_summary_path: str
    consistency_score_summary_path: str
    baseline_eer: float
    consistency_eer: float
    eer_delta: float
    baseline_min_dcf: float
    consistency_min_dcf: float
    min_dcf_delta: float
    baseline_score_gap: float | None
    consistency_score_gap: float | None
    score_gap_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CAMPPlusConsistencyRunArtifacts:
    output_root: str
    checkpoint_path: str
    training_summary_path: str
    consistency_summary_path: str
    embeddings_path: str
    embedding_metadata_jsonl_path: str
    embedding_metadata_parquet_path: str
    trials_path: str
    scores_path: str
    score_summary_path: str
    reproducibility_path: str
    report_path: str
    comparison_json_path: str
    comparison_markdown_path: str
    robust_dev_ablation_json_path: str | None
    robust_dev_ablation_markdown_path: str | None
    training_summary: TrainingSummary
    consistency_epochs: tuple[ConsistencyEpochBreakdown, ...]
    embedding_summary: EmbeddingExportSummary
    score_summary: ScoreSummary
    verification_report: Any
    comparison: ConsistencyBaselineComparisonSummary
    robust_dev_ablation: WrittenConsistencyAblationReport | None
    tracking_run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "checkpoint_path": self.checkpoint_path,
            "training_summary_path": self.training_summary_path,
            "consistency_summary_path": self.consistency_summary_path,
            "embeddings_path": self.embeddings_path,
            "embedding_metadata_jsonl_path": self.embedding_metadata_jsonl_path,
            "embedding_metadata_parquet_path": self.embedding_metadata_parquet_path,
            "trials_path": self.trials_path,
            "scores_path": self.scores_path,
            "score_summary_path": self.score_summary_path,
            "reproducibility_path": self.reproducibility_path,
            "report_path": self.report_path,
            "comparison_json_path": self.comparison_json_path,
            "comparison_markdown_path": self.comparison_markdown_path,
            "robust_dev_ablation_json_path": self.robust_dev_ablation_json_path,
            "robust_dev_ablation_markdown_path": self.robust_dev_ablation_markdown_path,
            "training_summary": self.training_summary.to_dict(),
            "consistency_epochs": [epoch.to_dict() for epoch in self.consistency_epochs],
            "embedding_summary": self.embedding_summary.to_dict(),
            "score_summary": self.score_summary.to_dict(),
            "verification_report": (
                None if self.verification_report is None else self.verification_report.to_dict()
            ),
            "comparison": self.comparison.to_dict(),
            "robust_dev_ablation": (
                None if self.robust_dev_ablation is None else self.robust_dev_ablation.to_dict()
            ),
            "tracking_run_dir": self.tracking_run_dir,
        }


def run_campp_consistency(
    config: CAMPPlusConsistencyConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
    reference_loader: Callable[..., tuple[Path, Any, Any]] = load_campp_encoder_from_checkpoint,
) -> CAMPPlusConsistencyRunArtifacts:
    prepare_demo_artifacts_if_needed(
        project=config.project,
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        enabled=config.data.generate_demo_artifacts_if_missing,
    )
    validate_training_precision(
        config.project.training.precision,
        baseline_name="CAM++ consistency",
    )
    if config.project.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive for CAM++ consistency runs.")

    set_global_seed(
        config.project.runtime.seed,
        deterministic=config.project.reproducibility.deterministic,
        pythonhashseed=config.project.reproducibility.pythonhashseed,
    )
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
    audio_request = AudioLoadRequest.from_config(
        config.project.normalization,
        vad=config.project.vad,
    )

    tracker_run = None
    if config.project.tracking.enabled:
        tracker = build_tracker(config=config.project)
        tracker_run = tracker.start_run(kind="campp-consistency", config=config.to_dict())
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
        checkpoint_path=config.student.checkpoint,
        model=model,
        classifier=classifier,
        project_root=project_root,
        candidate_names=(
            "campp_stage3_encoder.pt",
            "campp_stage2_encoder.pt",
            "campp_stage1_encoder.pt",
            "campp_encoder.pt",
        ),
        source_label="Stage-3",
    )

    criterion = ArcMarginLoss(
        scale=config.objective.scale,
        margin=(
            config.stage3.margin_schedule.start_margin
            if config.stage3.margin_schedule.enabled
            else config.objective.margin
        ),
        easy_margin=config.objective.easy_margin,
    )
    training_runtime = build_training_runtime(
        parameters=[*model.parameters(), *classifier.parameters()],
        optimization_config=config.optimization,
        precision=config.project.training.precision,
        device=device,
        max_epochs=config.project.training.max_epochs,
    )
    augmentation_runtime = TrainingAugmentationRuntime.from_project_config(
        project_root=project_root,
        scheduler_config=config.project.augmentation_scheduler,
        silence_config=config.project.silence_augmentation,
        total_epochs=config.project.training.max_epochs,
    )
    if (
        augmentation_runtime.scheduler is None
        or not augmentation_runtime.has_effective_augmentation
    ):
        raise ValueError(
            "CAM++ consistency requires a live augmentation scheduler "
            "and available corruption banks."
        )
    chunking_phases = build_fixed_crop_phases(
        enabled=config.stage3.crop_curriculum.enabled,
        start_crop_seconds=config.stage3.crop_curriculum.start_crop_seconds,
        end_crop_seconds=config.stage3.crop_curriculum.end_crop_seconds,
        curriculum_epochs=config.stage3.crop_curriculum.curriculum_epochs,
        base_chunking=config.project.chunking,
    )

    consistency_epochs: list[ConsistencyEpochBreakdown] = []
    schedule_rows: list[dict[str, float | int]] = []
    hard_negative_log: list[dict[str, object]] = []
    paired_examples_across_run = 0
    current_phase_index = -1
    dataset = None
    sampler = None
    loader = None

    for epoch in range(config.project.training.max_epochs):
        phase_index = phase_for_epoch(
            epoch,
            curriculum_enabled=config.stage3.crop_curriculum.enabled,
            curriculum_epochs=config.stage3.crop_curriculum.curriculum_epochs,
            n_phases=len(chunking_phases),
        )
        if phase_index != current_phase_index:
            current_phase_index = phase_index
            dataset, sampler, loader = build_consistency_dataloader(
                rows=train_rows,
                speaker_to_index=speaker_to_index,
                project=config.project,
                chunking_request=chunking_phases[phase_index],
                active_runtime=augmentation_runtime,
                device=device,
                hard_negative_fraction=config.stage3.hard_negative.hard_negative_fraction,
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
        if (
            config.stage3.hard_negative.enabled
            and epoch % config.stage3.hard_negative.mining_interval_epochs == 0
        ):
            speaker_weights, mining_entry = mine_hard_negatives(
                model=model,
                rows=train_rows,
                project_root=project_root,
                audio_request=audio_request,
                feature_request=feature_request,
                base_chunking=config.project.chunking,
                device=device,
                top_k=config.stage3.hard_negative.top_k_per_speaker,
                max_rows=config.stage3.hard_negative.max_train_rows_for_mining,
                seed=config.project.runtime.seed,
                epoch=epoch,
            )
            sampler.update_speaker_weights(speaker_weights)
            hard_negative_log.append(mining_entry)

        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        current_crop_seconds = chunking_phases[current_phase_index].train_min_crop_seconds
        metrics = run_consistency_batches(
            model=model,
            classifier=classifier,
            criterion=criterion,
            training_runtime=training_runtime,
            loader=loader,
            device=device,
            clean_classification_weight=config.consistency.clean_classification_weight,
            corrupted_classification_weight=config.consistency.corrupted_classification_weight,
            embedding_weight=config.consistency.embedding_weight,
            score_weight=config.consistency.score_weight,
        )
        if metrics.total_examples <= 0:
            raise ValueError("Consistency loader produced zero examples.")
        paired_examples_across_run += metrics.total_paired_examples
        paired_ratio = metrics.total_paired_examples / metrics.total_examples
        epoch_summary = ConsistencyEpochBreakdown(
            epoch=epoch + 1,
            mean_loss=round(metrics.total_loss / metrics.total_examples, 6),
            mean_clean_classification_loss=round(
                metrics.total_clean_classification_loss / metrics.total_examples,
                6,
            ),
            mean_corrupted_classification_loss=round(
                0.0
                if metrics.total_paired_examples == 0
                else metrics.total_corrupted_classification_loss / metrics.total_paired_examples,
                6,
            ),
            mean_embedding_loss=round(
                0.0
                if metrics.total_paired_examples == 0
                else metrics.total_embedding_loss / metrics.total_paired_examples,
                6,
            ),
            mean_score_loss=round(
                0.0
                if metrics.total_paired_examples == 0
                else metrics.total_score_loss / metrics.total_paired_examples,
                6,
            ),
            clean_accuracy=round(metrics.total_clean_correct / metrics.total_examples, 6),
            corrupted_accuracy=round(
                0.0
                if metrics.total_paired_examples == 0
                else metrics.total_corrupted_correct / metrics.total_paired_examples,
                6,
            ),
            paired_examples=metrics.total_paired_examples,
            paired_ratio=round(paired_ratio, 6),
            learning_rate=round(training_runtime.current_learning_rate(), 8),
        )
        consistency_epochs.append(epoch_summary)
        schedule_rows.append(
            {
                "epoch": epoch + 1,
                "phase_index": current_phase_index,
                "crop_seconds": round(float(current_crop_seconds), 6),
                "margin": round(float(current_margin), 6),
                "paired_examples": metrics.total_paired_examples,
                "paired_ratio": round(paired_ratio, 6),
            }
        )
        training_runtime.step_scheduler(mean_loss=epoch_summary.mean_loss)
        if tracker_run is not None:
            tracker_run.log_metrics(
                {
                    "train_loss": epoch_summary.mean_loss,
                    "clean_classification_loss": epoch_summary.mean_clean_classification_loss,
                    "corrupted_classification_loss": (
                        epoch_summary.mean_corrupted_classification_loss
                    ),
                    "embedding_loss": epoch_summary.mean_embedding_loss,
                    "score_loss": epoch_summary.mean_score_loss,
                    "clean_accuracy": epoch_summary.clean_accuracy,
                    "corrupted_accuracy": epoch_summary.corrupted_accuracy,
                    "paired_ratio": epoch_summary.paired_ratio,
                    "margin": round(float(current_margin), 6),
                    "crop_seconds": round(float(current_crop_seconds), 6),
                },
                step=epoch + 1,
            )

    if paired_examples_across_run <= 0:
        raise ValueError(
            "Consistency run never sampled a corrupted pair. Adjust augmentation probabilities "
            "or verify the corruption-bank manifests exist."
        )

    checkpoint_path = output_root / config.data.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "model_config": asdict(config.model),
            "baseline_config": config.to_dict(),
            "speaker_to_index": dict(speaker_to_index),
        },
        checkpoint_path,
    )
    if hard_negative_log:
        (output_root / HARD_NEGATIVE_LOG_FILE_NAME).write_text(
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
        epochs=tuple(
            EpochSummary(
                epoch=epoch.epoch,
                mean_loss=epoch.mean_loss,
                accuracy=epoch.clean_accuracy,
                learning_rate=epoch.learning_rate,
            )
            for epoch in consistency_epochs
        ),
    )
    training_summary_path = output_root / TRAINING_SUMMARY_FILE_NAME
    training_summary_path.write_text(
        json.dumps(training_summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    consistency_summary_path = output_root / CONSISTENCY_SUMMARY_FILE_NAME
    consistency_summary_path.write_text(
        json.dumps(
            {
                "base_stage3_config_path": config.base_stage3_config_path,
                "student_checkpoint": config.student.checkpoint,
                "comparison_checkpoint": config.student.resolved_comparison_checkpoint,
                "consistency": config.consistency.to_dict(),
                "robust_dev": config.robust_dev.to_dict(),
                "epochs": [epoch.to_dict() for epoch in consistency_epochs],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_root / CONSISTENCY_SCHEDULE_FILE_NAME).write_text(
        json.dumps(
            {
                "student_checkpoint": config.student.checkpoint,
                "comparison_checkpoint": config.student.resolved_comparison_checkpoint,
                "crop_curriculum": asdict(config.stage3.crop_curriculum),
                "margin_schedule": asdict(config.stage3.margin_schedule),
                "epochs": schedule_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
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
        embedding_source="campp_consistency",
    )
    trials_path, trial_rows = load_or_generate_trials(
        output_root=output_root,
        configured_trials_manifest=config.data.trials_manifest,
        metadata_rows=metadata_rows,
        project_root=project_root,
    )
    build_default_cohort_bank(
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

    baseline_reference_root = output_root / "baseline_reference"
    baseline_reference_root.mkdir(parents=True, exist_ok=True)
    baseline_score_summary, baseline_verification_report = _evaluate_reference_checkpoint(
        checkpoint_path=config.student.resolved_comparison_checkpoint,
        output_root=baseline_reference_root,
        dev_rows=dev_rows,
        manifest_path=config.data.dev_manifest,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking=config.project.chunking,
        device=device,
        trials_path=trials_path,
        trial_rows=trial_rows,
        loader=reference_loader,
    )
    comparison = _build_comparison_summary(
        baseline_checkpoint_path=config.student.resolved_comparison_checkpoint,
        consistency_checkpoint_path=str(checkpoint_path),
        baseline_score_summary=baseline_score_summary,
        consistency_score_summary=score_summary,
        baseline_verification_report=baseline_verification_report,
        consistency_verification_report=verification_report,
    )
    comparison_json_path = output_root / BASELINE_COMPARISON_JSON_NAME
    comparison_json_path.write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison_markdown_path = output_root / BASELINE_COMPARISON_MARKDOWN_NAME
    comparison_markdown_path.write_text(
        _render_comparison_markdown(
            comparison=comparison,
            project_root=project_root,
        ),
        encoding="utf-8",
    )

    robust_dev_ablation = None
    if config.robust_dev.enabled:
        robust_dev_ablation = write_consistency_ablation_report(
            title="CAM++ consistency robust-dev ablation",
            ticket_id="KVA-534",
            output_root=output_root / "robust_dev_ablation",
            project_root=project_root,
            device=device_override or config.project.runtime.device or "auto",
            catalog_path=config.robust_dev.catalog_path,
            suite_ids=config.robust_dev.suite_ids,
            clean_weight=config.robust_dev.clean_weight,
            corrupted_weight=config.robust_dev.corrupted_weight,
            baseline_checkpoint_path=config.student.resolved_comparison_checkpoint,
            baseline_label="Stage-3 baseline",
            baseline_clean_report_markdown_path=baseline_verification_report.report_markdown_path,
            baseline_clean_eer=baseline_verification_report.summary.metrics.eer,
            baseline_clean_min_dcf=baseline_verification_report.summary.metrics.min_dcf,
            baseline_clean_score_gap=baseline_score_summary.score_gap,
            consistency_checkpoint_path=str(output_root),
            consistency_label="CAM++ consistency",
            consistency_clean_report_markdown_path=verification_report.report_markdown_path,
            consistency_clean_eer=verification_report.summary.metrics.eer,
            consistency_clean_min_dcf=verification_report.summary.metrics.min_dcf,
            consistency_clean_score_gap=score_summary.score_gap,
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
            title="CAM++ Consistency Report",
            provenance=config.provenance,
            training_summary=training_summary,
            embedding_summary=embedding_summary,
            score_summary=score_summary,
            verification_report=verification_report,
            output_root=output_root,
            project_root=project_root,
        )
        + _render_consistency_appendix(
            config=config,
            consistency_epochs=consistency_epochs,
            comparison=comparison,
            comparison_markdown_path=comparison_markdown_path,
            robust_dev_ablation=robust_dev_ablation,
            project_root=project_root,
        ),
        encoding="utf-8",
    )

    if tracker_run is not None:
        final_epoch = consistency_epochs[-1]
        tracker_run.log_metrics(
            {
                "train_loss": final_epoch.mean_loss,
                "clean_classification_loss": final_epoch.mean_clean_classification_loss,
                "corrupted_classification_loss": final_epoch.mean_corrupted_classification_loss,
                "embedding_loss": final_epoch.mean_embedding_loss,
                "score_loss": final_epoch.mean_score_loss,
                "clean_accuracy": final_epoch.clean_accuracy,
                "corrupted_accuracy": final_epoch.corrupted_accuracy,
                "paired_ratio": final_epoch.paired_ratio,
                "score_gap": score_summary.score_gap or 0.0,
                "eer": verification_report.summary.metrics.eer,
                "min_dcf": verification_report.summary.metrics.min_dcf,
                "eer_delta": comparison.eer_delta,
                "min_dcf_delta": comparison.min_dcf_delta,
            },
            step=config.project.training.max_epochs,
        )
        artifact_paths = [
            checkpoint_path,
            training_summary_path,
            consistency_summary_path,
            output_root / CONSISTENCY_SCHEDULE_FILE_NAME,
            Path(embedding_summary.embeddings_path),
            Path(embedding_summary.metadata_jsonl_path),
            Path(embedding_summary.metadata_parquet_path),
            Path(score_summary.scores_path),
            score_summary_path,
            Path(verification_report.report_json_path),
            Path(verification_report.report_markdown_path),
            Path(verification_report.slice_dashboard_path),
            Path(baseline_verification_report.report_json_path),
            Path(baseline_verification_report.report_markdown_path),
            comparison_json_path,
            comparison_markdown_path,
            reproducibility_path,
            report_path,
        ]
        if robust_dev_ablation is not None:
            artifact_paths.extend(
                [
                    Path(robust_dev_ablation.report_json_path),
                    Path(robust_dev_ablation.report_markdown_path),
                ]
            )
        for artifact_path in artifact_paths:
            tracker_run.log_artifact(artifact_path)
        tracker_run.finish(
            summary={
                "checkpoint_path": str(checkpoint_path),
                "score_gap": score_summary.score_gap,
                "trial_count": score_summary.trial_count,
                "eer": verification_report.summary.metrics.eer,
                "min_dcf": verification_report.summary.metrics.min_dcf,
                "eer_delta": comparison.eer_delta,
                "min_dcf_delta": comparison.min_dcf_delta,
            }
        )

    return CAMPPlusConsistencyRunArtifacts(
        output_root=str(output_root),
        checkpoint_path=str(checkpoint_path),
        training_summary_path=str(training_summary_path),
        consistency_summary_path=str(consistency_summary_path),
        embeddings_path=embedding_summary.embeddings_path,
        embedding_metadata_jsonl_path=embedding_summary.metadata_jsonl_path,
        embedding_metadata_parquet_path=embedding_summary.metadata_parquet_path,
        trials_path=str(trials_path),
        scores_path=score_summary.scores_path,
        score_summary_path=str(score_summary_path),
        reproducibility_path=str(reproducibility_path),
        report_path=str(report_path),
        comparison_json_path=str(comparison_json_path),
        comparison_markdown_path=str(comparison_markdown_path),
        robust_dev_ablation_json_path=(
            None if robust_dev_ablation is None else robust_dev_ablation.report_json_path
        ),
        robust_dev_ablation_markdown_path=(
            None if robust_dev_ablation is None else robust_dev_ablation.report_markdown_path
        ),
        training_summary=training_summary,
        consistency_epochs=tuple(consistency_epochs),
        embedding_summary=embedding_summary,
        score_summary=score_summary,
        verification_report=verification_report,
        comparison=comparison,
        robust_dev_ablation=robust_dev_ablation,
        tracking_run_dir=(None if tracker_run is None else str(tracker_run.run_dir)),
    )


def _evaluate_reference_checkpoint(
    *,
    checkpoint_path: str,
    output_root: Path,
    dev_rows: list[Any],
    manifest_path: str,
    project_root: Path,
    audio_request: AudioLoadRequest,
    feature_request: FbankExtractionRequest,
    chunking: Any,
    device: torch.device,
    trials_path: Path,
    trial_rows: list[dict[str, Any]],
    loader: Callable[..., tuple[Path, Any, Any]],
) -> tuple[ScoreSummary, Any]:
    _, _, checkpoint_model = loader(
        torch=torch,
        checkpoint_path=checkpoint_path,
        project_root=project_root,
    )
    checkpoint_model = checkpoint_model.to(device)
    embedding_summary, metadata_rows = export_dev_embeddings(
        output_root=output_root,
        model=checkpoint_model,
        rows=dev_rows,
        manifest_path=manifest_path,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking=chunking,
        device=device,
        embedding_source="campp_consistency_reference",
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
    return score_summary, verification_report


def _build_comparison_summary(
    *,
    baseline_checkpoint_path: str,
    consistency_checkpoint_path: str,
    baseline_score_summary: ScoreSummary,
    consistency_score_summary: ScoreSummary,
    baseline_verification_report: Any,
    consistency_verification_report: Any,
) -> ConsistencyBaselineComparisonSummary:
    baseline_metrics = baseline_verification_report.summary.metrics
    consistency_metrics = consistency_verification_report.summary.metrics
    baseline_score_gap = baseline_score_summary.score_gap
    consistency_score_gap = consistency_score_summary.score_gap
    score_gap_delta = None
    if baseline_score_gap is not None and consistency_score_gap is not None:
        score_gap_delta = round(consistency_score_gap - baseline_score_gap, 6)
    return ConsistencyBaselineComparisonSummary(
        baseline_checkpoint_path=baseline_checkpoint_path,
        consistency_checkpoint_path=consistency_checkpoint_path,
        baseline_report_path=baseline_verification_report.report_markdown_path,
        consistency_report_path=consistency_verification_report.report_markdown_path,
        baseline_score_summary_path=baseline_score_summary.scores_path,
        consistency_score_summary_path=consistency_score_summary.scores_path,
        baseline_eer=baseline_metrics.eer,
        consistency_eer=consistency_metrics.eer,
        eer_delta=round(consistency_metrics.eer - baseline_metrics.eer, 6),
        baseline_min_dcf=baseline_metrics.min_dcf,
        consistency_min_dcf=consistency_metrics.min_dcf,
        min_dcf_delta=round(consistency_metrics.min_dcf - baseline_metrics.min_dcf, 6),
        baseline_score_gap=baseline_score_gap,
        consistency_score_gap=consistency_score_gap,
        score_gap_delta=score_gap_delta,
    )


def _render_comparison_markdown(
    *,
    comparison: ConsistencyBaselineComparisonSummary,
    project_root: Path,
) -> str:
    baseline_report = relative_to_project(
        Path(comparison.baseline_report_path),
        project_root=project_root,
    )
    consistency_report = relative_to_project(
        Path(comparison.consistency_report_path),
        project_root=project_root,
    )
    lines = [
        "# CAM++ Consistency Baseline Comparison",
        "",
        f"- Baseline checkpoint: `{comparison.baseline_checkpoint_path}`",
        f"- Consistency checkpoint: `{comparison.consistency_checkpoint_path}`",
        f"- Baseline report: `{baseline_report}`",
        f"- Consistency report: `{consistency_report}`",
        "",
        "| Metric | Baseline | Consistency | Delta (consistency - baseline) |",
        "| --- | --- | --- | --- |",
        (
            f"| EER | `{comparison.baseline_eer}` | `{comparison.consistency_eer}` | "
            f"`{comparison.eer_delta}` |"
        ),
        (
            f"| MinDCF | `{comparison.baseline_min_dcf}` | "
            f"`{comparison.consistency_min_dcf}` | `{comparison.min_dcf_delta}` |"
        ),
        (
            f"| Score gap | `{comparison.baseline_score_gap}` | "
            f"`{comparison.consistency_score_gap}` | `{comparison.score_gap_delta}` |"
        ),
        "",
    ]
    return "\n".join(lines)


def _render_consistency_appendix(
    *,
    config: CAMPPlusConsistencyConfig,
    consistency_epochs: list[ConsistencyEpochBreakdown],
    comparison: ConsistencyBaselineComparisonSummary,
    comparison_markdown_path: Path,
    robust_dev_ablation: WrittenConsistencyAblationReport | None,
    project_root: Path,
) -> str:
    comparison_report = relative_to_project(
        comparison_markdown_path,
        project_root=project_root,
    )
    lines = [
        "",
        "## Consistency Setup",
        "",
        f"- Base stage-3 config: `{config.base_stage3_config_path}`",
        f"- Stage-3 warm start: `{config.student.checkpoint}`",
        f"- Baseline comparison checkpoint: `{config.student.resolved_comparison_checkpoint}`",
        (
            "- Loss weights: "
            f"clean_classification={config.consistency.clean_classification_weight}, "
            f"corrupted_classification={config.consistency.corrupted_classification_weight}, "
            f"embedding={config.consistency.embedding_weight}, "
            f"score={config.consistency.score_weight}"
        ),
        "",
        "## Consistency Epochs",
        "",
        (
            "| Epoch | Total | Clean cls | Corrupted cls | Embedding | Score | "
            "Clean acc | Corrupted acc | Paired ratio | LR |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for epoch in consistency_epochs:
        lines.append(
            "| "
            f"{epoch.epoch} | "
            f"{epoch.mean_loss} | "
            f"{epoch.mean_clean_classification_loss} | "
            f"{epoch.mean_corrupted_classification_loss} | "
            f"{epoch.mean_embedding_loss} | "
            f"{epoch.mean_score_loss} | "
            f"{epoch.clean_accuracy} | "
            f"{epoch.corrupted_accuracy} | "
            f"{epoch.paired_ratio} | "
            f"{epoch.learning_rate} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- Comparison report: `{comparison_report}`",
            f"- EER delta (consistency - baseline): `{comparison.eer_delta}`",
            f"- MinDCF delta (consistency - baseline): `{comparison.min_dcf_delta}`",
            f"- Score-gap delta (consistency - baseline): `{comparison.score_gap_delta}`",
            "",
        ]
    )
    if robust_dev_ablation is not None:
        ablation_report = relative_to_project(
            Path(robust_dev_ablation.report_markdown_path),
            project_root=project_root,
        )
        lines.extend(
            [
                "## Robust-Dev Ablation",
                "",
                f"- Robust-dev report: `{ablation_report}`",
                (f"- Winner: `{robust_dev_ablation.report.summary.winner_candidate_id}`"),
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "BASELINE_COMPARISON_JSON_NAME",
    "BASELINE_COMPARISON_MARKDOWN_NAME",
    "CAMPPlusConsistencyRunArtifacts",
    "ConsistencyBaselineComparisonSummary",
    "run_campp_consistency",
]
