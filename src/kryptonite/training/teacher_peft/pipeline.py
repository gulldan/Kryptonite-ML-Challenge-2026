"""Runnable Hugging Face PEFT speaker pipeline with staged fine-tuning support."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    build_verification_evaluation_report_from_arrays,
    write_verification_evaluation_report,
)
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
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    SpeakerBaselineRunArtifacts,
    TrainingSummary,
    build_default_cohort_bank,
    build_fixed_train_chunking_request,
    load_or_generate_trials,
    prepare_demo_artifacts_if_needed,
    render_markdown_report,
    resolve_device,
    score_trials_detailed,
)

from .config import TeacherPeftConfig
from .data import ManifestWaveformDataset, WaveformTrainingBatch, collate_waveform_examples
from .model import (
    TeacherPeftEncoder,
    build_teacher_peft_backbone,
    count_trainable_parameters,
    load_teacher_feature_extractor,
    prepare_teacher_backbone_for_training,
    resolve_hidden_size,
    write_teacher_checkpoint,
)
from .progress import emit_progress
from .runtime import export_teacher_embeddings, train_teacher_epochs

REPORT_FILE_NAME = "teacher_peft_report.md"
TeacherPeftRunArtifacts = SpeakerBaselineRunArtifacts


def run_teacher_peft(
    config: TeacherPeftConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
    feature_extractor_factory: Callable[..., Any] = load_teacher_feature_extractor,
    backbone_factory: Callable[..., nn.Module] = build_teacher_peft_backbone,
    feature_extractor_override: Any | None = None,
    encoder_override: TeacherPeftEncoder | None = None,
    classifier_state_dict: dict[str, torch.Tensor] | None = None,
    classifier_speaker_to_index: dict[str, int] | None = None,
    run_id_override: str | None = None,
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
    feature_extractor = feature_extractor_override
    encoder = encoder_override
    if feature_extractor is None or encoder is None:
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
            pooling_mode=config.model.pooling_mode,
            mfa_num_layers=config.model.mfa_num_layers,
            layer_adapter_enabled=config.model.layer_adapter_enabled,
            adapter_dim=config.model.adapter_dim,
        ).to(device)
    else:
        prepare_teacher_backbone_for_training(
            backbone=encoder.backbone,
            model_config=config.model,
            peft_only=classifier_state_dict is None and config.model.freeze_feature_encoder,
        )
        encoder = encoder.to(device)

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
            collate_fn=partial(
                collate_waveform_examples,
                feature_extractor=feature_extractor,
                sample_rate_hz=config.project.normalization.target_sample_rate_hz,
            ),
            persistent_workers=config.project.runtime.num_workers > 0,
            prefetch_factor=2 if config.project.runtime.num_workers > 0 else None,
        ),
    )

    trainable_parameters, total_parameters = count_trainable_parameters(encoder)

    tracker_run = None
    if config.project.tracking.enabled:
        tracker = build_tracker(config=config.project)
        tracker_run = tracker.start_run(kind="teacher-peft", config=config.to_dict())
        run_id = tracker_run.run_id
    else:
        run_id = run_id_override or create_run_id()

    output_root = resolve_project_path(str(project_root), config.data.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    train_batches = len(loader)
    emit_progress(
        "[teacher-peft] init "
        f"run_id={run_id} device={device} model={config.model.model_id} "
        f"train_rows={len(train_rows)} dev_rows={len(dev_rows)} speakers={len(speaker_to_index)} "
        f"batch_size={config.project.training.batch_size} "
        f"eval_batch_size={config.project.training.eval_batch_size} "
        f"accumulation={config.optimization.gradient_accumulation_steps} "
        f"epochs={config.project.training.max_epochs} "
        f"train_batches={train_batches} "
        f"trainable_params={trainable_parameters} total_params={total_parameters} "
        f"output_root={output_root}",
    )

    classifier = CosineClassifier(
        config.model.embedding_dim,
        num_classes=len(speaker_to_index),
        num_blocks=config.objective.classifier_blocks,
        hidden_dim=config.objective.classifier_hidden_dim,
        subcenters_per_class=config.objective.subcenters_per_class,
    ).to(device)
    if classifier_state_dict is not None:
        if classifier_speaker_to_index != speaker_to_index:
            raise ValueError(
                "Cannot initialize classifier from checkpoint because the training "
                "speaker index differs from the checkpoint speaker index."
            )
        classifier.load_state_dict(classifier_state_dict)
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
    emit_progress(
        f"[teacher-peft] phase=train complete epochs={len(epoch_summaries)} "
        f"output_root={output_root}"
    )

    checkpoint_dir = output_root / config.data.checkpoint_name
    emit_progress(f"[teacher-peft] phase=checkpoint start dir={checkpoint_dir}")
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
    emit_progress(
        f"[teacher-peft] phase=checkpoint complete checkpoint_dir={checkpoint_dir} "
        f"training_summary={training_summary_path}"
    )

    emit_progress(
        f"[teacher-peft] phase=embedding-export start dev_rows={len(dev_rows)} "
        f"eval_batch_size={config.project.training.eval_batch_size}"
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
    emit_progress(
        f"[teacher-peft] phase=embedding-export complete "
        f"utterances={embedding_summary.utterance_count} "
        f"embeddings_path={embedding_summary.embeddings_path}"
    )
    emit_progress("[teacher-peft] phase=trials start")
    trial_artifacts = load_or_generate_trials(
        output_root=output_root,
        configured_trials_manifest=config.data.trials_manifest,
        metadata_rows=metadata_rows,
        project_root=project_root,
        emit_progress=emit_progress,
    )
    trials_path = Path(trial_artifacts.trials_path)
    emit_progress(
        f"[teacher-peft] phase=trials complete trials_path={trials_path} "
        f"trial_count={trial_artifacts.trial_count}"
    )
    cohort_bank = build_default_cohort_bank(
        output_root=output_root,
        embedding_summary=embedding_summary,
        train_manifest_path=config.data.train_manifest,
        trials_path=trials_path,
        project_root=project_root,
    )
    emit_progress(
        f"[teacher-peft] phase=cohort complete embeddings_path={cohort_bank.embeddings_path}"
    )
    emit_progress("[teacher-peft] phase=scoring start")
    scored_trials = score_trials_detailed(
        output_root=output_root,
        trials_path=trials_path,
        metadata_rows=metadata_rows,
        trial_rows=None,
        trial_count_hint=trial_artifacts.trial_count,
        emit_progress=emit_progress,
    )
    score_summary = scored_trials.summary
    score_summary_path = output_root / SCORE_SUMMARY_FILE_NAME
    score_summary_path.write_text(
        json.dumps(score_summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    emit_progress(
        f"[teacher-peft] phase=scoring complete score_gap={score_summary.score_gap} "
        f"scores_path={score_summary.scores_path}"
    )
    emit_progress("[teacher-peft] phase=verification start")
    verification_report = write_verification_evaluation_report(
        build_verification_evaluation_report_from_arrays(
            labels=scored_trials.labels,
            scores=scored_trials.scores,
            scores_path=score_summary.scores_path,
            trials_path=trials_path,
            metadata_path=embedding_summary.metadata_parquet_path,
            metadata_rows=metadata_rows,
            emit_progress=emit_progress,
        ),
        output_root=output_root,
    )
    emit_progress(
        "[teacher-peft] phase=verification complete "
        f"eer={verification_report.summary.metrics.eer:.6f} "
        f"min_dcf={verification_report.summary.metrics.min_dcf:.6f}"
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
    emit_progress(f"[teacher-peft] phase=repro complete path={reproducibility_path}")

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
    emit_progress(f"[teacher-peft] phase=report complete path={report_path}")

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
            reproducibility_path,
            report_path,
            *[path for path in checkpoint_files if path.is_file()],
        ]
        for optional_path in (
            getattr(verification_report, "slice_dashboard_path", None),
            getattr(verification_report, "slice_breakdown_path", None),
            getattr(verification_report, "error_analysis_json_path", None),
            getattr(verification_report, "error_analysis_markdown_path", None),
        ):
            if optional_path is not None:
                artifact_paths.append(Path(str(optional_path)))
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
