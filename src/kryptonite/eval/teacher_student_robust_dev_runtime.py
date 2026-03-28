"""Runtime helpers for teacher-vs-student robust-dev evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch

from kryptonite.config import ChunkingConfig, FeaturesConfig, NormalizationConfig, VADConfig
from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.features import FbankExtractionRequest
from kryptonite.models.campp.checkpoint import (
    load_campp_checkpoint_payload,
    load_campp_encoder_from_checkpoint,
    resolve_campp_checkpoint_path,
)
from kryptonite.models.eres2netv2.checkpoint import (
    load_eres2netv2_checkpoint_payload,
    load_eres2netv2_encoder_from_checkpoint,
    resolve_eres2netv2_checkpoint_path,
)

from .teacher_student_robust_dev_config import (
    TeacherStudentRobustDevCandidateConfig,
    TeacherStudentRobustDevConfig,
)
from .teacher_student_robust_dev_models import (
    SUITE_TRIALS_FILE_NAME,
    CandidateEvidence,
    CorruptedSuiteEntry,
    TeacherStudentRobustDevCostSummary,
    TeacherStudentRobustDevSuiteEvaluation,
)
from .teacher_student_robust_dev_utils import (
    load_json_object,
    optional_float,
    optional_int,
    parameter_count,
    require_mapping,
    state_dict_parameter_count,
    trainable_parameter_count,
)
from .verification_data import load_verification_score_rows, load_verification_trial_rows
from .verification_report import (
    build_verification_evaluation_report,
    write_verification_evaluation_report,
)

CandidateRuntimeResult = tuple[
    TeacherStudentRobustDevCandidateConfig,
    CandidateEvidence,
    tuple[TeacherStudentRobustDevSuiteEvaluation, ...],
]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    normalization: NormalizationConfig
    vad: VADConfig
    features: FeaturesConfig
    chunking: ChunkingConfig
    train_manifest_path: str
    max_dev_rows: int | None
    eval_batch_size: int
    device: str


@dataclass(frozen=True, slots=True)
class CostFields:
    total_parameters: int | None
    trainable_parameters: int | None
    embedding_dim: int | None
    checkpoint_size_bytes: int


def evaluate_candidate(
    *,
    candidate: TeacherStudentRobustDevCandidateConfig,
    config: TeacherStudentRobustDevConfig,
    suites: tuple[CorruptedSuiteEntry, ...],
    project_root: Path,
) -> CandidateRuntimeResult:
    from kryptonite.training.speaker_baseline import (
        TRAINING_SUMMARY_FILE_NAME,
        relative_to_project,
    )

    run_root = resolve_project_path(str(project_root), candidate.run_root)
    baseline_config, runtime_config, seed_cost = load_candidate_artifacts(
        candidate=candidate,
        run_root=run_root,
        project_root=project_root,
    )
    training_summary = load_json_object(run_root / TRAINING_SUMMARY_FILE_NAME)
    clean_report = load_json_object(run_root / "verification_eval_report.json")
    clean_summary = require_mapping(clean_report.get("summary"), "summary")
    clean_metrics = require_mapping(clean_summary.get("metrics"), "summary.metrics")
    clean_stats = require_mapping(clean_summary.get("score_statistics"), "summary.score_statistics")

    suite_results, measured_cost = evaluate_candidate_suites(
        candidate=candidate,
        runtime=runtime_config,
        suites=suites,
        device=config.device,
        project_root=project_root,
        report_output_root=(
            resolve_project_path(str(project_root), config.output_root)
            / "candidates"
            / candidate.candidate_id
        ),
    )
    cost = replace(
        seed_cost,
        total_parameters=measured_cost.total_parameters
        if measured_cost.total_parameters is not None
        else seed_cost.total_parameters,
        trainable_parameters=measured_cost.trainable_parameters
        if measured_cost.trainable_parameters is not None
        else seed_cost.trainable_parameters,
        embedding_dim=measured_cost.embedding_dim
        if measured_cost.embedding_dim is not None
        else seed_cost.embedding_dim,
    )
    project_config = require_mapping(baseline_config.get("project"), "baseline_config.project")
    optimization_config = require_mapping(
        baseline_config.get("optimization"), "baseline_config.optimization"
    )
    evidence = CandidateEvidence(
        run_root=run_root,
        clean_report_markdown_path=relative_to_project(
            run_root / "verification_eval_report.md",
            project_root=project_root,
        ),
        clean_trial_count=int(clean_metrics.get("trial_count", 0)),
        clean_eer=float(clean_metrics.get("eer", 0.0)),
        clean_min_dcf=float(clean_metrics.get("min_dcf", 0.0)),
        clean_score_gap=optional_float(clean_stats.get("score_gap")),
        cost=TeacherStudentRobustDevCostSummary(
            training_device=str(
                training_summary.get(
                    "device",
                    require_mapping(project_config.get("runtime"), "project.runtime").get(
                        "device",
                        "unknown",
                    ),
                )
            ),
            precision=str(
                require_mapping(project_config.get("training"), "project.training").get(
                    "precision",
                    "unknown",
                )
            ),
            train_batch_size=int(
                require_mapping(project_config.get("training"), "project.training").get(
                    "batch_size",
                    0,
                )
            ),
            eval_batch_size=int(
                require_mapping(project_config.get("training"), "project.training").get(
                    "eval_batch_size",
                    0,
                )
            ),
            gradient_accumulation_steps=int(
                optimization_config.get("gradient_accumulation_steps", 1)
            ),
            effective_batch_size=int(
                require_mapping(project_config.get("training"), "project.training").get(
                    "batch_size",
                    0,
                )
            )
            * int(optimization_config.get("gradient_accumulation_steps", 1)),
            max_epochs=int(
                require_mapping(project_config.get("training"), "project.training").get(
                    "max_epochs",
                    0,
                )
            ),
            train_row_count=int(training_summary.get("train_row_count", 0)),
            dev_row_count=int(training_summary.get("dev_row_count", 0)),
            total_parameters=cost.total_parameters,
            trainable_parameters=cost.trainable_parameters,
            checkpoint_size_bytes=cost.checkpoint_size_bytes,
            embedding_dim=cost.embedding_dim,
        ),
        train_manifest_path=runtime_config.train_manifest_path,
        max_dev_rows=runtime_config.max_dev_rows,
    )
    return candidate, evidence, suite_results


def load_candidate_artifacts(
    *,
    candidate: TeacherStudentRobustDevCandidateConfig,
    run_root: Path,
    project_root: Path,
) -> tuple[dict[str, Any], RuntimeConfig, CostFields]:
    if candidate.family == "campp":
        checkpoint_path = resolve_campp_checkpoint_path(
            checkpoint_path=run_root, project_root=project_root
        )
        payload = load_campp_checkpoint_payload(torch=torch, checkpoint_path=checkpoint_path)
        baseline_config = require_mapping(payload.get("baseline_config"), "baseline_config")
        total_parameters = state_dict_parameter_count(payload.get("model_state_dict"))
        embedding_dim = optional_int(
            require_mapping(payload.get("model_config"), "model_config").get("embedding_dim")
        )
        checkpoint_size_bytes = checkpoint_path.stat().st_size
        cost = CostFields(total_parameters, total_parameters, embedding_dim, checkpoint_size_bytes)
    elif candidate.family == "eres2netv2":
        checkpoint_path = resolve_eres2netv2_checkpoint_path(
            checkpoint_path=run_root, project_root=project_root
        )
        payload = load_eres2netv2_checkpoint_payload(torch=torch, checkpoint_path=checkpoint_path)
        baseline_config = require_mapping(payload.get("baseline_config"), "baseline_config")
        total_parameters = state_dict_parameter_count(payload.get("model_state_dict"))
        embedding_dim = optional_int(
            require_mapping(payload.get("model_config"), "model_config").get("embedding_size")
        )
        checkpoint_size_bytes = checkpoint_path.stat().st_size
        cost = CostFields(total_parameters, total_parameters, embedding_dim, checkpoint_size_bytes)
    elif candidate.family == "teacher_peft":
        from kryptonite.training.teacher_peft import resolve_teacher_peft_checkpoint_path

        checkpoint_dir = resolve_teacher_peft_checkpoint_path(
            checkpoint_path=run_root,
            project_root=project_root,
        )
        metadata = load_json_object(checkpoint_dir / "checkpoint_metadata.json")
        baseline_config = require_mapping(metadata.get("baseline_config"), "baseline_config")
        embedding_dim = optional_int(
            require_mapping(metadata.get("model"), "model").get("embedding_dim")
        )
        checkpoint_size_bytes = sum(
            path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
        )
        cost = CostFields(None, None, embedding_dim, checkpoint_size_bytes)
    else:
        raise ValueError(f"Unsupported family {candidate.family!r}.")
    return baseline_config, build_runtime_config(baseline_config), cost


def build_runtime_config(baseline_config: dict[str, Any]) -> RuntimeConfig:
    project_config = require_mapping(baseline_config.get("project"), "baseline_config.project")
    data_config = require_mapping(baseline_config.get("data"), "baseline_config.data")
    return RuntimeConfig(
        normalization=NormalizationConfig(
            **dict(
                require_mapping(
                    project_config.get("normalization"),
                    "project.normalization",
                )
            )
        ),
        vad=VADConfig(**dict(require_mapping(project_config.get("vad"), "project.vad"))),
        features=FeaturesConfig(
            **dict(require_mapping(project_config.get("features"), "project.features"))
        ),
        chunking=ChunkingConfig(
            **dict(require_mapping(project_config.get("chunking"), "project.chunking"))
        ),
        train_manifest_path=str(data_config.get("train_manifest", "")),
        max_dev_rows=optional_int(data_config.get("max_dev_rows")),
        eval_batch_size=int(
            require_mapping(project_config.get("training"), "project.training").get(
                "eval_batch_size",
                1,
            )
        ),
        device=str(
            require_mapping(project_config.get("runtime"), "project.runtime").get(
                "device",
                "auto",
            )
        ),
    )


def evaluate_candidate_suites(
    *,
    candidate: TeacherStudentRobustDevCandidateConfig,
    runtime: RuntimeConfig,
    suites: tuple[CorruptedSuiteEntry, ...],
    device: str,
    project_root: Path,
    report_output_root: Path,
) -> tuple[tuple[TeacherStudentRobustDevSuiteEvaluation, ...], CostFields]:
    from kryptonite.training.speaker_baseline import export_dev_embeddings, resolve_device

    runtime_device = resolve_device(device if device != "auto" else runtime.device)
    audio_request = AudioLoadRequest.from_config(runtime.normalization, vad=runtime.vad)
    feature_request = FbankExtractionRequest.from_config(runtime.features)
    measured_cost: CostFields
    exporter: Callable[[Path, list[Any], str], tuple[Any, list[dict[str, Any]]]]

    if candidate.family == "campp":
        _, model_config, model = load_campp_encoder_from_checkpoint(
            torch=torch,
            checkpoint_path=candidate.run_root,
            project_root=project_root,
        )
        model = model.to(runtime_device)
        measured_cost = CostFields(
            total_parameters=parameter_count(model),
            trainable_parameters=parameter_count(model),
            embedding_dim=getattr(model_config, "embedding_dim", None),
            checkpoint_size_bytes=0,
        )

        def export_suite_embeddings(
            output_root: Path,
            rows: list[Any],
            manifest_path: str,
        ) -> tuple[Any, list[dict[str, Any]]]:
            return export_dev_embeddings(
                output_root=output_root,
                model=model,
                rows=rows,
                manifest_path=manifest_path,
                project_root=project_root,
                audio_request=audio_request,
                feature_request=feature_request,
                chunking=runtime.chunking,
                device=runtime_device,
                embedding_source=f"teacher_student_robust_dev:{candidate.candidate_id}",
            )

        exporter = export_suite_embeddings
    elif candidate.family == "eres2netv2":
        _, model_config, model = load_eres2netv2_encoder_from_checkpoint(
            torch=torch,
            checkpoint_path=candidate.run_root,
            project_root=project_root,
        )
        model = model.to(runtime_device)
        measured_cost = CostFields(
            total_parameters=parameter_count(model),
            trainable_parameters=parameter_count(model),
            embedding_dim=getattr(model_config, "embedding_size", None),
            checkpoint_size_bytes=0,
        )

        def export_suite_embeddings(
            output_root: Path,
            rows: list[Any],
            manifest_path: str,
        ) -> tuple[Any, list[dict[str, Any]]]:
            return export_dev_embeddings(
                output_root=output_root,
                model=model,
                rows=rows,
                manifest_path=manifest_path,
                project_root=project_root,
                audio_request=audio_request,
                feature_request=feature_request,
                chunking=runtime.chunking,
                device=runtime_device,
                embedding_source=f"teacher_student_robust_dev:{candidate.candidate_id}",
            )

        exporter = export_suite_embeddings
    elif candidate.family == "teacher_peft":
        from kryptonite.training.teacher_peft import load_teacher_peft_encoder_from_checkpoint
        from kryptonite.training.teacher_peft.pipeline import export_teacher_embeddings

        _, metadata, feature_extractor, model = load_teacher_peft_encoder_from_checkpoint(
            checkpoint_path=candidate.run_root,
            project_root=project_root,
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            trainable=True,
        )
        model = model.to(runtime_device)
        measured_cost = CostFields(
            total_parameters=parameter_count(model),
            trainable_parameters=trainable_parameter_count(model),
            embedding_dim=optional_int(
                require_mapping(metadata.get("model"), "model").get("embedding_dim")
            ),
            checkpoint_size_bytes=0,
        )

        def export_suite_embeddings(
            output_root: Path,
            rows: list[Any],
            manifest_path: str,
        ) -> tuple[Any, list[dict[str, Any]]]:
            return export_teacher_embeddings(
                output_root=output_root,
                model=model,
                feature_extractor=feature_extractor,
                rows=rows,
                manifest_path=manifest_path,
                project_root=project_root,
                audio_request=audio_request,
                sample_rate_hz=runtime.normalization.target_sample_rate_hz,
                chunking=runtime.chunking,
                eval_batch_size=max(1, runtime.eval_batch_size),
                device=runtime_device,
                embedding_source=f"teacher_student_robust_dev:{candidate.candidate_id}",
            )

        exporter = export_suite_embeddings
    else:
        raise ValueError(f"Unsupported family {candidate.family!r}.")

    suites_output = tuple(
        evaluate_corrupted_suite(
            suite=suite,
            exporter=exporter,
            runtime=runtime,
            project_root=project_root,
            output_root=report_output_root / "robust_dev" / suite.suite_id,
        )
        for suite in suites
    )
    return suites_output, measured_cost


def evaluate_corrupted_suite(
    *,
    suite: CorruptedSuiteEntry,
    exporter: Callable[[Path, list[Any], str], tuple[Any, list[dict[str, Any]]]],
    runtime: RuntimeConfig,
    project_root: Path,
    output_root: Path,
) -> TeacherStudentRobustDevSuiteEvaluation:
    from kryptonite.training.manifest_speaker_data import load_manifest_rows
    from kryptonite.training.speaker_baseline import (
        SCORE_SUMMARY_FILE_NAME,
        build_default_cohort_bank,
        relative_to_project,
        score_trials,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    suite_rows = load_manifest_rows(
        suite.manifest_path,
        project_root=project_root,
        limit=runtime.max_dev_rows,
    )
    embedding_summary, metadata_rows = exporter(output_root, suite_rows, suite.manifest_path)
    trials_path, trial_rows = resolve_suite_trials(
        suite=suite,
        output_root=output_root,
        metadata_rows=metadata_rows,
        project_root=project_root,
    )
    build_default_cohort_bank(
        output_root=output_root,
        embedding_summary=embedding_summary,
        train_manifest_path=runtime.train_manifest_path,
        trials_path=trials_path,
        project_root=project_root,
    )
    score_summary = score_trials(
        output_root=output_root,
        trials_path=trials_path,
        metadata_rows=metadata_rows,
        trial_rows=trial_rows,
    )
    (output_root / SCORE_SUMMARY_FILE_NAME).write_text(
        json.dumps(score_summary.to_dict(), indent=2, sort_keys=True) + "\n",
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
    metrics = verification_report.summary.metrics
    return TeacherStudentRobustDevSuiteEvaluation(
        suite_id=suite.suite_id,
        family=suite.family,
        manifest_path=suite.manifest_path,
        trials_path=relative_to_project(trials_path, project_root=project_root),
        output_root=relative_to_project(output_root, project_root=project_root),
        report_markdown_path=relative_to_project(
            Path(verification_report.report_markdown_path),
            project_root=project_root,
        ),
        trial_count=metrics.trial_count,
        eer=metrics.eer,
        min_dcf=metrics.min_dcf,
        score_gap=score_summary.score_gap,
    )


def load_corrupted_suites(
    *,
    project_root: Path,
    catalog_path: str,
    suite_ids: tuple[str, ...],
) -> tuple[CorruptedSuiteEntry, ...]:
    catalog = load_json_object(resolve_project_path(str(project_root), catalog_path))
    suite_rows = catalog.get("suites")
    if not isinstance(suite_rows, list):
        raise ValueError("Corrupted suite catalog must contain a `suites` list.")
    suites = tuple(
        CorruptedSuiteEntry(
            suite_id=str(item.get("suite_id", "")).strip(),
            family=str(item.get("family", "unknown")).strip(),
            description=str(item.get("description", "")).strip(),
            manifest_path=str(item.get("manifest_path", "")).strip(),
            trial_manifest_paths=tuple(
                str(path).strip()
                for path in cast(list[object], item.get("trial_manifest_paths", []))
            ),
        )
        for item in suite_rows
        if isinstance(item, dict)
    )
    if suite_ids:
        lookup = {suite.suite_id: suite for suite in suites}
        missing = [suite_id for suite_id in suite_ids if suite_id not in lookup]
        if missing:
            raise ValueError(f"Missing corrupted suites in catalog: {missing}")
        suites = tuple(lookup[suite_id] for suite_id in suite_ids)
    if not suites:
        raise ValueError("No corrupted suites selected.")
    return suites


def resolve_suite_trials(
    *,
    suite: CorruptedSuiteEntry,
    output_root: Path,
    metadata_rows: list[dict[str, Any]],
    project_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    from kryptonite.training.speaker_baseline import load_or_generate_trials

    output_root.mkdir(parents=True, exist_ok=True)
    if not suite.trial_manifest_paths:
        return load_or_generate_trials(
            output_root=output_root,
            configured_trials_manifest=None,
            metadata_rows=metadata_rows,
            project_root=project_root,
        )
    merged_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for trial_manifest_path in suite.trial_manifest_paths:
        trial_path = resolve_project_path(str(project_root), trial_manifest_path)
        for row in load_verification_trial_rows(trial_path):
            key = trial_identity(row)
            if key in seen:
                continue
            seen.add(key)
            merged_rows.append(dict(row))
    if not merged_rows:
        raise ValueError(f"Suite {suite.suite_id!r} does not contain any trial rows.")
    trials_path = output_root / SUITE_TRIALS_FILE_NAME
    trials_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged_rows),
        encoding="utf-8",
    )
    return trials_path, merged_rows


def trial_identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("left_id", row.get("left_audio", ""))),
        str(row.get("right_id", row.get("right_audio", ""))),
        int(row.get("label", 0)),
    )


__all__ = [
    "CostFields",
    "RuntimeConfig",
    "evaluate_candidate",
    "load_corrupted_suites",
    "load_json_object",
    "optional_float",
    "optional_int",
    "require_mapping",
]
