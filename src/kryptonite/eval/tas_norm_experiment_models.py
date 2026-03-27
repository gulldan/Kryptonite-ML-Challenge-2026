"""Datamodels for reproducible TAS-norm experiment reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tas_norm import TasNormModel

TAS_NORM_EXPERIMENT_JSON_NAME = "tas_norm_experiment.json"
TAS_NORM_EXPERIMENT_MARKDOWN_NAME = "tas_norm_experiment.md"
VERIFICATION_AS_NORM_EVAL_SCORES_JSONL_NAME = "verification_scores_as_norm_eval.jsonl"


@dataclass(frozen=True, slots=True)
class TasNormArtifactRef:
    label: str
    configured_path: str
    resolved_path: str
    exists: bool
    kind: str
    sha256: str | None
    file_count: int
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "configured_path": self.configured_path,
            "resolved_path": self.resolved_path,
            "exists": self.exists,
            "kind": self.kind,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class TasNormMetricSnapshot:
    trial_count: int
    positive_count: int
    negative_count: int
    eer: float
    min_dcf: float
    mean_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_count": self.trial_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "eer": self.eer,
            "min_dcf": self.min_dcf,
            "mean_score": self.mean_score,
        }


@dataclass(frozen=True, slots=True)
class TasNormSplitSummary:
    eval_fraction: float
    split_seed: int
    train_trial_count: int
    train_positive_count: int
    train_negative_count: int
    eval_trial_count: int
    eval_positive_count: int
    eval_negative_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_fraction": self.eval_fraction,
            "split_seed": self.split_seed,
            "train_trial_count": self.train_trial_count,
            "train_positive_count": self.train_positive_count,
            "train_negative_count": self.train_negative_count,
            "eval_trial_count": self.eval_trial_count,
            "eval_positive_count": self.eval_positive_count,
            "eval_negative_count": self.eval_negative_count,
        }


@dataclass(frozen=True, slots=True)
class TasNormExperimentCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class TasNormExperimentSummary:
    decision: str
    passed_check_count: int
    failed_check_count: int
    eval_winner: str
    key_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "passed_check_count": self.passed_check_count,
            "failed_check_count": self.failed_check_count,
            "eval_winner": self.eval_winner,
            "key_blockers": list(self.key_blockers),
        }


@dataclass(frozen=True, slots=True)
class TasNormExperimentReport:
    title: str
    report_id: str
    candidate_label: str
    summary_text: str
    output_root: str
    source_config_path: str | None
    source_config_sha256: str | None
    implementation_scope: str
    cohort_bank_output_root: str
    cohort_built_during_run: bool
    feature_names: tuple[str, ...]
    top_k: int
    effective_top_k: int
    cohort_size: int
    embedding_dim: int
    floored_std_count: int
    excluded_same_speaker_count: int
    split: TasNormSplitSummary
    raw_train: TasNormMetricSnapshot
    raw_eval: TasNormMetricSnapshot
    as_norm_train: TasNormMetricSnapshot
    as_norm_eval: TasNormMetricSnapshot
    tas_norm_train: TasNormMetricSnapshot
    tas_norm_eval: TasNormMetricSnapshot
    model: TasNormModel
    training_config: dict[str, object]
    gates: dict[str, object]
    artifacts: tuple[TasNormArtifactRef, ...]
    checks: tuple[TasNormExperimentCheck, ...]
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]
    summary: TasNormExperimentSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "candidate_label": self.candidate_label,
            "summary_text": self.summary_text,
            "output_root": self.output_root,
            "source_config_path": self.source_config_path,
            "source_config_sha256": self.source_config_sha256,
            "implementation_scope": self.implementation_scope,
            "cohort_bank_output_root": self.cohort_bank_output_root,
            "cohort_built_during_run": self.cohort_built_during_run,
            "feature_names": list(self.feature_names),
            "top_k": self.top_k,
            "effective_top_k": self.effective_top_k,
            "cohort_size": self.cohort_size,
            "embedding_dim": self.embedding_dim,
            "floored_std_count": self.floored_std_count,
            "excluded_same_speaker_count": self.excluded_same_speaker_count,
            "split": self.split.to_dict(),
            "raw_train": self.raw_train.to_dict(),
            "raw_eval": self.raw_eval.to_dict(),
            "as_norm_train": self.as_norm_train.to_dict(),
            "as_norm_eval": self.as_norm_eval.to_dict(),
            "tas_norm_train": self.tas_norm_train.to_dict(),
            "tas_norm_eval": self.tas_norm_eval.to_dict(),
            "model": self.model.to_dict(),
            "training_config": dict(self.training_config),
            "gates": dict(self.gates),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "checks": [check.to_dict() for check in self.checks],
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BuiltTasNormExperiment:
    report: TasNormExperimentReport
    as_norm_eval_score_rows: list[dict[str, Any]]
    tas_norm_eval_score_rows: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class WrittenTasNormExperimentReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    as_norm_eval_scores_path: str
    tas_norm_eval_scores_path: str
    model_json_path: str
    source_config_copy_path: str | None
    summary: TasNormExperimentSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "as_norm_eval_scores_path": self.as_norm_eval_scores_path,
            "tas_norm_eval_scores_path": self.tas_norm_eval_scores_path,
            "model_json_path": self.model_json_path,
            "source_config_copy_path": self.source_config_copy_path,
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "BuiltTasNormExperiment",
    "TAS_NORM_EXPERIMENT_JSON_NAME",
    "TAS_NORM_EXPERIMENT_MARKDOWN_NAME",
    "TasNormArtifactRef",
    "TasNormExperimentCheck",
    "TasNormExperimentReport",
    "TasNormExperimentSummary",
    "TasNormMetricSnapshot",
    "TasNormSplitSummary",
    "VERIFICATION_AS_NORM_EVAL_SCORES_JSONL_NAME",
    "WrittenTasNormExperimentReport",
]
