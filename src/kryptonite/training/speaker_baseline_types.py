"""Shared types and constants for manifest-backed speaker baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch

from kryptonite.eval import (
    WrittenIdentificationEvaluationReport,
    WrittenVerificationEvaluationReport,
)

EMBEDDINGS_FILE_NAME = "dev_embeddings.npz"
EMBEDDING_METADATA_JSONL_NAME = "dev_embedding_metadata.jsonl"
EMBEDDING_METADATA_PARQUET_NAME = "dev_embedding_metadata.parquet"
TRAINING_SUMMARY_FILE_NAME = "training_summary.json"
SCORES_FILE_NAME = "dev_scores.jsonl"
TRIALS_FILE_NAME = "dev_trials.jsonl"
SCORE_SUMMARY_FILE_NAME = "score_summary.json"
REPRODUCIBILITY_FILE_NAME = "reproducibility_snapshot.json"


class EpochAwareDataset(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class EpochAwareBatchSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True, slots=True)
class EpochSummary:
    epoch: int
    mean_loss: float
    accuracy: float
    learning_rate: float
    is_best_checkpoint: bool = False


@dataclass(frozen=True, slots=True)
class EarlyStoppingSummary:
    enabled: bool
    monitor: str
    min_delta: float
    patience_epochs: int
    min_epochs: int
    restore_best: bool
    stop_train_accuracy: float | None
    stopped: bool
    reason: str | None
    best_epoch: int | None
    best_value: float | None
    restored_best: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingLoopResult:
    summaries: list[EpochSummary]
    early_stopping: EarlyStoppingSummary | None
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    best_classifier_state_dict: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    device: str
    train_manifest: str
    dev_manifest: str
    provenance_ruleset: str
    provenance_initialization: str
    speaker_count: int
    train_row_count: int
    dev_row_count: int
    checkpoint_path: str
    epochs: tuple[EpochSummary, ...]
    early_stopping: EarlyStoppingSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "train_manifest": self.train_manifest,
            "dev_manifest": self.dev_manifest,
            "provenance_ruleset": self.provenance_ruleset,
            "provenance_initialization": self.provenance_initialization,
            "speaker_count": self.speaker_count,
            "train_row_count": self.train_row_count,
            "dev_row_count": self.dev_row_count,
            "checkpoint_path": self.checkpoint_path,
            "epochs": [asdict(epoch) for epoch in self.epochs],
            "early_stopping": (
                None if self.early_stopping is None else self.early_stopping.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class EmbeddingExportSummary:
    manifest_path: str
    embedding_dim: int
    utterance_count: int
    speaker_count: int
    embeddings_path: str
    metadata_jsonl_path: str
    metadata_parquet_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    trials_path: str
    scores_path: str
    trial_count: int
    positive_count: int
    negative_count: int
    missing_embedding_count: int
    mean_positive_score: float | None
    mean_negative_score: float | None
    score_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialManifestArtifacts:
    trials_path: str
    trial_count: int


@dataclass(frozen=True, slots=True)
class ScoredTrialsArtifacts:
    summary: ScoreSummary
    labels: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True, slots=True)
class SpeakerBaselineRunArtifacts:
    output_root: str
    checkpoint_path: str
    training_summary_path: str
    embeddings_path: str
    embedding_metadata_jsonl_path: str
    embedding_metadata_parquet_path: str
    trials_path: str
    scores_path: str
    score_summary_path: str
    reproducibility_path: str
    report_path: str
    training_summary: TrainingSummary
    embedding_summary: EmbeddingExportSummary
    score_summary: ScoreSummary
    verification_report: WrittenVerificationEvaluationReport | None = None
    identification_report: WrittenIdentificationEvaluationReport | None = None
    tracking_run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "checkpoint_path": self.checkpoint_path,
            "training_summary_path": self.training_summary_path,
            "embeddings_path": self.embeddings_path,
            "embedding_metadata_jsonl_path": self.embedding_metadata_jsonl_path,
            "embedding_metadata_parquet_path": self.embedding_metadata_parquet_path,
            "trials_path": self.trials_path,
            "scores_path": self.scores_path,
            "score_summary_path": self.score_summary_path,
            "reproducibility_path": self.reproducibility_path,
            "report_path": self.report_path,
            "training_summary": self.training_summary.to_dict(),
            "embedding_summary": self.embedding_summary.to_dict(),
            "score_summary": self.score_summary.to_dict(),
            "verification_report": (
                None if self.verification_report is None else self.verification_report.to_dict()
            ),
            "identification_report": (
                None if self.identification_report is None else self.identification_report.to_dict()
            ),
            "tracking_run_dir": self.tracking_run_dir,
        }
