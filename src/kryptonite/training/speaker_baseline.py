"""Shared runtime helpers for manifest-backed speaker baseline recipes."""

from __future__ import annotations

from .baseline_reporting import relative_to_project, render_markdown_report
from .speaker_baseline_embeddings import (
    _load_manifest_metadata_lookup,
    _lookup_manifest_metadata_row,
    export_dev_embeddings,
)
from .speaker_baseline_training import (
    build_fixed_train_chunking_request,
    prepare_demo_artifacts_if_needed,
    resolve_device,
    train_epochs,
    write_checkpoint,
)
from .speaker_baseline_trials import (
    build_default_cohort_bank,
    load_or_generate_trials,
    mean_or_none,
    score_trials,
    score_trials_detailed,
)
from .speaker_baseline_types import (
    EMBEDDING_METADATA_JSONL_NAME,
    EMBEDDING_METADATA_PARQUET_NAME,
    EMBEDDINGS_FILE_NAME,
    REPRODUCIBILITY_FILE_NAME,
    SCORE_SUMMARY_FILE_NAME,
    SCORES_FILE_NAME,
    TRAINING_SUMMARY_FILE_NAME,
    TRIALS_FILE_NAME,
    EarlyStoppingSummary,
    EmbeddingExportSummary,
    EpochAwareBatchSampler,
    EpochAwareDataset,
    EpochSummary,
    ScoredTrialsArtifacts,
    ScoreSummary,
    SpeakerBaselineRunArtifacts,
    TrainingLoopResult,
    TrainingSummary,
    TrialManifestArtifacts,
)

__all__ = [
    "_load_manifest_metadata_lookup",
    "_lookup_manifest_metadata_row",
    "EMBEDDINGS_FILE_NAME",
    "EMBEDDING_METADATA_JSONL_NAME",
    "EMBEDDING_METADATA_PARQUET_NAME",
    "REPRODUCIBILITY_FILE_NAME",
    "SCORE_SUMMARY_FILE_NAME",
    "SCORES_FILE_NAME",
    "TRAINING_SUMMARY_FILE_NAME",
    "TRIALS_FILE_NAME",
    "EmbeddingExportSummary",
    "EarlyStoppingSummary",
    "EpochAwareBatchSampler",
    "EpochAwareDataset",
    "EpochSummary",
    "ScoredTrialsArtifacts",
    "ScoreSummary",
    "SpeakerBaselineRunArtifacts",
    "TrialManifestArtifacts",
    "TrainingLoopResult",
    "TrainingSummary",
    "build_default_cohort_bank",
    "build_fixed_train_chunking_request",
    "export_dev_embeddings",
    "load_or_generate_trials",
    "mean_or_none",
    "prepare_demo_artifacts_if_needed",
    "relative_to_project",
    "render_markdown_report",
    "resolve_device",
    "score_trials",
    "score_trials_detailed",
    "train_epochs",
    "write_checkpoint",
]
