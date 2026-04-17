"""Reusable EDA helpers for the Kryptonite speaker-recognition challenge."""

from kryptonite.eda.audio_stats import compute_audio_stats_table
from kryptonite.eda.domain import assign_domain_buckets
from kryptonite.eda.manifest import load_test_manifest, load_train_manifest
from kryptonite.eda.retrieval import evaluate_retrieval_embeddings
from kryptonite.eda.speaker_stats import (
    build_dataset_summary,
    build_speaker_stats,
    simulate_speaker_disjoint_split,
)
from kryptonite.eda.submission import validate_submission

__all__ = [
    "assign_domain_buckets",
    "build_dataset_summary",
    "build_speaker_stats",
    "compute_audio_stats_table",
    "evaluate_retrieval_embeddings",
    "load_test_manifest",
    "load_train_manifest",
    "simulate_speaker_disjoint_split",
    "validate_submission",
]
