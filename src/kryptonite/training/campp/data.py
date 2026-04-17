"""Compatibility re-exports for manifest-backed speaker-baseline data helpers."""

from __future__ import annotations

from kryptonite.training.manifest_speaker_data import (
    ManifestSpeakerDataset,
    TrainingBatch,
    TrainingExample,
    build_speaker_index,
    collate_training_examples,
    load_manifest_rows,
)

__all__ = [
    "ManifestSpeakerDataset",
    "TrainingBatch",
    "TrainingExample",
    "build_speaker_index",
    "collate_training_examples",
    "load_manifest_rows",
]
