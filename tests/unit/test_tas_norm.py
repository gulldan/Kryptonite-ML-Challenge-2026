from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kryptonite.eval import (
    CohortEmbeddingBankSelection,
    build_cohort_embedding_bank,
    fit_tas_norm_model,
    predict_tas_norm_probabilities,
    prepare_tas_norm_feature_batch,
)


def test_prepare_tas_norm_feature_batch_surfaces_as_norm_features(tmp_path: Path) -> None:
    embeddings_path, metadata_path = _write_eval_embeddings(tmp_path)
    cohort_root = _build_cohort_bank(tmp_path)

    batch = prepare_tas_norm_feature_batch(
        [
            {"left_id": "alpha:enroll", "right_id": "alpha:test", "label": 1, "score": 0.95},
            {"left_id": "alpha:enroll", "right_id": "beta:test", "label": 0, "score": 0.60},
            {"left_id": "beta:enroll", "right_id": "beta:test", "label": 1, "score": 0.90},
            {"left_id": "beta:enroll", "right_id": "alpha:test", "label": 0, "score": 0.55},
        ],
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        cohort_bank_root=cohort_root,
        top_k=2,
    )

    assert batch.feature_names[0] == "raw_score"
    assert batch.feature_names[1] == "as_norm_score"
    assert batch.feature_matrix.shape == (4, 6)
    assert batch.effective_top_k == 2
    assert batch.excluded_same_speaker_count >= 2
    assert all("as_norm_score" in row for row in batch.rows)


def test_fit_tas_norm_model_learns_simple_separation() -> None:
    feature_matrix = np.asarray(
        [
            [0.90, 2.00, 0.10, 0.20, 0.01, 0.02],
            [0.85, 1.80, 0.12, 0.18, 0.01, 0.02],
            [0.80, 1.60, 0.15, 0.19, 0.02, 0.02],
            [0.20, -1.80, 0.60, 0.40, 0.20, 0.10],
            [0.25, -1.60, 0.58, 0.42, 0.18, 0.09],
            [0.30, -1.50, 0.62, 0.39, 0.19, 0.11],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float64)

    model = fit_tas_norm_model(feature_matrix, labels)
    probabilities = predict_tas_norm_probabilities(feature_matrix, model)

    assert probabilities[:3].min() > 0.5
    assert probabilities[3:].max() < 0.5
    assert model.training.best_loss < model.training.initial_loss


def _write_eval_embeddings(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "artifacts" / "eval" / "run-001"
    output_root.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_root / "dev_embeddings.npz"
    metadata_path = output_root / "dev_embedding_metadata.jsonl"
    np.savez(
        embeddings_path,
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        point_ids=np.asarray(["eval-1", "eval-2", "eval-3", "eval-4"], dtype=str),
    )
    metadata_rows = [
        {
            "atlas_point_id": "eval-1",
            "trial_item_id": "alpha:enroll",
            "utterance_id": "alpha:enroll",
            "speaker_id": "alpha",
            "role": "enrollment",
            "audio_path": "datasets/fixture/alpha_enroll.wav",
        },
        {
            "atlas_point_id": "eval-2",
            "trial_item_id": "alpha:test",
            "utterance_id": "alpha:test",
            "speaker_id": "alpha",
            "role": "test",
            "audio_path": "datasets/fixture/alpha_test.wav",
        },
        {
            "atlas_point_id": "eval-3",
            "trial_item_id": "beta:enroll",
            "utterance_id": "beta:enroll",
            "speaker_id": "beta",
            "role": "enrollment",
            "audio_path": "datasets/fixture/beta_enroll.wav",
        },
        {
            "atlas_point_id": "eval-4",
            "trial_item_id": "beta:test",
            "utterance_id": "beta:test",
            "speaker_id": "beta",
            "role": "test",
            "audio_path": "datasets/fixture/beta_test.wav",
        },
    ]
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    return embeddings_path, metadata_path


def _build_cohort_bank(tmp_path: Path) -> Path:
    output_root = tmp_path / "artifacts" / "cohort-source"
    output_root.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_root / "source_embeddings.npz"
    metadata_path = output_root / "source_metadata.jsonl"
    np.savez(
        embeddings_path,
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, -0.2],
            ],
            dtype=np.float32,
        ),
        point_ids=np.asarray(["cohort-1", "cohort-2", "cohort-3", "cohort-4"], dtype=str),
    )
    metadata_rows = [
        {
            "atlas_point_id": "cohort-1",
            "trial_item_id": "cohort-1",
            "utterance_id": "cohort-1",
            "speaker_id": "alpha",
            "audio_path": "datasets/fixture/cohort_alpha.wav",
        },
        {
            "atlas_point_id": "cohort-2",
            "trial_item_id": "cohort-2",
            "utterance_id": "cohort-2",
            "speaker_id": "beta",
            "audio_path": "datasets/fixture/cohort_beta.wav",
        },
        {
            "atlas_point_id": "cohort-3",
            "trial_item_id": "cohort-3",
            "utterance_id": "cohort-3",
            "speaker_id": "gamma",
            "audio_path": "datasets/fixture/cohort_gamma.wav",
        },
        {
            "atlas_point_id": "cohort-4",
            "trial_item_id": "cohort-4",
            "utterance_id": "cohort-4",
            "speaker_id": "delta",
            "audio_path": "datasets/fixture/cohort_delta.wav",
        },
    ]
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    built = build_cohort_embedding_bank(
        project_root=tmp_path,
        output_root="artifacts/eval/cohort-bank",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        selection=CohortEmbeddingBankSelection(),
    )
    return Path(built.output_root)
