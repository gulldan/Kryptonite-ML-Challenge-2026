from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kryptonite.eval import (
    CohortEmbeddingBankSelection,
    apply_as_norm_to_verification_scores,
    build_cohort_embedding_bank,
)


def test_apply_as_norm_to_verification_scores_rewrites_scores_and_keeps_raw_scores(
    tmp_path: Path,
) -> None:
    eval_embeddings_path, eval_metadata_path = _write_eval_embeddings(tmp_path)
    cohort_root = _build_cohort_bank(tmp_path)

    result = apply_as_norm_to_verification_scores(
        [
            {"left_id": "alpha:enroll", "right_id": "alpha:test", "label": 1, "score": 1.0},
            {"left_id": "alpha:enroll", "right_id": "beta:test", "label": 0, "score": 0.0},
            {"left_id": "beta:enroll", "right_id": "beta:test", "label": 1, "score": 1.0},
            {"left_id": "beta:enroll", "right_id": "alpha:test", "label": 0, "score": 0.0},
        ],
        embeddings_path=eval_embeddings_path,
        metadata_path=eval_metadata_path,
        cohort_bank_root=cohort_root,
        top_k=2,
    )

    assert result.summary.method == "as-norm"
    assert result.summary.trial_count == 4
    assert result.summary.effective_top_k == 2
    assert result.summary.unique_identifier_count == 4
    assert result.summary.floored_std_count == 0
    assert all(row["score_normalization"] == "as-norm" for row in result.score_rows)
    assert [row["raw_score"] for row in result.score_rows] == [1.0, 0.0, 1.0, 0.0]

    alpha_embedding = np.asarray([1.0, 0.0], dtype=np.float64)
    beta_embedding = np.asarray([0.0, 1.0], dtype=np.float64)
    cohort_embeddings = np.asarray(
        [
            [1.0, 1.0],
            [0.2, 1.0],
            [1.0, -0.4],
        ],
        dtype=np.float64,
    )
    normalized_cohort = cohort_embeddings / np.linalg.norm(
        cohort_embeddings,
        axis=1,
        keepdims=True,
    )

    alpha_scores = normalized_cohort @ alpha_embedding
    beta_scores = normalized_cohort @ beta_embedding
    alpha_top = np.sort(alpha_scores)[-2:]
    beta_top = np.sort(beta_scores)[-2:]
    alpha_mean = float(alpha_top.mean())
    alpha_std = float(alpha_top.std(ddof=0))
    beta_mean = float(beta_top.mean())
    beta_std = float(beta_top.std(ddof=0))

    expected_scores = [
        (1.0 - alpha_mean) / alpha_std,
        0.5 * (((0.0 - alpha_mean) / alpha_std) + ((0.0 - beta_mean) / beta_std)),
        (1.0 - beta_mean) / beta_std,
        0.5 * (((0.0 - beta_mean) / beta_std) + ((0.0 - alpha_mean) / alpha_std)),
    ]
    actual_scores = [row["score"] for row in result.score_rows]
    assert actual_scores == pytest.approx(expected_scores, abs=1e-7)


def test_apply_as_norm_to_verification_scores_requires_resolvable_identifiers(
    tmp_path: Path,
) -> None:
    eval_embeddings_path, eval_metadata_path = _write_eval_embeddings(tmp_path)
    cohort_root = _build_cohort_bank(tmp_path)

    with pytest.raises(ValueError, match="could not resolve an embedding"):
        apply_as_norm_to_verification_scores(
            [
                {
                    "left_id": "alpha:enroll",
                    "right_id": "missing:test",
                    "label": 0,
                    "score": 0.0,
                }
            ],
            embeddings_path=eval_embeddings_path,
            metadata_path=eval_metadata_path,
            cohort_bank_root=cohort_root,
            top_k=2,
        )


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
                [1.0, 1.0],
                [0.2, 1.0],
                [1.0, -0.4],
            ],
            dtype=np.float32,
        ),
        point_ids=np.asarray(["cohort-1", "cohort-2", "cohort-3"], dtype=str),
    )
    metadata_rows = [
        {
            "atlas_point_id": "cohort-1",
            "trial_item_id": "cohort-1",
            "utterance_id": "cohort-1",
            "speaker_id": "speaker-1",
            "audio_path": "datasets/fixture/cohort_1.wav",
        },
        {
            "atlas_point_id": "cohort-2",
            "trial_item_id": "cohort-2",
            "utterance_id": "cohort-2",
            "speaker_id": "speaker-2",
            "audio_path": "datasets/fixture/cohort_2.wav",
        },
        {
            "atlas_point_id": "cohort-3",
            "trial_item_id": "cohort-3",
            "utterance_id": "cohort-3",
            "speaker_id": "speaker-3",
            "audio_path": "datasets/fixture/cohort_3.wav",
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
