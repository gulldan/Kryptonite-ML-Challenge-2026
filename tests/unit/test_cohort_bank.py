from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kryptonite.eval import (
    CohortEmbeddingBankSelection,
    build_cohort_embedding_bank,
    load_cohort_embedding_bank,
)


def test_build_cohort_embedding_bank_writes_normalized_artifacts(tmp_path: Path) -> None:
    embeddings_path, metadata_path = _write_source_embeddings(tmp_path)
    trials_path = _write_jsonl(
        tmp_path / "artifacts" / "manifests" / "dev_trials.jsonl",
        [
            {"left_id": "speaker_charlie:utt-1", "right_id": "speaker_delta:utt-1", "label": 0},
        ],
    )
    train_manifest = _write_jsonl(
        tmp_path / "artifacts" / "manifests" / "train_manifest.jsonl",
        [
            {"speaker_id": "speaker_alpha", "audio_path": "datasets/fixture/train_a.wav"},
            {"speaker_id": "speaker_bravo", "audio_path": "datasets/fixture/train_b.wav"},
        ],
    )

    built = build_cohort_embedding_bank(
        project_root=tmp_path,
        output_root="artifacts/eval/cohort-bank",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        selection=CohortEmbeddingBankSelection(
            trial_paths=(str(trials_path),),
            validation_manifest_paths=(str(train_manifest),),
            max_embeddings_per_speaker=1,
            strict_speaker_disjointness=True,
        ),
    )

    assert Path(built.embeddings_path).is_file()
    assert Path(built.metadata_jsonl_path).is_file()
    assert Path(built.metadata_parquet_path).is_file()
    assert Path(built.summary_path).is_file()
    assert built.summary.selected_row_count == 2
    assert built.summary.selected_speaker_count == 2
    assert built.summary.overlapping_validation_speakers == ()
    assert built.summary.trial_overlap_fallback_used is False

    loaded = load_cohort_embedding_bank(built.output_root)
    norms = np.linalg.norm(loaded.embeddings, axis=1)
    assert norms.tolist() == pytest.approx([1.0, 1.0])
    assert len(loaded.metadata_rows) == 2
    assert loaded.metadata_rows[0]["cohort_row_index"] == 0


def test_build_cohort_embedding_bank_uses_trial_overlap_fallback_when_needed(
    tmp_path: Path,
) -> None:
    embeddings_path = tmp_path / "artifacts" / "baselines" / "run-001" / "dev_embeddings.npz"
    metadata_path = tmp_path / "artifacts" / "baselines" / "run-001" / "dev_metadata.jsonl"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        embeddings_path,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        point_ids=np.asarray(["p-1", "p-2"], dtype=str),
    )
    _write_jsonl(
        metadata_path,
        [
            {
                "atlas_point_id": "p-1",
                "trial_item_id": "speaker_one:utt-1",
                "utterance_id": "speaker_one:utt-1",
                "speaker_id": "speaker_one",
                "audio_path": "datasets/fixture/one.wav",
            },
            {
                "atlas_point_id": "p-2",
                "trial_item_id": "speaker_two:utt-1",
                "utterance_id": "speaker_two:utt-1",
                "speaker_id": "speaker_two",
                "audio_path": "datasets/fixture/two.wav",
            },
        ],
    )
    trials_path = _write_jsonl(
        tmp_path / "artifacts" / "manifests" / "trials.jsonl",
        [
            {"left_id": "speaker_one:utt-1", "right_id": "speaker_two:utt-1", "label": 0},
        ],
    )

    built = build_cohort_embedding_bank(
        project_root=tmp_path,
        output_root="artifacts/eval/cohort-bank",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        selection=CohortEmbeddingBankSelection(
            trial_paths=(str(trials_path),),
            allow_trial_overlap_fallback=True,
        ),
    )

    assert built.summary.selected_row_count == 2
    assert built.summary.trial_overlap_fallback_used is True


def test_build_cohort_embedding_bank_rejects_validation_speaker_overlap(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "artifacts" / "baselines" / "run-001" / "dev_embeddings.npz"
    metadata_path = tmp_path / "artifacts" / "baselines" / "run-001" / "dev_metadata.jsonl"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        embeddings_path,
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        point_ids=np.asarray(["p-1"], dtype=str),
    )
    _write_jsonl(
        metadata_path,
        [
            {
                "atlas_point_id": "p-1",
                "trial_item_id": "speaker_alpha:utt-1",
                "utterance_id": "speaker_alpha:utt-1",
                "speaker_id": "speaker_alpha",
                "audio_path": "datasets/fixture/alpha.wav",
            }
        ],
    )
    train_manifest = _write_jsonl(
        tmp_path / "artifacts" / "manifests" / "train_manifest.jsonl",
        [{"speaker_id": "speaker_alpha", "audio_path": "datasets/fixture/train.wav"}],
    )

    with pytest.raises(ValueError, match="overlaps speakers"):
        build_cohort_embedding_bank(
            project_root=tmp_path,
            output_root="artifacts/eval/cohort-bank",
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            selection=CohortEmbeddingBankSelection(
                validation_manifest_paths=(str(train_manifest),),
                strict_speaker_disjointness=True,
            ),
        )


def _write_source_embeddings(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "artifacts" / "baselines" / "run-001"
    output_root.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_root / "dev_embeddings.npz"
    metadata_path = output_root / "dev_embedding_metadata.jsonl"
    np.savez(
        embeddings_path,
        embeddings=np.asarray(
            [
                [3.0, 0.0],
                [1.0, 1.0],
                [0.0, 4.0],
                [2.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        point_ids=np.asarray(["p-1", "p-2", "p-3", "p-4", "p-5"], dtype=str),
    )
    _write_jsonl(
        metadata_path,
        [
            {
                "atlas_point_id": "p-1",
                "trial_item_id": "speaker_charlie:utt-1",
                "utterance_id": "speaker_charlie:utt-1",
                "speaker_id": "speaker_charlie",
                "dataset": "fixture",
                "split": "dev",
                "role": "enrollment",
                "audio_path": "datasets/fixture/charlie_1.wav",
            },
            {
                "atlas_point_id": "p-2",
                "trial_item_id": "speaker_charlie:utt-2",
                "utterance_id": "speaker_charlie:utt-2",
                "speaker_id": "speaker_charlie",
                "dataset": "fixture",
                "split": "dev",
                "role": "test",
                "audio_path": "datasets/fixture/charlie_2.wav",
            },
            {
                "atlas_point_id": "p-3",
                "trial_item_id": "speaker_delta:utt-1",
                "utterance_id": "speaker_delta:utt-1",
                "speaker_id": "speaker_delta",
                "dataset": "fixture",
                "split": "dev",
                "role": "enrollment",
                "audio_path": "datasets/fixture/delta_1.wav",
            },
            {
                "atlas_point_id": "p-4",
                "trial_item_id": "speaker_echo:utt-1",
                "utterance_id": "speaker_echo:utt-1",
                "speaker_id": "speaker_echo",
                "dataset": "fixture",
                "split": "dev",
                "role": "enrollment",
                "audio_path": "datasets/fixture/echo_1.wav",
            },
            {
                "atlas_point_id": "p-5",
                "trial_item_id": "speaker_echo:utt-2",
                "utterance_id": "speaker_echo:utt-2",
                "speaker_id": "speaker_echo",
                "dataset": "fixture",
                "split": "dev",
                "role": "test",
                "audio_path": "datasets/fixture/echo_2.wav",
            },
        ],
    )
    return embeddings_path, metadata_path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path
