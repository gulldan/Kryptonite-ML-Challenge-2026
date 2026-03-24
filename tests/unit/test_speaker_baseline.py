from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kryptonite.training.speaker_baseline import score_trials


def test_score_trials_resolves_bare_audio_names_from_metadata_audio_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts" / "baselines" / "campp" / "run-001"
    output_root.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_root / "dev_embeddings.npz"
    np.savez(
        embeddings_path,
        embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        point_ids=np.asarray(["utt-00000", "utt-00001"], dtype=str),
    )

    metadata_rows = [
        {
            "trial_item_id": "speaker_alpha:utt-1",
            "utterance_id": "speaker_alpha:utt-1",
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000001.wav",
        },
        {
            "trial_item_id": "speaker_alpha:utt-2",
            "utterance_id": "speaker_alpha:utt-2",
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000002.wav",
        },
    ]
    trial_rows = [
        {
            "left_audio": "ffsvc22_dev_000001.wav",
            "right_audio": "ffsvc22_dev_000002.wav",
            "label": 1,
        }
    ]
    trials_path = output_root / "dev_trials.jsonl"
    trials_path.write_text("".join(json.dumps(row) + "\n" for row in trial_rows), encoding="utf-8")

    summary = score_trials(
        output_root=output_root,
        trials_path=trials_path,
        metadata_rows=metadata_rows,
        trial_rows=trial_rows,
        embeddings_path=embeddings_path,
    )

    assert summary.trial_count == 1
    assert summary.positive_count == 1
    assert summary.missing_embedding_count == 0
    assert summary.mean_positive_score == 1.0
