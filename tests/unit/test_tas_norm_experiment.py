from __future__ import annotations

import json
from pathlib import Path

from kryptonite.eval import (
    build_tas_norm_experiment_report,
    load_tas_norm_experiment_config,
    write_tas_norm_experiment_report,
)


def test_tas_norm_experiment_builder_and_writer_emit_report_bundle(tmp_path: Path) -> None:
    config_path = _write_fixture_config(tmp_path)

    config = load_tas_norm_experiment_config(config_path=config_path)
    built = build_tas_norm_experiment_report(config, config_path=config_path, project_root=tmp_path)
    written = write_tas_norm_experiment_report(built)

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")

    assert payload["report_id"] == "fixture-tas-norm"
    assert payload["summary"]["decision"] == "no_go"
    assert "## Quality Snapshot" in markdown
    assert Path(written.as_norm_eval_scores_path).is_file()
    assert Path(written.tas_norm_eval_scores_path).is_file()
    assert Path(written.model_json_path).is_file()


def test_checked_in_tas_norm_config_builds_against_repository() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "release" / "tas-norm-experiment.toml"

    config = load_tas_norm_experiment_config(config_path=config_path)
    built = build_tas_norm_experiment_report(config, config_path=config_path)

    assert built.report.report_id == "kryptonite-2026-tas-norm"
    assert built.report.summary.decision == "no_go"
    assert built.report.summary.eval_winner in {"raw", "as-norm", "tas-norm"}


def _write_fixture_config(tmp_path: Path) -> Path:
    artifacts_root = tmp_path / "artifacts" / "fixture"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_fixture_scores(artifacts_root)
    _write_fixture_embeddings(artifacts_root)

    config_path = tmp_path / "configs" / "release" / "tas-norm-experiment.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'title = "Fixture TAS-norm experiment"',
                'report_id = "fixture-tas-norm"',
                'candidate_label = "fixture"',
                'summary = "Fixture TAS experiment"',
                'output_root = "artifacts/release-decisions/fixture-tas-norm"',
                "",
                "[artifacts]",
                'scores_path = "artifacts/fixture/dev_scores.jsonl"',
                'trials_path = "artifacts/fixture/dev_trials.jsonl"',
                'metadata_path = "artifacts/fixture/dev_embedding_metadata.jsonl"',
                'embeddings_path = "artifacts/fixture/dev_embeddings.npz"',
                (
                    "cohort_bank_output_root = "
                    '"artifacts/release-decisions/fixture-tas-norm/cohort-bank"'
                ),
                "",
                "[cohort_selection]",
                "allow_trial_overlap_fallback = true",
                "strict_speaker_disjointness = false",
                "min_embeddings_per_speaker = 1",
                "",
                "[tas_norm]",
                "top_k = 2",
                "std_epsilon = 1e-6",
                "exclude_matching_speakers = true",
                "eval_fraction = 0.5",
                "split_seed = 99",
                "",
                "[training]",
                "learning_rate = 0.05",
                "max_steps = 200",
                "l2_regularization = 0.01",
                "early_stopping_patience = 20",
                "min_relative_loss_improvement = 1e-4",
                "balance_classes = true",
                "",
                "[gates]",
                "min_train_trials = 4",
                "min_eval_trials = 4",
                "min_train_positives = 2",
                "min_train_negatives = 2",
                "min_eval_positives = 2",
                "min_eval_negatives = 2",
                "min_eer_gain_vs_raw = 0.001",
                "min_min_dcf_gain_vs_raw = 0.001",
                "min_eer_gain_vs_as_norm = 0.001",
                "min_min_dcf_gain_vs_as_norm = 0.001",
                "max_train_eval_eer_gap = 0.25",
                "max_train_eval_min_dcf_gap = 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_fixture_scores(artifacts_root: Path) -> None:
    score_rows = [
        {"left_id": "alpha:enroll", "right_id": "alpha:test", "label": 1, "score": 0.99},
        {"left_id": "alpha:enroll", "right_id": "beta:test", "label": 0, "score": 0.10},
        {"left_id": "beta:enroll", "right_id": "beta:test", "label": 1, "score": 0.98},
        {"left_id": "beta:enroll", "right_id": "alpha:test", "label": 0, "score": 0.12},
        {"left_id": "gamma:enroll", "right_id": "gamma:test", "label": 1, "score": 0.97},
        {"left_id": "gamma:enroll", "right_id": "delta:test", "label": 0, "score": 0.15},
        {"left_id": "delta:enroll", "right_id": "delta:test", "label": 1, "score": 0.96},
        {"left_id": "delta:enroll", "right_id": "gamma:test", "label": 0, "score": 0.16},
    ]
    (artifacts_root / "dev_scores.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in score_rows),
        encoding="utf-8",
    )
    (artifacts_root / "dev_trials.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in score_rows),
        encoding="utf-8",
    )


def _write_fixture_embeddings(artifacts_root: Path) -> None:
    import numpy as np

    embeddings_path = artifacts_root / "dev_embeddings.npz"
    metadata_path = artifacts_root / "dev_embedding_metadata.jsonl"
    np.savez(
        embeddings_path,
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.7, 0.7],
                [0.7, 0.7],
                [0.9, -0.2],
                [0.9, -0.2],
            ],
            dtype=np.float32,
        ),
        point_ids=np.asarray(
            [
                "eval-1",
                "eval-2",
                "eval-3",
                "eval-4",
                "eval-5",
                "eval-6",
                "eval-7",
                "eval-8",
            ],
            dtype=str,
        ),
    )
    metadata_rows = [
        {
            "atlas_point_id": "eval-1",
            "trial_item_id": "alpha:enroll",
            "utterance_id": "alpha:enroll",
            "speaker_id": "alpha",
            "audio_path": "datasets/fixture/alpha_enroll.wav",
        },
        {
            "atlas_point_id": "eval-2",
            "trial_item_id": "alpha:test",
            "utterance_id": "alpha:test",
            "speaker_id": "alpha",
            "audio_path": "datasets/fixture/alpha_test.wav",
        },
        {
            "atlas_point_id": "eval-3",
            "trial_item_id": "beta:enroll",
            "utterance_id": "beta:enroll",
            "speaker_id": "beta",
            "audio_path": "datasets/fixture/beta_enroll.wav",
        },
        {
            "atlas_point_id": "eval-4",
            "trial_item_id": "beta:test",
            "utterance_id": "beta:test",
            "speaker_id": "beta",
            "audio_path": "datasets/fixture/beta_test.wav",
        },
        {
            "atlas_point_id": "eval-5",
            "trial_item_id": "gamma:enroll",
            "utterance_id": "gamma:enroll",
            "speaker_id": "gamma",
            "audio_path": "datasets/fixture/gamma_enroll.wav",
        },
        {
            "atlas_point_id": "eval-6",
            "trial_item_id": "gamma:test",
            "utterance_id": "gamma:test",
            "speaker_id": "gamma",
            "audio_path": "datasets/fixture/gamma_test.wav",
        },
        {
            "atlas_point_id": "eval-7",
            "trial_item_id": "delta:enroll",
            "utterance_id": "delta:enroll",
            "speaker_id": "delta",
            "audio_path": "datasets/fixture/delta_enroll.wav",
        },
        {
            "atlas_point_id": "eval-8",
            "trial_item_id": "delta:test",
            "utterance_id": "delta:test",
            "speaker_id": "delta",
            "audio_path": "datasets/fixture/delta_test.wav",
        },
    ]
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
