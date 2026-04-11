from __future__ import annotations

import json

import numpy as np
import polars as pl
import soundfile as sf

from kryptonite.data.participant_manifests import build_participant_training_manifests
from kryptonite.eda import (
    assign_domain_buckets,
    build_dataset_summary,
    build_speaker_stats,
    compute_audio_stats_table,
    evaluate_retrieval_embeddings,
    load_train_manifest,
    simulate_speaker_disjoint_split,
    validate_submission,
)
from kryptonite.eda.community import LabelPropagationConfig, label_propagation_rerank


def test_audio_profile_summary_and_split_artifacts(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    audio_dir = dataset_root / "train" / "speaker_a"
    audio_dir.mkdir(parents=True)
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = 0.2 * np.sin(2 * np.pi * 440.0 * time)
    sf.write(audio_dir / "00000.flac", waveform, sample_rate)
    (dataset_root / "train.csv").write_text(
        "speaker_id,filepath\nspeaker_a,train/speaker_a/00000.flac\n",
        encoding="utf-8",
    )

    manifest = load_train_manifest(dataset_root)
    file_stats = compute_audio_stats_table(manifest, analysis_seconds=1.0, workers=1)
    speaker_stats = build_speaker_stats(file_stats)
    summary = build_dataset_summary(file_stats, speaker_stats)

    assert file_stats.height == 1
    assert file_stats.row(0, named=True)["sample_rate_hz"] == sample_rate
    assert file_stats.row(0, named=True)["duration_s"] == 1.0
    assert file_stats.row(0, named=True)["rms_dbfs"] > -25.0
    assert speaker_stats.row(0, named=True)["n_utts"] == 1
    assert summary["file_count"] == 1
    assert summary["speaker_count"] == 1

    split_stats = simulate_speaker_disjoint_split(
        pl.DataFrame(
            {
                "speaker_id": ["a", "b"],
                "n_utts": [12, 3],
                "total_duration_s": [24.0, 6.0],
                "mean_duration_s": [2.0, 2.0],
            }
        ),
        val_fraction=0.5,
        min_val_utts=11,
        seed=1,
    )
    assert split_stats["eligible_val_speaker_count"] == 1
    assert split_stats["val_speaker_count"] == 1


def test_domain_buckets_mark_quality_conditions() -> None:
    domains = assign_domain_buckets(
        pl.DataFrame(
            {
                "split": ["train", "train", "train"],
                "speaker_id": ["a", "b", "c"],
                "filepath": ["a.flac", "b.flac", "c.flac"],
                "resolved_path": ["a.flac", "b.flac", "c.flac"],
                "error": [None, None, None],
                "duration_s": [1.0, 5.0, 5.0],
                "clipping_frac": [0.0, 0.02, 0.0],
                "peak_dbfs": [-3.0, 0.0, -6.0],
                "silence_ratio_40db": [0.0, 0.0, 0.0],
                "rms_dbfs": [-20.0, -20.0, -20.0],
                "narrowband_proxy": [0.0, 0.0, 0.8],
                "rolloff95_hz": [6000.0, 6000.0, 2500.0],
                "spectral_flatness": [0.1, 0.1, 0.1],
            }
        )
    )

    assert domains.get_column("domain_cluster").to_list() == [
        "short",
        "clipped",
        "narrowband_like",
    ]


def test_retrieval_evaluation_uses_self_match_exclusion() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    result = evaluate_retrieval_embeddings(
        embeddings,
        labels=["a", "a", "b", "b"],
        filepaths=["a0", "a1", "b0", "b1"],
        top_k=1,
    )

    assert result.summary["precision_at_1"] == 1.0
    assert result.summary["top1_accuracy"] == 1.0
    assert result.confused_speaker_pairs.is_empty()


def test_submission_validator_checks_paths_and_neighbours(tmp_path) -> None:
    template = tmp_path / "test_public.csv"
    template.write_text("filepath\na.flac\nb.flac\nc.flac\n", encoding="utf-8")
    valid = tmp_path / "submission.csv"
    valid.write_text(
        'filepath,neighbours\na.flac,"1,2"\nb.flac,"0,2"\nc.flac,"0,1"\n',
        encoding="utf-8",
    )

    report = validate_submission(template_csv=template, submission_csv=valid, k=2)

    assert report["passed"] is True
    assert report["invalid_row_count"] == 0

    invalid = tmp_path / "bad_submission.csv"
    invalid.write_text(
        'filepath,neighbours\na.flac,"0,0"\nb.flac,"0,2"\nc.flac,"0,1"\n',
        encoding="utf-8",
    )

    bad_report = validate_submission(template_csv=template, submission_csv=invalid, k=2)

    assert bad_report["passed"] is False
    assert bad_report["invalid_row_count"] == 1


def test_participant_split_csvs_convert_to_training_manifests(tmp_path) -> None:
    train_csv = tmp_path / "train_split.csv"
    dev_csv = tmp_path / "val_split.csv"
    train_csv.write_text(
        "speaker_id,filepath\nspeaker_a,train/speaker_a/00000.flac\n",
        encoding="utf-8",
    )
    dev_csv.write_text(
        "speaker_id,filepath\nspeaker_b,train/speaker_b/00000.flac\n",
        encoding="utf-8",
    )

    result = build_participant_training_manifests(
        train_split_csv=train_csv,
        dev_split_csv=dev_csv,
        output_dir=tmp_path / "manifests",
        project_root=tmp_path,
        dataset_name="participants_test",
    )

    train_manifest = tmp_path / result["train_manifest"]
    dev_manifest = tmp_path / result["dev_manifest"]
    assert train_manifest.is_file()
    assert dev_manifest.is_file()
    train_row = json.loads(train_manifest.read_text(encoding="utf-8"))
    dev_row = json.loads(dev_manifest.read_text(encoding="utf-8"))
    assert train_row["audio_path"] == "datasets/Для участников/train/speaker_a/00000.flac"
    assert dev_row["split"] == "dev"


def test_label_propagation_rerank_prefers_mutual_label_candidates() -> None:
    indices = np.array(
        [
            [1, 2, 3, 4],
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [4, 0, 1, 2],
            [3, 0, 1, 2],
        ],
        dtype=np.int64,
    )
    scores = np.array(
        [
            [0.9, 0.8, 0.2, 0.1],
            [0.9, 0.8, 0.2, 0.1],
            [0.8, 0.8, 0.2, 0.1],
            [0.9, 0.2, 0.2, 0.1],
            [0.9, 0.2, 0.2, 0.1],
        ],
        dtype=np.float32,
    )

    top_indices, _, meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=LabelPropagationConfig(
            "test_labelprop",
            edge_top=2,
            reciprocal_top=2,
            rank_top=4,
            label_min_size=2,
            label_max_size=4,
            label_min_candidates=1,
        ),
        top_k=2,
    )

    assert set(top_indices[0]) == {1, 2}
    assert meta["label_used_share"] > 0.0
