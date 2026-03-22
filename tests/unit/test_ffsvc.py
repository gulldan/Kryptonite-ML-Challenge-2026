from __future__ import annotations

import json
from pathlib import Path

from kryptonite.data.ffsvc import (
    build_speaker_splits,
    load_ffsvc_trials,
    parse_ffsvc_utterance,
    prepare_ffsvc2022_surrogate,
)


def test_parse_ffsvc_utterance_extracts_core_fields() -> None:
    utterance = parse_ffsvc_utterance(
        original_name="F0336_336I-1.5M_1_0009_fast",
        ffsvc_name="ffsvc22_dev_000001",
    )

    assert utterance.speaker_id == "0336"
    assert utterance.source_prefix == "F"
    assert utterance.capture_condition == "I-1.5M"
    assert utterance.session_index == "1"
    assert utterance.utterance_index == "0009"
    assert utterance.pace == "fast"
    assert utterance.audio_filename == "ffsvc22_dev_000001.wav"


def test_parse_ffsvc_utterance_accepts_three_digit_condition_prefix() -> None:
    utterance = parse_ffsvc_utterance(
        original_name="F0046_046PAD0.25M_1_0198_normal",
        ffsvc_name="ffsvc22_dev_000002",
    )

    assert utterance.speaker_id == "0046"
    assert utterance.capture_condition == "PAD0.25M"
    assert utterance.pace == "normal"


def test_load_ffsvc_trials_parses_labels(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.txt"
    trials_path.write_text(
        "1 ffsvc22_dev_000001.wav ffsvc22_dev_000002.wav\n"
        "0 ffsvc22_dev_000003.wav ffsvc22_dev_000004.wav\n"
    )

    trials = load_ffsvc_trials(trials_path)

    assert [trial.label for trial in trials] == [1, 0]


def test_prepare_ffsvc2022_surrogate_writes_manifests_and_split_trials(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "ffsvc2022-surrogate"
    metadata_root = dataset_root / "metadata"
    audio_root = dataset_root / "raw" / "dev"
    manifests_root = tmp_path / "artifacts" / "manifests"
    metadata_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)
    manifests_root.mkdir(parents=True)

    metadata_root.joinpath("dev_meta_list.txt").write_text(
        "Original_Name FFSVC2022_Name\n"
        "F0101_101I1M_1_0001_normal ffsvc22_dev_000001\n"
        "S0101_101I3M_1_0002_normal ffsvc22_dev_000002\n"
        "T0202_202I1M_1_0003_fast ffsvc22_dev_000003\n"
        "F0202_202I3M_1_0004_slow ffsvc22_dev_000004\n"
    )
    metadata_root.joinpath("trials_dev_keys.txt").write_text(
        "1 ffsvc22_dev_000001.wav ffsvc22_dev_000002.wav\n"
        "0 ffsvc22_dev_000001.wav ffsvc22_dev_000003.wav\n"
    )
    for filename in (
        "ffsvc22_dev_000001.wav",
        "ffsvc22_dev_000002.wav",
        "ffsvc22_dev_000003.wav",
        "ffsvc22_dev_000004.wav",
    ):
        audio_root.joinpath(filename).write_bytes(b"RIFFtest")

    artifacts = prepare_ffsvc2022_surrogate(
        project_root=str(tmp_path),
        dataset_root="datasets",
        manifests_root="artifacts/manifests",
        dev_speaker_count=1,
        seed=7,
    )

    speaker_split = json.loads(Path(artifacts.speaker_split_file).read_text())
    train_manifest_lines = Path(artifacts.train_manifest_file).read_text().splitlines()
    dev_manifest_lines = Path(artifacts.dev_manifest_file).read_text().splitlines()
    split_trial_lines = Path(artifacts.split_trials_file).read_text().splitlines()

    assert artifacts.utterance_count == 4
    assert artifacts.official_trial_count == 2
    assert len(speaker_split["train_speakers"]) == 1
    assert len(speaker_split["dev_speakers"]) == 1
    assert len(train_manifest_lines) == 2
    assert len(dev_manifest_lines) == 2
    assert len(split_trial_lines) == 1


def test_build_speaker_splits_requires_non_trivial_holdout() -> None:
    utterances = [
        parse_ffsvc_utterance(
            original_name="F0101_101I1M_1_0001_normal",
            ffsvc_name="ffsvc22_dev_000001",
        ),
        parse_ffsvc_utterance(
            original_name="F0202_202I1M_1_0001_normal",
            ffsvc_name="ffsvc22_dev_000002",
        ),
    ]

    try:
        build_speaker_splits(utterances=utterances, dev_speaker_count=2, seed=42)
    except ValueError as exc:
        assert "dev_speaker_count" in str(exc)
    else:
        raise AssertionError("Expected build_speaker_splits to reject full holdout.")
