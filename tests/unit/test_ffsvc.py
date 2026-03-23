from __future__ import annotations

import json
import wave
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
        "S0303_303I1M_1_0005_fast ffsvc22_dev_000005\n"
        "T0303_303I3M_1_0006_slow ffsvc22_dev_000006\n"
    )
    metadata_root.joinpath("trials_dev_keys.txt").write_text(
        "1 ffsvc22_dev_000001.wav ffsvc22_dev_000002.wav\n"
        "1 ffsvc22_dev_000003.wav ffsvc22_dev_000004.wav\n"
        "1 ffsvc22_dev_000005.wav ffsvc22_dev_000006.wav\n"
        "0 ffsvc22_dev_000001.wav ffsvc22_dev_000003.wav\n"
        "0 ffsvc22_dev_000002.wav ffsvc22_dev_000004.wav\n"
        "0 ffsvc22_dev_000004.wav ffsvc22_dev_000005.wav\n"
    )
    for filename in (
        "ffsvc22_dev_000001.wav",
        "ffsvc22_dev_000002.wav",
        "ffsvc22_dev_000003.wav",
        "ffsvc22_dev_000004.wav",
        "ffsvc22_dev_000005.wav",
        "ffsvc22_dev_000006.wav",
    ):
        _write_wav(audio_root / filename)

    artifacts = prepare_ffsvc2022_surrogate(
        project_root=str(tmp_path),
        dataset_root="datasets",
        manifests_root="artifacts/manifests",
        dev_speaker_count=2,
        seed=7,
        trial_target_per_bucket=1,
    )

    speaker_split = json.loads(Path(artifacts.speaker_split_file).read_text())
    speaker_split_summary = json.loads(Path(artifacts.speaker_split_summary_file).read_text())
    split_trial_summary = json.loads(Path(artifacts.split_trial_summary_file).read_text())
    manifest_inventory = json.loads(Path(artifacts.manifest_inventory_file).read_text())
    train_manifest_lines = Path(artifacts.train_manifest_file).read_text().splitlines()
    dev_manifest_lines = Path(artifacts.dev_manifest_file).read_text().splitlines()
    split_trial_lines = Path(artifacts.split_trials_file).read_text().splitlines()

    assert artifacts.utterance_count == 6
    assert artifacts.official_trial_count == 6
    assert len(speaker_split["train_speakers"]) == 1
    assert len(speaker_split["dev_speakers"]) == 2
    assert len(train_manifest_lines) == 2
    assert len(dev_manifest_lines) == 4
    assert len(split_trial_lines) == artifacts.split_trial_count
    assert artifacts.split_trial_count == split_trial_summary["trial_count"]
    train_entry = json.loads(train_manifest_lines[0])
    split_trial_entry = json.loads(split_trial_lines[0])
    assert train_entry["schema_version"] == "kryptonite.manifest.v1"
    assert train_entry["record_type"] == "utterance"
    assert train_entry["source_dataset"] == "ffsvc2022"
    assert train_entry["duration_seconds"] == 0.25
    assert train_entry["sample_rate_hz"] == 8000
    assert train_entry["num_channels"] == 1
    assert ":" in train_entry["session_id"]
    assert split_trial_entry["duration_bucket"] == "short"
    assert "channel_relation" in split_trial_entry
    assert "domain_relation" in split_trial_entry
    assert manifest_inventory["dataset"] == "ffsvc2022-surrogate"
    assert manifest_inventory["manifest_tables"][0]["jsonl_path"].endswith("all_manifest.jsonl")
    assert manifest_inventory["manifest_tables"][0]["csv_path"].endswith("all_manifest.csv")
    assert manifest_inventory["manifest_tables"][0]["speaker_count"] == 3
    assert manifest_inventory["manifest_tables"][0]["row_count"] == 6
    assert {
        table["jsonl_path"].split("/")[-1] for table in manifest_inventory["auxiliary_tables"]
    } == {"official_dev_trials.jsonl", "speaker_disjoint_dev_trials.jsonl"}
    assert {file["path"].split("/")[-1] for file in manifest_inventory["auxiliary_files"]} == {
        "speaker_splits.json",
        "speaker_disjoint_dev_trials_summary.json",
        "speaker_split_summary.json",
    }
    assert speaker_split_summary["requested_dev_speaker_count"] == 2
    assert speaker_split_summary["trial_target_per_bucket"] == 1
    assert speaker_split_summary["manifest_summary"]["is_valid"] is True
    assert speaker_split_summary["trial_coverage"]["is_valid"] is True
    assert speaker_split_summary["trial_coverage"]["positive_trial_count"] > 0
    assert speaker_split_summary["trial_coverage"]["negative_trial_count"] > 0
    assert speaker_split_summary["trial_coverage"]["negative_trial_requirement_enabled"] is True
    assert speaker_split_summary["trial_balance"]["is_valid"] is True
    assert split_trial_summary["label_counts"]["positive"] > 0
    assert split_trial_summary["label_counts"]["negative"] > 0
    assert split_trial_summary["is_valid"] is True
    assert speaker_split_summary["trial_coverage"]["missing_dev_speakers"] == []


def test_prepare_ffsvc2022_surrogate_quarantines_known_duplicate_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "ffsvc2022-surrogate"
    metadata_root = dataset_root / "metadata"
    audio_root = dataset_root / "raw" / "dev"
    manifests_root = tmp_path / "artifacts" / "manifests"
    metadata_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)
    manifests_root.mkdir(parents=True)

    metadata_root.joinpath("dev_meta_list.txt").write_text(
        "Original_Name FFSVC2022_Name\n"
        "S0449_449I1M_1_0212_normal ffsvc22_dev_002177\n"
        "S0449_449I1M_1_0211_normal ffsvc22_dev_043388\n"
        "S0449_449PAD5M_1_0234_normal ffsvc22_dev_063743\n"
        "S0449_449PAD5M_1_0233_normal ffsvc22_dev_063782\n"
        "F0202_202I1M_1_0001_normal ffsvc22_dev_000001\n"
        "F0202_202I3M_1_0002_normal ffsvc22_dev_000002\n"
        "F0303_303I1M_1_0003_normal ffsvc22_dev_000003\n"
        "F0303_303I3M_1_0004_normal ffsvc22_dev_000004\n"
    )
    metadata_root.joinpath("trials_dev_keys.txt").write_text(
        "1 ffsvc22_dev_000001.wav ffsvc22_dev_000002.wav\n"
        "1 ffsvc22_dev_000003.wav ffsvc22_dev_000004.wav\n"
    )
    for filename in (
        "ffsvc22_dev_000001.wav",
        "ffsvc22_dev_000002.wav",
        "ffsvc22_dev_000003.wav",
        "ffsvc22_dev_000004.wav",
        "ffsvc22_dev_002177.wav",
        "ffsvc22_dev_043388.wav",
        "ffsvc22_dev_063743.wav",
        "ffsvc22_dev_063782.wav",
    ):
        _write_wav(audio_root / filename)

    artifacts = prepare_ffsvc2022_surrogate(
        project_root=str(tmp_path),
        dataset_root="datasets",
        manifests_root="artifacts/manifests",
        dev_speaker_count=2,
        seed=0,
        trial_target_per_bucket=1,
    )

    all_entries = _read_jsonl(Path(artifacts.all_manifest_file))
    train_entries = _read_jsonl(Path(artifacts.train_manifest_file))
    dev_entries = _read_jsonl(Path(artifacts.dev_manifest_file))
    quarantine_entries = _read_jsonl(Path(artifacts.quarantine_manifest_file))

    assert artifacts.source_utterance_count == 8
    assert artifacts.utterance_count == 6
    assert artifacts.quarantined_utterance_count == 2
    assert {entry["utterance_id"] for entry in all_entries} == {
        "ffsvc22_dev_000001",
        "ffsvc22_dev_000002",
        "ffsvc22_dev_000003",
        "ffsvc22_dev_000004",
        "ffsvc22_dev_043388",
        "ffsvc22_dev_063782",
    }
    assert {entry["utterance_id"] for entry in train_entries} == {
        "ffsvc22_dev_000003",
        "ffsvc22_dev_000004",
    }
    assert {entry["utterance_id"] for entry in dev_entries} == {
        "ffsvc22_dev_000001",
        "ffsvc22_dev_000002",
        "ffsvc22_dev_043388",
        "ffsvc22_dev_063782",
    }
    assert {entry["utterance_id"] for entry in quarantine_entries} == {
        "ffsvc22_dev_002177",
        "ffsvc22_dev_063743",
    }
    assert {entry["quality_issue_code"] for entry in quarantine_entries} == {
        "duplicate_audio_content"
    }
    assert {entry["duplicate_policy"] for entry in quarantine_entries} == {"quarantine"}
    assert {entry["schema_version"] for entry in quarantine_entries} == {"kryptonite.manifest.v1"}
    assert {
        (entry["utterance_id"], entry["duplicate_canonical_utterance_id"])
        for entry in quarantine_entries
    } == {
        ("ffsvc22_dev_002177", "ffsvc22_dev_043388"),
        ("ffsvc22_dev_063743", "ffsvc22_dev_063782"),
    }


def test_prepare_ffsvc2022_surrogate_omits_quarantined_duplicates_from_split_trials(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets" / "ffsvc2022-surrogate"
    metadata_root = dataset_root / "metadata"
    audio_root = dataset_root / "raw" / "dev"
    manifests_root = tmp_path / "artifacts" / "manifests"
    metadata_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)
    manifests_root.mkdir(parents=True)

    metadata_root.joinpath("dev_meta_list.txt").write_text(
        "Original_Name FFSVC2022_Name\n"
        "S0449_449I1M_1_0212_normal ffsvc22_dev_002177\n"
        "S0449_449I1M_1_0211_normal ffsvc22_dev_043388\n"
        "S0449_449PAD5M_1_0234_normal ffsvc22_dev_063743\n"
        "S0449_449PAD5M_1_0233_normal ffsvc22_dev_063782\n"
        "F0202_202I1M_1_0001_normal ffsvc22_dev_000001\n"
        "F0202_202I3M_1_0002_normal ffsvc22_dev_000002\n"
        "F0303_303I1M_1_0003_normal ffsvc22_dev_000003\n"
        "F0303_303I3M_1_0004_normal ffsvc22_dev_000004\n"
    )
    metadata_root.joinpath("trials_dev_keys.txt").write_text(
        "1 ffsvc22_dev_043388.wav ffsvc22_dev_063782.wav\n"
        "1 ffsvc22_dev_002177.wav ffsvc22_dev_063782.wav\n"
        "1 ffsvc22_dev_000003.wav ffsvc22_dev_000004.wav\n"
    )
    for filename in (
        "ffsvc22_dev_000001.wav",
        "ffsvc22_dev_000002.wav",
        "ffsvc22_dev_000003.wav",
        "ffsvc22_dev_000004.wav",
        "ffsvc22_dev_002177.wav",
        "ffsvc22_dev_043388.wav",
        "ffsvc22_dev_063743.wav",
        "ffsvc22_dev_063782.wav",
    ):
        _write_wav(audio_root / filename)

    artifacts = prepare_ffsvc2022_surrogate(
        project_root=str(tmp_path),
        dataset_root="datasets",
        manifests_root="artifacts/manifests",
        dev_speaker_count=2,
        seed=0,
        trial_target_per_bucket=1,
    )

    split_trials = _read_jsonl(Path(artifacts.split_trials_file))

    assert artifacts.official_trial_count == 3
    assert artifacts.split_trial_count > 0
    assert not {
        audio_name
        for trial in split_trials
        for audio_name in (trial["left_audio"], trial["right_audio"])
    } & {"ffsvc22_dev_002177.wav", "ffsvc22_dev_063743.wav"}


def test_prepare_ffsvc2022_surrogate_requires_multiple_dev_speakers_for_balanced_trials(
    tmp_path: Path,
) -> None:
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
        "S0303_303I1M_1_0005_fast ffsvc22_dev_000005\n"
        "T0303_303I3M_1_0006_slow ffsvc22_dev_000006\n"
    )
    metadata_root.joinpath("trials_dev_keys.txt").write_text(
        "1 ffsvc22_dev_000001.wav ffsvc22_dev_000002.wav\n"
        "1 ffsvc22_dev_000003.wav ffsvc22_dev_000004.wav\n"
        "1 ffsvc22_dev_000005.wav ffsvc22_dev_000006.wav\n"
    )
    for filename in (
        "ffsvc22_dev_000001.wav",
        "ffsvc22_dev_000002.wav",
        "ffsvc22_dev_000003.wav",
        "ffsvc22_dev_000004.wav",
        "ffsvc22_dev_000005.wav",
        "ffsvc22_dev_000006.wav",
    ):
        _write_wav(audio_root / filename)

    try:
        prepare_ffsvc2022_surrogate(
            project_root=str(tmp_path),
            dataset_root="datasets",
            manifests_root="artifacts/manifests",
            dev_speaker_count=1,
            seed=7,
            trial_target_per_bucket=1,
        )
    except ValueError as exc:
        assert "at least two held-out speakers" in str(exc)
    else:
        raise AssertionError(
            "Expected balanced verification-trial generation to reject one-speaker dev split."
        )


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_wav(path: Path, *, sample_rate: int = 8000, duration_seconds: float = 0.25) -> None:
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)
