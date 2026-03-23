from __future__ import annotations

import json
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

from kryptonite.eda.dataset_leakage import (
    build_dataset_leakage_report,
    write_dataset_leakage_report,
)


def test_build_dataset_leakage_report_flags_missing_train_and_dev_splits(
    tmp_path: Path,
) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests"
    audio_root = tmp_path / "datasets" / "demo-speaker-recognition" / "speaker_alpha"
    manifests_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    audio_path = audio_root / "sample.wav"
    _write_wav(audio_path, duration_seconds=1.0)
    _write_jsonl(
        manifests_root / "demo_manifest.jsonl",
        [
            {
                "audio_path": "datasets/demo-speaker-recognition/speaker_alpha/sample.wav",
                "role": "test",
                "speaker_id": "speaker_alpha",
            }
        ],
    )

    report = build_dataset_leakage_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )

    finding_codes = {finding.code for finding in report.findings}

    assert report.finding_counts_by_severity["critical"] >= 1
    assert "missing_required_split" in finding_codes
    assert report.split_counts == {"demo": 1}


def test_build_dataset_leakage_report_detects_duplicates_and_overlap(tmp_path: Path) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests" / "ffsvc2022-surrogate"
    audio_root = tmp_path / "datasets" / "ffsvc2022-surrogate" / "raw" / "dev"
    manifests_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    train_a = audio_root / "ffsvc22_dev_000001.wav"
    train_b = audio_root / "ffsvc22_dev_000002.wav"
    dev_duplicate_content = audio_root / "ffsvc22_dev_000003.wav"
    dev_duplicate_path = audio_root / "ffsvc22_dev_000004.wav"
    extra_dev = audio_root / "ffsvc22_dev_000005.wav"

    _write_wav(train_a, duration_seconds=1.0, frame_pattern=b"\x01\x02")
    _write_wav(train_b, duration_seconds=1.0, frame_pattern=b"\x03\x04")
    _write_wav(dev_duplicate_content, duration_seconds=1.0, frame_pattern=b"\x01\x02")
    _write_wav(dev_duplicate_path, duration_seconds=1.0, frame_pattern=b"\x05\x06")
    _write_wav(extra_dev, duration_seconds=1.0, frame_pattern=b"\x07\x08")

    train_rows = [
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000001.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "1",
            "speaker_id": "0101",
            "split": "train",
            "utterance_id": "utt-1",
        },
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000002.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "2",
            "speaker_id": "0202",
            "split": "train",
            "utterance_id": "utt-2",
        },
    ]
    dev_rows = [
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000003.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "1",
            "speaker_id": "0101",
            "split": "dev",
            "utterance_id": "utt-3",
        },
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000002.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "2",
            "speaker_id": "0202",
            "split": "dev",
            "utterance_id": "utt-4",
        },
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000004.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "3",
            "speaker_id": "0303",
            "split": "dev",
            "utterance_id": "utt-5",
        },
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000005.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "4",
            "speaker_id": "0404",
            "split": "train",
            "utterance_id": "utt-6",
        },
    ]

    _write_jsonl(manifests_root / "train_manifest.jsonl", train_rows)
    _write_jsonl(manifests_root / "dev_manifest.jsonl", dev_rows)
    _write_jsonl(manifests_root / "all_manifest.jsonl", [*train_rows, *dev_rows])
    _write_jsonl(manifests_root / "quarantine_manifest.jsonl", [train_rows[0]])
    _write_jsonl(
        manifests_root / "speaker_disjoint_dev_trials.jsonl",
        [
            {
                "label": 0,
                "left_audio": "ffsvc22_dev_000001.wav",
                "right_audio": "ffsvc22_dev_000004.wav",
            }
        ],
    )

    report = build_dataset_leakage_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )

    finding_codes = {finding.code for finding in report.findings}
    written = write_dataset_leakage_report(
        report=report,
        output_root=tmp_path / "artifacts" / "eda" / "dataset-leakage",
    )
    payload = json.loads(Path(written.json_path).read_text())
    markdown = Path(written.markdown_path).read_text()

    assert report.data_manifest_count == 3
    assert "cross_split_audio_overlap" in finding_codes
    assert "duplicate_audio_content" in finding_codes
    assert "speaker_overlap" in finding_codes
    assert "session_overlap" in finding_codes
    assert "manifest_split_mismatch" in finding_codes
    assert "speaker_disjoint_trial_split_violation" in finding_codes
    assert payload["finding_count"] >= 6
    assert "Dataset Leakage Audit" in markdown
    assert "Different paths resolve to identical audio content" in markdown


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_wav(
    path: Path,
    *,
    duration_seconds: float,
    sample_rate: int = 16000,
    frame_pattern: bytes = b"\x00\x00",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frame_pattern * frame_count)
