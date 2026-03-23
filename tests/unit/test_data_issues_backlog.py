from __future__ import annotations

import json
import math
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

from kryptonite.eda.data_issues_backlog import (
    build_data_issues_backlog_report,
    write_data_issues_backlog_report,
)


def test_build_data_issues_backlog_report_classifies_cleanup_actions(tmp_path: Path) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests" / "ffsvc2022-surrogate"
    audio_root = tmp_path / "datasets" / "ffsvc2022-surrogate" / "raw" / "dev"
    manifests_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    canonical_audio = audio_root / "ffsvc22_dev_000001.wav"
    quiet_audio = audio_root / "ffsvc22_dev_000002.wav"
    duplicate_audio = audio_root / "ffsvc22_dev_000003.wav"

    _write_tone_wav(canonical_audio, sample_rate=16_000, amplitude=8_000)
    _write_tone_wav(quiet_audio, sample_rate=16_000, amplitude=1_000)
    _write_problematic_wav(duplicate_audio, sample_rate=8_000, silent_prefix_seconds=0.7)

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
            "session_index": "3",
            "speaker_id": "0303",
            "split": "dev",
            "utterance_id": "utt-3",
        }
    ]
    quarantine_rows = [
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_999999.wav",
            "dataset": "ffsvc2022-surrogate",
            "duplicate_canonical_utterance_id": "utt-3",
            "duplicate_policy": "quarantine",
            "quality_issue_code": "duplicate_audio_content",
            "speaker_id": "0303",
            "split": "train",
            "utterance_id": "utt-999",
        }
    ]

    _write_jsonl(manifests_root / "train_manifest.jsonl", train_rows)
    _write_jsonl(manifests_root / "dev_manifest.jsonl", dev_rows)
    _write_jsonl(manifests_root / "all_manifest.jsonl", [*train_rows, *dev_rows])
    _write_jsonl(
        manifests_root / "speaker_disjoint_dev_trials.jsonl",
        [
            {
                "label": 1,
                "left_audio": "ffsvc22_dev_000003.wav",
                "right_audio": "ffsvc22_dev_000003.wav",
            }
        ],
    )
    _write_jsonl(manifests_root / "quarantine_manifest.jsonl", quarantine_rows)

    report = build_data_issues_backlog_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )
    written = write_data_issues_backlog_report(
        report=report,
        output_root=tmp_path / "artifacts" / "eda" / "data-issues-backlog",
    )

    payload = json.loads(Path(written.json_path).read_text())
    markdown = Path(written.markdown_path).read_text()
    action_by_code = {issue.code: issue.action for issue in report.issues}

    assert action_by_code["quarantined_rows"] == "quarantine"
    assert action_by_code["mixed_sample_rates"] == "fix"
    assert action_by_code["silence_heavy_tail"] == "keep"
    assert action_by_code["low_level_recordings"] == "keep"
    assert report.issue_counts_by_action["quarantine"] >= 1
    assert report.quarantine_row_count == 1
    assert "Data Issues Backlog" in markdown
    assert "Stop Rules" in markdown
    assert payload["issue_count"] == report.issue_count


def test_build_data_issues_backlog_report_blocks_when_required_splits_are_missing(
    tmp_path: Path,
) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests"
    audio_root = tmp_path / "datasets" / "demo-speaker-recognition" / "speaker_alpha"
    manifests_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    audio_path = audio_root / "sample.wav"
    _write_tone_wav(audio_path, sample_rate=16_000, amplitude=8_000)
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

    report = build_data_issues_backlog_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )

    issue_by_code = {issue.code: issue for issue in report.issues}

    assert issue_by_code["missing_required_split"].action == "fix"
    assert any("train/dev manifests exist" in rule for rule in report.stop_rules)
    assert report.issue_counts_by_severity["critical"] >= 1


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_tone_wav(
    path: Path,
    *,
    sample_rate: int,
    amplitude: int,
    duration_seconds: float = 1.0,
) -> None:
    frame_count = int(sample_rate * duration_seconds)
    samples = [
        round(amplitude * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
        for index in range(frame_count)
    ]
    _write_wav_samples(path, sample_rate=sample_rate, samples=samples)


def _write_problematic_wav(
    path: Path,
    *,
    sample_rate: int,
    silent_prefix_seconds: float,
    duration_seconds: float = 1.0,
) -> None:
    silent_frames = int(sample_rate * silent_prefix_seconds)
    active_frames = max(1, int(sample_rate * duration_seconds) - silent_frames)
    samples = [0] * silent_frames
    samples.extend(32_767 if index % 2 == 0 else -32_767 for index in range(active_frames))
    _write_wav_samples(path, sample_rate=sample_rate, samples=samples)


def _write_wav_samples(path: Path, *, sample_rate: int, samples: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
        )
