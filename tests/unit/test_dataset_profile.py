from __future__ import annotations

import json
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

from kryptonite.eda.dataset_profile import (
    build_dataset_profile_report,
    write_dataset_profile_report,
)


def test_build_dataset_profile_report_deduplicates_overlapping_manifests(tmp_path: Path) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests"
    ffsvc_root = manifests_root / "ffsvc2022-surrogate"
    datasets_root = tmp_path / "datasets"
    ffsvc_audio_root = datasets_root / "ffsvc2022-surrogate" / "raw" / "dev"
    demo_audio_root = datasets_root / "demo-speaker-recognition"

    ffsvc_root.mkdir(parents=True)
    ffsvc_audio_root.mkdir(parents=True)
    demo_audio_root.mkdir(parents=True)

    train_a = ffsvc_audio_root / "ffsvc22_dev_000001.wav"
    train_b = ffsvc_audio_root / "ffsvc22_dev_000002.wav"
    dev_a = ffsvc_audio_root / "ffsvc22_dev_000003.wav"
    demo_a = demo_audio_root / "speaker_alpha" / "enroll_01.wav"
    demo_a.parent.mkdir(parents=True)

    _write_wav(train_a, duration_seconds=1.0)
    _write_wav(train_b, duration_seconds=2.0)
    _write_wav(dev_a, duration_seconds=3.0)
    _write_wav(demo_a, duration_seconds=1.5)

    train_rows = [
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000001.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "1",
            "speaker_id": "0101",
            "split": "train",
        },
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000002.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "2",
            "speaker_id": "0202",
            "split": "train",
        },
    ]
    dev_rows = [
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000003.wav",
            "dataset": "ffsvc2022-surrogate",
            "session_index": "1",
            "speaker_id": "0303",
            "split": "dev",
        }
    ]
    demo_rows = [
        {
            "audio_path": "datasets/demo-speaker-recognition/speaker_alpha/enroll_01.wav",
            "duration_seconds": 1.5,
            "role": "enrollment",
            "sample_rate_hz": 16000,
            "speaker_id": "speaker_alpha",
        }
    ]

    _write_jsonl(ffsvc_root / "train_manifest.jsonl", train_rows)
    _write_jsonl(ffsvc_root / "dev_manifest.jsonl", dev_rows)
    _write_jsonl(ffsvc_root / "all_manifest.jsonl", [*train_rows, *dev_rows])
    _write_jsonl(manifests_root / "demo_manifest.jsonl", demo_rows)
    _write_jsonl(
        ffsvc_root / "official_dev_trials.jsonl",
        [{"label": 1, "left_audio": "a.wav", "right_audio": "b.wav"}],
    )

    report = build_dataset_profile_report(
        project_root=tmp_path, manifests_root="artifacts/manifests"
    )

    split_counts = {summary.name: summary.summary.entry_count for summary in report.split_summaries}
    dataset_counts = {
        summary.name: summary.summary.entry_count for summary in report.dataset_summaries
    }

    assert report.manifest_count == 4
    assert report.raw_entry_count == 7
    assert report.duplicate_entry_count == 3
    assert report.total_summary.entry_count == 4
    assert report.total_summary.unique_speakers == 4
    assert report.total_summary.unique_sessions == 3
    assert split_counts == {"train": 2, "dev": 1, "demo": 1}
    assert dataset_counts == {"ffsvc2022-surrogate": 3, "demo-speaker-recognition": 1}
    assert report.ignored_manifests[0].manifest_path.endswith("official_dev_trials.jsonl")
    assert report.warnings == ["3 overlapping rows were deduplicated across manifests."]

    written = write_dataset_profile_report(
        report=report,
        output_root=tmp_path / "artifacts" / "eda" / "dataset-profile",
    )
    markdown = Path(written.markdown_path).read_text()
    payload = json.loads(Path(written.json_path).read_text())

    assert "Rows By Split" in markdown
    assert "Duration Histogram" in markdown
    assert "ffsvc2022-surrogate/train_manifest.jsonl" in markdown
    assert payload["duplicate_entry_count"] == 3


def test_build_dataset_profile_report_tracks_missing_audio_and_manifest_duration(
    tmp_path: Path,
) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests"
    manifests_root.mkdir(parents=True)
    _write_jsonl(
        manifests_root / "demo_manifest.jsonl",
        [
            {
                "audio_path": "datasets/demo-speaker-recognition/speaker_alpha/missing.wav",
                "duration_seconds": 2.5,
                "role": "test",
                "sample_rate_hz": 8000,
                "speaker_id": "speaker_alpha",
            }
        ],
    )

    report = build_dataset_profile_report(
        project_root=tmp_path, manifests_root="artifacts/manifests"
    )

    assert report.total_summary.entry_count == 1
    assert report.total_summary.missing_audio_file_count == 1
    assert report.total_summary.duration_source_counts == {"manifest": 1}
    assert report.total_summary.sample_rate_counts == {"8000": 1}
    assert report.total_summary.duration_summary.total == 2.5
    assert report.warnings == [
        "Expected dataset coverage is incomplete. Missing splits: train, dev.",
        "1 rows point to missing audio files.",
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_wav(path: Path, *, duration_seconds: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)
