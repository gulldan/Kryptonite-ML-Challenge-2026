from __future__ import annotations

import json
import math
import wave
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path

from kryptonite.eda.dataset_audio_quality import (
    build_dataset_audio_quality_report,
    write_dataset_audio_quality_report,
)


def test_build_dataset_audio_quality_report_surfaces_waveform_flags(tmp_path: Path) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests" / "ffsvc2022-surrogate"
    audio_root = tmp_path / "datasets" / "ffsvc2022-surrogate" / "raw" / "dev"
    manifests_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    normal_audio = audio_root / "ffsvc22_dev_000001.wav"
    quiet_audio = audio_root / "ffsvc22_dev_000002.wav"
    problematic_audio = audio_root / "ffsvc22_dev_000003.wav"

    _write_tone_wav(normal_audio, sample_rate=16_000, amplitude=8_000)
    _write_tone_wav(quiet_audio, sample_rate=16_000, amplitude=1_000)
    _write_problematic_wav(problematic_audio, sample_rate=8_000, silent_prefix_seconds=0.7)

    train_rows = [
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000001.wav",
            "dataset": "ffsvc2022-surrogate",
            "source_prefix": "A",
            "capture_condition": "cond-clean",
            "session_index": "1",
            "speaker_id": "0101",
            "split": "train",
            "utterance_id": "utt-1",
        },
        {
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000002.wav",
            "dataset": "ffsvc2022-surrogate",
            "source_prefix": "B",
            "capture_condition": "cond-quiet",
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
            "source_prefix": "C",
            "capture_condition": "cond-tail",
            "session_index": "3",
            "speaker_id": "0303",
            "split": "dev",
            "utterance_id": "utt-3",
        }
    ]

    _write_jsonl(manifests_root / "train_manifest.jsonl", train_rows)
    _write_jsonl(manifests_root / "dev_manifest.jsonl", dev_rows)
    _write_jsonl(manifests_root / "all_manifest.jsonl", [*train_rows, *dev_rows])
    _write_jsonl(
        manifests_root / "speaker_disjoint_dev_trials.jsonl",
        [{"label": 1, "left_audio": "a.wav", "right_audio": "b.wav"}],
    )
    _write_jsonl(manifests_root / "quarantine_manifest.jsonl", [train_rows[0]])

    report = build_dataset_audio_quality_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )
    written = write_dataset_audio_quality_report(
        report=report,
        output_root=tmp_path / "artifacts" / "eda" / "dataset-audio-quality",
    )

    payload = json.loads(Path(written.json_path).read_text())
    markdown = Path(written.markdown_path).read_text()
    split_counts = {summary.name: summary.summary.entry_count for summary in report.split_summaries}

    assert report.raw_entry_count == 6
    assert report.duplicate_entry_count == 3
    assert report.total_summary.entry_count == 3
    assert report.total_summary.waveform_metrics_count == 3
    assert split_counts == {"train": 2, "dev": 1}
    assert report.total_summary.flag_counts["low_loudness"] == 1
    assert report.total_summary.flag_counts["non_16k_sample_rate"] == 1
    assert report.total_summary.flag_counts["high_silence_ratio"] == 1
    assert report.total_summary.flag_counts["clipping_risk"] == 1
    assert {pattern.code for pattern in report.patterns} >= {
        "mixed_sample_rates",
        "silence_heavy_tail",
        "low_level_recordings",
        "clipping_present",
    }
    assert "Dataset Audio Quality Report" in markdown
    assert "Key Patterns" in markdown
    assert payload["total_summary"]["flag_counts"]["high_silence_ratio"] == 1


def test_build_dataset_audio_quality_report_tracks_missing_audio_with_manifest_fallback(
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
                "sample_rate_hz": 8_000,
                "speaker_id": "speaker_alpha",
            }
        ],
    )

    report = build_dataset_audio_quality_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )

    assert report.total_summary.entry_count == 1
    assert report.total_summary.missing_audio_file_count == 1
    assert report.total_summary.waveform_metrics_count == 0
    assert report.total_summary.duration_summary.total == 2.5
    assert report.total_summary.flag_counts["missing_audio_file"] == 1
    assert report.total_summary.flag_counts["non_16k_sample_rate"] == 1
    assert report.warnings == [
        "Expected dataset coverage is incomplete. Missing splits: train, dev.",
        "1 rows point to missing audio files.",
    ]


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
    _write_wav_samples(path, sample_rate=sample_rate, channels=1, samples=samples)


def _write_problematic_wav(
    path: Path,
    *,
    sample_rate: int,
    silent_prefix_seconds: float,
    duration_seconds: float = 1.0,
) -> None:
    frame_count = int(sample_rate * duration_seconds)
    silent_frames = int(sample_rate * silent_prefix_seconds)
    samples = [0] * silent_frames + [32_767] * (frame_count - silent_frames)
    _write_wav_samples(path, sample_rate=sample_rate, channels=1, samples=samples)


def _write_wav_samples(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    samples: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = array("h")
    for sample in samples:
        for _ in range(channels):
            values.append(sample)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(values.tobytes())
