from __future__ import annotations

import json
import math
import wave
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import soundfile as sf

from kryptonite.eda.audio_quality.inspection import (
    analyze_pcm_chunk,
    maximum_possible_amplitude,
    pcm_samples,
)
from kryptonite.eda.dataset_audio_quality import (
    build_dataset_audio_quality_report,
    write_dataset_audio_quality_report,
)


def _encode_pcm_24bit_le(samples: Sequence[int]) -> bytes:
    encoded = bytearray()
    for sample in samples:
        value = sample if sample >= 0 else (1 << 24) + sample
        encoded.extend(
            (
                value & 0xFF,
                (value >> 8) & 0xFF,
                (value >> 16) & 0xFF,
            )
        )
    return bytes(encoded)


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
    rows_payload = _read_jsonl(Path(written.rows_path))
    flagged_rows_payload = _read_jsonl(Path(written.flagged_rows_path))
    split_counts = {summary.name: summary.summary.entry_count for summary in report.split_summaries}

    assert report.raw_entry_count == 6
    assert report.duplicate_entry_count == 3
    assert report.total_summary.entry_count == 3
    assert report.flagged_record_count == 2
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
    assert len(rows_payload) == 3
    assert len(flagged_rows_payload) == 2
    assert {row["audio_path"] for row in flagged_rows_payload} == {
        "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000002.wav",
        "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000003.wav",
    }


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


def test_build_dataset_audio_quality_report_handles_zero_signal_flac_and_mp3_inputs(
    tmp_path: Path,
) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root = tmp_path / "datasets" / "demo"
    manifests_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    zero_signal_audio = audio_root / "zero.flac"
    tone_audio = audio_root / "tone.mp3"
    _write_float_audio(zero_signal_audio, format_name="FLAC", sample_rate=16_000, amplitude=0.0)
    _write_float_audio(tone_audio, format_name="MP3", sample_rate=16_000, amplitude=0.25)

    rows = [
        {
            "audio_path": "datasets/demo/zero.flac",
            "dataset": "demo",
            "speaker_id": "speaker-zero",
            "split": "train",
            "utterance_id": "utt-zero",
        },
        {
            "audio_path": "datasets/demo/tone.mp3",
            "dataset": "demo",
            "speaker_id": "speaker-tone",
            "split": "dev",
            "utterance_id": "utt-tone",
        },
    ]
    _write_jsonl(manifests_root / "train_manifest.jsonl", [rows[0]])
    _write_jsonl(manifests_root / "dev_manifest.jsonl", [rows[1]])
    _write_jsonl(manifests_root / "all_manifest.jsonl", rows)

    report = build_dataset_audio_quality_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )

    assert report.total_summary.entry_count == 2
    assert report.total_summary.waveform_metrics_count == 2
    assert report.total_summary.flag_counts["zero_signal"] == 1
    assert report.total_summary.audio_format_counts["flac"] == 1
    assert report.total_summary.audio_format_counts["mp3"] == 1
    assert "zero_signal_rows" in {pattern.code for pattern in report.patterns}


def test_build_dataset_audio_quality_report_marks_broken_headers_as_audio_read_error(
    tmp_path: Path,
) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root = tmp_path / "datasets" / "demo"
    manifests_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    broken_audio = audio_root / "broken.wav"
    broken_audio.write_bytes(b"not-a-real-wave-header")
    _write_jsonl(
        manifests_root / "train_manifest.jsonl",
        [
            {
                "audio_path": "datasets/demo/broken.wav",
                "dataset": "demo",
                "speaker_id": "speaker-broken",
                "split": "train",
                "utterance_id": "utt-broken",
            }
        ],
    )

    report = build_dataset_audio_quality_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )

    assert report.total_summary.entry_count == 1
    assert report.total_summary.waveform_metrics_count == 0
    assert report.total_summary.audio_inspection_error_count == 1
    assert report.total_summary.flag_counts["audio_read_error"] == 1
    assert "1 audio files could not be fully inspected." in report.warnings


@pytest.mark.parametrize(
    ("sample_width_bytes", "frames", "expected"),
    [
        (1, bytes([0, 128, 255]), [-128, 0, 127]),
        (2, array("h", [-32_768, -123, 0, 123, 32_767]).tobytes(), [-32_768, -123, 0, 123, 32_767]),
        (
            3,
            _encode_pcm_24bit_le([-8_388_608, -456_789, 0, 456_789, 8_388_607]),
            [-8_388_608, -456_789, 0, 456_789, 8_388_607],
        ),
        (
            4,
            array("i", [-123_456_789, -1, 0, 1, 123_456_789]).tobytes(),
            [-123_456_789, -1, 0, 1, 123_456_789],
        ),
    ],
)
def test_pcm_samples_decodes_supported_widths(
    sample_width_bytes: int,
    frames: bytes,
    expected: list[int],
) -> None:
    assert list(pcm_samples(frames=frames, sample_width_bytes=sample_width_bytes)) == expected


def test_analyze_pcm_chunk_tracks_silence_and_clipping_flags() -> None:
    silent_frames = array("h", [0, 0, 0, 0]).tobytes()
    clipped_frames = array("h", [32_767, -32_767, 32_767, -32_767]).tobytes()
    max_amplitude = maximum_possible_amplitude(2)

    silent = analyze_pcm_chunk(
        frames=silent_frames,
        sample_width_bytes=2,
        max_possible_amplitude=max_amplitude,
    )
    clipped = analyze_pcm_chunk(
        frames=clipped_frames,
        sample_width_bytes=2,
        max_possible_amplitude=max_amplitude,
    )

    assert silent is not None
    assert silent.sample_count == 4
    assert silent.sum_of_squares == 0.0
    assert silent.signed_sum == 0.0
    assert silent.peak_amplitude == 0
    assert silent.is_silent is True
    assert silent.is_clipped is False

    assert clipped is not None
    assert clipped.sample_count == 4
    assert clipped.peak_amplitude == 32_767
    assert clipped.is_silent is False
    assert clipped.is_clipped is True


def test_analyze_pcm_chunk_preserves_legacy_integer_rms_and_average_semantics() -> None:
    frames = array("h", [-1, -2]).tobytes()

    stats = analyze_pcm_chunk(
        frames=frames,
        sample_width_bytes=2,
        max_possible_amplitude=maximum_possible_amplitude(2),
    )

    assert stats is not None
    assert stats.sample_count == 2
    assert stats.sum_of_squares == 2.0
    assert stats.signed_sum == -4.0
    assert stats.peak_amplitude == 2


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def _write_float_audio(
    path: Path,
    *,
    format_name: str,
    sample_rate: int,
    amplitude: float,
    duration_seconds: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(sample_rate * duration_seconds)
    samples = [
        amplitude * math.sin(2.0 * math.pi * 220.0 * index / sample_rate)
        for index in range(frame_count)
    ]
    sf.write(path, samples, sample_rate, format=format_name)


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
