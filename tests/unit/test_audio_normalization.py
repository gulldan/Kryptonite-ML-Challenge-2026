from __future__ import annotations

import json
import math
import wave
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import soundfile as sf

from kryptonite.data.normalization import AudioNormalizationPolicy, normalize_audio_manifest_bundle


def test_normalize_audio_manifest_bundle_rewrites_manifests_and_quarantines_failures(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets" / "ffsvc2022-surrogate" / "raw" / "dev"
    source_root = tmp_path / "artifacts" / "manifests" / "ffsvc2022-surrogate"
    dataset_root.mkdir(parents=True)
    source_root.mkdir(parents=True)

    stereo_offset_audio = dataset_root / "ffsvc22_dev_000001.wav"
    clean_audio = dataset_root / "ffsvc22_dev_000002.wav"

    _write_stereo_problematic_wav(stereo_offset_audio, sample_rate=8_000)
    _write_sine_wav(clean_audio, sample_rate=16_000, amplitude=6_000)

    train_rows = [
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "ffsvc2022-surrogate",
            "source_dataset": "ffsvc2022",
            "speaker_id": "0101",
            "utterance_id": "utt-1",
            "session_id": "0101:1",
            "split": "train",
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000001.wav",
            "duration_seconds": 1.0,
            "sample_rate_hz": 8_000,
            "num_channels": 2,
            "source_prefix": "F",
        }
    ]
    dev_rows = [
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "ffsvc2022-surrogate",
            "source_dataset": "ffsvc2022",
            "speaker_id": "0202",
            "utterance_id": "utt-2",
            "session_id": "0202:1",
            "split": "dev",
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000002.wav",
            "duration_seconds": 1.0,
            "sample_rate_hz": 16_000,
            "num_channels": 1,
            "source_prefix": "S",
        },
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "ffsvc2022-surrogate",
            "source_dataset": "ffsvc2022",
            "speaker_id": "0303",
            "utterance_id": "utt-3",
            "session_id": "0303:1",
            "split": "dev",
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/ffsvc22_dev_000003.wav",
            "duration_seconds": 1.0,
            "sample_rate_hz": 16_000,
            "num_channels": 1,
            "source_prefix": "T",
        },
    ]
    all_rows = [*train_rows, *dev_rows]
    quarantine_rows = [
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "ffsvc2022-surrogate",
            "source_dataset": "ffsvc2022",
            "speaker_id": "0999",
            "utterance_id": "quarantine-1",
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/quarantine.wav",
            "quality_issue_code": "duplicate_audio_content",
        }
    ]
    trial_rows = [
        {
            "label": 1,
            "left_audio": "ffsvc22_dev_000001.wav",
            "right_audio": "ffsvc22_dev_000002.wav",
        }
    ]

    _write_jsonl(source_root / "train_manifest.jsonl", train_rows)
    _write_jsonl(source_root / "dev_manifest.jsonl", dev_rows)
    _write_jsonl(source_root / "all_manifest.jsonl", all_rows)
    _write_jsonl(source_root / "quarantine_manifest.jsonl", quarantine_rows)
    _write_jsonl(source_root / "speaker_disjoint_dev_trials.jsonl", trial_rows)
    source_root.joinpath("speaker_splits.json").write_text(json.dumps({"dev_speakers": ["0202"]}))

    summary = normalize_audio_manifest_bundle(
        project_root=str(tmp_path),
        dataset_root="datasets",
        source_manifests_root="artifacts/manifests/ffsvc2022-surrogate",
        output_root="artifacts/preprocessed/ffsvc2022-surrogate",
        policy=AudioNormalizationPolicy(
            target_sample_rate_hz=16_000,
            target_channels=1,
            output_format="wav",
            output_pcm_bits_per_sample=16,
            peak_headroom_db=6.0,
            dc_offset_threshold=0.01,
            clipped_sample_threshold=0.999,
        ),
    )

    normalized_train_rows = _read_jsonl(
        tmp_path
        / "artifacts"
        / "preprocessed"
        / "ffsvc2022-surrogate"
        / "manifests"
        / "train_manifest.jsonl"
    )
    normalized_dev_rows = _read_jsonl(
        tmp_path
        / "artifacts"
        / "preprocessed"
        / "ffsvc2022-surrogate"
        / "manifests"
        / "dev_manifest.jsonl"
    )
    quarantine_output_rows = _read_jsonl(
        tmp_path
        / "artifacts"
        / "preprocessed"
        / "ffsvc2022-surrogate"
        / "manifests"
        / "quarantine_manifest.jsonl"
    )
    report_payload = json.loads(
        (
            tmp_path
            / "artifacts"
            / "preprocessed"
            / "ffsvc2022-surrogate"
            / "reports"
            / "audio_normalization_report.json"
        ).read_text()
    )
    normalized_waveform, normalized_sample_rate = sf.read(
        str(tmp_path / cast(str, normalized_train_rows[0]["audio_path"])),
        always_2d=True,
        dtype="float32",
    )

    assert summary.source_row_count == 3
    assert summary.normalized_row_count == 2
    assert summary.generated_quarantine_row_count == 1
    assert summary.carried_quarantine_row_count == 1
    assert summary.resampled_row_count == 1
    assert summary.downmixed_row_count == 1
    assert summary.dc_offset_fixed_row_count == 1
    assert summary.peak_scaled_row_count == 1
    assert summary.source_clipping_row_count == 1
    assert summary.quarantine_issue_counts == {"missing_audio_file": 1}
    assert report_payload["normalized_audio_count"] == 2

    normalized_train = normalized_train_rows[0]
    assert normalized_train["sample_rate_hz"] == 16_000
    assert normalized_train["num_channels"] == 1
    assert normalized_train["channel"] == "mono"
    assert normalized_train["normalization_profile"] == "16000hz-1ch-pcm16-wav"
    assert normalized_train["normalization_resampled"] is True
    assert normalized_train["normalization_downmixed"] is True
    assert normalized_train["normalization_dc_offset_removed"] is True
    assert normalized_train["normalization_peak_scaled"] is True
    assert normalized_train["source_sample_rate_hz"] == 8_000
    assert normalized_train["source_num_channels"] == 2
    assert cast(float, normalized_train["source_clipped_sample_ratio"]) > 0.0

    assert len(normalized_dev_rows) == 1
    assert normalized_dev_rows[0]["utterance_id"] == "utt-2"

    assert {row["utterance_id"] for row in quarantine_output_rows} == {"quarantine-1", "utt-3"}
    generated_quarantine = next(
        row for row in quarantine_output_rows if row["utterance_id"] == "utt-3"
    )
    assert generated_quarantine["quality_issue_code"] == "missing_audio_file"
    assert generated_quarantine["quarantine_stage"] == "audio_normalization"

    assert normalized_sample_rate == 16_000
    assert normalized_waveform.shape[1] == 1
    assert float(abs(normalized_waveform).max()) <= (10 ** (-6.0 / 20.0)) + 1e-3

    with wave.open(
        str(
            tmp_path
            / "artifacts"
            / "preprocessed"
            / "ffsvc2022-surrogate"
            / "audio"
            / "ffsvc2022-surrogate"
            / "raw"
            / "dev"
            / "ffsvc22_dev_000001.wav"
        ),
        "rb",
    ) as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_sine_wav(
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
    _write_pcm_wav(path, sample_rate=sample_rate, channels=1, samples=samples)


def _write_stereo_problematic_wav(path: Path, *, sample_rate: int) -> None:
    frame_count = sample_rate
    left_channel = []
    right_channel = []
    for index in range(frame_count):
        tone = round(12_000 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
        clipped = 32_767 if index % 20 == 0 else max(-32_768, min(32_767, tone + 9_000))
        left_channel.append(clipped)
        right_channel.append(
            32_767 if index % 16 == 0 else max(-32_768, min(32_767, tone + 18_000))
        )

    interleaved: list[int] = []
    for left, right in zip(left_channel, right_channel, strict=True):
        interleaved.extend((left, right))
    _write_pcm_wav(path, sample_rate=sample_rate, channels=2, samples=interleaved)


def _write_pcm_wav(path: Path, *, sample_rate: int, channels: int, samples: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(array("h", samples).tobytes())
