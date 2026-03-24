from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from kryptonite.config import NormalizationConfig
from kryptonite.data.loudness import (
    LoudnessNormalizationSettings,
    apply_loudness_normalization,
)
from kryptonite.eda.loudness_normalization import build_loudness_normalization_report


def test_apply_loudness_normalization_hits_target_without_waveform_distortion() -> None:
    waveform = _sine_wave(amplitude=0.01)
    normalized, decision = apply_loudness_normalization(
        waveform,
        settings=LoudnessNormalizationSettings(
            mode="rms",
            target_loudness_dbfs=-27.0,
            max_gain_db=20.0,
            max_attenuation_db=12.0,
            peak_headroom_db=1.0,
        ),
    )

    assert decision.applied is True
    assert decision.gain_clamped is False
    assert decision.source_rms_dbfs == pytest.approx(-43.0103, abs=0.2)
    assert decision.output_rms_dbfs == pytest.approx(-27.0, abs=0.2)
    assert decision.alignment_error == pytest.approx(0.0, abs=1e-7)
    assert decision.degradation_check_passed is True
    assert normalized.shape == waveform.shape


def test_apply_loudness_normalization_respects_gain_cap() -> None:
    waveform = _sine_wave(amplitude=0.001)
    _, decision = apply_loudness_normalization(
        waveform,
        settings=LoudnessNormalizationSettings(
            mode="rms",
            target_loudness_dbfs=-27.0,
            max_gain_db=20.0,
            max_attenuation_db=12.0,
            peak_headroom_db=1.0,
        ),
    )

    assert decision.gain_clamped is True
    assert decision.applied_gain_db == pytest.approx(20.0, abs=1e-4)
    assert decision.output_rms_dbfs == pytest.approx(-43.0, abs=0.3)


def test_build_loudness_normalization_report_tracks_before_after_metrics(tmp_path: Path) -> None:
    audio_root = tmp_path / "datasets" / "demo"
    manifest_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    quiet = audio_root / "quiet.wav"
    loud = audio_root / "loud.wav"
    _write_wave(quiet, amplitude=0.01)
    _write_wave(loud, amplitude=0.2)

    rows = [
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "demo",
            "source_dataset": "demo",
            "speaker_id": "speaker-a",
            "utterance_id": "utt-quiet",
            "audio_path": "datasets/demo/quiet.wav",
        },
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "demo",
            "source_dataset": "demo",
            "speaker_id": "speaker-b",
            "utterance_id": "utt-loud",
            "audio_path": "datasets/demo/loud.wav",
        },
    ]
    manifest_path = manifest_root / "dev_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = build_loudness_normalization_report(
        project_root=tmp_path,
        manifest_path=manifest_path,
        normalization=NormalizationConfig(
            target_sample_rate_hz=16_000,
            target_channels=1,
            output_format="wav",
            output_pcm_bits_per_sample=16,
            peak_headroom_db=1.0,
            dc_offset_threshold=0.01,
            clipped_sample_threshold=0.999,
            loudness_mode="rms",
            target_loudness_dbfs=-27.0,
            max_loudness_gain_db=20.0,
            max_loudness_attenuation_db=12.0,
        ),
    )

    assert report.summary.row_count == 2
    assert report.summary.changed_row_count == 2
    assert report.summary.degradation_check_failed_row_count == 0
    assert report.summary.baseline_guard_passed is True
    assert report.summary.target_reached_row_count >= 1
    assert report.summary.mean_output_rms_dbfs is not None
    assert report.summary.mean_source_rms_dbfs is not None
    assert all(record.degradation_check_passed for record in report.records)


def _sine_wave(*, amplitude: float, sample_rate_hz: int = 16_000) -> np.ndarray:
    time = np.arange(sample_rate_hz, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = amplitude * np.sin(2.0 * np.pi * 220.0 * time)
    return waveform.reshape(1, -1)


def _write_wave(path: Path, *, amplitude: float) -> None:
    sf.write(path, _sine_wave(amplitude=amplitude).reshape(-1), 16_000, format="WAV")
