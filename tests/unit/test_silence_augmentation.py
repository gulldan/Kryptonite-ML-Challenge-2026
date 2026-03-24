from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.config import NormalizationConfig, SilenceAugmentationConfig, VADConfig
from kryptonite.data import analyze_silence_profile, apply_silence_augmentation
from kryptonite.eda.silence_augmentation import build_silence_augmentation_report


def test_apply_silence_augmentation_scales_detected_interior_pause() -> None:
    waveform = _waveform_with_interior_pause(pause_seconds=0.2)
    config = SilenceAugmentationConfig(
        enabled=True,
        pause_ratio_min=1.5,
        pause_ratio_max=1.5,
        min_detected_pause_seconds=0.08,
        max_perturbed_pause_seconds=0.6,
    )

    baseline = analyze_silence_profile(waveform, sample_rate_hz=16_000, config=config)
    augmented, decision = apply_silence_augmentation(
        waveform,
        sample_rate_hz=16_000,
        config=config,
        rng=random.Random(7),
    )
    updated = analyze_silence_profile(augmented, sample_rate_hz=16_000, config=config)

    assert decision.applied is True
    assert decision.perturbed_pause_count == 1
    assert decision.stretched_pause_count == 1
    assert decision.compressed_pause_count == 0
    assert augmented.shape[-1] == waveform.shape[-1] + 1_600
    assert updated.interior_pause_total_seconds > baseline.interior_pause_total_seconds


def test_apply_silence_augmentation_inserts_pause_and_boundary_padding() -> None:
    waveform = _speech_only_waveform(duration_seconds=1.0)
    config = SilenceAugmentationConfig(
        enabled=True,
        max_leading_padding_seconds=0.05,
        max_trailing_padding_seconds=0.05,
        max_inserted_pauses=1,
        min_inserted_pause_seconds=0.1,
        max_inserted_pause_seconds=0.1,
    )

    augmented, decision = apply_silence_augmentation(
        waveform,
        sample_rate_hz=16_000,
        config=config,
        rng=random.Random(3),
    )
    updated = analyze_silence_profile(augmented, sample_rate_hz=16_000, config=config)

    assert decision.applied is True
    assert decision.inserted_pause_count == 1
    assert decision.inserted_pause_total_seconds == 0.1
    assert decision.leading_padding_seconds > 0.0 or decision.trailing_padding_seconds > 0.0
    assert updated.interior_pause_count >= 1
    assert augmented.shape[-1] > waveform.shape[-1]


def test_build_silence_augmentation_report_tracks_pause_metrics(tmp_path: Path) -> None:
    audio_root = tmp_path / "datasets" / "demo"
    manifest_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    audio_path = audio_root / "utterance.wav"
    sf.write(audio_path, _waveform_with_interior_pause(pause_seconds=0.2).T, 16_000, format="WAV")
    manifest_path = manifest_root / "dev_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "demo",
                "source_dataset": "demo",
                "speaker_id": "speaker-a",
                "utterance_id": "utt-a",
                "audio_path": "datasets/demo/utterance.wav",
            }
        )
        + "\n"
    )

    report = build_silence_augmentation_report(
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
        ),
        vad=VADConfig(mode="none"),
        silence_augmentation=SilenceAugmentationConfig(
            enabled=True,
            pause_ratio_min=1.5,
            pause_ratio_max=1.5,
            min_detected_pause_seconds=0.08,
            max_perturbed_pause_seconds=0.6,
        ),
        seed=11,
    )

    assert report.summary.row_count == 1
    assert report.summary.changed_row_count == 1
    assert report.summary.rows_with_pause_perturbation == 1
    assert report.summary.mean_output_silence_ratio > report.summary.mean_input_silence_ratio
    assert report.records[0].perturbed_pause_count == 1


def _speech_only_waveform(*, duration_seconds: float) -> np.ndarray:
    sample_rate_hz = 16_000
    frame_count = int(sample_rate_hz * duration_seconds)
    time = np.arange(frame_count, dtype=np.float32) / np.float32(sample_rate_hz)
    speech = (0.25 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32, copy=False)
    return speech[np.newaxis, :]


def _waveform_with_interior_pause(*, pause_seconds: float) -> np.ndarray:
    sample_rate_hz = 16_000
    left = _speech_only_waveform(duration_seconds=1.0)
    pause = np.zeros((1, int(sample_rate_hz * pause_seconds)), dtype=np.float32)
    right = _speech_only_waveform(duration_seconds=1.0)
    return np.concatenate([left, pause, right], axis=1)
