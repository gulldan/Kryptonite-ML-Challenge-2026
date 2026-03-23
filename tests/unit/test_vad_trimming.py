from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kryptonite.data.vad as vad_module
from kryptonite.config import NormalizationConfig, VADConfig
from kryptonite.eda.vad_trimming import build_vad_trimming_report


def test_build_vad_trimming_report_compares_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_vad_segments(monkeypatch)
    audio_root = tmp_path / "datasets" / "demo"
    manifest_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    audio_path = audio_root / "utterance.wav"
    _write_trim_candidate_audio(audio_path)
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

    report = build_vad_trimming_report(
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
        vad=VADConfig(
            mode="light",
            min_output_duration_seconds=1.0,
            min_retained_ratio=0.4,
        ),
    )

    assert report.row_count == 1
    assert report.backend == "silero_vad_v6_onnx"
    assert report.provider == "auto"
    assert report.min_output_duration_seconds == 1.0
    assert report.min_retained_ratio == 0.4
    summaries = {summary.mode: summary for summary in report.summaries}
    assert summaries["none"].trimmed_row_count == 0
    assert summaries["light"].trimmed_row_count == 1
    assert summaries["aggressive"].trimmed_row_count == 1
    assert (
        summaries["aggressive"].total_output_duration_seconds
        < summaries["light"].total_output_duration_seconds
    )
    assert report.examples_by_mode["light"][0].removed_duration_seconds > 0.0


def _write_trim_candidate_audio(path: Path) -> None:
    sample_rate_hz = 16_000
    silence = np.zeros(4_800, dtype=np.float32)
    time = np.arange(32_000, dtype=np.float32) / np.float32(sample_rate_hz)
    speech = (0.25 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32, copy=False)
    waveform = np.concatenate([silence, speech, silence])
    sf.write(path, waveform, sample_rate_hz, format="WAV")


def _patch_vad_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_detect(
        waveform: np.ndarray,
        *,
        sample_rate_hz: int,
        settings: vad_module.VADSettings,
    ) -> list[dict[str, int]]:
        assert sample_rate_hz == 16_000
        assert waveform.ndim == 2
        assert settings.backend == "silero_vad_v6_onnx"
        assert settings.provider == "auto"
        if settings.mode == "light":
            return [{"start": 4_000, "end": 36_000}]
        if settings.mode == "aggressive":
            return [{"start": 5_500, "end": 34_500}]
        return []

    monkeypatch.setattr(vad_module, "_detect_speech_segments", fake_detect)
