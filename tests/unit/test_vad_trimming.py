from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.config import NormalizationConfig
from kryptonite.eda.vad_trimming import build_vad_trimming_report


def test_build_vad_trimming_report_compares_modes(tmp_path: Path) -> None:
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
    )

    assert report.row_count == 1
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
    time = np.arange(8_000, dtype=np.float32) / np.float32(sample_rate_hz)
    speech = (0.25 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32, copy=False)
    waveform = np.concatenate([silence, speech, silence])
    sf.write(path, waveform, sample_rate_hz, format="WAV")
