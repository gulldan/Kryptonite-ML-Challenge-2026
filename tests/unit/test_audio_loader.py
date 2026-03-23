from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kryptonite.data.vad as vad_module
from kryptonite.config import NormalizationConfig, VADConfig
from kryptonite.data import AudioLoadRequest, iter_manifest_audio, load_audio, load_manifest_audio
from kryptonite.data.schema import ManifestValidationError


def test_load_audio_applies_resampling_and_mono_fold_down(tmp_path: Path) -> None:
    audio_path = tmp_path / "datasets" / "demo" / "stereo.wav"
    audio_path.parent.mkdir(parents=True)
    _write_tone_audio(audio_path, format_name="WAV", sample_rate_hz=8_000, channels=2)

    request = AudioLoadRequest.from_config(
        NormalizationConfig(
            target_sample_rate_hz=16_000,
            target_channels=1,
            output_format="wav",
            output_pcm_bits_per_sample=16,
            peak_headroom_db=1.0,
            dc_offset_threshold=0.01,
            clipped_sample_threshold=0.999,
        )
    )
    loaded = load_audio(audio_path, project_root=tmp_path, request=request)

    assert loaded.source_format == "WAV"
    assert loaded.source_sample_rate_hz == 8_000
    assert loaded.sample_rate_hz == 16_000
    assert loaded.source_num_channels == 2
    assert loaded.num_channels == 1
    assert loaded.resampled is True
    assert loaded.downmixed is True
    assert loaded.waveform.shape == (1, 16_000)
    assert loaded.duration_seconds == pytest.approx(1.0, abs=1e-6)


def test_load_audio_reads_flac_window_without_loading_full_duration(tmp_path: Path) -> None:
    audio_path = tmp_path / "datasets" / "demo" / "long.flac"
    audio_path.parent.mkdir(parents=True)
    _write_tone_audio(audio_path, format_name="FLAC", sample_rate_hz=16_000, duration_seconds=4.0)

    loaded = load_audio(
        audio_path,
        project_root=tmp_path,
        request=AudioLoadRequest(
            target_sample_rate_hz=16_000,
            target_channels=1,
            start_seconds=1.25,
            duration_seconds=0.5,
        ),
    )

    assert loaded.source_format == "FLAC"
    assert loaded.resampled is False
    assert loaded.downmixed is False
    assert loaded.frame_count == 8_000
    assert loaded.duration_seconds == pytest.approx(0.5, abs=1e-6)
    assert loaded.source_duration_seconds == pytest.approx(4.0, abs=1e-6)


def test_load_audio_supports_mp3_inputs(tmp_path: Path) -> None:
    audio_path = tmp_path / "datasets" / "demo" / "sample.mp3"
    audio_path.parent.mkdir(parents=True)
    _write_tone_audio(audio_path, format_name="MP3", sample_rate_hz=16_000)

    loaded = load_audio(
        audio_path,
        project_root=tmp_path,
        request=AudioLoadRequest(target_sample_rate_hz=16_000, target_channels=1),
    )

    assert loaded.source_format == "MP3"
    assert loaded.sample_rate_hz == 16_000
    assert loaded.num_channels == 1
    assert loaded.frame_count == 16_000
    assert float(np.abs(loaded.waveform).max()) > 0.0


def test_manifest_audio_loading_uses_manifest_contract_and_line_numbers(tmp_path: Path) -> None:
    audio_root = tmp_path / "datasets" / "demo"
    manifest_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    wav_path = audio_root / "speaker-a.wav"
    flac_path = audio_root / "speaker-b.flac"
    _write_tone_audio(wav_path, format_name="WAV", sample_rate_hz=16_000)
    _write_tone_audio(flac_path, format_name="FLAC", sample_rate_hz=16_000)

    rows = [
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "demo",
            "source_dataset": "demo",
            "speaker_id": "speaker-a",
            "utterance_id": "utt-a",
            "audio_path": "datasets/demo/speaker-a.wav",
        },
        {
            "schema_version": "kryptonite.manifest.v1",
            "record_type": "utterance",
            "dataset": "demo",
            "source_dataset": "demo",
            "speaker_id": "speaker-b",
            "utterance_id": "utt-b",
            "audio_path": "datasets/demo/speaker-b.flac",
        },
    ]
    manifest_path = manifest_root / "train_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded_rows = list(
        iter_manifest_audio(
            "artifacts/manifests/demo/train_manifest.jsonl",
            project_root=tmp_path,
            request=AudioLoadRequest(target_sample_rate_hz=16_000, target_channels=1),
        )
    )

    assert [loaded.line_number for loaded in loaded_rows] == [1, 2]
    assert {loaded.row.utterance_id for loaded in loaded_rows} == {"utt-a", "utt-b"}
    assert all(
        loaded.manifest_path == "artifacts/manifests/demo/train_manifest.jsonl"
        for loaded in loaded_rows
    )

    direct = load_manifest_audio(rows[0], project_root=tmp_path)
    assert direct.row.audio_path == "datasets/demo/speaker-a.wav"
    assert direct.audio.source_format == "WAV"


def test_load_audio_rejects_offsets_past_end_of_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "datasets" / "demo" / "short.wav"
    audio_path.parent.mkdir(parents=True)
    _write_tone_audio(audio_path, format_name="WAV", sample_rate_hz=16_000)

    with pytest.raises(ValueError, match="past EOF"):
        load_audio(
            audio_path,
            project_root=tmp_path,
            request=AudioLoadRequest(start_seconds=2.0),
        )


def test_load_audio_can_trim_leading_and_trailing_silence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_vad_segments(
        monkeypatch,
        light=[{"start": 4_000, "end": 12_000}],
    )
    audio_path = tmp_path / "datasets" / "demo" / "trim-me.wav"
    audio_path.parent.mkdir(parents=True)
    _write_trim_candidate_audio(audio_path)

    request = AudioLoadRequest.from_config(
        NormalizationConfig(
            target_sample_rate_hz=16_000,
            target_channels=1,
            output_format="wav",
            output_pcm_bits_per_sample=16,
            peak_headroom_db=1.0,
            dc_offset_threshold=0.01,
            clipped_sample_threshold=0.999,
        ),
        vad=VADConfig(mode="light"),
    )
    loaded = load_audio(audio_path, project_root=tmp_path, request=request)

    assert loaded.vad_mode == "light"
    assert loaded.trim_applied is True
    assert loaded.trim_reason == "trimmed"
    assert loaded.vad_speech_detected is True
    assert loaded.duration_seconds < loaded.source_duration_seconds
    assert loaded.trimmed_leading_seconds > 0.0
    assert loaded.trimmed_trailing_seconds > 0.0


def test_aggressive_vad_trims_more_than_light(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_vad_segments(
        monkeypatch,
        light=[{"start": 4_000, "end": 12_000}],
        aggressive=[{"start": 5_500, "end": 10_500}],
    )
    audio_path = tmp_path / "datasets" / "demo" / "trim-hard.wav"
    audio_path.parent.mkdir(parents=True)
    _write_trim_candidate_audio(audio_path)

    base_config = NormalizationConfig(
        target_sample_rate_hz=16_000,
        target_channels=1,
        output_format="wav",
        output_pcm_bits_per_sample=16,
        peak_headroom_db=1.0,
        dc_offset_threshold=0.01,
        clipped_sample_threshold=0.999,
    )
    light = load_audio(
        audio_path,
        project_root=tmp_path,
        request=AudioLoadRequest.from_config(base_config, vad=VADConfig(mode="light")),
    )
    aggressive = load_audio(
        audio_path,
        project_root=tmp_path,
        request=AudioLoadRequest.from_config(base_config, vad=VADConfig(mode="aggressive")),
    )

    assert aggressive.duration_seconds < light.duration_seconds
    assert aggressive.trimmed_leading_seconds >= light.trimmed_leading_seconds
    assert aggressive.trimmed_trailing_seconds >= light.trimmed_trailing_seconds


def test_load_audio_keeps_silence_only_waveform_when_no_speech_is_detected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_vad_segments(monkeypatch, aggressive=[])
    audio_path = tmp_path / "datasets" / "demo" / "silence.wav"
    audio_path.parent.mkdir(parents=True)
    waveform = np.zeros(16_000, dtype=np.float32)
    sf.write(audio_path, waveform, 16_000, format="WAV")

    loaded = load_audio(
        audio_path,
        project_root=tmp_path,
        request=AudioLoadRequest(
            target_sample_rate_hz=16_000,
            target_channels=1,
            vad_mode="aggressive",
        ),
    )

    assert loaded.trim_applied is False
    assert loaded.trim_reason == "no_speech_detected"
    assert loaded.vad_speech_detected is False
    assert loaded.duration_seconds == pytest.approx(1.0, abs=1e-6)


def test_load_manifest_audio_rejects_rows_without_audio_path(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="audio_path"):
        load_manifest_audio(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "demo",
                "source_dataset": "demo",
                "speaker_id": "speaker-a",
            },
            project_root=tmp_path,
        )


def _write_tone_audio(
    path: Path,
    *,
    format_name: str,
    sample_rate_hz: int,
    channels: int = 1,
    duration_seconds: float = 1.0,
) -> None:
    frame_count = int(sample_rate_hz * duration_seconds)
    time = np.arange(frame_count, dtype=np.float32) / np.float32(sample_rate_hz)
    base = (0.3 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32, copy=False)
    if channels == 1:
        waveform = base
    else:
        waveform = np.stack(
            [
                base,
                (0.15 * np.sin(2.0 * np.pi * 330.0 * time)).astype(np.float32, copy=False),
            ],
            axis=1,
        )
    sf.write(path, waveform, sample_rate_hz, format=format_name)


def _write_trim_candidate_audio(path: Path) -> None:
    sample_rate_hz = 16_000
    silence = np.zeros(4_800, dtype=np.float32)
    time = np.arange(8_000, dtype=np.float32) / np.float32(sample_rate_hz)
    speech = (0.25 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32, copy=False)
    waveform = np.concatenate([silence, speech, silence])
    sf.write(path, waveform, sample_rate_hz, format="WAV")


def _patch_vad_segments(
    monkeypatch: pytest.MonkeyPatch,
    *,
    light: list[dict[str, int]] | None = None,
    aggressive: list[dict[str, int]] | None = None,
) -> None:
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
            return list(light or [])
        if settings.mode == "aggressive":
            return list(aggressive or [])
        return []

    monkeypatch.setattr(vad_module, "_detect_speech_segments", fake_detect)
