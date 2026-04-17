from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from kryptonite.config import FeaturesConfig, NormalizationConfig
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    build_fbank_parity_report,
    extract_fbank,
)
from kryptonite.features.campp_official import official_campp_fbank


def test_extract_fbank_returns_expected_shape_and_dtype() -> None:
    waveform = _sine_wave(duration_seconds=1.0)

    features = extract_fbank(
        waveform,
        sample_rate_hz=16_000,
        request=FbankExtractionRequest(
            num_mel_bins=80,
            output_dtype="float16",
        ),
    )

    assert features.shape == (99, 80)
    assert features.dtype == torch.float16


def test_extract_fbank_official_campp_frontend_matches_official_helper() -> None:
    waveform = torch.sin(torch.linspace(0.0, 12.0, steps=16_000))
    request = FbankExtractionRequest(frontend="official_campp", num_mel_bins=80)

    features = FbankExtractor(request=request).extract(waveform, sample_rate_hz=16_000)
    expected = official_campp_fbank(waveform, sample_rate_hz=16_000, num_mel_bins=80)

    assert features.dtype == torch.float32
    assert torch.allclose(features, expected)
    assert torch.allclose(features.mean(dim=0), torch.zeros(80), atol=1e-5)


@pytest.mark.parametrize("cmvn_mode", ["none", "sliding"])
def test_online_fbank_matches_offline_for_unaligned_chunks(cmvn_mode: str) -> None:
    generator = torch.Generator().manual_seed(7)
    waveform = torch.randn(1, 30_157, generator=generator, dtype=torch.float32) * 0.05
    request = FbankExtractionRequest(
        num_mel_bins=80,
        cmvn_mode=cmvn_mode,
        cmvn_window_frames=64,
        output_dtype="float32",
    )
    extractor = FbankExtractor(request=request)

    offline = extractor.extract(waveform, sample_rate_hz=16_000)
    online_extractor = extractor.create_online_extractor()
    chunks = [
        online_extractor.push(waveform[:, start : start + 1_733], sample_rate_hz=16_000)
        for start in range(0, int(waveform.shape[-1]), 1_733)
    ]
    chunks.append(online_extractor.flush())
    online = torch.cat([chunk for chunk in chunks if chunk.numel() > 0], dim=0)

    assert online.shape == offline.shape
    assert online.dtype == offline.dtype
    assert torch.allclose(online, offline, atol=1e-5, rtol=0.0)


def test_online_fbank_does_not_emit_extra_frame_on_exact_frame_boundary() -> None:
    waveform = _sine_wave(duration_seconds=0.035).astype(np.float32)
    extractor = FbankExtractor(request=FbankExtractionRequest(output_dtype="float32"))

    offline = extractor.extract(waveform, sample_rate_hz=16_000)
    online_extractor = extractor.create_online_extractor()
    online = torch.cat(
        [
            online_extractor.push(waveform[:, :320], sample_rate_hz=16_000),
            online_extractor.push(waveform[:, 320:], sample_rate_hz=16_000),
            online_extractor.flush(),
        ],
        dim=0,
    )

    assert offline.shape == (2, 80)
    assert online.shape == offline.shape
    assert torch.allclose(online, offline, atol=1e-5, rtol=0.0)


def test_extract_fbank_rejects_multichannel_waveforms() -> None:
    waveform = torch.randn(2, 1_600, dtype=torch.float32)

    with pytest.raises(ValueError, match="mono waveform"):
        extract_fbank(waveform, sample_rate_hz=16_000)


def test_build_fbank_parity_report_passes_on_manifest_audio(tmp_path: Path) -> None:
    audio_root = tmp_path / "datasets" / "demo"
    manifest_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    audio_path = audio_root / "utterance.wav"
    _write_wave(audio_path, duration_seconds=1.3)
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

    report = build_fbank_parity_report(
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
        features=FeaturesConfig(
            sample_rate_hz=16_000,
            num_mel_bins=80,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            fft_size=512,
            window_type="hann",
            f_min_hz=20.0,
            cmvn_mode="sliding",
            cmvn_window_frames=32,
            output_dtype="float32",
        ),
        chunk_duration_ms=137.0,
    )

    assert report.summary.row_count == 1
    assert report.summary.frame_mismatch_row_count == 0
    assert report.summary.passed is True
    assert report.records[0].max_abs_diff is not None
    assert report.records[0].max_abs_diff <= report.summary.atol


def _sine_wave(*, duration_seconds: float, sample_rate_hz: int = 16_000) -> np.ndarray:
    sample_count = round(duration_seconds * sample_rate_hz)
    time = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
    return waveform.reshape(1, -1)


def _write_wave(path: Path, *, duration_seconds: float) -> None:
    sf.write(path, _sine_wave(duration_seconds=duration_seconds).reshape(-1), 16_000, format="WAV")
