from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from kryptonite.config import FeatureCacheConfig, FeaturesConfig, NormalizationConfig
from kryptonite.data import AudioLoadRequest, iter_manifest_audio, load_audio
from kryptonite.features import (
    FbankExtractionRequest,
    FeatureCacheSettings,
    FeatureCacheStore,
    build_feature_cache_benchmark_report,
    materialize_feature_cache,
    write_feature_cache_benchmark_report,
)


def test_feature_cache_key_changes_when_source_or_request_changes(tmp_path: Path) -> None:
    audio_path = tmp_path / "datasets" / "demo" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    _write_wave(audio_path, duration_seconds=1.0)

    request = AudioLoadRequest.from_config(_normalization_config())
    loaded = load_audio(audio_path, project_root=tmp_path, request=request)
    store = FeatureCacheStore(
        root=tmp_path / "artifacts" / "cache",
        settings=FeatureCacheSettings(namespace="unit-test-cache"),
    )
    feature_request = FbankExtractionRequest(output_dtype="float32")
    first_key = store.build_key(loaded_audio=loaded, request=feature_request)

    new_mtime = Path(audio_path).stat().st_mtime_ns + 1_000_000
    os.utime(audio_path, ns=(new_mtime, new_mtime))
    loaded_after_touch = load_audio(audio_path, project_root=tmp_path, request=request)
    touched_key = store.build_key(loaded_audio=loaded_after_touch, request=feature_request)
    changed_request_key = store.build_key(
        loaded_audio=loaded_after_touch,
        request=FbankExtractionRequest(output_dtype="float16"),
    )

    assert first_key.cache_id != touched_key.cache_id
    assert touched_key.cache_id != changed_request_key.cache_id


def test_materialize_feature_cache_writes_then_hits_existing_entries(tmp_path: Path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path, durations=[1.0])
    request = AudioLoadRequest.from_config(_normalization_config())
    samples = list(iter_manifest_audio(manifest_path, project_root=tmp_path, request=request))
    store = FeatureCacheStore(
        root=tmp_path / "artifacts" / "cache",
        settings=FeatureCacheSettings(namespace="materialize-test"),
    )
    feature_request = FbankExtractionRequest(output_dtype="float32")

    first_report = materialize_feature_cache(samples, store=store, request=feature_request)
    second_report = materialize_feature_cache(samples, store=store, request=feature_request)
    key = store.build_key(loaded_audio=samples[0].audio, request=feature_request)
    cached = store.load(key)

    assert first_report.summary.row_count == 1
    assert first_report.summary.cache_write_count == 1
    assert first_report.summary.cache_hit_count == 0
    assert second_report.summary.cache_write_count == 0
    assert second_report.summary.cache_hit_count == 1
    assert tuple(cached.shape) == (99, 80)
    assert cached.dtype == torch.float32


def test_build_feature_cache_benchmark_report_on_cpu(tmp_path: Path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path, durations=[1.0, 1.3])

    report = build_feature_cache_benchmark_report(
        project_root=tmp_path,
        cache_root="artifacts/cache",
        manifest_path=manifest_path,
        normalization=_normalization_config(),
        features=_features_config(),
        feature_cache=FeatureCacheConfig(
            namespace="benchmark-test",
            benchmark_device="cpu",
            benchmark_warmup_iterations=0,
            benchmark_iterations=1,
        ),
    )
    written = write_feature_cache_benchmark_report(
        report=report,
        output_root=tmp_path / "artifacts" / "eda" / "feature-cache",
    )

    scenario_names = {scenario.name for scenario in report.benchmarks}

    assert report.selected_device == "cpu"
    assert report.materialization.summary.row_count == 2
    assert scenario_names == {
        "cpu_precompute_write",
        "cpu_runtime_extract",
        "cpu_cache_read",
    }
    assert report.policy.train_policy == "precompute_cpu"
    assert Path(written.json_path).is_file()
    assert Path(written.markdown_path).is_file()
    assert Path(written.rows_path).is_file()


def _write_manifest_fixture(tmp_path: Path, *, durations: list[float]) -> Path:
    audio_root = tmp_path / "datasets" / "demo"
    manifest_root = tmp_path / "artifacts" / "manifests" / "demo"
    audio_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    rows: list[dict[str, str]] = []
    for index, duration in enumerate(durations, start=1):
        audio_path = audio_root / f"utterance-{index}.wav"
        _write_wave(audio_path, duration_seconds=duration)
        rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "demo",
                "source_dataset": "demo",
                "speaker_id": f"speaker-{index}",
                "utterance_id": f"utt-{index}",
                "audio_path": f"datasets/demo/{audio_path.name}",
            }
        )

    manifest_path = manifest_root / "dev_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest_path


def _normalization_config() -> NormalizationConfig:
    return NormalizationConfig(
        target_sample_rate_hz=16_000,
        target_channels=1,
        output_format="wav",
        output_pcm_bits_per_sample=16,
        peak_headroom_db=1.0,
        dc_offset_threshold=0.01,
        clipped_sample_threshold=0.999,
    )


def _features_config() -> FeaturesConfig:
    return FeaturesConfig(
        sample_rate_hz=16_000,
        num_mel_bins=80,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        fft_size=512,
        window_type="hann",
        f_min_hz=20.0,
        cmvn_mode="none",
        output_dtype="float32",
    )


def _write_wave(path: Path, *, duration_seconds: float, sample_rate_hz: int = 16_000) -> None:
    sample_count = round(duration_seconds * sample_rate_hz)
    time = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
    sf.write(path, waveform, sample_rate_hz, format="WAV")
