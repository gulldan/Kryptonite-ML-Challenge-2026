from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import soundfile as sf

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.data import AudioLoadRequest
from kryptonite.data.schema import ManifestRow
from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest
from kryptonite.training.augmentation_runtime import TrainingAugmentationRuntime
from kryptonite.training.augmentation_scheduler import ScheduledAugmentation
from kryptonite.training.manifest_speaker_data import (
    ManifestSpeakerDataset,
    TrainingSampleRequest,
    build_speaker_index,
    collate_training_examples,
    load_manifest_rows,
)
from kryptonite.training.production_dataloader import (
    BalancedSpeakerBatchSampler,
    build_production_train_dataloader,
)


def test_balanced_speaker_batch_sampler_is_resumable_and_balanced() -> None:
    rows = [
        _manifest_row("speaker_alpha", "alpha_0.wav"),
        _manifest_row("speaker_alpha", "alpha_1.wav"),
        _manifest_row("speaker_bravo", "bravo_0.wav"),
        _manifest_row("speaker_bravo", "bravo_1.wav"),
        _manifest_row("speaker_charlie", "charlie_0.wav"),
        _manifest_row("speaker_charlie", "charlie_1.wav"),
    ]
    sampler = BalancedSpeakerBatchSampler(
        rows=rows,
        batch_size=3,
        seed=17,
        chunking_request=UtteranceChunkingRequest(
            train_min_crop_seconds=1.0,
            train_max_crop_seconds=2.0,
            train_num_crops=1,
        ),
    )
    sampler.set_epoch(0)
    iterator = iter(sampler)
    first_batch = next(iterator)
    state = sampler.state_dict()
    remaining_batches = list(iterator)

    resumed = BalancedSpeakerBatchSampler(
        rows=rows,
        batch_size=3,
        seed=17,
        chunking_request=UtteranceChunkingRequest(
            train_min_crop_seconds=1.0,
            train_max_crop_seconds=2.0,
            train_num_crops=1,
        ),
    )
    resumed.set_epoch(0)
    resumed.load_state_dict(state)

    assert list(iter(resumed)) == remaining_batches
    assert len(first_batch) == 3
    assert len({rows[item.row_index].speaker_id for item in first_batch}) == 3
    assert len({item.crop_seconds for item in first_batch}) == 1


def test_manifest_speaker_dataset_supports_mixed_clean_and_corrupted_requests(
    tmp_path: Path,
) -> None:
    train_manifest = _write_train_manifest(tmp_path)
    _write_noise_bank(tmp_path)
    config = _load_test_config(
        tmp_path,
        overrides=[
            "runtime.num_workers=0",
            "features.num_mel_bins=16",
            "chunking.train_min_crop_seconds=0.5",
            "chunking.train_max_crop_seconds=0.5",
            "chunking.train_num_crops=1",
        ],
    )
    rows = load_manifest_rows(train_manifest, project_root=tmp_path)
    speaker_to_index = build_speaker_index(rows)
    runtime = TrainingAugmentationRuntime.from_project_config(
        project_root=tmp_path,
        scheduler_config=config.augmentation_scheduler,
        silence_config=config.silence_augmentation,
        total_epochs=1,
    )
    dataset = ManifestSpeakerDataset(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project_root=tmp_path,
        audio_request=AudioLoadRequest.from_config(config.normalization, vad=config.vad),
        feature_request=FbankExtractionRequest.from_config(config.features),
        chunking_request=UtteranceChunkingRequest.from_config(config.chunking),
        seed=config.runtime.seed,
        augmentation_runtime=runtime,
    )
    candidate_id = next(iter(runtime.noise_candidates))
    clean_example = dataset[
        TrainingSampleRequest(
            row_index=0,
            request_seed=11,
            crop_seconds=0.5,
        )
    ]
    corrupted_example = dataset[
        TrainingSampleRequest(
            row_index=1,
            request_seed=12,
            crop_seconds=0.5,
            clean_sample=False,
            recipe_stage="warmup",
            recipe_intensity="light",
            augmentations=(
                ScheduledAugmentation(
                    family="noise",
                    candidate_id=candidate_id,
                    label="noise/stationary/light",
                    severity="light",
                    metadata={},
                ),
            ),
        )
    ]
    batch = collate_training_examples([clean_example, corrupted_example])

    assert batch.features.shape[0] == 2
    assert batch.features.shape[-1] == 16
    assert batch.clean_sample_mask.tolist() == [True, False]
    assert batch.recipe_intensities == ("clean", "light")
    assert batch.recipe_stages == ("steady", "warmup")
    assert batch.augmentation_traces[0] == ()
    assert len(batch.augmentation_traces[1]) == 1
    metadata = cast(dict[str, object], batch.augmentation_traces[1][0]["metadata"])
    assert "target_snr_db" in metadata


def test_build_production_train_dataloader_supports_variable_crop_and_scheduler(
    tmp_path: Path,
) -> None:
    train_manifest = _write_train_manifest(tmp_path)
    _write_noise_bank(tmp_path)
    config = _load_test_config(
        tmp_path,
        overrides=[
            "runtime.num_workers=0",
            "training.batch_size=2",
            "training.max_epochs=1",
            "features.num_mel_bins=16",
            "chunking.train_min_crop_seconds=0.5",
            "chunking.train_max_crop_seconds=0.75",
            "chunking.train_num_crops=1",
            "augmentation_scheduler.clean_probability_start=0.0",
            "augmentation_scheduler.clean_probability_end=0.0",
            "augmentation_scheduler.light_probability_start=1.0",
            "augmentation_scheduler.light_probability_end=1.0",
            "augmentation_scheduler.medium_probability_start=0.0",
            "augmentation_scheduler.medium_probability_end=0.0",
            "augmentation_scheduler.heavy_probability_start=0.0",
            "augmentation_scheduler.heavy_probability_end=0.0",
        ],
    )
    rows = load_manifest_rows(train_manifest, project_root=tmp_path)
    speaker_to_index = build_speaker_index(rows)
    dataset, sampler, loader = build_production_train_dataloader(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project=config,
        total_epochs=1,
        pin_memory=False,
    )
    dataset.set_epoch(0)
    sampler.set_epoch(0)
    batch = next(iter(loader))

    assert batch.features.shape[0] == 2
    assert batch.features.shape[-1] == 16
    assert batch.clean_sample_mask.tolist() == [False, False]
    assert all(stage == "warmup" for stage in batch.recipe_stages)
    assert all(intensity == "light" for intensity in batch.recipe_intensities)
    assert len(set(batch.crop_seconds)) == 1
    assert all(batch.augmentation_traces)


def _load_test_config(tmp_path: Path, *, overrides: list[str]) -> ProjectConfig:
    return load_project_config(
        config_path=Path("configs/base.toml"),
        overrides=[
            f'paths.project_root="{tmp_path.as_posix()}"',
            "tracking.enabled=false",
            *overrides,
        ],
    )


def _write_train_manifest(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "datasets" / "fixture"
    manifest_root = tmp_path / "artifacts" / "manifests"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    specs = [
        ("speaker_alpha", "train_a_0.wav", 220.0),
        ("speaker_alpha", "train_a_1.wav", 233.0),
        ("speaker_bravo", "train_b_0.wav", 330.0),
        ("speaker_bravo", "train_b_1.wav", 347.0),
    ]
    for speaker_id, file_name, frequency in specs:
        _write_tone(dataset_root / file_name, frequency_hz=frequency, duration_seconds=1.0)
        rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "fixture",
                "source_dataset": "fixture",
                "speaker_id": speaker_id,
                "utterance_id": f"{speaker_id}:{Path(file_name).stem}",
                "split": "train",
                "audio_path": f"datasets/fixture/{file_name}",
                "channel": "mono",
            }
        )

    manifest_path = manifest_root / "train_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest_path


def _write_noise_bank(tmp_path: Path) -> None:
    audio_root = tmp_path / "artifacts" / "corruptions" / "noise-bank" / "audio"
    manifest_root = tmp_path / "artifacts" / "corruptions" / "noise-bank" / "manifests"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    noise_path = audio_root / "stationary.wav"
    _write_noise(noise_path, sample_rate_hz=16_000, duration_seconds=2.0)
    manifest_root.joinpath("noise_bank_manifest.jsonl").write_text(
        json.dumps(
            {
                "noise_id": "noise_stationary_light",
                "normalized_audio_path": "artifacts/corruptions/noise-bank/audio/stationary.wav",
                "severity": "light",
                "sampling_weight": 1.0,
                "category": "stationary",
                "mix_mode": "additive",
                "recommended_snr_db_min": 8.0,
                "recommended_snr_db_max": 12.0,
                "tags": ["fixture"],
            }
        )
        + "\n"
    )


def _manifest_row(speaker_id: str, file_name: str) -> ManifestRow:
    return ManifestRow(
        dataset="fixture",
        source_dataset="fixture",
        speaker_id=speaker_id,
        audio_path=f"datasets/fixture/{file_name}",
        utterance_id=f"{speaker_id}:{Path(file_name).stem}",
    )


def _write_tone(
    path: Path,
    *,
    frequency_hz: float,
    sample_rate_hz: int = 16_000,
    duration_seconds: float,
) -> None:
    sample_count = int(sample_rate_hz * duration_seconds)
    timeline = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = 0.3 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform, sample_rate_hz, format="WAV")


def _write_noise(
    path: Path,
    *,
    sample_rate_hz: int,
    duration_seconds: float,
) -> None:
    sample_count = int(sample_rate_hz * duration_seconds)
    rng = np.random.default_rng(123)
    waveform = 0.1 * rng.standard_normal(sample_count, dtype=np.float32)
    sf.write(path, waveform, sample_rate_hz, format="WAV")
