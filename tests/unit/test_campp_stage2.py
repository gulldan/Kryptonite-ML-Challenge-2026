from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.data import ManifestRow
from kryptonite.features import UtteranceChunkingRequest
from kryptonite.training import load_campp_baseline_config, run_campp_baseline
from kryptonite.training.campp import load_campp_stage2_config, run_campp_stage2
from kryptonite.training.campp.stage2_config import Stage2UtteranceCurriculumConfig
from kryptonite.training.campp.stage2_pipeline import _phase_for_epoch
from kryptonite.training.campp.stage2_sampler import Stage2BatchSampler


def test_stage2_sampler_hard_negative_fraction_increases_boosted_speaker_frequency() -> None:
    rows = _speaker_rows(speaker_count=8, rows_per_speaker=3)
    chunking_request = _fixed_chunking_request()

    uniform_sampler = Stage2BatchSampler(
        rows=rows,
        batch_size=4,
        seed=7,
        chunking_request=chunking_request,
        hard_negative_fraction=0.0,
        batches_per_epoch=24,
    )
    uniform_sampler.update_speaker_weights({"speaker_00": 16.0})
    uniform_counts = _count_speaker_occurrences(uniform_sampler, rows=rows, epoch=0)

    biased_sampler = Stage2BatchSampler(
        rows=rows,
        batch_size=4,
        seed=7,
        chunking_request=chunking_request,
        hard_negative_fraction=1.0,
        batches_per_epoch=24,
    )
    biased_sampler.update_speaker_weights({"speaker_00": 16.0})
    biased_counts = _count_speaker_occurrences(biased_sampler, rows=rows, epoch=0)

    assert biased_counts["speaker_00"] > uniform_counts["speaker_00"]


def test_stage2_sampler_resets_missing_speaker_weights_to_uniform() -> None:
    rows = _speaker_rows(speaker_count=8, rows_per_speaker=3)
    sampler = Stage2BatchSampler(
        rows=rows,
        batch_size=4,
        seed=17,
        chunking_request=_fixed_chunking_request(),
        hard_negative_fraction=1.0,
        batches_per_epoch=24,
    )
    sampler.update_speaker_weights({"speaker_00": 16.0})
    boosted_counts = _count_speaker_occurrences(sampler, rows=rows, epoch=0)

    sampler.update_speaker_weights({})
    reset_counts = _count_speaker_occurrences(sampler, rows=rows, epoch=1)

    assert boosted_counts["speaker_00"] > reset_counts["speaker_00"]


def test_phase_for_epoch_respects_curriculum_epochs() -> None:
    curriculum = Stage2UtteranceCurriculumConfig(
        enabled=True,
        short_crop_seconds=1.5,
        long_crop_seconds=4.0,
        curriculum_epochs=2,
    )

    assert [_phase_for_epoch(epoch, curriculum=curriculum, n_phases=3) for epoch in range(7)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        2,
    ]


def test_campp_stage2_smoke_run_writes_checkpoint_scores_and_hard_negative_log(
    tmp_path: Path,
) -> None:
    train_manifest, dev_manifest = _write_manifest_fixtures(tmp_path)
    stage1_config_path = _write_campp_stage1_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
    )
    stage1_config = load_campp_baseline_config(
        config_path=stage1_config_path,
        env_file=tmp_path / ".env",
    )
    stage1_artifacts = run_campp_baseline(
        stage1_config,
        config_path=stage1_config_path,
        device_override="cpu",
    )

    stage2_config_path = _write_campp_stage2_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        stage1_checkpoint=Path(stage1_artifacts.output_root),
    )
    stage2_config = load_campp_stage2_config(
        config_path=stage2_config_path,
        env_file=tmp_path / ".env",
    )
    artifacts = run_campp_stage2(
        stage2_config, config_path=stage2_config_path, device_override="cpu"
    )

    assert Path(artifacts.checkpoint_path).is_file()
    assert Path(artifacts.embeddings_path).is_file()
    assert Path(artifacts.embedding_metadata_jsonl_path).is_file()
    assert Path(artifacts.embedding_metadata_parquet_path).is_file()
    assert Path(artifacts.scores_path).is_file()
    assert Path(artifacts.score_summary_path).is_file()
    assert Path(artifacts.report_path).is_file()
    assert artifacts.training_summary.provenance_initialization == "pretrained"

    hard_negative_log_path = Path(artifacts.output_root) / "hard_negative_mining_log.jsonl"
    assert hard_negative_log_path.is_file()
    hard_negative_rows = _read_jsonl(hard_negative_log_path)
    assert hard_negative_rows
    assert hard_negative_rows[0]["status"] == "ok"
    speakers_mined = hard_negative_rows[0]["speakers_mined"]
    assert isinstance(speakers_mined, int)
    assert speakers_mined >= 2

    report_text = Path(artifacts.report_path).read_text(encoding="utf-8")
    assert "# CAM++ Stage-2 Report" in report_text
    assert "## Verification Eval" in report_text


def _speaker_rows(*, speaker_count: int, rows_per_speaker: int) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for speaker_index in range(speaker_count):
        speaker_id = f"speaker_{speaker_index:02d}"
        for row_index in range(rows_per_speaker):
            rows.append(
                ManifestRow(
                    dataset="fixture",
                    source_dataset="fixture",
                    speaker_id=speaker_id,
                    audio_path=f"datasets/fixture/{speaker_id}_{row_index}.wav",
                    utterance_id=f"{speaker_id}:{row_index}",
                )
            )
    return rows


def _fixed_chunking_request() -> UtteranceChunkingRequest:
    return UtteranceChunkingRequest(
        train_min_crop_seconds=1.0,
        train_max_crop_seconds=1.0,
        train_num_crops=1,
    )


def _count_speaker_occurrences(
    sampler: Stage2BatchSampler,
    *,
    rows: list[ManifestRow],
    epoch: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    sampler.set_epoch(epoch)
    for batch in sampler:
        for request in batch:
            counts[rows[request.row_index].speaker_id] += 1
    return counts


def _write_campp_stage1_config(tmp_path: Path, *, train_manifest: Path, dev_manifest: Path) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "campp-stage1.toml"
    config_path.write_text(
        "\n".join(
            [
                f'base_config = "{Path("configs/base.toml").resolve().as_posix()}"',
                "project_overrides = [",
                f"  'paths.project_root=\"{tmp_path.as_posix()}\"',",
                "  'tracking.enabled=false',",
                "  'runtime.num_workers=0',",
                "  'training.batch_size=2',",
                "  'training.eval_batch_size=2',",
                "  'training.max_epochs=1',",
                "  'chunking.train_min_crop_seconds=0.5',",
                "  'chunking.train_max_crop_seconds=0.5',",
                "  'chunking.train_num_crops=1',",
                "  'features.num_mel_bins=16',",
                "]",
                "",
                "[data]",
                f'train_manifest = "{train_manifest.as_posix()}"',
                f'dev_manifest = "{dev_manifest.as_posix()}"',
                'output_root = "artifacts/baselines/campp"',
                'trials_manifest = ""',
                'checkpoint_name = "campp_encoder.pt"',
                "generate_demo_artifacts_if_missing = false",
                "",
                "[model]",
                "feat_dim = 16",
                "embedding_size = 32",
                "growth_rate = 8",
                "bottleneck_scale = 2",
                "init_channels = 16",
                "head_channels = 8",
                "head_res_blocks = [1, 1]",
                "block_layers = [2, 2, 2]",
                "block_kernel_sizes = [3, 3, 3]",
                "block_dilations = [1, 1, 2]",
                "memory_efficient = false",
                "",
                "[objective]",
                "classifier_hidden_dim = 16",
                "scale = 16.0",
                "margin = 0.2",
                "",
                "[optimization]",
                "learning_rate = 0.05",
                "min_learning_rate = 0.01",
                "weight_decay = 0.0",
                "warmup_epochs = 0",
                "grad_clip_norm = 5.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_campp_stage2_config(
    tmp_path: Path,
    *,
    train_manifest: Path,
    dev_manifest: Path,
    stage1_checkpoint: Path,
) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "campp-stage2.toml"
    config_path.write_text(
        "\n".join(
            [
                f'base_config = "{Path("configs/base.toml").resolve().as_posix()}"',
                "project_overrides = [",
                f"  'paths.project_root=\"{tmp_path.as_posix()}\"',",
                "  'tracking.enabled=false',",
                "  'runtime.num_workers=0',",
                "  'training.batch_size=2',",
                "  'training.eval_batch_size=2',",
                "  'training.max_epochs=1',",
                "  'chunking.train_min_crop_seconds=0.5',",
                "  'chunking.train_max_crop_seconds=0.5',",
                "  'chunking.train_num_crops=1',",
                "  'features.num_mel_bins=16',",
                "  'augmentation_scheduler.enabled=false',",
                "  'silence_augmentation.enabled=false',",
                "]",
                "",
                "[data]",
                f'train_manifest = "{train_manifest.as_posix()}"',
                f'dev_manifest = "{dev_manifest.as_posix()}"',
                'output_root = "artifacts/baselines/campp-stage2"',
                'trials_manifest = ""',
                'checkpoint_name = "campp_stage2_encoder.pt"',
                "generate_demo_artifacts_if_missing = false",
                "",
                "[model]",
                "feat_dim = 16",
                "embedding_size = 32",
                "growth_rate = 8",
                "bottleneck_scale = 2",
                "init_channels = 16",
                "head_channels = 8",
                "head_res_blocks = [1, 1]",
                "block_layers = [2, 2, 2]",
                "block_kernel_sizes = [3, 3, 3]",
                "block_dilations = [1, 1, 2]",
                "memory_efficient = false",
                "",
                "[objective]",
                "classifier_hidden_dim = 16",
                "scale = 16.0",
                "margin = 0.2",
                "",
                "[optimization]",
                "learning_rate = 0.05",
                "min_learning_rate = 0.01",
                "weight_decay = 0.0",
                "warmup_epochs = 0",
                "grad_clip_norm = 5.0",
                "",
                "[provenance]",
                'ruleset = "standard"',
                'initialization = "pretrained"',
                'pretrained_resources = ["campp_encoder.pt"]',
                "",
                "[stage2]",
                f'stage1_checkpoint = "{stage1_checkpoint.as_posix()}"',
                "",
                "[stage2.hard_negative]",
                "enabled = true",
                "mining_interval_epochs = 1",
                "top_k_per_speaker = 1",
                "hard_negative_fraction = 0.5",
                "",
                "[stage2.utterance_curriculum]",
                "enabled = true",
                "short_crop_seconds = 0.5",
                "long_crop_seconds = 0.5",
                "curriculum_epochs = 1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_manifest_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "datasets" / "fixture"
    manifest_root = tmp_path / "artifacts" / "manifests"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, object]] = []
    dev_rows: list[dict[str, object]] = []
    train_specs = [
        ("speaker_alpha", "train_a_0.wav", 220.0),
        ("speaker_alpha", "train_a_1.wav", 233.0),
        ("speaker_bravo", "train_b_0.wav", 330.0),
        ("speaker_bravo", "train_b_1.wav", 347.0),
    ]
    for speaker_id, file_name, frequency in train_specs:
        _write_tone(dataset_root / file_name, frequency_hz=frequency)
        train_rows.append(
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

    dev_specs = [
        ("speaker_alpha", "enrollment", "dev_a_enroll.wav", 241.0),
        ("speaker_alpha", "test", "dev_a_test.wav", 251.0),
        ("speaker_bravo", "enrollment", "dev_b_enroll.wav", 361.0),
        ("speaker_bravo", "test", "dev_b_test.wav", 371.0),
    ]
    for speaker_id, role, file_name, frequency in dev_specs:
        _write_tone(dataset_root / file_name, frequency_hz=frequency)
        dev_rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "fixture",
                "source_dataset": "fixture",
                "speaker_id": speaker_id,
                "utterance_id": f"{speaker_id}:{Path(file_name).stem}",
                "split": "dev",
                "role": role,
                "audio_path": f"datasets/fixture/{file_name}",
                "channel": "mono",
                "corruption_suite": "dev_snr",
                "corruption_family": "noise",
                "corruption_severity": "light",
                "corruption_metadata": {"corruption_category": "stationary"},
            }
        )

    train_manifest = manifest_root / "train_manifest.jsonl"
    dev_manifest = manifest_root / "dev_manifest.jsonl"
    train_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8"
    )
    dev_manifest.write_text("".join(json.dumps(row) + "\n" for row in dev_rows), encoding="utf-8")
    return train_manifest, dev_manifest


def _write_tone(
    path: Path,
    *,
    frequency_hz: float,
    sample_rate_hz: int = 16_000,
    duration_seconds: float = 1.2,
) -> None:
    sample_count = int(sample_rate_hz * duration_seconds)
    timeline = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = 0.3 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform, sample_rate_hz, format="WAV")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
