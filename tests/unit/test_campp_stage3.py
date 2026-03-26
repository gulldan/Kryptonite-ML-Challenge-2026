from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.training import load_campp_baseline_config, run_campp_baseline
from kryptonite.training.campp import (
    load_campp_stage2_config,
    load_campp_stage3_config,
    run_campp_stage2,
    run_campp_stage3,
)
from kryptonite.training.campp.finetune_common import margin_for_epoch


def test_margin_for_epoch_reaches_target_margin_after_ramp() -> None:
    margins = [
        margin_for_epoch(
            epoch,
            enabled=True,
            start_margin=0.3,
            end_margin=0.45,
            ramp_epochs=3,
        )
        for epoch in range(5)
    ]

    assert margins == [0.3, 0.375, 0.45, 0.45, 0.45]


def test_campp_stage3_smoke_run_writes_checkpoint_scores_and_schedule(tmp_path: Path) -> None:
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
    stage2_artifacts = run_campp_stage2(
        stage2_config,
        config_path=stage2_config_path,
        device_override="cpu",
    )

    stage3_config_path = _write_campp_stage3_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        stage2_checkpoint=Path(stage2_artifacts.output_root),
    )
    stage3_config = load_campp_stage3_config(
        config_path=stage3_config_path,
        env_file=tmp_path / ".env",
    )
    artifacts = run_campp_stage3(
        stage3_config,
        config_path=stage3_config_path,
        device_override="cpu",
    )

    assert Path(artifacts.checkpoint_path).is_file()
    assert Path(artifacts.embeddings_path).is_file()
    assert Path(artifacts.embedding_metadata_jsonl_path).is_file()
    assert Path(artifacts.embedding_metadata_parquet_path).is_file()
    assert Path(artifacts.scores_path).is_file()
    assert Path(artifacts.score_summary_path).is_file()
    assert Path(artifacts.report_path).is_file()
    cohort_summary_path = Path(artifacts.output_root) / "cohort_summary.json"
    assert cohort_summary_path.is_file()
    assert artifacts.training_summary.provenance_initialization == "pretrained"

    schedule_path = Path(artifacts.output_root) / "stage3_schedule.json"
    assert schedule_path.is_file()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert [row["margin"] for row in schedule["epochs"]] == [0.3, 0.45]
    assert [row["crop_seconds"] for row in schedule["epochs"]] == [0.5, 0.625]

    report_text = Path(artifacts.report_path).read_text(encoding="utf-8")
    assert "# CAM++ Stage-3 Report" in report_text
    assert "## Cohort Bank" in report_text
    assert "## Stage-3 Schedule" in report_text
    cohort_summary = json.loads(cohort_summary_path.read_text(encoding="utf-8"))
    assert cohort_summary["selected_row_count"] == 4
    assert cohort_summary["selected_speaker_count"] == 2
    assert cohort_summary["trial_overlap_fallback_used"] is True
    assert cohort_summary["overlapping_validation_speakers"] == []


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


def _write_campp_stage3_config(
    tmp_path: Path,
    *,
    train_manifest: Path,
    dev_manifest: Path,
    stage2_checkpoint: Path,
) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "campp-stage3.toml"
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
                "  'training.max_epochs=2',",
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
                'output_root = "artifacts/baselines/campp-stage3"',
                'trials_manifest = ""',
                'checkpoint_name = "campp_stage3_encoder.pt"',
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
                "margin = 0.3",
                "",
                "[optimization]",
                "learning_rate = 0.02",
                "min_learning_rate = 0.005",
                "weight_decay = 0.0",
                "warmup_epochs = 0",
                "grad_clip_norm = 5.0",
                "",
                "[provenance]",
                'ruleset = "standard"',
                'initialization = "pretrained"',
                'pretrained_resources = ["campp_stage2_encoder.pt"]',
                "",
                "[stage3]",
                f'stage2_checkpoint = "{stage2_checkpoint.as_posix()}"',
                "",
                "[stage3.hard_negative]",
                "enabled = false",
                "mining_interval_epochs = 1",
                "top_k_per_speaker = 1",
                "hard_negative_fraction = 0.25",
                "",
                "[stage3.crop_curriculum]",
                "enabled = true",
                "start_crop_seconds = 0.5",
                "end_crop_seconds = 0.75",
                "curriculum_epochs = 1",
                "",
                "[stage3.margin_schedule]",
                "enabled = true",
                "start_margin = 0.3",
                "end_margin = 0.45",
                "ramp_epochs = 2",
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
        ("speaker_charlie", "enrollment", "dev_c_enroll.wav", 241.0),
        ("speaker_charlie", "test", "dev_c_test.wav", 251.0),
        ("speaker_delta", "enrollment", "dev_d_enroll.wav", 361.0),
        ("speaker_delta", "test", "dev_d_test.wav", 371.0),
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
