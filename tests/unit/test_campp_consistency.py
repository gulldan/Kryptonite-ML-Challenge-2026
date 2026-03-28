from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.training import load_campp_baseline_config, run_campp_baseline
from kryptonite.training.campp import (
    load_campp_consistency_config,
    load_campp_stage2_config,
    load_campp_stage3_config,
    run_campp_consistency,
    run_campp_stage2,
    run_campp_stage3,
)


def test_campp_consistency_smoke_run_writes_ablation_and_report(tmp_path: Path) -> None:
    train_manifest, dev_manifest = _write_manifest_fixtures(tmp_path)
    catalog_path = _write_corrupted_suite_catalog(
        tmp_path,
        dev_manifest=dev_manifest,
    )
    _write_noise_bank_fixture(tmp_path)

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
    stage3_artifacts = run_campp_stage3(
        stage3_config,
        config_path=stage3_config_path,
        device_override="cpu",
    )

    consistency_config_path = _write_campp_consistency_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        student_checkpoint=Path(stage3_artifacts.output_root),
        base_stage3_config=stage3_config_path,
        catalog_path=catalog_path,
    )
    consistency_config = load_campp_consistency_config(
        config_path=consistency_config_path,
        env_file=tmp_path / ".env",
    )
    artifacts = run_campp_consistency(
        consistency_config,
        config_path=consistency_config_path,
        device_override="cpu",
    )

    assert Path(artifacts.checkpoint_path).is_file()
    assert Path(artifacts.consistency_summary_path).is_file()
    assert Path(artifacts.comparison_json_path).is_file()
    assert Path(artifacts.comparison_markdown_path).is_file()
    assert Path(artifacts.report_path).is_file()
    assert Path(artifacts.embeddings_path).is_file()
    assert artifacts.robust_dev_ablation_json_path is not None
    assert artifacts.robust_dev_ablation_markdown_path is not None
    assert Path(artifacts.robust_dev_ablation_json_path).is_file()
    assert Path(artifacts.robust_dev_ablation_markdown_path).is_file()
    assert artifacts.consistency_epochs[-1].paired_ratio > 0.0
    assert (
        artifacts.comparison.baseline_checkpoint_path
        == Path(stage3_artifacts.output_root).as_posix()
    )

    comparison_payload = json.loads(
        Path(artifacts.comparison_json_path).read_text(encoding="utf-8")
    )
    assert (
        comparison_payload["baseline_checkpoint_path"]
        == Path(stage3_artifacts.output_root).as_posix()
    )
    assert "eer_delta" in comparison_payload

    ablation_payload = json.loads(
        Path(artifacts.robust_dev_ablation_json_path).read_text(encoding="utf-8")
    )
    assert ablation_payload["baseline"]["candidate_id"] == "stage3_baseline"
    assert ablation_payload["consistency"]["candidate_id"] == "campp_consistency"
    assert ablation_payload["summary"]["corrupted_suite_ids"] == ["fixture_noise"]

    report_text = Path(artifacts.report_path).read_text(encoding="utf-8")
    assert "# CAM++ Consistency Report" in report_text
    assert "## Consistency Setup" in report_text
    assert "## Robust-Dev Ablation" in report_text

    ablation_text = Path(artifacts.robust_dev_ablation_markdown_path).read_text(encoding="utf-8")
    assert "CAM++ Consistency Robust-Dev Ablation" in ablation_text
    assert "Per-Suite Deltas" in ablation_text


def _write_campp_consistency_config(
    tmp_path: Path,
    *,
    train_manifest: Path,
    dev_manifest: Path,
    student_checkpoint: Path,
    base_stage3_config: Path,
    catalog_path: Path,
) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "campp-consistency.toml"
    config_path.write_text(
        "\n".join(
            [
                f'base_stage3_config = "{base_stage3_config.as_posix()}"',
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
                "  'augmentation_scheduler.enabled=true',",
                "  'augmentation_scheduler.max_augmentations_per_sample=1',",
                "  'augmentation_scheduler.clean_probability_start=0.0',",
                "  'augmentation_scheduler.clean_probability_end=0.0',",
                "  'augmentation_scheduler.light_probability_start=1.0',",
                "  'augmentation_scheduler.light_probability_end=1.0',",
                "  'augmentation_scheduler.medium_probability_start=0.0',",
                "  'augmentation_scheduler.medium_probability_end=0.0',",
                "  'augmentation_scheduler.heavy_probability_start=0.0',",
                "  'augmentation_scheduler.heavy_probability_end=0.0',",
                "  'silence_augmentation.enabled=false',",
                "]",
                "",
                "[data]",
                f'train_manifest = "{train_manifest.as_posix()}"',
                f'dev_manifest = "{dev_manifest.as_posix()}"',
                'output_root = "artifacts/baselines/campp-consistency"',
                'trials_manifest = ""',
                'checkpoint_name = "campp_consistency_encoder.pt"',
                "generate_demo_artifacts_if_missing = false",
                "",
                "[optimization]",
                'optimizer_name = "adamw"',
                'scheduler_name = "cosine"',
                "learning_rate = 0.0005",
                "min_learning_rate = 0.0001",
                "weight_decay = 0.0",
                "warmup_epochs = 0",
                "gradient_accumulation_steps = 1",
                "grad_clip_norm = 5.0",
                "",
                "[student]",
                f'checkpoint = "{student_checkpoint.as_posix()}"',
                "",
                "[consistency]",
                "clean_classification_weight = 1.0",
                "corrupted_classification_weight = 0.5",
                "embedding_weight = 0.2",
                "score_weight = 0.1",
                "",
                "[robust_dev]",
                "enabled = true",
                f'catalog_path = "{catalog_path.as_posix()}"',
                "clean_weight = 0.25",
                "corrupted_weight = 0.75",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


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
            }
        )

    train_manifest = manifest_root / "train_manifest.jsonl"
    dev_manifest = manifest_root / "dev_manifest.jsonl"
    train_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    dev_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in dev_rows),
        encoding="utf-8",
    )
    return train_manifest, dev_manifest


def _write_noise_bank_fixture(tmp_path: Path) -> None:
    audio_root = tmp_path / "artifacts" / "corruptions" / "noise-bank" / "audio"
    manifest_root = tmp_path / "artifacts" / "corruptions" / "noise-bank" / "manifests"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    noise_path = audio_root / "fixture_noise.wav"
    _write_noise(noise_path)
    manifest_path = manifest_root / "noise_bank_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "noise_id": "fixture_noise",
                "normalized_audio_path": "artifacts/corruptions/noise-bank/audio/fixture_noise.wav",
                "severity": "light",
                "sampling_weight": 1.0,
                "category": "stationary",
                "mix_mode": "additive",
                "recommended_snr_db_min": 8.0,
                "recommended_snr_db_max": 8.0,
                "tags": ["fixture"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_corrupted_suite_catalog(tmp_path: Path, *, dev_manifest: Path) -> Path:
    catalog_root = tmp_path / "artifacts" / "eval" / "corrupted-dev-suites"
    catalog_root.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_root / "corrupted_dev_suites_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "suite_id": "fixture_noise",
                        "family": "noise",
                        "description": "Fixture corrupted-dev suite",
                        "manifest_path": dev_manifest.as_posix(),
                        "trial_manifest_paths": [],
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return catalog_path


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


def _write_noise(
    path: Path,
    *,
    sample_rate_hz: int = 16_000,
    duration_seconds: float = 1.2,
) -> None:
    sample_count = int(sample_rate_hz * duration_seconds)
    rng = np.random.default_rng(7)
    waveform = 0.05 * rng.standard_normal(sample_count).astype(np.float32)
    sf.write(path, waveform, sample_rate_hz, format="WAV")
