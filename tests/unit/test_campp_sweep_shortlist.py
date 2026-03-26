from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.training import load_campp_baseline_config, run_campp_baseline
from kryptonite.training.campp import (
    load_campp_stage2_config,
    load_campp_sweep_shortlist_config,
    run_campp_stage2,
    run_campp_sweep_shortlist,
)


def test_campp_sweep_shortlist_runs_candidates_and_ranks_robust_dev(tmp_path: Path) -> None:
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

    suite_manifest = _write_fake_corrupted_suite_manifest(
        tmp_path,
        source_manifest=dev_manifest,
    )
    suite_catalog = _write_fake_suite_catalog(
        tmp_path,
        suite_manifest=suite_manifest,
        trials_manifest=Path(stage2_artifacts.trials_path),
    )
    stage3_config_path = _write_campp_stage3_base_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        stage2_checkpoint=Path(stage2_artifacts.output_root),
    )
    shortlist_config_path = _write_sweep_shortlist_config(
        tmp_path,
        base_stage3_config=stage3_config_path,
        suite_catalog=suite_catalog,
    )

    shortlist_config = load_campp_sweep_shortlist_config(config_path=shortlist_config_path)
    artifacts = run_campp_sweep_shortlist(
        shortlist_config,
        config_path=shortlist_config_path,
        env_file=tmp_path / ".env",
        device_override="cpu",
    )

    assert Path(tmp_path / artifacts.report_json_path).is_file()
    assert Path(tmp_path / artifacts.report_markdown_path).is_file()
    assert artifacts.summary.corrupted_suite_ids == ("dev_snr",)
    assert artifacts.summary.executed_candidate_count == 2
    assert len(artifacts.candidates) == 2
    assert [candidate.rank for candidate in artifacts.candidates] == [1, 2]
    assert artifacts.summary.winner_candidate_id in {"mean_pooling", "max_pooling"}

    for candidate in artifacts.candidates:
        assert candidate.clean_eer is not None
        assert candidate.clean_min_dcf is not None
        assert len(candidate.suites) == 2
        assert candidate.suites[0].suite_id == "clean_dev"
        assert candidate.suites[1].suite_id == "dev_snr"
        assert Path(tmp_path / candidate.run_output_root).is_dir()
        assert Path(tmp_path / candidate.run_report_path).is_file()
        assert Path(tmp_path / candidate.suites[1].report_markdown_path).is_file()


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


def _write_campp_stage3_base_config(
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
                "  'training.max_epochs=1',",
                "  'chunking.train_min_crop_seconds=0.5',",
                "  'chunking.train_max_crop_seconds=0.5',",
                "  'chunking.train_num_crops=1',",
                "  'chunking.eval_max_full_utterance_seconds=0.75',",
                "  'chunking.eval_chunk_seconds=0.75',",
                "  'chunking.eval_chunk_overlap_seconds=0.1',",
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
                "ramp_epochs = 1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_sweep_shortlist_config(
    tmp_path: Path,
    *,
    base_stage3_config: Path,
    suite_catalog: Path,
) -> Path:
    config_path = tmp_path / "configs" / "training" / "campp-stage3-sweep-shortlist.toml"
    config_path.write_text(
        "\n".join(
            [
                f'base_stage3_config_path = "{base_stage3_config.as_posix()}"',
                'output_root = "artifacts/sweeps/campp-stage3-shortlist"',
                "",
                "[selection]",
                "clean_weight = 0.25",
                "corrupted_weight = 0.75",
                "eer_weight = 0.7",
                "min_dcf_weight = 0.3",
                "",
                "[budget]",
                "max_candidates = 2",
                'notes = ["bounded smoke shortlist"]',
                "",
                "[corrupted_suites]",
                f'catalog_path = "{suite_catalog.as_posix()}"',
                "run_clean_dev = true",
                'suite_ids = ["dev_snr"]',
                "",
                "[[candidates]]",
                'candidate_id = "mean_pooling"',
                'description = "Mean pooling reference candidate"',
                "project_overrides = [",
                "  'training.batch_size=2',",
                "  'training.eval_batch_size=2',",
                "  'chunking.eval_chunk_seconds=0.75',",
                "  'chunking.eval_max_full_utterance_seconds=0.75',",
                "  'chunking.eval_chunk_overlap_seconds=0.1',",
                "  'chunking.eval_pooling=\"mean\"',",
                "]",
                "",
                "[candidates.margin_schedule]",
                "start_margin = 0.30",
                "end_margin = 0.45",
                "ramp_epochs = 1",
                "",
                "[candidates.crop_curriculum]",
                "start_crop_seconds = 0.5",
                "end_crop_seconds = 0.75",
                "curriculum_epochs = 1",
                "",
                "[[candidates]]",
                'candidate_id = "max_pooling"',
                'description = "Max pooling alternative candidate"',
                "project_overrides = [",
                "  'training.batch_size=2',",
                "  'training.eval_batch_size=2',",
                "  'chunking.eval_chunk_seconds=0.75',",
                "  'chunking.eval_max_full_utterance_seconds=0.75',",
                "  'chunking.eval_chunk_overlap_seconds=0.1',",
                "  'chunking.eval_pooling=\"max\"',",
                "]",
                "",
                "[candidates.margin_schedule]",
                "start_margin = 0.35",
                "end_margin = 0.50",
                "ramp_epochs = 1",
                "",
                "[candidates.crop_curriculum]",
                "start_crop_seconds = 0.5",
                "end_crop_seconds = 0.75",
                "curriculum_epochs = 1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_fake_corrupted_suite_manifest(tmp_path: Path, *, source_manifest: Path) -> Path:
    suite_manifest = tmp_path / "artifacts" / "manifests" / "dev_snr_manifest.jsonl"
    suite_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in source_manifest.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload["corruption_suite"] = "dev_snr"
        payload["noise_slice"] = "heavy"
        rows.append(payload)
    suite_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return suite_manifest


def _write_fake_suite_catalog(
    tmp_path: Path,
    *,
    suite_manifest: Path,
    trials_manifest: Path,
) -> Path:
    catalog_path = tmp_path / "artifacts" / "eval" / "corrupted-dev-suites" / "catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suites": [
            {
                "suite_id": "dev_snr",
                "family": "noise",
                "description": "Fixture corrupted suite",
                "manifest_path": suite_manifest.as_posix(),
                "trial_manifest_paths": [trials_manifest.as_posix()],
            }
        ]
    }
    catalog_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return catalog_path


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
        ("speaker_alpha", "dev_enroll_a.wav", 225.0, "enrollment"),
        ("speaker_alpha", "dev_test_a.wav", 228.0, "test"),
        ("speaker_bravo", "dev_enroll_b.wav", 335.0, "enrollment"),
        ("speaker_bravo", "dev_test_b.wav", 338.0, "test"),
    ]
    for speaker_id, file_name, frequency, role in dev_specs:
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
                "duration_seconds": 1.0,
            }
        )

    train_manifest = manifest_root / "train_manifest.jsonl"
    dev_manifest = manifest_root / "dev_manifest.jsonl"
    train_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    dev_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dev_rows),
        encoding="utf-8",
    )
    return train_manifest, dev_manifest


def _write_tone(path: Path, *, frequency_hz: float) -> None:
    sample_rate = 16_000
    timeline = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    waveform = 0.2 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform, sample_rate)
