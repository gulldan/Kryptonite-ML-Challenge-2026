from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.training import load_campp_baseline_config, run_campp_baseline
from kryptonite.training.campp import (
    load_campp_model_selection_config,
    load_campp_stage2_config,
    load_campp_sweep_shortlist_config,
    run_campp_model_selection,
    run_campp_stage2,
    run_campp_sweep_shortlist,
)


def test_campp_model_selection_evaluates_raw_and_averaged_variants(tmp_path: Path) -> None:
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
    shortlist_artifacts = run_campp_sweep_shortlist(
        shortlist_config,
        config_path=shortlist_config_path,
        env_file=tmp_path / ".env",
        device_override="cpu",
    )

    selection_config_path = _write_model_selection_config(
        tmp_path,
        shortlist_report=Path(tmp_path / shortlist_artifacts.report_json_path),
    )
    selection_config = load_campp_model_selection_config(config_path=selection_config_path)
    artifacts = run_campp_model_selection(
        selection_config,
        config_path=selection_config_path,
        env_file=tmp_path / ".env",
        device_override="cpu",
    )

    assert Path(tmp_path / artifacts.report_json_path).is_file()
    assert Path(tmp_path / artifacts.report_markdown_path).is_file()
    assert Path(tmp_path / artifacts.final_checkpoint_path).is_file()
    assert (
        artifacts.summary.shortlist_winner_candidate_id
        == shortlist_artifacts.candidates[0].candidate_id
    )
    assert any(variant.variant_id == "winner_raw" for variant in artifacts.variants)
    averaged_variants = [
        variant for variant in artifacts.variants if variant.uses_checkpoint_averaging
    ]
    assert len(averaged_variants) == 1
    averaged_variant = averaged_variants[0]
    assert averaged_variant.variant_id == "top2_uniform_average"
    assert len(averaged_variant.source_candidate_ids) == 2
    assert Path(tmp_path / averaged_variant.checkpoint_path).is_file()
    assert all(suite.trial_count > 0 for suite in averaged_variant.suites)


def test_model_selection_config_rejects_invalid_candidate_counts(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "training" / "campp-stage3-model-selection.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'shortlist_report_path = "artifacts/report.json"',
                'output_root = "artifacts/model-selection/campp-stage3"',
                "",
                "[averaging]",
                "enabled = true",
                "candidate_counts = [1]",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_campp_model_selection_config(config_path=config_path)
    except ValueError as exc:
        assert "candidate_counts" in str(exc)
    else:
        raise AssertionError("Expected invalid candidate_counts to raise ValueError.")


def _write_model_selection_config(tmp_path: Path, *, shortlist_report: Path) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "campp-stage3-model-selection.toml"
    config_path.write_text(
        "\n".join(
            [
                f'shortlist_report_path = "{shortlist_report.as_posix()}"',
                'output_root = "artifacts/model-selection/campp-stage3"',
                "",
                "[averaging]",
                "enabled = true",
                "candidate_counts = [2]",
                'weights = "uniform"',
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
                "eer_weight = 0.70",
                "min_dcf_weight = 0.30",
                "",
                "[budget]",
                "max_candidates = 2",
                'notes = ["unit-test shortlist"]',
                "",
                "[corrupted_suites]",
                f'catalog_path = "{suite_catalog.as_posix()}"',
                "run_clean_dev = true",
                'suite_ids = ["dev_snr"]',
                "",
                "[[candidates]]",
                'candidate_id = "mean_pooling"',
                'description = "Reference mean-pooling stage-3 candidate."',
                'project_overrides = ["chunking.eval_pooling=\\"mean\\""]',
                "",
                "[[candidates]]",
                'candidate_id = "max_pooling"',
                'description = "Alternative max-pooling stage-3 candidate."',
                'project_overrides = ["chunking.eval_pooling=\\"max\\""]',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_fake_suite_catalog(
    tmp_path: Path,
    *,
    suite_manifest: Path,
    trials_manifest: Path,
) -> Path:
    output_root = tmp_path / "artifacts" / "eval" / "corrupted-dev-suites"
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_path = output_root / "corrupted_dev_suites_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "suite_id": "dev_snr",
                        "family": "noise",
                        "description": "Synthetic unit-test corrupted suite.",
                        "manifest_path": suite_manifest.as_posix(),
                        "trial_manifest_paths": [trials_manifest.as_posix()],
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return catalog_path


def _write_fake_corrupted_suite_manifest(tmp_path: Path, *, source_manifest: Path) -> Path:
    output_root = tmp_path / "artifacts" / "eval" / "corrupted-dev-suites"
    output_root.mkdir(parents=True, exist_ok=True)
    suite_manifest = output_root / "dev_snr_manifest.jsonl"

    rows = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transformed_rows: list[dict[str, object]] = []
    for row in rows:
        audio_path = Path(str(row["audio_path"]))
        suite_audio_path = audio_path.with_name(f"{audio_path.stem}_dev_snr.wav")
        waveform, sample_rate_hz = sf.read(audio_path)
        scaled = np.asarray(waveform, dtype=np.float32) * 0.85
        sf.write(suite_audio_path, scaled, sample_rate_hz)
        transformed_rows.append(
            {
                **row,
                "audio_path": suite_audio_path.as_posix(),
                "source_dataset": "fixture-dev-snr",
            }
        )

    suite_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transformed_rows),
        encoding="utf-8",
    )
    return suite_manifest


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
    dev_specs = [
        ("speaker_alpha", "dev_enroll_a.wav", 220.0, "enrollment"),
        ("speaker_alpha", "dev_test_a.wav", 233.0, "test"),
        ("speaker_bravo", "dev_enroll_b.wav", 330.0, "enrollment"),
        ("speaker_bravo", "dev_test_b.wav", 347.0, "test"),
    ]

    for speaker_id, filename, frequency in train_specs:
        audio_path = dataset_root / filename
        _write_wave_fixture(audio_path, frequency=frequency)
        train_rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "fixture-train",
                "source_dataset": "fixture-train",
                "split": "train",
                "speaker_id": speaker_id,
                "utterance_id": audio_path.stem,
                "audio_path": audio_path.as_posix(),
                "duration_seconds": 0.96,
                "sample_rate_hz": 16_000,
                "role": "train",
                "channel": "mic",
            }
        )

    for speaker_id, filename, frequency, role in dev_specs:
        audio_path = dataset_root / filename
        _write_wave_fixture(audio_path, frequency=frequency)
        dev_rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "fixture-dev",
                "source_dataset": "fixture-dev",
                "split": "dev",
                "speaker_id": speaker_id,
                "utterance_id": audio_path.stem,
                "audio_path": audio_path.as_posix(),
                "duration_seconds": 0.96,
                "sample_rate_hz": 16_000,
                "role": role,
                "channel": "mic",
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


def _write_wave_fixture(path: Path, *, frequency: float) -> None:
    sample_rate_hz = 16_000
    seconds = 0.96
    sample_count = int(sample_rate_hz * seconds)
    time = np.linspace(0.0, seconds, sample_count, endpoint=False)
    waveform = (
        0.45 * np.sin(2.0 * np.pi * frequency * time)
        + 0.12 * np.sin(2.0 * np.pi * (frequency * 1.5) * time)
    ).astype(np.float32)
    sf.write(path, waveform, sample_rate_hz)
