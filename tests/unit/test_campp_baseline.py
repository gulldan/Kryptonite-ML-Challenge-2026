from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from kryptonite.models import ArcMarginLoss, CAMPPlusConfig, CAMPPlusEncoder, CosineClassifier
from kryptonite.training import load_campp_baseline_config, run_campp_baseline


def test_campp_encoder_produces_embeddings_and_margin_loss() -> None:
    encoder = CAMPPlusEncoder(
        CAMPPlusConfig(
            feat_dim=16,
            embedding_size=32,
            growth_rate=8,
            bottleneck_scale=2,
            init_channels=16,
            head_channels=8,
            head_res_blocks=(1, 1),
            block_layers=(2, 2, 2),
            block_kernel_sizes=(3, 3, 3),
            block_dilations=(1, 1, 2),
            memory_efficient=False,
        )
    )
    classifier = CosineClassifier(32, num_classes=3)
    criterion = ArcMarginLoss(scale=16.0, margin=0.2)

    features = torch.randn(4, 80, 16, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    embeddings = encoder(features)
    logits = classifier(embeddings)
    loss = criterion(logits, labels)

    assert embeddings.shape == (4, 32)
    assert logits.shape == (4, 3)
    assert torch.isfinite(loss)


def test_campp_baseline_smoke_run_writes_checkpoint_embeddings_and_scores(tmp_path: Path) -> None:
    train_manifest, dev_manifest = _write_manifest_fixtures(tmp_path)
    config_path = _write_campp_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
    )

    config = load_campp_baseline_config(config_path=config_path, env_file=tmp_path / ".env")
    artifacts = run_campp_baseline(config, config_path=config_path, device_override="cpu")

    assert Path(artifacts.checkpoint_path).is_file()
    assert Path(artifacts.embeddings_path).is_file()
    assert Path(artifacts.embedding_metadata_jsonl_path).is_file()
    assert Path(artifacts.embedding_metadata_parquet_path).is_file()
    assert Path(artifacts.scores_path).is_file()
    assert Path(artifacts.score_summary_path).is_file()
    assert Path(artifacts.report_path).is_file()
    assert artifacts.verification_report is not None
    assert Path(artifacts.verification_report.report_json_path).is_file()
    assert Path(artifacts.verification_report.report_markdown_path).is_file()
    assert artifacts.training_summary.epochs[-1].mean_loss > 0.0
    assert artifacts.training_summary.provenance_ruleset == "standard"
    assert artifacts.training_summary.provenance_initialization == "from_scratch"
    assert artifacts.score_summary.trial_count > 0
    assert artifacts.score_summary.positive_count > 0
    assert artifacts.score_summary.negative_count > 0
    assert (
        artifacts.verification_report.summary.metrics.trial_count
        == artifacts.score_summary.trial_count
    )

    payload = np.load(artifacts.embeddings_path)
    assert payload["embeddings"].shape == (4, 32)

    report_text = Path(artifacts.report_path).read_text()
    assert "# CAM++ Baseline Report" in report_text
    assert "- Ruleset: `standard`" in report_text
    assert "## Verification Eval" in report_text


def _write_campp_config(tmp_path: Path, *, train_manifest: Path, dev_manifest: Path) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "campp-test.toml"
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
                'output_root = "artifacts/baselines/campp-test"',
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
        )
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
            }
        )

    train_manifest = manifest_root / "train_manifest.jsonl"
    dev_manifest = manifest_root / "dev_manifest.jsonl"
    train_manifest.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    dev_manifest.write_text("".join(json.dumps(row) + "\n" for row in dev_rows))
    return train_manifest, dev_manifest


def _write_tone(path: Path, *, frequency_hz: float, sample_rate_hz: int = 16_000) -> None:
    sample_count = int(sample_rate_hz * 0.5)
    timeline = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = 0.3 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform, sample_rate_hz, format="WAV")
