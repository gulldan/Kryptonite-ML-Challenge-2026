from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from kryptonite.training import load_campp_baseline_config, run_campp_baseline
from kryptonite.training.campp import (
    load_campp_distillation_config,
    load_campp_stage2_config,
    load_campp_stage3_config,
    run_campp_distillation,
    run_campp_stage2,
    run_campp_stage3,
)
from kryptonite.training.teacher_peft import TeacherPeftEncoder


class FakeFeatureExtractor:
    def __call__(
        self,
        waveforms: list[np.ndarray],
        *,
        sampling_rate: int,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del sampling_rate, padding, return_tensors
        max_length = max(len(waveform) for waveform in waveforms)
        input_values = torch.zeros((len(waveforms), max_length), dtype=torch.float32)
        attention_mask = torch.zeros((len(waveforms), max_length), dtype=torch.int32)
        for index, waveform in enumerate(waveforms):
            length = len(waveform)
            input_values[index, :length] = torch.tensor(waveform, dtype=torch.float32)
            attention_mask[index, :length] = 1
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
        }


class FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.frame_projection = torch.nn.Linear(1, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(
        self,
        *,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: object,
    ) -> SimpleNamespace:
        del output_hidden_states, return_dict, attention_mask
        assert input_values is not None
        hidden = self.frame_projection(input_values.unsqueeze(-1))
        return SimpleNamespace(last_hidden_state=hidden)

    def _get_feature_vector_attention_mask(
        self,
        sequence_length: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return attention_mask[:, :sequence_length].to(dtype=torch.bool)


def test_campp_distillation_smoke_run_writes_comparison_and_report(tmp_path: Path) -> None:
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
    stage3_artifacts = run_campp_stage3(
        stage3_config,
        config_path=stage3_config_path,
        device_override="cpu",
    )

    teacher_checkpoint_dir = (
        tmp_path / "artifacts" / "baselines" / "teacher-peft" / "run-001" / "teacher_peft"
    )
    teacher_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    distillation_config_path = _write_campp_distillation_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        student_checkpoint=Path(stage3_artifacts.output_root),
        teacher_checkpoint=teacher_checkpoint_dir,
        base_stage3_config=stage3_config_path,
    )
    distillation_config = load_campp_distillation_config(
        config_path=distillation_config_path,
        env_file=tmp_path / ".env",
    )
    artifacts = run_campp_distillation(
        distillation_config,
        config_path=distillation_config_path,
        device_override="cpu",
        teacher_loader=_build_fake_teacher_loader(teacher_checkpoint_dir),
    )

    assert Path(artifacts.checkpoint_path).is_file()
    assert Path(artifacts.distillation_summary_path).is_file()
    assert Path(artifacts.comparison_json_path).is_file()
    assert Path(artifacts.comparison_markdown_path).is_file()
    assert Path(artifacts.report_path).is_file()
    assert Path(artifacts.embeddings_path).is_file()
    assert artifacts.training_summary.epochs[-1].mean_loss > 0.0
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

    report_text = Path(artifacts.report_path).read_text(encoding="utf-8")
    assert "# CAM++ Distillation Report" in report_text
    assert "## Distillation Setup" in report_text
    assert "## Baseline Comparison" in report_text


def _build_fake_teacher_loader(checkpoint_dir: Path):
    def loader(
        *,
        checkpoint_path: str | Path,
        project_root: str | Path = ".",
        token: str | None = None,
        trainable: bool = False,
    ):
        del checkpoint_path, project_root, token, trainable
        encoder = TeacherPeftEncoder(
            backbone=FakeBackbone(hidden_size=32),
            hidden_size=32,
            embedding_dim=32,
            projection_dropout=0.1,
        )
        metadata = {
            "model": {
                "model_id": "fixture/wavlm",
                "revision": "main",
                "embedding_dim": 32,
            }
        }
        return checkpoint_dir, metadata, FakeFeatureExtractor(), encoder

    return loader


def _write_campp_distillation_config(
    tmp_path: Path,
    *,
    train_manifest: Path,
    dev_manifest: Path,
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    base_stage3_config: Path,
) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "campp-distillation.toml"
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
                "  'augmentation_scheduler.enabled=false',",
                "  'silence_augmentation.enabled=false',",
                "]",
                "",
                "[data]",
                f'train_manifest = "{train_manifest.as_posix()}"',
                f'dev_manifest = "{dev_manifest.as_posix()}"',
                'output_root = "artifacts/baselines/campp-distillation"',
                'trials_manifest = ""',
                'checkpoint_name = "campp_distilled_encoder.pt"',
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
                "[teacher]",
                f'checkpoint = "{teacher_checkpoint.as_posix()}"',
                "",
                "[distillation]",
                "classification_weight = 1.0",
                "embedding_weight = 0.25",
                "score_weight = 0.1",
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
                'pretrained_resources = ["campp_stage2_encoder.pt"]',
                "",
                "[stage3]",
                f'stage2_checkpoint = "{stage2_checkpoint.as_posix()}"',
                "",
                "[stage3.hard_negative]",
                "enabled = false",
                "mining_interval_epochs = 1",
                "top_k_per_speaker = 1",
                "hard_negative_fraction = 0.5",
                "",
                "[stage3.crop_curriculum]",
                "enabled = true",
                "start_crop_seconds = 0.5",
                "end_crop_seconds = 0.625",
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
    train_manifest.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    dev_manifest.write_text("".join(json.dumps(row) + "\n" for row in dev_rows))
    return train_manifest, dev_manifest


def _write_tone(path: Path, *, frequency_hz: float, sample_rate_hz: int = 16_000) -> None:
    sample_count = int(sample_rate_hz * 0.5)
    timeline = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = 0.3 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform, sample_rate_hz, format="WAV")
