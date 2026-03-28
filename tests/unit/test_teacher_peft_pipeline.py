from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from kryptonite.training import load_teacher_peft_config, run_teacher_peft


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

    def save_pretrained(self, output_dir: Path | str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")


class FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.frame_projection = torch.nn.Linear(1, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.gradient_checkpointing_enabled = False
        self.feature_encoder_frozen = False

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

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self) -> None:
        return None

    def freeze_feature_encoder(self) -> None:
        self.feature_encoder_frozen = True

    def _get_feature_vector_attention_mask(
        self,
        sequence_length: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return attention_mask[:, :sequence_length].to(dtype=torch.bool)

    def save_pretrained(self, output_dir: Path | str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        torch.save(self.state_dict(), path / "adapter_model.bin")


def test_teacher_peft_smoke_run_writes_checkpoint_embeddings_and_scores(tmp_path: Path) -> None:
    train_manifest, dev_manifest = _write_manifest_fixtures(tmp_path)
    config_path = _write_teacher_config(
        tmp_path,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
    )

    config = load_teacher_peft_config(config_path=config_path, env_file=tmp_path / ".env")
    artifacts = run_teacher_peft(
        config,
        config_path=config_path,
        device_override="cpu",
        feature_extractor_factory=lambda **_: FakeFeatureExtractor(),
        backbone_factory=lambda **_: FakeBackbone(),
    )

    checkpoint_dir = Path(artifacts.checkpoint_path)
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "adapter" / "adapter_config.json").is_file()
    assert (checkpoint_dir / "feature_extractor" / "preprocessor_config.json").is_file()
    assert (checkpoint_dir / "heads.pt").is_file()
    assert (checkpoint_dir / "checkpoint_metadata.json").is_file()
    assert Path(artifacts.embeddings_path).is_file()
    assert Path(artifacts.embedding_metadata_jsonl_path).is_file()
    assert Path(artifacts.embedding_metadata_parquet_path).is_file()
    assert Path(artifacts.scores_path).is_file()
    assert Path(artifacts.score_summary_path).is_file()
    assert Path(artifacts.report_path).is_file()
    assert artifacts.verification_report is not None
    assert artifacts.training_summary.epochs[-1].mean_loss > 0.0
    assert artifacts.training_summary.provenance_initialization == "pretrained"
    assert artifacts.score_summary.trial_count > 0
    assert artifacts.score_summary.positive_count > 0
    assert artifacts.score_summary.negative_count > 0

    payload = np.load(artifacts.embeddings_path)
    assert payload["embeddings"].shape == (4, 256)
    report_text = Path(artifacts.report_path).read_text(encoding="utf-8")
    assert "# Teacher PEFT Report" in report_text
    assert "- Initialization: `pretrained`" in report_text
    assert "## Verification Eval" in report_text

    checkpoint_metadata = json.loads((checkpoint_dir / "checkpoint_metadata.json").read_text())
    assert checkpoint_metadata["model"]["model_id"] == "microsoft/wavlm-base-plus"
    assert checkpoint_metadata["adapter"]["target_modules"] == ["all-linear"]


def _write_teacher_config(tmp_path: Path, *, train_manifest: Path, dev_manifest: Path) -> Path:
    config_root = tmp_path / "configs" / "training"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "teacher-peft-test.toml"
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
                "]",
                "",
                "[data]",
                f'train_manifest = "{train_manifest.as_posix()}"',
                f'dev_manifest = "{dev_manifest.as_posix()}"',
                'output_root = "artifacts/baselines/teacher-peft-test"',
                "generate_demo_artifacts_if_missing = false",
                'checkpoint_name = "teacher_peft"',
                "",
                "[model]",
                'model_id = "microsoft/wavlm-base-plus"',
                "embedding_dim = 256",
                "projection_dropout = 0.1",
                "",
                "[adapter]",
                "rank = 8",
                "alpha = 16",
                "dropout = 0.0",
                'target_modules = ["all-linear"]',
                "",
                "[objective]",
                "classifier_hidden_dim = 64",
                "scale = 16.0",
                "margin = 0.2",
                "",
                "[optimization]",
                'optimizer_name = "adamw"',
                'scheduler_name = "cosine"',
                "learning_rate = 0.001",
                "min_learning_rate = 0.0001",
                "weight_decay = 0.0",
                "warmup_epochs = 0",
                "gradient_accumulation_steps = 1",
                "grad_clip_norm = 1.0",
                "",
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
    train_manifest.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    dev_manifest.write_text("".join(json.dumps(row) + "\n" for row in dev_rows))
    return train_manifest, dev_manifest


def _write_tone(path: Path, *, frequency_hz: float, sample_rate_hz: int = 16_000) -> None:
    sample_count = int(sample_rate_hz * 0.5)
    timeline = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate_hz)
    waveform = 0.3 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform, sample_rate_hz, format="WAV")
