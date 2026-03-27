from pathlib import Path

from kryptonite.config import load_dotenv, load_project_config, mask_secret, parse_override_value


def test_load_project_config_uses_defaults() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    assert config.paths.dataset_root == "datasets"
    assert config.runtime.seed == 42
    assert config.reproducibility.deterministic is True
    assert config.reproducibility.pythonhashseed == 42
    assert config.training.precision == "fp32"
    assert config.backends.allow_tensorrt is False
    assert config.export.boundary == "encoder_only"
    assert config.export.input_name == "encoder_input"
    assert config.export.output_name == "embedding"
    assert config.tracking.backend == "local"
    assert config.normalization.target_sample_rate_hz == 16000
    assert config.normalization.target_channels == 1
    assert config.normalization.output_format == "wav"
    assert config.normalization.loudness_mode == "none"
    assert config.normalization.target_loudness_dbfs == -27.0
    assert config.normalization.max_loudness_gain_db == 20.0
    assert config.normalization.max_loudness_attenuation_db == 12.0
    assert config.vad.mode == "none"
    assert config.vad.backend == "silero_vad_v6_onnx"
    assert config.vad.provider == "auto"
    assert config.vad.min_output_duration_seconds == 1.0
    assert config.vad.min_retained_ratio == 0.4
    assert config.silence_augmentation.enabled is False
    assert config.silence_augmentation.max_leading_padding_seconds == 0.15
    assert config.silence_augmentation.max_trailing_padding_seconds == 0.20
    assert config.silence_augmentation.max_inserted_pauses == 2
    assert config.silence_augmentation.pause_ratio_min == 0.9
    assert config.silence_augmentation.pause_ratio_max == 1.4
    assert config.silence_augmentation.analysis_frame_ms == 20.0
    assert config.augmentation_scheduler.enabled is True
    assert config.augmentation_scheduler.warmup_epochs == 2
    assert config.augmentation_scheduler.ramp_epochs == 3
    assert config.augmentation_scheduler.clean_probability_start == 0.7
    assert config.augmentation_scheduler.heavy_probability_end == 0.2
    assert config.augmentation_scheduler.family_weights.codec == 0.8
    assert config.features.sample_rate_hz == 16000
    assert config.features.num_mel_bins == 80
    assert config.features.frame_length_ms == 25.0
    assert config.features.frame_shift_ms == 10.0
    assert config.features.fft_size == 512
    assert config.features.window_type == "hann"
    assert config.features.cmvn_mode == "none"
    assert config.features.output_dtype == "float32"
    assert config.chunking.train_min_crop_seconds == 1.0
    assert config.chunking.train_max_crop_seconds == 4.0
    assert config.chunking.train_num_crops == 1
    assert config.chunking.train_short_utterance_policy == "repeat_pad"
    assert config.chunking.eval_pooling == "mean"
    assert config.chunking.demo_pooling == "mean"
    assert config.deployment.model_bundle_root == "artifacts/model-bundle"
    assert config.deployment.demo_subset_root == "artifacts/demo-subset"
    assert config.deployment.enrollment_cache_root == "artifacts/enrollment-cache"
    assert config.deployment.require_artifacts is False
    assert config.resolved_secrets["wandb_api_key"] is None


def test_load_project_config_applies_overrides_and_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=test-wandb-token\n")

    config = load_project_config(
        config_path=Path("configs/base.toml"),
        overrides=[
            "runtime.seed=7",
            "training.batch_size=8",
            'training.precision="bf16"',
            "backends.allow_tensorrt=true",
            "runtime.log_level=DEBUG",
            "normalization.output_format=flac",
            "normalization.loudness_mode=rms",
            "normalization.target_loudness_dbfs=-24.0",
            "normalization.max_loudness_gain_db=18.0",
            "vad.mode=light",
            "vad.provider=cpu",
            "vad.min_output_duration_seconds=1.25",
            "vad.min_retained_ratio=0.5",
            "silence_augmentation.enabled=true",
            "silence_augmentation.max_leading_padding_seconds=0.15",
            "silence_augmentation.max_inserted_pauses=2",
            "silence_augmentation.pause_ratio_max=1.4",
            "augmentation_scheduler.warmup_epochs=1",
            "augmentation_scheduler.light_probability_end=0.2",
            "augmentation_scheduler.heavy_probability_end=0.3",
            "augmentation_scheduler.family_weights.silence=0.9",
            "features.num_mel_bins=96",
            "features.cmvn_mode=sliding",
            "features.output_dtype=float16",
            "chunking.train_num_crops=3",
            "chunking.train_short_utterance_policy=zero_pad",
            "chunking.eval_chunk_overlap_seconds=0.5",
            "chunking.demo_pooling=max",
        ],
        env_file=env_file,
    )

    assert config.runtime.seed == 7
    assert config.training.batch_size == 8
    assert config.training.precision == "bf16"
    assert config.runtime.log_level == "DEBUG"
    assert config.backends.allow_tensorrt is True
    assert config.export.boundary == "encoder_only"
    assert config.export.input_name == "encoder_input"
    assert config.export.output_name == "embedding"
    assert config.normalization.output_format == "flac"
    assert config.normalization.loudness_mode == "rms"
    assert config.normalization.target_loudness_dbfs == -24.0
    assert config.normalization.max_loudness_gain_db == 18.0
    assert config.vad.mode == "light"
    assert config.vad.provider == "cpu"
    assert config.vad.min_output_duration_seconds == 1.25
    assert config.vad.min_retained_ratio == 0.5
    assert config.silence_augmentation.enabled is True
    assert config.silence_augmentation.max_leading_padding_seconds == 0.15
    assert config.silence_augmentation.max_inserted_pauses == 2
    assert config.silence_augmentation.pause_ratio_max == 1.4
    assert config.augmentation_scheduler.warmup_epochs == 1
    assert config.augmentation_scheduler.light_probability_end == 0.2
    assert config.augmentation_scheduler.heavy_probability_end == 0.3
    assert config.augmentation_scheduler.family_weights.silence == 0.9
    assert config.features.num_mel_bins == 96
    assert config.features.cmvn_mode == "sliding"
    assert config.features.output_dtype == "float16"
    assert config.chunking.train_num_crops == 3
    assert config.chunking.train_short_utterance_policy == "zero_pad"
    assert config.chunking.eval_chunk_overlap_seconds == 0.5
    assert config.chunking.demo_pooling == "max"
    assert config.deployment.enrollment_cache_root == "artifacts/enrollment-cache"
    assert config.deployment.require_artifacts is False
    assert config.resolved_secrets["wandb_api_key"] == "test-wandb-token"


def test_load_dotenv_and_mask_secret() -> None:
    assert load_dotenv(Path("does-not-exist.env")) == {}
    assert mask_secret(None) is None
    assert mask_secret("abcd") == "****"
    assert mask_secret("abcdefgh") == "ab****gh"


def test_parse_override_value_supports_toml_literals_and_raw_strings() -> None:
    assert parse_override_value("123") == 123
    assert parse_override_value("true") is True
    assert parse_override_value('"fp16"') == "fp16"
    assert parse_override_value("datasets") == "datasets"
