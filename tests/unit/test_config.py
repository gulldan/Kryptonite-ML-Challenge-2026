from pathlib import Path

from kryptonite.config import load_dotenv, load_project_config, mask_secret, parse_override_value


def test_load_project_config_uses_defaults() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    assert config.paths.dataset_root == "datasets"
    assert config.runtime.seed == 42
    assert config.reproducibility.deterministic is True
    assert config.reproducibility.pythonhashseed == 42
    assert config.backends.allow_tensorrt is False
    assert config.tracking.backend == "local"
    assert config.normalization.target_sample_rate_hz == 16000
    assert config.normalization.target_channels == 1
    assert config.normalization.output_format == "wav"
    assert config.vad.mode == "none"
    assert config.vad.backend == "silero_vad_v6_onnx"
    assert config.vad.provider == "auto"
    assert config.deployment.model_bundle_root == "artifacts/model-bundle"
    assert config.deployment.demo_subset_root == "artifacts/demo-subset"
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
            "backends.allow_tensorrt=true",
            "runtime.log_level=DEBUG",
            "normalization.output_format=flac",
            "vad.mode=light",
            "vad.provider=cpu",
        ],
        env_file=env_file,
    )

    assert config.runtime.seed == 7
    assert config.training.batch_size == 8
    assert config.runtime.log_level == "DEBUG"
    assert config.backends.allow_tensorrt is True
    assert config.normalization.output_format == "flac"
    assert config.vad.mode == "light"
    assert config.vad.provider == "cpu"
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
