from pathlib import Path

import pytest

from kryptonite.config import load_project_config


@pytest.mark.parametrize(
    ("config_path", "expected_device", "expected_tracking_enabled"),
    [
        ("research/configs/deployment/train.toml", "auto", True),
        ("research/configs/deployment/infer.toml", "cpu", False),
        ("research/configs/deployment/infer-gpu.toml", "cuda", False),
    ],
)
def test_load_deployment_profiles(
    config_path: str,
    expected_device: str,
    expected_tracking_enabled: bool,
) -> None:
    config = load_project_config(config_path=Path(config_path))

    assert config.paths.dataset_root == "datasets"
    assert config.runtime.device == expected_device
    assert config.backends.inference == "torch"
    assert config.tracking.enabled is expected_tracking_enabled
    assert config.deployment.model_bundle_root == "artifacts/model-bundle"
