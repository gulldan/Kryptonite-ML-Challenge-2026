from __future__ import annotations

import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve.deployment import build_infer_artifact_report
from kryptonite.training.deployment import build_training_artifact_report


def test_generate_demo_artifacts_creates_expected_files(tmp_path: Path) -> None:
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
        ],
    )

    generated = generate_demo_artifacts(config=config)

    manifest_file = Path(generated.manifest_file)
    subset_file = Path(generated.subset_file)
    model_file = Path(generated.model_file)
    metadata_file = Path(generated.metadata_file)

    assert generated.clip_count == 6
    assert manifest_file.is_file()
    assert subset_file.is_file()
    assert model_file.is_file()
    assert metadata_file.is_file()
    assert len(manifest_file.read_text().splitlines()) == 6

    subset_payload = json.loads(subset_file.read_text())
    assert len(subset_payload["enrollment"]) == 4
    assert len(subset_payload["test"]) == 2


def test_generated_demo_artifacts_satisfy_strict_preflight(tmp_path: Path) -> None:
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
        ],
    )

    generate_demo_artifacts(config=config)

    training_report = build_training_artifact_report(config=config, strict=True)
    infer_report = build_infer_artifact_report(config=config, strict=True)

    assert training_report.passed is True
    assert infer_report.passed is True
    assert training_report.missing_required == []
    assert infer_report.missing_required == []
