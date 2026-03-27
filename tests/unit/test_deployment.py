from __future__ import annotations

from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve.deployment import build_infer_artifact_report
from kryptonite.serve.enrollment_cache import (
    ENROLLMENT_EMBEDDINGS_NPZ_NAME,
    ENROLLMENT_METADATA_PARQUET_NAME,
    ENROLLMENT_SUMMARY_JSON_NAME,
)
from kryptonite.training.deployment import build_training_artifact_report


def test_training_artifact_report_is_advisory_by_default(tmp_path: Path) -> None:
    config = load_project_config(
        config_path=Path("configs/deployment/train.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="manifests"',
        ],
    )

    report = build_training_artifact_report(config=config, strict=False)

    assert report.passed is True
    assert report.missing_required == ["dataset_root", "manifests_root", "demo_manifest_file"]


def test_infer_artifact_report_requires_real_paths_in_strict_mode(tmp_path: Path) -> None:
    manifests_root = tmp_path / "manifests"
    model_bundle_root = tmp_path / "model-bundle"
    demo_subset_root = tmp_path / "demo-subset"
    enrollment_cache_root = tmp_path / "enrollment-cache"
    enrollment_root = demo_subset_root / "enrollment"
    test_root = demo_subset_root / "test"

    manifests_root.mkdir()
    model_bundle_root.mkdir()
    enrollment_cache_root.mkdir()
    enrollment_root.mkdir(parents=True)
    test_root.mkdir(parents=True)

    (manifests_root / "demo_manifest.jsonl").write_text("{}\n")
    (model_bundle_root / "model.onnx").write_text("fake-model")
    (model_bundle_root / "metadata.json").write_text("{}")
    (enrollment_root / "speaker-a-enroll.wav").write_text("fake-audio")
    (test_root / "speaker-a-test.wav").write_text("fake-audio")
    (demo_subset_root / "demo_subset.json").write_text("{}")
    (enrollment_cache_root / ENROLLMENT_EMBEDDINGS_NPZ_NAME).write_text("fake-embeddings")
    (enrollment_cache_root / ENROLLMENT_METADATA_PARQUET_NAME).write_text("fake-metadata")
    (enrollment_cache_root / ENROLLMENT_SUMMARY_JSON_NAME).write_text("{}")

    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.manifests_root="manifests"',
            'deployment.model_bundle_root="model-bundle"',
            'deployment.demo_subset_root="demo-subset"',
            'deployment.enrollment_cache_root="enrollment-cache"',
        ],
    )

    report = build_infer_artifact_report(config=config, strict=True)

    assert report.passed is True
    assert report.missing_required == []


def test_gpu_compose_override_uses_gpu_image_and_profile() -> None:
    compose_override = Path("compose.gpu.yml").read_text()
    dockerfile = Path("deployment/docker/infer.gpu.Dockerfile").read_text()
    default_command = (
        'CMD ["python", "apps/api/main.py", "--config", '
        '"configs/deployment/infer-gpu.toml"'
    )

    assert "deployment/docker/infer.gpu.Dockerfile" in compose_override
    assert "configs/deployment/infer-gpu.toml" in compose_override
    assert "gpus: all" in compose_override
    assert default_command in dockerfile
