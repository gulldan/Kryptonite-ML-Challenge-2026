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
    manifest_csv_file = manifest_file.with_suffix(".csv")
    manifest_inventory_file = Path(generated.manifest_inventory_file)
    subset_file = Path(generated.subset_file)
    model_file = Path(generated.model_file)
    metadata_file = Path(generated.metadata_file)
    enrollment_embeddings_file = Path(generated.enrollment_embeddings_file)
    enrollment_summary_file = Path(generated.enrollment_summary_file)

    assert generated.clip_count == 6
    assert manifest_file.is_file()
    assert manifest_csv_file.is_file()
    assert manifest_inventory_file.is_file()
    assert subset_file.is_file()
    assert model_file.is_file()
    assert metadata_file.is_file()
    assert enrollment_embeddings_file.is_file()
    assert enrollment_summary_file.is_file()
    assert len(manifest_file.read_text().splitlines()) == 6

    first_manifest_row = json.loads(manifest_file.read_text().splitlines()[0])
    manifest_inventory = json.loads(manifest_inventory_file.read_text())
    subset_payload = json.loads(subset_file.read_text())
    assert first_manifest_row["schema_version"] == "kryptonite.manifest.v1"
    assert first_manifest_row["record_type"] == "utterance"
    assert first_manifest_row["split"] == "demo"
    assert first_manifest_row["num_channels"] == 1
    assert manifest_inventory["dataset"] == "demo-speaker-recognition"
    assert manifest_inventory["manifest_tables"][0]["row_count"] == 6
    assert manifest_inventory["manifest_tables"][0]["speaker_count"] == 2
    assert manifest_inventory["manifest_tables"][0]["csv_path"].endswith("demo_manifest.csv")
    assert manifest_inventory["auxiliary_files"][0]["path"].endswith("demo_subset.json")
    assert len(subset_payload["enrollment"]) == 4
    assert len(subset_payload["test"]) == 2
    metadata_payload = json.loads(metadata_file.read_text())
    assert metadata_payload["enrollment_cache_compatibility_id"]
    assert metadata_payload["model_version"] == "demo-onnx-stub-v1"
    assert metadata_payload["input_name"] == "encoder_input"
    assert metadata_payload["output_name"] == "embedding"
    assert metadata_payload["export_boundary"]["boundary"] == "encoder_only"
    assert metadata_payload["export_boundary"]["input_tensor"]["layout"] == "BTF"


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
