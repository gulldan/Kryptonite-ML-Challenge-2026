from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import create_http_server
from kryptonite.serve.enrollment_cache import load_enrollment_embedding_cache


def test_generated_demo_artifacts_include_loadable_enrollment_cache(tmp_path: Path) -> None:
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
            'deployment.enrollment_cache_root="artifacts/enrollment-cache"',
        ],
    )

    generate_demo_artifacts(config=config)
    loaded = load_enrollment_embedding_cache(tmp_path / "artifacts" / "enrollment-cache")

    assert loaded.summary.enrollment_count == 2
    assert loaded.summary.source_row_count == 4
    assert loaded.summary.embedding_dim == 160
    assert loaded.summary.counts_by_role == {"enrollment": 4}
    assert loaded.summary.compatibility_id == "demo-speaker-recognition-cache-v1"
    assert loaded.embeddings.shape == (2, 160)

    metadata_by_id = {row["enrollment_id"]: row for row in loaded.metadata_rows}
    assert metadata_by_id["speaker_alpha"]["sample_count"] == 2
    assert metadata_by_id["speaker_bravo"]["sample_count"] == 2


def test_create_http_server_rejects_incompatible_enrollment_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
            'deployment.enrollment_cache_root="artifacts/enrollment-cache"',
        ],
    )
    generated = generate_demo_artifacts(config=config)

    metadata_path = Path(generated.metadata_file)
    metadata_payload = json.loads(metadata_path.read_text())
    metadata_payload["enrollment_cache_compatibility_id"] = "mismatched-cache-id"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))

    def fake_load_module(module_name: str) -> object:
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: "1.0.0")

    with pytest.raises(RuntimeError, match="Enrollment cache compatibility mismatch"):
        create_http_server(host="127.0.0.1", port=0, config=config)
