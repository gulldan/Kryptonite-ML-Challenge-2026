from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import build_service_metadata, create_http_server
from kryptonite.serve.deployment import build_infer_artifact_report


def test_build_serve_runtime_report_marks_missing_required_backend(monkeypatch) -> None:
    config = load_project_config(config_path=Path("configs/deployment/infer.toml"))

    monkeypatch.setattr(
        serve_runtime,
        "_load_module",
        lambda module_name: (_ for _ in ()).throw(ImportError(f"{module_name} missing")),
    )
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: None)

    report = serve_runtime.build_serve_runtime_report(config=config)

    assert report.passed is False
    assert report.selected_backend == "onnxruntime"
    assert report.missing_required == ["onnxruntime"]


def test_create_http_server_uses_selected_backend_metadata(monkeypatch, tmp_path) -> None:
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

    def fake_load_module(module_name: str) -> object:
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: "1.0.0")

    report = serve_runtime.build_serve_runtime_report(config=config)
    artifact_report = build_infer_artifact_report(config=config, strict=False)
    payload = build_service_metadata(config=config, report=report, artifact_report=artifact_report)

    assert report.passed is True
    assert payload["selected_backend"] == "onnxruntime"
    assert payload["status"] == "ok"
    assert payload["artifacts"]["scope"] == "infer"
    assert payload["artifacts"]["strict"] is False

    server = create_http_server(host="127.0.0.1", port=0, config=config)
    try:
        assert server.server_address[1] > 0
    finally:
        server.server_close()


def test_create_http_server_fails_when_backend_is_unavailable(monkeypatch) -> None:
    config = load_project_config(config_path=Path("configs/deployment/infer.toml"))

    monkeypatch.setattr(
        serve_runtime,
        "_load_module",
        lambda module_name: (_ for _ in ()).throw(ImportError(f"{module_name} missing")),
    )
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: None)

    with pytest.raises(RuntimeError, match="Serve runtime smoke: FAIL"):
        create_http_server(host="127.0.0.1", port=0, config=config)


def test_create_http_server_fails_when_required_artifacts_are_missing(
    monkeypatch, tmp_path
) -> None:
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[f'paths.project_root="{tmp_path}"'],
    )

    def fake_load_module(module_name: str) -> object:
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: "1.0.0")

    with pytest.raises(RuntimeError, match="Artifact preflight \\(infer\\): FAIL"):
        create_http_server(
            host="127.0.0.1",
            port=0,
            config=config,
            require_artifacts=True,
        )
