from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import build_service_metadata, create_http_server
from kryptonite.serve.deployment import build_infer_artifact_report
from kryptonite.serve.enrollment_cache import MODEL_BUNDLE_METADATA_NAME, load_model_bundle_metadata


def test_build_serve_runtime_report_resolves_torch_fallback_for_auto_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_demo_config(tmp_path)
    _patch_runtime_modules(monkeypatch, include_torch=True, onnx_providers=["CPUExecutionProvider"])

    metadata = load_model_bundle_metadata(
        tmp_path / "artifacts" / "model-bundle" / MODEL_BUNDLE_METADATA_NAME
    )
    report = serve_runtime.build_serve_runtime_report(config=config, model_metadata=metadata)

    assert report.passed is True
    assert report.requested_backend == "auto"
    assert report.selected_backend == "torch"
    assert report.selected_provider is None
    assert "PyTorch fallback" in report.selection_reason
    assert any(
        step.backend == "onnxruntime" and step.selected is False and "not validated" in step.reason
        for step in report.selection_trace
    )


def test_build_serve_runtime_report_fails_when_auto_chain_has_no_eligible_backend(
    monkeypatch,
) -> None:
    config = load_project_config(config_path=Path("configs/deployment/infer.toml"))
    monkeypatch.setattr(
        serve_runtime,
        "_load_module",
        lambda module_name: (_ for _ in ()).throw(ImportError(f"{module_name} missing")),
    )
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: None)

    report = serve_runtime.build_serve_runtime_report(config=config)

    assert report.passed is False
    assert report.requested_backend == "auto"
    assert report.selected_backend is None
    assert report.selection_error == (
        "No eligible inference backend candidates were available in the configured fallback chain."
    )


def test_explicit_onnxruntime_request_fails_until_bundle_validates_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_demo_config(
        tmp_path,
        overrides=['backends.inference="onnxruntime"'],
    )
    _patch_runtime_modules(monkeypatch, include_torch=True, onnx_providers=["CPUExecutionProvider"])
    metadata = load_model_bundle_metadata(
        tmp_path / "artifacts" / "model-bundle" / MODEL_BUNDLE_METADATA_NAME
    )

    report = serve_runtime.build_serve_runtime_report(config=config, model_metadata=metadata)

    assert report.passed is False
    assert report.requested_backend == "onnxruntime"
    assert report.selected_backend is None
    assert report.selection_error == "Requested onnxruntime backend is not available."
    assert "not validated" in report.selection_reason


def test_create_http_server_uses_resolved_backend_metadata(monkeypatch, tmp_path) -> None:
    config = _build_demo_config(tmp_path)
    _patch_runtime_modules(monkeypatch, include_torch=True, onnx_providers=["CPUExecutionProvider"])

    metadata = load_model_bundle_metadata(
        tmp_path / "artifacts" / "model-bundle" / MODEL_BUNDLE_METADATA_NAME
    )
    report = serve_runtime.build_serve_runtime_report(config=config, model_metadata=metadata)
    artifact_report = build_infer_artifact_report(config=config, strict=False)
    payload = build_service_metadata(config=config, report=report, artifact_report=artifact_report)

    assert report.passed is True
    assert payload["requested_backend"] == "auto"
    assert payload["selected_backend"] == "torch"
    assert payload["selected_provider"] is None
    assert payload["status"] == "ok"
    assert payload["artifacts"]["scope"] == "infer"
    assert payload["artifacts"]["strict"] is False

    server = create_http_server(host="127.0.0.1", port=0, config=config)
    try:
        assert server.server_address[1] > 0
    finally:
        server.server_close()


def test_create_http_server_fails_when_no_backend_candidate_is_available(monkeypatch) -> None:
    config = load_project_config(config_path=Path("configs/deployment/infer.toml"))

    monkeypatch.setattr(
        serve_runtime,
        "_load_module",
        lambda module_name: (_ for _ in ()).throw(ImportError(f"{module_name} missing")),
    )
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: None)

    with pytest.raises(
        RuntimeError,
        match="No eligible inference backend candidates were available",
    ):
        create_http_server(host="127.0.0.1", port=0, config=config)


def test_create_http_server_fails_when_required_artifacts_are_missing(
    monkeypatch, tmp_path
) -> None:
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[f'paths.project_root="{tmp_path}"'],
    )
    _patch_runtime_modules(monkeypatch, include_torch=True, onnx_providers=["CPUExecutionProvider"])

    with pytest.raises(RuntimeError, match="Artifact preflight \\(infer\\): FAIL"):
        create_http_server(
            host="127.0.0.1",
            port=0,
            config=config,
            require_artifacts=True,
        )


def _build_demo_config(tmp_path: Path, *, overrides: list[str] | None = None):
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
            'deployment.enrollment_cache_root="artifacts/enrollment-cache"',
            *(overrides or []),
        ],
    )
    generate_demo_artifacts(config=config)
    return config


def _patch_runtime_modules(
    monkeypatch,
    *,
    include_torch: bool,
    onnx_providers: list[str],
) -> None:
    def fake_load_module(module_name: str) -> object:
        if module_name == "torch" and include_torch:
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                version=SimpleNamespace(cuda=None),
            )
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: onnx_providers)
        raise ImportError(f"{module_name} missing")

    def fake_distribution_version(distribution: str) -> str | None:
        versions = {
            "torch": "2.10.0",
            "onnxruntime": "1.24.4",
            "tensorrt-cu12": None,
        }
        return versions.get(distribution)

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", fake_distribution_version)
