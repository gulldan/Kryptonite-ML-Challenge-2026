from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import create_http_server


def test_health_endpoint_reports_selected_backend(monkeypatch, tmp_path: Path) -> None:
    config = _build_demo_config(tmp_path)

    def fake_load_module(module_name: str) -> object:
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: "1.0.0")

    server = create_http_server(host="127.0.0.1", port=0, config=config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        server.wait_started()
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/health") as response:
            payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/openapi.json") as response:
            openapi_payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert payload["service"] == "kryptonite-infer"
    assert payload["selected_backend"] == "onnxruntime"
    assert payload["status"] == "ok"
    assert payload["artifacts"]["scope"] == "infer"
    assert payload["artifacts"]["strict"] is False
    assert payload["enrollment_cache"]["loaded"] is True
    assert payload["enrollment_cache"]["enrollment_count"] == 2
    assert payload["enrollment_cache"]["runtime_store"]["enabled"] is True
    assert payload["enrollment_cache"]["runtime_store"]["enrollment_count"] == 0
    assert payload["telemetry"]["enabled"] is True
    assert payload["telemetry"]["metrics_enabled"] is True
    assert payload["telemetry"]["metrics_path"] == "/metrics"
    assert payload["inferencer"]["implementation"] == "feature_statistics"
    assert payload["model_bundle"]["loaded"] is True
    assert payload["model_bundle"]["model_version"] == "demo-onnx-stub-v1"
    assert payload["model_bundle"]["input_name"] == "encoder_input"
    assert payload["model_bundle"]["output_name"] == "embedding"
    assert payload["model_bundle"]["export_boundary"]["boundary"] == "encoder_only"
    assert payload["model_bundle"]["export_boundary"]["frontend_location"] == "runtime"
    assert "/health" in openapi_payload["paths"]
    assert "/metrics" in openapi_payload["paths"]
    assert "/demo/api/state" in openapi_payload["paths"]
    assert "/demo/api/compare" in openapi_payload["paths"]
    assert "/demo/api/enroll" in openapi_payload["paths"]
    assert "/demo/api/verify" in openapi_payload["paths"]
    assert "/embed" in openapi_payload["paths"]
    assert "/enroll" in openapi_payload["paths"]
    assert "/verify" in openapi_payload["paths"]
    assert "/benchmark" in openapi_payload["paths"]


def _build_demo_config(tmp_path: Path):
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
    return config
