from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import Inferencer, create_http_server

ALPHA_AUDIO_PATH = "artifacts/demo-subset/test/speaker_alpha-test_01.wav"
BRAVO_AUDIO_PATH = "artifacts/demo-subset/test/speaker_bravo-test_01.wav"


@pytest.mark.parametrize(
    ("requested_backend", "resolved_backend"),
    (("auto", "torch"), ("torch", "torch")),
)
def test_end_to_end_regression_suite_exercises_backend_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requested_backend: str,
    resolved_backend: str,
) -> None:
    pytest.importorskip("torch")
    config = _build_demo_config(tmp_path, requested_backend=requested_backend)
    _patch_runtime_backend_probe(monkeypatch)

    inferencer = Inferencer.from_config(config=config)
    server, thread = _start_server(config=config)
    try:
        health_payload = _get_json(f"http://127.0.0.1:{server.server_address[1]}/health")
        openapi_payload = _get_json(f"http://127.0.0.1:{server.server_address[1]}/openapi.json")
        embed_status, embed_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/embed",
            {"audio_path": ALPHA_AUDIO_PATH, "stage": "demo"},
        )
        verify_status, verify_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {
                "enrollment_id": "speaker_alpha",
                "audio_path": ALPHA_AUDIO_PATH,
                "stage": "demo",
                "threshold": 0.995,
            },
        )
        negative_status, negative_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {
                "enrollment_id": "speaker_alpha",
                "audio_path": BRAVO_AUDIO_PATH,
                "stage": "demo",
                "threshold": 0.995,
            },
        )
        benchmark_status, benchmark_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/benchmark",
            {
                "audio_paths": [ALPHA_AUDIO_PATH, BRAVO_AUDIO_PATH],
                "stage": "eval",
                "iterations": 2,
                "warmup_iterations": 1,
            },
        )
        invalid_status, invalid_payload = _post_json_allow_error(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {"enrollment_id": "speaker_alpha"},
        )
        metrics_payload = _get_text(f"http://127.0.0.1:{server.server_address[1]}/metrics")
    finally:
        _stop_server(server, thread)

    local_embed_payload = inferencer.embed_audio_paths(
        audio_paths=[ALPHA_AUDIO_PATH],
        stage="demo",
    )
    local_verify_payload = inferencer.verify_audio_paths(
        enrollment_id="speaker_alpha",
        audio_paths=[ALPHA_AUDIO_PATH],
        stage="demo",
        threshold=0.995,
    )

    assert health_payload["requested_backend"] == requested_backend
    assert health_payload["selected_backend"] == resolved_backend
    assert health_payload["selected_provider"] is None
    assert health_payload["telemetry"]["backend"] == resolved_backend
    assert health_payload["telemetry"]["implementation"] == "feature_statistics"
    assert health_payload["model_bundle"]["model_version"] == "demo-onnx-stub-v1"
    assert "/embed" in openapi_payload["paths"]
    assert "/verify" in openapi_payload["paths"]
    assert "/benchmark" in openapi_payload["paths"]
    assert "/metrics" in openapi_payload["paths"]

    assert embed_status == 200
    assert embed_payload == local_embed_payload
    assert embed_payload["backend"]["implementation"] == "feature_statistics"

    assert verify_status == 200
    assert verify_payload == local_verify_payload
    assert verify_payload["scores"][0] > negative_payload["scores"][0]
    assert verify_payload["decisions"] == [True]

    assert negative_status == 200
    assert negative_payload["decisions"] == [False]

    assert benchmark_status == 200
    assert benchmark_payload["mode"] == "benchmark"
    assert benchmark_payload["stage"] == "eval"
    assert benchmark_payload["iterations"] == 2
    assert benchmark_payload["warmup_iterations"] == 1
    assert benchmark_payload["backend"]["implementation"] == "feature_statistics"
    assert benchmark_payload["total_chunk_count"] >= 2
    assert benchmark_payload["mean_iteration_seconds"] >= 0.0
    assert benchmark_payload["mean_ms_per_audio"] >= 0.0

    assert invalid_status == 422
    assert invalid_payload["status"] == "error"

    assert (
        'kryptonite_http_requests_total{method="POST",path="/embed",status="200"} 1'
        in metrics_payload
    )
    assert (
        'kryptonite_http_requests_total{method="POST",path="/verify",status="200"} 2'
        in metrics_payload
    )
    assert (
        'kryptonite_http_requests_total{method="POST",path="/verify",status="422"} 1'
        in metrics_payload
    )
    assert (
        _inference_operations_metric_line(
            backend=resolved_backend,
            operation="embed",
            stage="demo",
            count=1,
        )
        in metrics_payload
    )
    assert (
        _inference_operations_metric_line(
            backend=resolved_backend,
            operation="verify",
            stage="demo",
            count=2,
        )
        in metrics_payload
    )
    assert (
        _inference_operations_metric_line(
            backend=resolved_backend,
            operation="benchmark",
            stage="eval",
            count=1,
        )
        in metrics_payload
    )
    assert (
        'kryptonite_validation_errors_total{error_type="RequestValidationError",'
        'path="/verify",status="422"} 1' in metrics_payload
    )


def _build_demo_config(tmp_path: Path, *, requested_backend: str) -> ProjectConfig:
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
            'deployment.enrollment_cache_root="artifacts/enrollment-cache"',
            f'backends.inference="{requested_backend}"',
            "backends.allow_torch=true",
            "backends.allow_onnx=true",
            "backends.allow_tensorrt=false",
        ],
    )
    generate_demo_artifacts(config=config)
    return config


def _patch_runtime_backend_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_modules = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda=None),
        ),
        "onnxruntime": SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
        "tensorrt": SimpleNamespace(Logger=object),
    }
    distribution_versions = {
        "torch": "2.10.0",
        "onnxruntime": "1.24.4",
        "tensorrt-cu12": "10.1.0",
    }

    def fake_load_module(module_name: str) -> object:
        if module_name in backend_modules:
            return backend_modules[module_name]
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(
        serve_runtime,
        "_distribution_version",
        lambda distribution: distribution_versions.get(distribution),
    )


def _start_server(
    *,
    config: ProjectConfig,
) -> tuple[Any, threading.Thread]:
    server = create_http_server(host="127.0.0.1", port=0, config=config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.wait_started()
    return server, thread


def _stop_server(server: Any, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, payload: dict[str, object]) -> tuple[int, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json_allow_error(url: str, payload: dict[str, object]) -> tuple[int, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _inference_operations_metric_line(
    *,
    backend: str,
    operation: str,
    stage: str,
    count: int,
) -> str:
    return (
        "kryptonite_inference_operations_total"
        f'{{backend="{backend}",model_version="demo-onnx-stub-v1",'
        f'operation="{operation}",stage="{stage}"}} {count}'
    )
