from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import create_http_server


def test_pairwise_scoring_endpoint_returns_cosine_scores(monkeypatch, tmp_path: Path) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/score/pairwise",
            {
                "left": [[3.0, 0.0], [1.0, 1.0]],
                "right": [[9.0, 0.0], [1.0, -1.0]],
            },
        )
    finally:
        _stop_server(server, thread)

    assert status == 200
    assert payload["mode"] == "pairwise"
    assert payload["embedding_dim"] == 2
    assert payload["scores"] == [1.0, 0.0]


def test_one_to_many_scoring_endpoint_returns_ranked_matches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/score/one-to-many",
            {
                "queries": [[1.0, 0.0]],
                "references": [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]],
                "query_ids": ["probe-a"],
                "reference_ids": ["ref-perfect", "ref-close", "ref-opposite"],
                "top_k": 2,
            },
        )
    finally:
        _stop_server(server, thread)

    assert status == 200
    assert payload["mode"] == "one_to_many"
    assert payload["scores"][0][0] == 1.0
    assert payload["top_matches"] == [
        {
            "query_id": "probe-a",
            "matches": [
                {"reference_id": "ref-perfect", "score": 1.0},
                {"reference_id": "ref-close", "score": 0.70710678},
            ],
        }
    ]


def test_embed_endpoint_returns_runtime_embeddings_from_audio_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/embed",
            {
                "audio_paths": ["artifacts/demo-subset/test/speaker_alpha-test_01.wav"],
            },
        )
    finally:
        _stop_server(server, thread)

    assert status == 200
    assert payload["mode"] == "embed"
    assert payload["embedding_dim"] == 160
    assert payload["item_count"] == 1
    assert payload["backend"]["implementation"] == "feature_statistics"
    assert len(payload["items"][0]["embedding"]) == 160


def test_metrics_endpoint_reports_http_inference_and_validation_counters(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        status, _ = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/embed",
            {
                "audio_paths": ["artifacts/demo-subset/test/speaker_alpha-test_01.wav"],
            },
        )
        invalid_status, invalid_payload = _post_json_allow_error(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {"enrollment_id": "speaker_alpha"},
        )
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/metrics") as response:
            metrics_payload = response.read().decode("utf-8")
            content_type = response.headers["Content-Type"]
    finally:
        _stop_server(server, thread)

    assert status == 200
    assert invalid_status == 422
    assert invalid_payload["status"] == "error"
    assert content_type.startswith("text/plain")
    assert (
        'kryptonite_http_requests_total{method="POST",path="/embed",status="200"} 1'
        in metrics_payload
    )
    assert (
        'kryptonite_inference_operations_total{backend="onnxruntime",'
        'model_version="demo-onnx-stub-v1",operation="embed",stage="demo"} 1' in metrics_payload
    )
    assert (
        'kryptonite_validation_errors_total{error_type="RequestValidationError",'
        'path="/verify",status="422"} 1' in metrics_payload
    )


def test_embed_endpoint_emits_structured_telemetry_logs(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="kryptonite.serve.telemetry")

    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        status, _ = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/embed",
            {
                "audio_paths": ["artifacts/demo-subset/test/speaker_alpha-test_01.wav"],
            },
        )
    finally:
        _stop_server(server, thread)

    assert status == 200
    payloads = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "kryptonite.serve.telemetry" and record.getMessage().startswith("{")
    ]
    inference_logs = [
        payload
        for payload in payloads
        if payload.get("event") == "inference_operation" and payload.get("operation") == "embed"
    ]

    assert inference_logs
    log_payload = inference_logs[-1]
    assert log_payload["backend"] == "onnxruntime"
    assert log_payload["implementation"] == "feature_statistics"
    assert log_payload["model_version"] == "demo-onnx-stub-v1"
    assert log_payload["audio_count"] == 1
    assert log_payload["total_audio_duration_seconds"] > 0.0
    assert log_payload["total_chunk_count"] >= 1
    assert log_payload["latency_ms"] >= 0.0


def test_enroll_and_verify_endpoints_share_the_same_scoring_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/enrollments") as response:
            initial_enrollments_payload = json.loads(response.read().decode("utf-8"))
        enroll_status, enroll_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/enroll",
            {
                "enrollment_id": "speaker-charlie",
                "embeddings": [[2.0, 0.0], [6.0, 0.0]],
                "metadata": {"source": "integration-test"},
            },
        )
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/enrollments") as response:
            enrollments_payload = json.loads(response.read().decode("utf-8"))
        verify_status, verify_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {
                "enrollment_id": "speaker-charlie",
                "probe": [1.0, 0.0],
                "threshold": 0.9,
            },
        )
    finally:
        _stop_server(server, thread)

    assert enroll_status == 201
    assert enroll_payload["sample_count"] == 2
    assert initial_enrollments_payload["enrollment_count"] == 2
    assert enrollments_payload["enrollment_count"] == 3
    assert {enrollment["enrollment_id"] for enrollment in enrollments_payload["enrollments"]} == {
        "speaker_alpha",
        "speaker_bravo",
        "speaker-charlie",
    }
    assert verify_status == 200
    assert verify_payload["mode"] == "verify"
    assert verify_payload["scores"] == [1.0]
    assert verify_payload["decisions"] == [True]


def test_enrollments_persist_across_server_restart(monkeypatch, tmp_path: Path) -> None:
    first_server, first_thread = _start_server(monkeypatch, tmp_path)
    try:
        enroll_status, _ = _post_json(
            f"http://127.0.0.1:{first_server.server_address[1]}/enroll",
            {
                "enrollment_id": "speaker-charlie",
                "embeddings": [[2.0, 0.0], [6.0, 0.0]],
                "metadata": {"source": "restart-test"},
            },
        )
    finally:
        _stop_server(first_server, first_thread)

    second_server, second_thread = _start_server(monkeypatch, tmp_path)
    try:
        with urlopen(f"http://127.0.0.1:{second_server.server_address[1]}/enrollments") as response:
            enrollments_payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{second_server.server_address[1]}/health") as response:
            health_payload = json.loads(response.read().decode("utf-8"))
        verify_status, verify_payload = _post_json(
            f"http://127.0.0.1:{second_server.server_address[1]}/verify",
            {
                "enrollment_id": "speaker-charlie",
                "probe": [1.0, 0.0],
                "threshold": 0.9,
            },
        )
    finally:
        _stop_server(second_server, second_thread)

    assert enroll_status == 201
    assert enrollments_payload["enrollment_count"] == 3
    assert {enrollment["enrollment_id"] for enrollment in enrollments_payload["enrollments"]} == {
        "speaker_alpha",
        "speaker_bravo",
        "speaker-charlie",
    }
    assert health_payload["enrollment_cache"]["runtime_store"]["loaded"] is True
    assert health_payload["enrollment_cache"]["runtime_store"]["enrollment_count"] == 1
    assert verify_status == 200
    assert verify_payload["scores"] == [1.0]
    assert verify_payload["decisions"] == [True]


def test_verify_endpoint_accepts_audio_paths_against_preloaded_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        alpha_status, alpha_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {
                "enrollment_id": "speaker_alpha",
                "audio_path": "artifacts/demo-subset/test/speaker_alpha-test_01.wav",
            },
        )
        bravo_status, bravo_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {
                "enrollment_id": "speaker_alpha",
                "audio_path": "artifacts/demo-subset/test/speaker_bravo-test_01.wav",
            },
        )
    finally:
        _stop_server(server, thread)

    assert alpha_status == 200
    assert bravo_status == 200
    assert alpha_payload["scores"][0] > bravo_payload["scores"][0]
    assert alpha_payload["backend"]["implementation"] == "feature_statistics"
    assert alpha_payload["probe_items"][0]["chunk_count"] >= 1


def _start_server(
    monkeypatch,
    tmp_path: Path,
) -> tuple[Any, threading.Thread]:
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
    server.wait_started()
    return server, thread


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


def _stop_server(server: Any, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


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
