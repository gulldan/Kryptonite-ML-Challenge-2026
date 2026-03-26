from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.serve import create_http_server


def test_pairwise_scoring_endpoint_returns_cosine_scores(monkeypatch) -> None:
    server, thread = _start_server(monkeypatch)
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


def test_one_to_many_scoring_endpoint_returns_ranked_matches(monkeypatch) -> None:
    server, thread = _start_server(monkeypatch)
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


def test_enroll_and_verify_endpoints_share_the_same_scoring_state(monkeypatch) -> None:
    server, thread = _start_server(monkeypatch)
    try:
        enroll_status, enroll_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/enroll",
            {
                "enrollment_id": "speaker-alpha",
                "embeddings": [[2.0, 0.0], [6.0, 0.0]],
                "metadata": {"source": "integration-test"},
            },
        )
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/enrollments") as response:
            enrollments_payload = json.loads(response.read().decode("utf-8"))
        verify_status, verify_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/verify",
            {
                "enrollment_id": "speaker-alpha",
                "probe": [1.0, 0.0],
                "threshold": 0.9,
            },
        )
    finally:
        _stop_server(server, thread)

    assert enroll_status == 201
    assert enroll_payload["sample_count"] == 2
    assert enrollments_payload["enrollment_count"] == 1
    assert enrollments_payload["enrollments"][0]["enrollment_id"] == "speaker-alpha"
    assert verify_status == 200
    assert verify_payload["mode"] == "verify"
    assert verify_payload["scores"] == [1.0]
    assert verify_payload["decisions"] == [True]


def _start_server(monkeypatch) -> tuple[ThreadingHTTPServer, threading.Thread]:
    config = load_project_config(config_path=Path("configs/deployment/infer.toml"))

    def fake_load_module(module_name: str) -> object:
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: "1.0.0")

    server = create_http_server(host="127.0.0.1", port=0, config=config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
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
