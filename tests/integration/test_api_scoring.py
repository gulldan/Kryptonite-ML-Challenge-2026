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
) -> tuple[ThreadingHTTPServer, threading.Thread]:
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
