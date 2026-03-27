from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import create_http_server


def test_demo_page_and_state_endpoint_expose_runtime_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/demo") as response:
            demo_html = response.read().decode("utf-8")
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/demo/api/state") as response:
            state_payload = json.loads(response.read().decode("utf-8"))
    finally:
        _stop_server(server, thread)

    assert "Kryptonite Speaker Demo" in demo_html
    assert state_payload["service"]["requested_backend"] == "auto"
    assert state_payload["service"]["selected_backend"] == "torch"
    assert state_payload["threshold"]["source"] == "builtin_default"
    assert state_payload["threshold"]["value"] == 0.995
    assert state_payload["default_stage"] == "demo"
    assert state_payload["enrollment_count"] == 2


def test_demo_compare_and_verify_accept_base64_audio_uploads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, thread = _start_server(monkeypatch, tmp_path)
    try:
        alpha_probe = _audio_upload(
            tmp_path / "artifacts/demo-subset/test/speaker_alpha-test_01.wav"
        )
        compare_status, compare_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/demo/api/compare",
            {
                "left_audio": alpha_probe,
                "right_audio": alpha_probe,
                "threshold": 0.995,
            },
        )
        verify_status, verify_payload = _post_json(
            f"http://127.0.0.1:{server.server_address[1]}/demo/api/verify",
            {
                "enrollment_id": "speaker_alpha",
                "audio_file": alpha_probe,
                "threshold": 0.995,
            },
        )
    finally:
        _stop_server(server, thread)

    assert compare_status == 200
    assert compare_payload["mode"] == "demo_compare"
    assert compare_payload["decision"] is True
    assert compare_payload["score"] > 0.999999
    assert compare_payload["backend"]["implementation"] == "feature_statistics"
    assert compare_payload["latency_ms"] >= 0.0

    assert verify_status == 200
    assert verify_payload["mode"] == "demo_verify"
    assert verify_payload["enrollment_id"] == "speaker_alpha"
    assert verify_payload["decision"] is True
    assert verify_payload["score"] >= verify_payload["threshold"]["value"]
    assert verify_payload["backend"]["implementation"] == "feature_statistics"


def test_demo_enrollment_persists_across_server_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_server, first_thread = _start_server(monkeypatch, tmp_path)
    try:
        enroll_status, enroll_payload = _post_json(
            f"http://127.0.0.1:{first_server.server_address[1]}/demo/api/enroll",
            {
                "enrollment_id": "speaker_charlie",
                "audio_files": [
                    _audio_upload(
                        tmp_path / "artifacts/demo-subset/enrollment/speaker_alpha-enroll_01.wav"
                    )
                ],
            },
        )
    finally:
        _stop_server(first_server, first_thread)

    second_server, second_thread = _start_server(monkeypatch, tmp_path)
    try:
        with urlopen(
            f"http://127.0.0.1:{second_server.server_address[1]}/demo/api/state"
        ) as response:
            state_payload = json.loads(response.read().decode("utf-8"))
    finally:
        _stop_server(second_server, second_thread)

    assert enroll_status == 200
    assert enroll_payload["mode"] == "demo_enroll"
    assert enroll_payload["replaced"] is False
    assert enroll_payload["sample_count"] == 1
    assert state_payload["enrollment_count"] == 3
    assert {enrollment["enrollment_id"] for enrollment in state_payload["enrollments"]} == {
        "speaker_alpha",
        "speaker_bravo",
        "speaker_charlie",
    }
    assert state_payload["service"]["runtime_store"]["loaded"] is True


def _audio_upload(path: Path) -> dict[str, str]:
    return {
        "filename": path.name,
        "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _start_server(
    monkeypatch,
    tmp_path: Path,
):
    config = _build_demo_config(tmp_path)

    def fake_load_module(module_name: str) -> object:
        if module_name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                version=SimpleNamespace(cuda=None),
            )
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


def _stop_server(server, thread: threading.Thread) -> None:
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
