from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.serve import create_http_server


def test_health_endpoint_reports_selected_backend(monkeypatch) -> None:
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
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/healthz") as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert payload["service"] == "kryptonite-infer"
    assert payload["selected_backend"] == "onnxruntime"
    assert payload["status"] == "ok"
    assert payload["artifacts"]["scope"] == "infer"
    assert payload["artifacts"]["strict"] is False
