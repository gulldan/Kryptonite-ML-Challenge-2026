from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from kryptonite.serve.triton_smoke import run_triton_infer_smoke


def test_run_triton_infer_smoke_probes_ready_and_infer(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "name": "encoder_input",
                        "shape": [1, 12, 80],
                        "datatype": "FP32",
                        "data": [[[0.1 for _ in range(80)] for _ in range(12)]],
                    }
                ],
                "outputs": [{"name": "embedding"}],
            }
        ),
        encoding="utf-8",
    )

    server = HTTPServer(("127.0.0.1", 0), _TritonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = run_triton_infer_smoke(
            server_url=f"http://127.0.0.1:{server.server_port}",
            model_name="kryptonite_encoder",
            request_path=request_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.model_name == "kryptonite_encoder"
    assert result.output_name == "embedding"
    assert result.output_shape == (1, 160)
    assert result.output_datatype == "FP32"
    assert result.ready_latency_seconds >= 0.0
    assert result.infer_latency_seconds >= 0.0


class _TritonHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/v2/health/ready", "/v2/models/kryptonite_encoder/ready"}:
            self.send_response(200)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v2/models/kryptonite_encoder/infer":
            self.send_response(404)
            self.end_headers()
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        assert payload["inputs"][0]["name"] == "encoder_input"
        assert payload["outputs"][0]["name"] == "embedding"

        body = json.dumps(
            {
                "model_name": "kryptonite_encoder",
                "outputs": [
                    {
                        "name": "embedding",
                        "datatype": "FP32",
                        "shape": [1, 160],
                        "data": [[0.0] * 160],
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return
