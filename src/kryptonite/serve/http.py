"""Thin HTTP adapter for the current inference runtime."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kryptonite.config import ProjectConfig

from .runtime import (
    build_serve_runtime_report,
    build_service_metadata,
    render_serve_runtime_report,
)


def create_http_server(
    *,
    host: str,
    port: int,
    config: ProjectConfig,
) -> ThreadingHTTPServer:
    report = build_serve_runtime_report(config=config)
    if not report.passed:
        raise RuntimeError(render_serve_runtime_report(report))

    payload = build_service_metadata(config=config, report=report)
    return ThreadingHTTPServer((host, port), _build_handler(payload))


def run_http_server(
    *,
    host: str,
    port: int,
    config: ProjectConfig,
) -> None:
    server = create_http_server(host=host, port=port, config=config)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_handler(payload: dict[str, object]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/healthz", "/readyz"}:
                self._write_json(HTTPStatus.OK, payload)
                return
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {
                    "status": "not_found",
                    "path": self.path,
                },
            )

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, status: HTTPStatus, data: dict[str, object]) -> None:
            body = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler
