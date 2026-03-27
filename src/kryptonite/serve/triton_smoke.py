"""Thin HTTP smoke helpers for Triton's KServe-compatible infer API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class TritonSmokeResult:
    server_url: str
    model_name: str
    ready_latency_seconds: float
    infer_latency_seconds: float
    output_name: str
    output_shape: tuple[int, ...]
    output_datatype: str

    def to_dict(self) -> dict[str, object]:
        return {
            "server_url": self.server_url,
            "model_name": self.model_name,
            "ready_latency_seconds": round(self.ready_latency_seconds, 8),
            "infer_latency_seconds": round(self.infer_latency_seconds, 8),
            "output_name": self.output_name,
            "output_shape": list(self.output_shape),
            "output_datatype": self.output_datatype,
        }


def run_triton_infer_smoke(
    *,
    server_url: str,
    model_name: str,
    request_path: Path | str,
    timeout_seconds: float = 10.0,
) -> TritonSmokeResult:
    normalized_server_url = server_url.rstrip("/")
    if not normalized_server_url:
        raise ValueError("server_url must be a non-empty URL.")
    if not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive.")

    ready_started = time.perf_counter()
    _read_json(
        f"{normalized_server_url}/v2/health/ready",
        timeout_seconds=timeout_seconds,
    )
    _read_json(
        f"{normalized_server_url}/v2/models/{model_name}/ready",
        timeout_seconds=timeout_seconds,
    )
    ready_latency_seconds = time.perf_counter() - ready_started

    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    infer_started = time.perf_counter()
    infer_response = _post_json(
        f"{normalized_server_url}/v2/models/{model_name}/infer",
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    infer_latency_seconds = time.perf_counter() - infer_started

    outputs = infer_response.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError("Triton infer response does not include a non-empty outputs list.")
    first_output = outputs[0]
    if not isinstance(first_output, dict):
        raise RuntimeError("Triton infer response outputs entries must be JSON objects.")
    output_name = first_output.get("name")
    output_shape = first_output.get("shape")
    output_datatype = first_output.get("datatype")
    if not isinstance(output_name, str) or not output_name:
        raise RuntimeError("Triton infer response output is missing a valid name.")
    if not isinstance(output_shape, list) or not all(
        isinstance(item, int) for item in output_shape
    ):
        raise RuntimeError("Triton infer response output is missing a valid integer shape.")
    if not isinstance(output_datatype, str) or not output_datatype:
        raise RuntimeError("Triton infer response output is missing a valid datatype.")

    return TritonSmokeResult(
        server_url=normalized_server_url,
        model_name=model_name,
        ready_latency_seconds=ready_latency_seconds,
        infer_latency_seconds=infer_latency_seconds,
        output_name=output_name,
        output_shape=tuple(output_shape),
        output_datatype=output_datatype,
    )


def render_triton_smoke_result(result: TritonSmokeResult) -> str:
    return "\n".join(
        [
            "Triton infer smoke: PASS",
            f"Server: {result.server_url}",
            f"Model: {result.model_name}",
            f"Ready latency: {result.ready_latency_seconds:.6f}s",
            f"Infer latency: {result.infer_latency_seconds:.6f}s",
            f"Output: {result.output_name} shape={list(result.output_shape)} "
            f"datatype={result.output_datatype}",
        ]
    )


def _read_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:  # pragma: no cover - depends on remote server behavior
        raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
    except URLError as exc:  # pragma: no cover - depends on remote server behavior
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc
    if not body.strip():
        return {}
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(payload).__name__}.")
    return payload


def _post_json(url: str, *, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:  # pragma: no cover - depends on remote server behavior
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:  # pragma: no cover - depends on remote server behavior
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(parsed).__name__}.")
    return parsed


__all__ = [
    "TritonSmokeResult",
    "render_triton_smoke_result",
    "run_triton_infer_smoke",
]
