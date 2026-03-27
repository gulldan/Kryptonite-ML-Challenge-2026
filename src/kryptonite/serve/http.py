"""FastAPI transport adapter for runtime metadata and unified inference flows."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from kryptonite.config import ProjectConfig
from kryptonite.deployment import resolve_project_path

from .api_models import (
    BenchmarkAudioRequest,
    DemoCompareRequest,
    DemoEnrollmentRequest,
    DemoVerifyRequest,
    EmbedAudioRequest,
    EnrollmentRequest,
    OneToManyScoringRequest,
    PairwiseScoringRequest,
    VerifyRequest,
)
from .demo import (
    build_demo_state,
    resolve_demo_threshold,
    run_demo_compare,
    run_demo_enroll,
    run_demo_verify,
)
from .demo_ui import render_demo_page
from .inferencer import Inferencer
from .scoring_service import EnrollmentNotFoundError
from .telemetry import PROMETHEUS_CONTENT_TYPE, ServiceTelemetry


def create_http_server(
    *,
    host: str,
    port: int,
    config: ProjectConfig,
    require_artifacts: bool = False,
) -> FastAPIServerHandle:
    app = create_http_app(config=config, require_artifacts=require_artifacts)
    return FastAPIServerHandle(app=app, host=host, port=port)


def create_http_app(
    *,
    config: ProjectConfig,
    require_artifacts: bool = False,
) -> FastAPI:
    try:
        inferencer = Inferencer.from_config(
            config=config,
            require_artifacts=require_artifacts,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    telemetry = ServiceTelemetry.from_inferencer(
        config=config,
        service="kryptonite-infer",
        backend=inferencer.selected_backend,
        implementation=inferencer.inferencer_implementation,
        model_version=inferencer.model_version,
    )
    telemetry.record_service_start(
        log_level=config.runtime.log_level,
        strict_artifacts=require_artifacts or config.deployment.require_artifacts,
    )
    return _build_app(inferencer, config=config, telemetry=telemetry)


def run_http_server(
    *,
    host: str,
    port: int,
    config: ProjectConfig,
    require_artifacts: bool = False,
) -> None:
    server = create_http_server(
        host=host,
        port=port,
        config=config,
        require_artifacts=require_artifacts,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_app(
    inferencer: Inferencer,
    *,
    config: ProjectConfig,
    telemetry: ServiceTelemetry,
) -> FastAPI:
    app = FastAPI(
        title="Kryptonite Inference API",
        version="0.1.0",
        summary="Thin FastAPI adapter over the unified Kryptonite inferencer.",
    )
    demo_threshold = resolve_demo_threshold(config=config)
    demo_root = resolve_project_path(config.paths.project_root, config.deployment.demo_subset_root)

    @app.exception_handler(EnrollmentNotFoundError)
    async def _handle_enrollment_not_found(
        _request: object,
        exc: EnrollmentNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content=_error_payload(str(exc)))

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = _sanitize_validation_errors(exc.errors())
        telemetry.record_validation_error(
            path=_resolve_request_path(request),
            error_type=type(exc).__name__,
            status_code=422,
            message="Request validation failed.",
            details=errors,
        )
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "Request validation failed.",
                details={"errors": errors},
            ),
        )

    @app.exception_handler(ValueError)
    async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        telemetry.record_validation_error(
            path=_resolve_request_path(request),
            error_type=type(exc).__name__,
            status_code=400,
            message=str(exc),
        )
        return JSONResponse(status_code=400, content=_error_payload(str(exc)))

    @app.middleware("http")
    async def _telemetry_middleware(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            telemetry.record_http_request(
                method=request.method,
                path=_resolve_request_path(request),
                status_code=500,
                duration_seconds=time.perf_counter() - started,
            )
            raise

        telemetry.record_http_request(
            method=request.method,
            path=_resolve_request_path(request),
            status_code=response.status_code,
            duration_seconds=time.perf_counter() - started,
        )
        return response

    @app.get("/healthz", include_in_schema=False)
    @app.get("/readyz", include_in_schema=False)
    @app.get("/health", tags=["health"])
    async def health() -> dict[str, Any]:
        payload = inferencer.health_payload()
        payload["telemetry"] = telemetry.summary()
        return payload

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    @app.get("/demo", tags=["demo"], response_class=HTMLResponse)
    async def demo_page() -> HTMLResponse:
        return HTMLResponse(render_demo_page())

    @app.get("/demo/api/state", tags=["demo"])
    async def demo_state() -> dict[str, object]:
        return build_demo_state(
            inferencer=inferencer,
            config=config,
            threshold=demo_threshold,
        )

    @app.post("/demo/api/compare", tags=["demo"])
    async def demo_compare(request: DemoCompareRequest) -> dict[str, object]:
        payload = run_demo_compare(
            inferencer=inferencer,
            request=request,
            default_threshold=demo_threshold,
        )
        _record_audio_operation_from_items(
            telemetry,
            operation="demo_compare",
            stage=str(payload["stage"]),
            items=(
                _as_mapping(payload["left_audio"]),
                _as_mapping(payload["right_audio"]),
            ),
            latency_seconds=_latency_seconds_from_payload(payload),
            extra={
                "decision": bool(payload["decision"]),
                "normalized": bool(payload["normalized"]),
            },
        )
        return payload

    @app.post("/demo/api/enroll", tags=["demo"])
    async def demo_enroll(request: DemoEnrollmentRequest) -> dict[str, object]:
        payload = run_demo_enroll(inferencer=inferencer, request=request)
        _record_audio_operation_from_items(
            telemetry,
            operation="demo_enroll",
            stage=str(payload["stage"]),
            items=_as_mapping_sequence(payload["audio_items"]),
            latency_seconds=_latency_seconds_from_payload(payload),
            extra={"replaced": bool(payload["replaced"])},
        )
        return payload

    @app.post("/demo/api/verify", tags=["demo"])
    async def demo_verify(request: DemoVerifyRequest) -> dict[str, object]:
        payload = run_demo_verify(
            inferencer=inferencer,
            request=request,
            default_threshold=demo_threshold,
        )
        _record_audio_operation_from_items(
            telemetry,
            operation="demo_verify",
            stage=str(payload["stage"]),
            items=(_as_mapping(payload["probe_audio"]),),
            latency_seconds=_latency_seconds_from_payload(payload),
            extra={
                "decision": bool(payload["decision"]),
                "normalized": bool(payload["normalized"]),
            },
        )
        return payload

    @app.get("/enrollments", tags=["enrollment"])
    async def enrollments() -> dict[str, Any]:
        return inferencer.list_enrollments()

    @app.post("/score/pairwise", tags=["scoring"])
    async def score_pairwise(request: PairwiseScoringRequest) -> dict[str, Any]:
        return inferencer.score_pairwise(
            left=request.left,
            right=request.right,
            normalize=request.normalize,
        )

    @app.post("/score/one-to-many", tags=["scoring"])
    async def score_one_to_many(request: OneToManyScoringRequest) -> dict[str, Any]:
        return inferencer.score_one_to_many(
            queries=request.queries,
            references=request.references,
            normalize=request.normalize,
            top_k=request.top_k,
            query_ids=request.query_ids,
            reference_ids=request.reference_ids,
        )

    @app.post("/embed", tags=["inference"])
    async def embed(request: EmbedAudioRequest) -> dict[str, Any]:
        started = time.perf_counter()
        payload = inferencer.embed_audio_paths(
            audio_paths=request.resolve_audio_paths(),
            stage=request.stage,
        )
        _record_audio_operation_from_items(
            telemetry,
            operation="embed",
            stage=str(payload["stage"]),
            items=_as_mapping_sequence(payload["items"]),
            latency_seconds=time.perf_counter() - started,
            total_chunk_count=_coerce_int(payload.get("total_chunk_count")),
        )
        return payload

    @app.post("/benchmark", tags=["inference"])
    async def benchmark(request: BenchmarkAudioRequest) -> dict[str, Any]:
        started = time.perf_counter()
        payload = inferencer.benchmark_audio_paths(
            audio_paths=request.resolve_audio_paths(),
            stage=request.stage,
            iterations=request.iterations,
            warmup_iterations=request.warmup_iterations,
        )
        telemetry.record_inference_operation(
            operation="benchmark",
            stage=str(payload["stage"]),
            audio_count=_coerce_int(payload.get("audio_count")),
            total_audio_duration_seconds=_coerce_float(payload.get("total_audio_duration_seconds")),
            total_chunk_count=_coerce_int(payload.get("total_chunk_count")),
            latency_seconds=time.perf_counter() - started,
            extra={
                "iterations": _coerce_int(payload.get("iterations")),
                "warmup_iterations": _coerce_int(payload.get("warmup_iterations")),
                "mean_iteration_seconds": _coerce_float(payload.get("mean_iteration_seconds")),
            },
        )
        return payload

    @app.post("/enroll", tags=["enrollment"])
    async def enroll(request: EnrollmentRequest, response: Response) -> dict[str, Any]:
        if request.uses_audio_paths:
            started = time.perf_counter()
            payload = inferencer.enroll_audio_paths(
                enrollment_id=request.enrollment_id,
                audio_paths=request.resolve_audio_paths(),
                stage=request.stage,
                metadata=request.metadata,
            )
            _record_audio_operation_from_items(
                telemetry,
                operation="enroll",
                stage=str(payload["stage"]),
                items=_as_mapping_sequence(payload["audio_items"]),
                latency_seconds=time.perf_counter() - started,
                extra={"replaced": bool(payload["replaced"])},
            )
        else:
            payload = inferencer.enroll_embeddings(
                enrollment_id=request.enrollment_id,
                embeddings=request.resolve_embeddings(),
                metadata=request.metadata,
            )
        response.status_code = (
            status.HTTP_200_OK if payload["replaced"] else status.HTTP_201_CREATED
        )
        return payload

    @app.post("/verify", tags=["inference"])
    async def verify(request: VerifyRequest) -> dict[str, Any]:
        if request.uses_audio_paths:
            started = time.perf_counter()
            payload = inferencer.verify_audio_paths(
                enrollment_id=request.enrollment_id,
                audio_paths=request.resolve_audio_paths(),
                stage=request.stage,
                normalize=request.normalize,
                threshold=request.threshold,
            )
            _record_audio_operation_from_items(
                telemetry,
                operation="verify",
                stage=str(payload["stage"]),
                items=_as_mapping_sequence(payload["probe_items"]),
                latency_seconds=time.perf_counter() - started,
                extra=_verify_operation_extra(payload),
            )
            return payload
        return inferencer.verify_embeddings(
            enrollment_id=request.enrollment_id,
            probes=request.resolve_probes(),
            normalize=request.normalize,
            threshold=request.threshold,
        )

    if telemetry.metrics_enabled:

        async def metrics() -> Response:
            return Response(
                content=telemetry.render_prometheus(),
                media_type=PROMETHEUS_CONTENT_TYPE,
            )

        app.add_api_route(
            telemetry.metrics_path,
            metrics,
            methods=["GET"],
            tags=["telemetry"],
        )

    if not demo_root.exists():
        app.extra["demo_warning"] = f"Configured demo subset root is missing: {demo_root}"
    return app


@dataclass(slots=True)
class FastAPIServerHandle:
    app: FastAPI
    host: str
    port: int
    server_address: tuple[str, int] = field(init=False)
    _socket: socket.socket = field(init=False, repr=False)
    _server: uvicorn.Server = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._socket = socket.create_server((self.host, self.port), backlog=2048)
        self.server_address = self._socket.getsockname()[:2]
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=int(self.server_address[1]),
            access_log=False,
            log_level="warning",
        )
        self._server = _EmbeddedUvicornServer(config)

    def serve_forever(self) -> None:
        try:
            asyncio.run(self._server.serve(sockets=[self._socket]))
        except KeyboardInterrupt:
            return

    def shutdown(self) -> None:
        self._server.should_exit = True

    def server_close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            return

    def wait_started(self, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.started:
                return
            time.sleep(0.01)
        raise TimeoutError("Timed out while waiting for the FastAPI server to start.")

    @property
    def started(self) -> bool:
        return bool(getattr(self._server, "started", False))


class _EmbeddedUvicornServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        return


def _error_payload(
    message: str,
    *,
    details: object | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "error",
        "message": message,
    }
    if details is not None:
        payload["details"] = details
    return payload


def _resolve_request_path(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None:
        for attribute in ("path", "path_format"):
            candidate = getattr(route, attribute, None)
            if isinstance(candidate, str) and candidate:
                return candidate
    return request.url.path


def _record_audio_operation_from_items(
    telemetry: ServiceTelemetry,
    *,
    operation: str,
    stage: str,
    items: Sequence[Mapping[str, object]],
    latency_seconds: float,
    total_chunk_count: int | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    resolved_total_chunk_count = (
        total_chunk_count
        if total_chunk_count is not None
        else sum(_coerce_int(item.get("chunk_count")) for item in items)
    )
    telemetry.record_inference_operation(
        operation=operation,
        stage=stage,
        audio_count=len(items),
        total_audio_duration_seconds=sum(
            _coerce_float(item.get("duration_seconds")) for item in items
        ),
        total_chunk_count=resolved_total_chunk_count,
        latency_seconds=latency_seconds,
        extra=None if extra is None else dict(extra),
    )


def _latency_seconds_from_payload(payload: Mapping[str, object]) -> float:
    return _coerce_float(payload.get("latency_ms")) / 1000.0


def _coerce_float(value: object | None) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _coerce_int(value: object | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    raise TypeError(f"Expected mapping payload, got {type(value).__name__}.")


def _as_mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"Expected sequence payload, got {type(value).__name__}.")
    return [_as_mapping(item) for item in value]


def _sanitize_validation_errors(errors: Sequence[Any]) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for raw_error in errors:
        if not isinstance(raw_error, Mapping):
            sanitized.append({"message": repr(raw_error)})
            continue
        normalized = {str(key): _json_safe(value) for key, value in raw_error.items()}
        sanitized.append(normalized)
    return sanitized


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_safe(item) for item in value]
    return repr(value)


def _verify_operation_extra(payload: Mapping[str, object]) -> dict[str, object]:
    extra: dict[str, object] = {"normalized": bool(payload.get("normalized"))}
    decisions = payload.get("decisions")
    if isinstance(decisions, Sequence) and not isinstance(decisions, str | bytes) and decisions:
        extra["decision"] = bool(decisions[0])
    return extra


__all__ = [
    "FastAPIServerHandle",
    "create_http_app",
    "create_http_server",
    "run_http_server",
]
