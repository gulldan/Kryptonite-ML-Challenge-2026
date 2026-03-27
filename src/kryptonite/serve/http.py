"""FastAPI transport adapter for runtime metadata and unified inference flows."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Response, status
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
    return _build_app(inferencer, config=config)


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


def _build_app(inferencer: Inferencer, *, config: ProjectConfig) -> FastAPI:
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

    @app.exception_handler(ValueError)
    async def _handle_value_error(_request: object, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content=_error_payload(str(exc)))

    @app.get("/healthz", include_in_schema=False)
    @app.get("/readyz", include_in_schema=False)
    @app.get("/health", tags=["health"])
    async def health() -> dict[str, Any]:
        return inferencer.health_payload()

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
        return run_demo_compare(
            inferencer=inferencer,
            request=request,
            default_threshold=demo_threshold,
        )

    @app.post("/demo/api/enroll", tags=["demo"])
    async def demo_enroll(request: DemoEnrollmentRequest) -> dict[str, object]:
        return run_demo_enroll(inferencer=inferencer, request=request)

    @app.post("/demo/api/verify", tags=["demo"])
    async def demo_verify(request: DemoVerifyRequest) -> dict[str, object]:
        return run_demo_verify(
            inferencer=inferencer,
            request=request,
            default_threshold=demo_threshold,
        )

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
        return inferencer.embed_audio_paths(
            audio_paths=request.resolve_audio_paths(),
            stage=request.stage,
        )

    @app.post("/benchmark", tags=["inference"])
    async def benchmark(request: BenchmarkAudioRequest) -> dict[str, Any]:
        return inferencer.benchmark_audio_paths(
            audio_paths=request.resolve_audio_paths(),
            stage=request.stage,
            iterations=request.iterations,
            warmup_iterations=request.warmup_iterations,
        )

    @app.post("/enroll", tags=["enrollment"])
    async def enroll(request: EnrollmentRequest, response: Response) -> dict[str, Any]:
        if request.uses_audio_paths:
            payload = inferencer.enroll_audio_paths(
                enrollment_id=request.enrollment_id,
                audio_paths=request.resolve_audio_paths(),
                stage=request.stage,
                metadata=request.metadata,
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
            return inferencer.verify_audio_paths(
                enrollment_id=request.enrollment_id,
                audio_paths=request.resolve_audio_paths(),
                stage=request.stage,
                normalize=request.normalize,
                threshold=request.threshold,
            )
        return inferencer.verify_embeddings(
            enrollment_id=request.enrollment_id,
            probes=request.resolve_probes(),
            normalize=request.normalize,
            threshold=request.threshold,
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
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "error",
        "message": message,
    }
    if details:
        payload["details"] = dict(details)
    return payload


__all__ = [
    "FastAPIServerHandle",
    "create_http_app",
    "create_http_server",
    "run_http_server",
]
