"""Thin HTTP adapter for runtime metadata and embedding-based scoring flows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from kryptonite.config import ProjectConfig
from kryptonite.deployment import render_artifact_report, resolve_project_path

from .deployment import build_infer_artifact_report
from .enrollment_cache import (
    ENROLLMENT_SUMMARY_JSON_NAME,
    MODEL_BUNDLE_METADATA_NAME,
    load_enrollment_embedding_cache,
    load_model_bundle_metadata,
    validate_enrollment_cache_compatibility,
)
from .runtime import (
    build_serve_runtime_report,
    build_service_metadata,
    render_serve_runtime_report,
)
from .scoring_service import EnrollmentNotFoundError, EnrollmentRecord, ScoringService


def create_http_server(
    *,
    host: str,
    port: int,
    config: ProjectConfig,
    require_artifacts: bool = False,
) -> ThreadingHTTPServer:
    report = build_serve_runtime_report(config=config)
    if not report.passed:
        raise RuntimeError(render_serve_runtime_report(report))

    artifact_report = build_infer_artifact_report(config=config, strict=require_artifacts)
    if not artifact_report.passed:
        raise RuntimeError(render_artifact_report(artifact_report))

    model_metadata_path = (
        resolve_project_path(config.paths.project_root, config.deployment.model_bundle_root)
        / MODEL_BUNDLE_METADATA_NAME
    )
    model_metadata = (
        load_model_bundle_metadata(model_metadata_path) if model_metadata_path.exists() else None
    )
    try:
        scoring_service, enrollment_cache_payload = _build_scoring_service(
            config=config,
            model_metadata=model_metadata,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    payload = build_service_metadata(
        config=config,
        report=report,
        artifact_report=artifact_report,
        enrollment_cache=enrollment_cache_payload,
    )
    return ThreadingHTTPServer((host, port), _build_handler(payload, scoring_service))


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


def _build_handler(
    payload: dict[str, object],
    scoring_service: ScoringService,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/healthz", "/readyz"}:
                self._write_json(HTTPStatus.OK, payload)
                return
            if path == "/enrollments":
                self._write_json(HTTPStatus.OK, scoring_service.list_enrollments())
                return
            self._write_not_found(path)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in {
                "/score/pairwise",
                "/score/one-to-many",
                "/enroll",
                "/verify",
            }:
                self._write_not_found(path)
                return

            try:
                request = self._read_json_object()
                status, response = self._dispatch_post(path=path, payload=request)
            except json.JSONDecodeError as exc:
                self._write_error(
                    HTTPStatus.BAD_REQUEST,
                    "Request body must contain valid JSON.",
                    details={"error": str(exc)},
                )
                return
            except EnrollmentNotFoundError as exc:
                self._write_error(HTTPStatus.NOT_FOUND, str(exc))
                return
            except ValueError as exc:
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
                return

            self._write_json(status, response)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _dispatch_post(
            self,
            *,
            path: str,
            payload: dict[str, object],
        ) -> tuple[HTTPStatus, dict[str, object]]:
            if path == "/score/pairwise":
                response = scoring_service.score_pairwise(
                    left=self._require_payload_value(payload, "left"),
                    right=self._require_payload_value(payload, "right"),
                    normalize=self._coerce_bool(payload.get("normalize"), default=True),
                )
                return HTTPStatus.OK, response

            if path == "/score/one-to-many":
                response = scoring_service.score_one_to_many(
                    queries=self._require_payload_value(payload, "queries"),
                    references=self._require_payload_value(payload, "references"),
                    normalize=self._coerce_bool(payload.get("normalize"), default=True),
                    top_k=self._coerce_optional_positive_int(payload.get("top_k")),
                    query_ids=self._coerce_optional_string_list(payload.get("query_ids")),
                    reference_ids=self._coerce_optional_string_list(payload.get("reference_ids")),
                )
                return HTTPStatus.OK, response

            if path == "/enroll":
                response = scoring_service.enroll(
                    enrollment_id=self._coerce_non_empty_string(
                        self._require_payload_value(payload, "enrollment_id"),
                        field_name="enrollment_id",
                    ),
                    embeddings=self._resolve_embedding_payload(
                        payload,
                        singular_key="embedding",
                        plural_key="embeddings",
                    ),
                    metadata=self._coerce_optional_mapping(payload.get("metadata")),
                )
                status = HTTPStatus.OK if response["replaced"] else HTTPStatus.CREATED
                return status, response

            response = scoring_service.verify(
                enrollment_id=self._coerce_non_empty_string(
                    self._require_payload_value(payload, "enrollment_id"),
                    field_name="enrollment_id",
                ),
                probes=self._resolve_embedding_payload(
                    payload,
                    singular_key="probe",
                    plural_key="probes",
                ),
                normalize=self._coerce_bool(payload.get("normalize"), default=True),
                threshold=self._coerce_optional_float(payload.get("threshold")),
            )
            return HTTPStatus.OK, response

        def _read_json_object(self) -> dict[str, object]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise ValueError("Content-Length header is required.")
            try:
                length = int(content_length)
            except ValueError as exc:
                raise ValueError("Content-Length header must be an integer.") from exc
            if length <= 0:
                raise ValueError("Request body must not be empty.")

            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Request JSON body must be an object.")
            return payload

        def _require_payload_value(
            self,
            payload: dict[str, object],
            field_name: str,
        ) -> object:
            if field_name not in payload:
                raise ValueError(f"{field_name} is required.")
            return payload[field_name]

        def _resolve_embedding_payload(
            self,
            payload: dict[str, object],
            *,
            singular_key: str,
            plural_key: str,
        ) -> object:
            if plural_key in payload:
                return payload[plural_key]
            if singular_key in payload:
                return payload[singular_key]
            raise ValueError(f"Either {singular_key} or {plural_key} is required.")

        def _coerce_bool(self, value: object, *, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            raise ValueError("Boolean fields must be encoded as true/false.")

        def _coerce_optional_float(self, value: object) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool):
                raise ValueError("threshold must be a number.")
            if isinstance(value, int | float):
                return float(value)
            raise ValueError("threshold must be a number.")

        def _coerce_optional_positive_int(self, value: object) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("top_k must be a positive integer.")
            if value <= 0:
                raise ValueError("top_k must be a positive integer.")
            return value

        def _coerce_non_empty_string(self, value: object, *, field_name: str) -> str:
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string.")
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{field_name} must not be empty.")
            return normalized

        def _coerce_optional_string_list(self, value: object) -> list[str] | None:
            if value is None:
                return None
            if not isinstance(value, list):
                raise ValueError("Identifier lists must be JSON arrays of strings.")
            normalized: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError("Identifier lists must contain only strings.")
                if not item.strip():
                    raise ValueError("Identifier lists must not contain empty strings.")
                normalized.append(item.strip())
            return normalized

        def _coerce_optional_mapping(self, value: object) -> Mapping[str, object] | None:
            if value is None:
                return None
            if not isinstance(value, dict):
                raise ValueError("metadata must be a JSON object.")
            normalized: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("metadata keys must be strings.")
                normalized[key] = item
            return normalized

        def _write_error(
            self,
            status: HTTPStatus,
            message: str,
            *,
            details: dict[str, object] | None = None,
        ) -> None:
            payload: dict[str, object] = {
                "status": "error",
                "message": message,
            }
            if details:
                payload["details"] = details
            self._write_json(status, payload)

        def _write_not_found(self, path: str) -> None:
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {
                    "status": "not_found",
                    "path": path,
                },
            )

        def _write_json(self, status: HTTPStatus, data: object) -> None:
            body = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _build_scoring_service(
    *,
    config: ProjectConfig,
    model_metadata: Mapping[str, object] | None,
) -> tuple[ScoringService, dict[str, object]]:
    cache_root = resolve_project_path(
        config.paths.project_root,
        config.deployment.enrollment_cache_root,
    )
    summary_path = cache_root / ENROLLMENT_SUMMARY_JSON_NAME
    if not summary_path.exists():
        return (
            ScoringService(),
            {
                "loaded": False,
                "cache_root": str(cache_root),
                "enrollment_count": 0,
            },
        )

    if model_metadata is None:
        raise ValueError(
            "Enrollment cache exists, but model bundle metadata is missing; "
            "cannot validate compatibility."
        )

    loaded_cache = load_enrollment_embedding_cache(cache_root)
    validate_enrollment_cache_compatibility(
        summary=loaded_cache.summary,
        model_metadata=model_metadata,
    )

    initial_enrollments = {
        enrollment_id: EnrollmentRecord(
            enrollment_id=enrollment_id,
            sample_count=int(metadata_row["sample_count"]),
            embedding_dim=int(embedding.shape[0]),
            embedding=embedding,
            metadata=dict(metadata_row),
        )
        for enrollment_id, embedding, metadata_row in zip(
            _resolve_enrollment_ids(loaded_cache.metadata_rows),
            loaded_cache.embeddings,
            loaded_cache.metadata_rows,
            strict=True,
        )
    }
    return (
        ScoringService(initial_enrollments=initial_enrollments),
        {
            "loaded": True,
            "cache_root": str(cache_root),
            "enrollment_count": loaded_cache.summary.enrollment_count,
            "compatibility_id": loaded_cache.summary.compatibility_id,
            "format_version": loaded_cache.summary.format_version,
            "source_manifest_path": loaded_cache.summary.source_manifest_path,
        },
    )


def _resolve_enrollment_ids(metadata_rows: list[dict[str, object]]) -> list[str]:
    enrollment_ids: list[str] = []
    for row in metadata_rows:
        enrollment_id = row.get("enrollment_id")
        if not isinstance(enrollment_id, str) or not enrollment_id.strip():
            raise ValueError("Enrollment cache metadata rows must define non-empty enrollment_id.")
        enrollment_ids.append(enrollment_id)
    return enrollment_ids
