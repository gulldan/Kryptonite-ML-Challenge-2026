"""Thin HTTP adapter for runtime metadata and unified inference flows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from kryptonite.config import ProjectConfig

from .inferencer import Inferencer
from .scoring_service import EnrollmentNotFoundError


def create_http_server(
    *,
    host: str,
    port: int,
    config: ProjectConfig,
    require_artifacts: bool = False,
) -> ThreadingHTTPServer:
    try:
        inferencer = Inferencer.from_config(
            config=config,
            require_artifacts=require_artifacts,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return ThreadingHTTPServer((host, port), _build_handler(inferencer))


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


def _build_handler(inferencer: Inferencer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/healthz", "/readyz"}:
                self._write_json(HTTPStatus.OK, inferencer.health_payload())
                return
            if path == "/enrollments":
                self._write_json(HTTPStatus.OK, inferencer.list_enrollments())
                return
            self._write_not_found(path)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in {
                "/embed",
                "/benchmark",
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
            if path == "/embed":
                response = inferencer.embed_audio_paths(
                    audio_paths=self._resolve_string_list_payload(
                        payload,
                        singular_key="audio_path",
                        plural_key="audio_paths",
                        field_name="audio_paths",
                    ),
                    stage=self._coerce_optional_stage(payload.get("stage")),
                )
                return HTTPStatus.OK, response

            if path == "/benchmark":
                response = inferencer.benchmark_audio_paths(
                    audio_paths=self._resolve_string_list_payload(
                        payload,
                        singular_key="audio_path",
                        plural_key="audio_paths",
                        field_name="audio_paths",
                    ),
                    stage=self._coerce_optional_stage(payload.get("stage")),
                    iterations=self._coerce_optional_positive_int(
                        payload.get("iterations"),
                        field_name="iterations",
                    )
                    or 3,
                    warmup_iterations=self._coerce_optional_non_negative_int(
                        payload.get("warmup_iterations"),
                        field_name="warmup_iterations",
                    )
                    or 1,
                )
                return HTTPStatus.OK, response

            if path == "/score/pairwise":
                response = inferencer.score_pairwise(
                    left=self._require_payload_value(payload, "left"),
                    right=self._require_payload_value(payload, "right"),
                    normalize=self._coerce_bool(payload.get("normalize"), default=True),
                )
                return HTTPStatus.OK, response

            if path == "/score/one-to-many":
                response = inferencer.score_one_to_many(
                    queries=self._require_payload_value(payload, "queries"),
                    references=self._require_payload_value(payload, "references"),
                    normalize=self._coerce_bool(payload.get("normalize"), default=True),
                    top_k=self._coerce_optional_positive_int(
                        payload.get("top_k"),
                        field_name="top_k",
                    ),
                    query_ids=self._coerce_optional_string_list(payload.get("query_ids")),
                    reference_ids=self._coerce_optional_string_list(payload.get("reference_ids")),
                )
                return HTTPStatus.OK, response

            if path == "/enroll":
                enrollment_id = self._coerce_non_empty_string(
                    self._require_payload_value(payload, "enrollment_id"),
                    field_name="enrollment_id",
                )
                if "embeddings" in payload or "embedding" in payload:
                    response = inferencer.enroll_embeddings(
                        enrollment_id=enrollment_id,
                        embeddings=self._resolve_embedding_payload(
                            payload,
                            singular_key="embedding",
                            plural_key="embeddings",
                        ),
                        metadata=self._coerce_optional_mapping(payload.get("metadata")),
                    )
                else:
                    response = inferencer.enroll_audio_paths(
                        enrollment_id=enrollment_id,
                        audio_paths=self._resolve_string_list_payload(
                            payload,
                            singular_key="audio_path",
                            plural_key="audio_paths",
                            field_name="audio_paths",
                        ),
                        stage=self._coerce_optional_stage(payload.get("stage")),
                        metadata=self._coerce_optional_mapping(payload.get("metadata")),
                    )
                status = HTTPStatus.OK if response["replaced"] else HTTPStatus.CREATED
                return status, response

            enrollment_id = self._coerce_non_empty_string(
                self._require_payload_value(payload, "enrollment_id"),
                field_name="enrollment_id",
            )
            if "probes" in payload or "probe" in payload:
                response = inferencer.verify_embeddings(
                    enrollment_id=enrollment_id,
                    probes=self._resolve_embedding_payload(
                        payload,
                        singular_key="probe",
                        plural_key="probes",
                    ),
                    normalize=self._coerce_bool(payload.get("normalize"), default=True),
                    threshold=self._coerce_optional_float(payload.get("threshold")),
                )
            else:
                response = inferencer.verify_audio_paths(
                    enrollment_id=enrollment_id,
                    audio_paths=self._resolve_string_list_payload(
                        payload,
                        singular_key="audio_path",
                        plural_key="audio_paths",
                        field_name="audio_paths",
                    ),
                    stage=self._coerce_optional_stage(payload.get("stage")),
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

        def _resolve_string_list_payload(
            self,
            payload: dict[str, object],
            *,
            singular_key: str,
            plural_key: str,
            field_name: str,
        ) -> list[str]:
            if plural_key in payload:
                values = payload[plural_key]
                if not isinstance(values, list):
                    raise ValueError(f"{field_name} must be a JSON array of strings.")
                normalized = self._coerce_optional_string_list(values)
                if not normalized:
                    raise ValueError(f"{field_name} must not be empty.")
                return normalized
            if singular_key in payload:
                return [
                    self._coerce_non_empty_string(
                        payload[singular_key],
                        field_name=singular_key,
                    )
                ]
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

        def _coerce_optional_positive_int(
            self,
            value: object,
            *,
            field_name: str,
        ) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be a positive integer.")
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
            return value

        def _coerce_optional_non_negative_int(
            self,
            value: object,
            *,
            field_name: str,
        ) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be a non-negative integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
            return value

        def _coerce_non_empty_string(self, value: object, *, field_name: str) -> str:
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string.")
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{field_name} must not be empty.")
            return normalized

        def _coerce_optional_stage(self, value: object) -> str | None:
            if value is None:
                return None
            return self._coerce_non_empty_string(value, field_name="stage").lower()

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
