"""Structured JSON logs and Prometheus-compatible metrics for serving flows."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kryptonite.config import ProjectConfig

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_LOGGER_NAME = "kryptonite.serve.telemetry"
_DEFAULT_HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    service: str
    backend: str
    implementation: str
    model_version: str


@dataclass(slots=True)
class _Histogram:
    buckets: tuple[float, ...]
    bucket_counts: list[int]
    count: int = 0
    total: float = 0.0

    @classmethod
    def create(cls, buckets: tuple[float, ...]) -> _Histogram:
        return cls(buckets=buckets, bucket_counts=[0 for _ in buckets])

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for index, upper_bound in enumerate(self.buckets):
            if value <= upper_bound:
                self.bucket_counts[index] += 1


class ServiceTelemetry:
    """Capture lightweight service telemetry without extra runtime dependencies."""

    def __init__(
        self,
        *,
        context: TelemetryContext,
        enabled: bool,
        structured_logs: bool,
        metrics_enabled: bool,
        metrics_path: str,
        log_level: str,
    ) -> None:
        self._context = context
        self._enabled = enabled
        self._structured_logs = enabled and structured_logs
        self._metrics_enabled = enabled and metrics_enabled
        self.metrics_path = metrics_path
        self._logger = _configure_json_logger(log_level)
        self._lock = threading.Lock()
        self._http_requests_total: dict[tuple[str, str, str], int] = {}
        self._http_request_duration: dict[tuple[str, str], _Histogram] = {}
        self._validation_errors_total: dict[tuple[str, str, str], int] = {}
        self._inference_operations_total: dict[tuple[str, str, str, str], int] = {}
        self._inference_operation_duration: dict[tuple[str, str, str, str], _Histogram] = {}
        self._inference_audio_duration_total: dict[tuple[str, str, str, str], float] = {}
        self._inference_inputs_total: dict[tuple[str, str, str, str], int] = {}
        self._inference_chunks_total: dict[tuple[str, str, str, str], int] = {}

    @classmethod
    def from_inferencer(
        cls,
        *,
        config: ProjectConfig,
        service: str,
        backend: str,
        implementation: str,
        model_version: str,
    ) -> ServiceTelemetry:
        return cls(
            context=TelemetryContext(
                service=service,
                backend=backend,
                implementation=implementation,
                model_version=model_version,
            ),
            enabled=config.telemetry.enabled,
            structured_logs=config.telemetry.structured_logs,
            metrics_enabled=config.telemetry.metrics_enabled,
            metrics_path=config.telemetry.metrics_path,
            log_level=config.runtime.log_level,
        )

    @property
    def metrics_enabled(self) -> bool:
        return self._metrics_enabled

    def summary(self) -> dict[str, object]:
        return {
            "enabled": self._enabled,
            "structured_logs": self._structured_logs,
            "metrics_enabled": self._metrics_enabled,
            "metrics_path": self.metrics_path if self._metrics_enabled else None,
            "backend": self._context.backend,
            "implementation": self._context.implementation,
            "model_version": self._context.model_version,
        }

    def record_service_start(self, *, log_level: str, strict_artifacts: bool) -> None:
        self._log_event(
            event="service_start",
            fields={
                "log_level": log_level,
                "metrics_path": self.metrics_path if self._metrics_enabled else None,
                "strict_artifacts": strict_artifacts,
            },
        )

    def record_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        if self._metrics_enabled:
            counter_key = (method, path, str(status_code))
            histogram_key = (method, path)
            with self._lock:
                self._http_requests_total[counter_key] = (
                    self._http_requests_total.get(counter_key, 0) + 1
                )
                histogram = self._http_request_duration.get(histogram_key)
                if histogram is None:
                    histogram = _Histogram.create(_DEFAULT_HISTOGRAM_BUCKETS)
                    self._http_request_duration[histogram_key] = histogram
                histogram.observe(duration_seconds)
        self._log_event(
            event="http_request",
            fields={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(duration_seconds * 1000.0, 3),
            },
        )

    def record_validation_error(
        self,
        *,
        path: str,
        error_type: str,
        status_code: int,
        message: str,
        details: Any | None = None,
    ) -> None:
        if self._metrics_enabled:
            with self._lock:
                counter_key = (path, error_type, str(status_code))
                self._validation_errors_total[counter_key] = (
                    self._validation_errors_total.get(counter_key, 0) + 1
                )
        payload: dict[str, object] = {
            "path": path,
            "error_type": error_type,
            "status_code": status_code,
            "message": message,
        }
        if details is not None:
            payload["details"] = details
        self._log_event(
            event="validation_error",
            level=logging.WARNING,
            fields=payload,
        )

    def record_inference_operation(
        self,
        *,
        operation: str,
        stage: str,
        audio_count: int,
        total_audio_duration_seconds: float,
        total_chunk_count: int,
        latency_seconds: float,
        extra: dict[str, object] | None = None,
    ) -> None:
        stage_label = stage or "unknown"
        metrics_key = (
            operation,
            stage_label,
            self._context.backend,
            self._context.model_version,
        )
        if self._metrics_enabled:
            with self._lock:
                self._inference_operations_total[metrics_key] = (
                    self._inference_operations_total.get(metrics_key, 0) + 1
                )
                histogram = self._inference_operation_duration.get(metrics_key)
                if histogram is None:
                    histogram = _Histogram.create(_DEFAULT_HISTOGRAM_BUCKETS)
                    self._inference_operation_duration[metrics_key] = histogram
                histogram.observe(latency_seconds)
                self._inference_audio_duration_total[metrics_key] = (
                    self._inference_audio_duration_total.get(metrics_key, 0.0)
                    + total_audio_duration_seconds
                )
                self._inference_inputs_total[metrics_key] = (
                    self._inference_inputs_total.get(metrics_key, 0) + audio_count
                )
                self._inference_chunks_total[metrics_key] = (
                    self._inference_chunks_total.get(metrics_key, 0) + total_chunk_count
                )

        payload: dict[str, object] = {
            "operation": operation,
            "stage": stage_label,
            "audio_count": audio_count,
            "total_audio_duration_seconds": round(total_audio_duration_seconds, 6),
            "total_chunk_count": total_chunk_count,
            "latency_ms": round(latency_seconds * 1000.0, 3),
        }
        if extra:
            payload.update(extra)
        self._log_event(event="inference_operation", fields=payload)

    def render_prometheus(self) -> str:
        if not self._metrics_enabled:
            return ""
        with self._lock:
            http_requests_total = dict(self._http_requests_total)
            http_request_duration = dict(self._http_request_duration)
            validation_errors_total = dict(self._validation_errors_total)
            inference_operations_total = dict(self._inference_operations_total)
            inference_operation_duration = dict(self._inference_operation_duration)
            inference_audio_duration_total = dict(self._inference_audio_duration_total)
            inference_inputs_total = dict(self._inference_inputs_total)
            inference_chunks_total = dict(self._inference_chunks_total)

        lines: list[str] = []
        lines.extend(
            [
                "# HELP kryptonite_http_requests_total Total HTTP requests handled by the service.",
                "# TYPE kryptonite_http_requests_total counter",
            ]
        )
        for key in sorted(http_requests_total):
            method, path, status = key
            lines.append(
                _metric_line(
                    "kryptonite_http_requests_total",
                    http_requests_total[key],
                    {"method": method, "path": path, "status": status},
                )
            )

        lines.extend(
            [
                "# HELP kryptonite_http_request_duration_seconds HTTP request latency in seconds.",
                "# TYPE kryptonite_http_request_duration_seconds histogram",
            ]
        )
        for key in sorted(http_request_duration):
            method, path = key
            histogram = http_request_duration[key]
            labels = {"method": method, "path": path}
            lines.extend(
                _render_histogram(
                    "kryptonite_http_request_duration_seconds",
                    histogram,
                    labels,
                )
            )

        lines.extend(
            [
                "# HELP kryptonite_validation_errors_total Validation errors returned by the API.",
                "# TYPE kryptonite_validation_errors_total counter",
            ]
        )
        for key in sorted(validation_errors_total):
            path, error_type, status = key
            lines.append(
                _metric_line(
                    "kryptonite_validation_errors_total",
                    validation_errors_total[key],
                    {"path": path, "error_type": error_type, "status": status},
                )
            )

        lines.extend(
            [
                "# HELP kryptonite_inference_operations_total "
                "Audio inference operations handled by the service.",
                "# TYPE kryptonite_inference_operations_total counter",
            ]
        )
        for key in sorted(inference_operations_total):
            operation, stage, backend, model_version = key
            lines.append(
                _metric_line(
                    "kryptonite_inference_operations_total",
                    inference_operations_total[key],
                    {
                        "operation": operation,
                        "stage": stage,
                        "backend": backend,
                        "model_version": model_version,
                    },
                )
            )

        lines.extend(
            [
                "# HELP kryptonite_inference_operation_duration_seconds "
                "Audio inference latency in seconds.",
                "# TYPE kryptonite_inference_operation_duration_seconds histogram",
            ]
        )
        for key in sorted(inference_operation_duration):
            operation, stage, backend, model_version = key
            histogram = inference_operation_duration[key]
            labels = {
                "operation": operation,
                "stage": stage,
                "backend": backend,
                "model_version": model_version,
            }
            lines.extend(
                _render_histogram(
                    "kryptonite_inference_operation_duration_seconds",
                    histogram,
                    labels,
                )
            )

        lines.extend(
            [
                "# HELP kryptonite_inference_audio_duration_seconds_total "
                "Total input audio duration observed by the service.",
                "# TYPE kryptonite_inference_audio_duration_seconds_total counter",
            ]
        )
        for key in sorted(inference_audio_duration_total):
            operation, stage, backend, model_version = key
            lines.append(
                _metric_line(
                    "kryptonite_inference_audio_duration_seconds_total",
                    inference_audio_duration_total[key],
                    {
                        "operation": operation,
                        "stage": stage,
                        "backend": backend,
                        "model_version": model_version,
                    },
                )
            )

        lines.extend(
            [
                "# HELP kryptonite_inference_input_audios_total "
                "Total audio inputs processed by the service.",
                "# TYPE kryptonite_inference_input_audios_total counter",
            ]
        )
        for key in sorted(inference_inputs_total):
            operation, stage, backend, model_version = key
            lines.append(
                _metric_line(
                    "kryptonite_inference_input_audios_total",
                    inference_inputs_total[key],
                    {
                        "operation": operation,
                        "stage": stage,
                        "backend": backend,
                        "model_version": model_version,
                    },
                )
            )

        lines.extend(
            [
                "# HELP kryptonite_inference_chunks_total "
                "Total audio chunks processed by the service.",
                "# TYPE kryptonite_inference_chunks_total counter",
            ]
        )
        for key in sorted(inference_chunks_total):
            operation, stage, backend, model_version = key
            lines.append(
                _metric_line(
                    "kryptonite_inference_chunks_total",
                    inference_chunks_total[key],
                    {
                        "operation": operation,
                        "stage": stage,
                        "backend": backend,
                        "model_version": model_version,
                    },
                )
            )

        return "\n".join(lines) + "\n"

    def _log_event(
        self,
        *,
        event: str,
        level: int = logging.INFO,
        fields: dict[str, object] | None = None,
    ) -> None:
        if not self._structured_logs:
            return
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "service": self._context.service,
            "backend": self._context.backend,
            "implementation": self._context.implementation,
            "model_version": self._context.model_version,
            **dict(fields or {}),
        }
        self._logger.log(level, json.dumps(payload, sort_keys=True, default=_json_default))


def resolve_model_version(model_metadata: dict[str, Any] | None) -> str:
    if not model_metadata:
        return "unknown"
    for key in ("model_version", "version", "release_tag", "artifact_version"):
        candidate = _coerce_string(model_metadata.get(key))
        if candidate is not None:
            return candidate
    compatibility_id = _coerce_string(model_metadata.get("enrollment_cache_compatibility_id"))
    if compatibility_id is not None:
        return compatibility_id
    model_file = _coerce_string(model_metadata.get("model_file"))
    if model_file is not None:
        return Path(model_file).stem or "unknown"
    inferencer_backend = _coerce_string(model_metadata.get("inferencer_backend"))
    embedding_stage = _coerce_string(model_metadata.get("embedding_stage"))
    if inferencer_backend is not None and embedding_stage is not None:
        return f"{inferencer_backend}:{embedding_stage}"
    return "unknown"


def _configure_json_logger(log_level: str) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_normalize_log_level(log_level))
    if not logging.getLogger().handlers:
        logging.basicConfig(level=_normalize_log_level(log_level), format="%(message)s")
    return logger


def _normalize_log_level(log_level: str) -> int:
    normalized = log_level.strip().upper()
    return getattr(logging, normalized, logging.INFO)


def _metric_line(name: str, value: int | float, labels: dict[str, str]) -> str:
    return f"{name}{_format_labels(labels)} {value}"


def _render_histogram(
    name: str,
    histogram: _Histogram,
    labels: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for upper_bound, bucket_count in zip(histogram.buckets, histogram.bucket_counts, strict=True):
        lines.append(
            _metric_line(
                f"{name}_bucket",
                bucket_count,
                {**labels, "le": _format_bucket_bound(upper_bound)},
            )
        )
    lines.append(_metric_line(f"{name}_bucket", histogram.count, {**labels, "le": "+Inf"}))
    lines.append(_metric_line(f"{name}_sum", round(histogram.total, 12), labels))
    lines.append(_metric_line(f"{name}_count", histogram.count, labels))
    return lines


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    rendered = ",".join(
        f'{key}="{_escape_label_value(value)}"' for key, value in sorted(labels.items())
    )
    return "{" + rendered + "}"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_bucket_bound(value: float) -> str:
    text = f"{value:.12g}"
    return text


def _coerce_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


__all__ = [
    "PROMETHEUS_CONTENT_TYPE",
    "ServiceTelemetry",
    "TelemetryContext",
    "resolve_model_version",
]
