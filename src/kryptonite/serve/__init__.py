"""Serving adapters and backend wrappers."""

from .deployment import build_infer_artifact_report
from .enrollment_cache import (
    ENROLLMENT_CACHE_FORMAT_VERSION,
    build_enrollment_embedding_cache,
    load_enrollment_embedding_cache,
    validate_enrollment_cache_compatibility,
)
from .http import create_http_app, create_http_server, run_http_server
from .inferencer import Inferencer
from .runtime import (
    ServeRuntimeReport,
    build_serve_runtime_report,
    build_service_metadata,
    render_serve_runtime_report,
)
from .scoring_service import EnrollmentNotFoundError, EnrollmentRecord, ScoringService

__all__ = [
    "EnrollmentNotFoundError",
    "EnrollmentRecord",
    "Inferencer",
    "ScoringService",
    "ENROLLMENT_CACHE_FORMAT_VERSION",
    "ServeRuntimeReport",
    "build_enrollment_embedding_cache",
    "build_infer_artifact_report",
    "build_service_metadata",
    "build_serve_runtime_report",
    "create_http_server",
    "create_http_app",
    "load_enrollment_embedding_cache",
    "render_serve_runtime_report",
    "run_http_server",
    "validate_enrollment_cache_compatibility",
]
