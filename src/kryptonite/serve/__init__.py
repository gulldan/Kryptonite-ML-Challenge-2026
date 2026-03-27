"""Serving adapters and backend wrappers."""

from .deployment import build_infer_artifact_report
from .enrollment_cache import (
    ENROLLMENT_CACHE_FORMAT_VERSION,
    build_enrollment_embedding_cache,
    load_enrollment_embedding_cache,
    validate_enrollment_cache_compatibility,
)
from .enrollment_store import (
    RUNTIME_ENROLLMENT_STORE_DB_NAME,
    RUNTIME_ENROLLMENT_STORE_FORMAT_VERSION,
    RuntimeEnrollmentStore,
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
from .stress_report import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_BENCHMARK_ITERATIONS,
    DEFAULT_VERIFY_THRESHOLD,
    DEFAULT_WARMUP_ITERATIONS,
    InferenceStressMemorySummary,
    InferenceStressReport,
    InferenceStressSummary,
    StressHardLimitSummary,
    WrittenInferenceStressReport,
    build_inference_stress_report,
    default_stress_report_output_root,
    generate_inference_stress_inputs,
    render_inference_stress_markdown,
    write_inference_stress_report,
)
from .stress_report import (
    REPORT_JSON_NAME as INFERENCE_STRESS_REPORT_JSON_NAME,
)
from .stress_report import (
    REPORT_MARKDOWN_NAME as INFERENCE_STRESS_REPORT_MARKDOWN_NAME,
)
from .telemetry import PROMETHEUS_CONTENT_TYPE, ServiceTelemetry, resolve_model_version
from .triton_repository import (
    DEFAULT_TRITON_MODEL_NAME,
    DEFAULT_TRITON_REPOSITORY_ROOT,
    BuiltTritonModelRepository,
    TritonDynamicBatchingConfig,
    TritonRepositoryRequest,
    build_triton_model_repository,
    build_triton_repository_source_report,
)
from .triton_smoke import TritonSmokeResult, render_triton_smoke_result, run_triton_infer_smoke

__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_BENCHMARK_ITERATIONS",
    "DEFAULT_TRITON_MODEL_NAME",
    "DEFAULT_TRITON_REPOSITORY_ROOT",
    "DEFAULT_VERIFY_THRESHOLD",
    "DEFAULT_WARMUP_ITERATIONS",
    "BuiltTritonModelRepository",
    "EnrollmentNotFoundError",
    "EnrollmentRecord",
    "Inferencer",
    "INFERENCE_STRESS_REPORT_JSON_NAME",
    "INFERENCE_STRESS_REPORT_MARKDOWN_NAME",
    "InferenceStressReport",
    "InferenceStressMemorySummary",
    "InferenceStressSummary",
    "ScoringService",
    "ENROLLMENT_CACHE_FORMAT_VERSION",
    "PROMETHEUS_CONTENT_TYPE",
    "ServeRuntimeReport",
    "RUNTIME_ENROLLMENT_STORE_DB_NAME",
    "RUNTIME_ENROLLMENT_STORE_FORMAT_VERSION",
    "RuntimeEnrollmentStore",
    "ServiceTelemetry",
    "TritonDynamicBatchingConfig",
    "TritonRepositoryRequest",
    "TritonSmokeResult",
    "build_enrollment_embedding_cache",
    "build_infer_artifact_report",
    "build_inference_stress_report",
    "build_service_metadata",
    "build_serve_runtime_report",
    "build_triton_model_repository",
    "build_triton_repository_source_report",
    "create_http_server",
    "create_http_app",
    "default_stress_report_output_root",
    "generate_inference_stress_inputs",
    "load_enrollment_embedding_cache",
    "render_triton_smoke_result",
    "render_serve_runtime_report",
    "render_inference_stress_markdown",
    "resolve_model_version",
    "run_triton_infer_smoke",
    "run_http_server",
    "StressHardLimitSummary",
    "validate_enrollment_cache_compatibility",
    "write_inference_stress_report",
    "WrittenInferenceStressReport",
]
