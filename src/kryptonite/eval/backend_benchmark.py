"""Public facade for reproducible PyTorch/ORT/TensorRT backend benchmarks."""

from .backend_benchmark_builder import build_backend_benchmark_report
from .backend_benchmark_config import (
    BackendBenchmarkArtifactsConfig,
    BackendBenchmarkConfig,
    BackendBenchmarkEvaluationConfig,
    BackendBenchmarkWorkloadConfig,
    load_backend_benchmark_config,
)
from .backend_benchmark_models import (
    BACKEND_BENCHMARK_REPORT_JSON_NAME,
    BACKEND_BENCHMARK_REPORT_MARKDOWN_NAME,
    BACKEND_BENCHMARK_WORKLOAD_ROWS_NAME,
    BackendBenchmarkBackendSummary,
    BackendBenchmarkPlotAsset,
    BackendBenchmarkReport,
    BackendBenchmarkSummary,
    BackendBenchmarkWorkloadResult,
    WrittenBackendBenchmarkReport,
)
from .backend_benchmark_rendering import (
    render_backend_benchmark_markdown,
    write_backend_benchmark_report,
)

__all__ = [
    "BACKEND_BENCHMARK_REPORT_JSON_NAME",
    "BACKEND_BENCHMARK_REPORT_MARKDOWN_NAME",
    "BACKEND_BENCHMARK_WORKLOAD_ROWS_NAME",
    "BackendBenchmarkArtifactsConfig",
    "BackendBenchmarkBackendSummary",
    "BackendBenchmarkConfig",
    "BackendBenchmarkEvaluationConfig",
    "BackendBenchmarkPlotAsset",
    "BackendBenchmarkReport",
    "BackendBenchmarkSummary",
    "BackendBenchmarkWorkloadConfig",
    "BackendBenchmarkWorkloadResult",
    "WrittenBackendBenchmarkReport",
    "build_backend_benchmark_report",
    "load_backend_benchmark_config",
    "render_backend_benchmark_markdown",
    "write_backend_benchmark_report",
]
