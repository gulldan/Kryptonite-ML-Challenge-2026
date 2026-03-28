"""Datamodels for reproducible PyTorch/ORT/TensorRT backend benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

BACKEND_BENCHMARK_REPORT_JSON_NAME = "backend_benchmark_report.json"
BACKEND_BENCHMARK_REPORT_MARKDOWN_NAME = "backend_benchmark_report.md"
BACKEND_BENCHMARK_WORKLOAD_ROWS_NAME = "backend_benchmark_workload_rows.jsonl"


@dataclass(frozen=True, slots=True)
class BackendBenchmarkPlotAsset:
    batch_size: int
    title: str
    path: str
    point_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "title": self.title,
            "path": self.path,
            "point_count": self.point_count,
        }


@dataclass(frozen=True, slots=True)
class BackendBenchmarkWorkloadResult:
    backend: str
    provider: str | None
    implementation: str
    version: str | None
    device: str
    workload_id: str
    description: str
    batch_size: int
    frame_count: int
    status: str
    initialization_seconds: float | None
    cold_start_seconds: float | None
    warm_mean_latency_ms: float | None
    warm_median_latency_ms: float | None
    warm_p95_latency_ms: float | None
    warm_stddev_latency_ms: float | None
    warm_latency_cv: float | None
    throughput_items_per_second: float | None
    throughput_frames_per_second: float | None
    process_rss_peak_mib: float | None
    process_rss_delta_mib: float | None
    process_gpu_peak_mib: float | None
    process_gpu_delta_mib: float | None
    mean_abs_diff: float | None
    max_abs_diff: float | None
    cosine_distance: float | None
    quality_passed: bool | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "provider": self.provider,
            "implementation": self.implementation,
            "version": self.version,
            "device": self.device,
            "workload_id": self.workload_id,
            "description": self.description,
            "batch_size": self.batch_size,
            "frame_count": self.frame_count,
            "status": self.status,
            "initialization_seconds": self.initialization_seconds,
            "cold_start_seconds": self.cold_start_seconds,
            "warm_mean_latency_ms": self.warm_mean_latency_ms,
            "warm_median_latency_ms": self.warm_median_latency_ms,
            "warm_p95_latency_ms": self.warm_p95_latency_ms,
            "warm_stddev_latency_ms": self.warm_stddev_latency_ms,
            "warm_latency_cv": self.warm_latency_cv,
            "throughput_items_per_second": self.throughput_items_per_second,
            "throughput_frames_per_second": self.throughput_frames_per_second,
            "process_rss_peak_mib": self.process_rss_peak_mib,
            "process_rss_delta_mib": self.process_rss_delta_mib,
            "process_gpu_peak_mib": self.process_gpu_peak_mib,
            "process_gpu_delta_mib": self.process_gpu_delta_mib,
            "mean_abs_diff": self.mean_abs_diff,
            "max_abs_diff": self.max_abs_diff,
            "cosine_distance": self.cosine_distance,
            "quality_passed": self.quality_passed,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BackendBenchmarkBackendSummary:
    backend: str
    provider: str | None
    implementation: str
    version: str | None
    device: str
    configured_workload_count: int
    successful_workload_count: int
    mean_initialization_seconds: float | None
    mean_cold_start_seconds: float | None
    mean_warm_latency_ms: float | None
    mean_throughput_items_per_second: float | None
    mean_throughput_frames_per_second: float | None
    max_warm_latency_cv: float | None
    peak_process_rss_mib: float | None
    peak_process_gpu_mib: float | None
    max_mean_abs_diff: float | None
    max_cosine_distance: float | None
    passed: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "provider": self.provider,
            "implementation": self.implementation,
            "version": self.version,
            "device": self.device,
            "configured_workload_count": self.configured_workload_count,
            "successful_workload_count": self.successful_workload_count,
            "mean_initialization_seconds": self.mean_initialization_seconds,
            "mean_cold_start_seconds": self.mean_cold_start_seconds,
            "mean_warm_latency_ms": self.mean_warm_latency_ms,
            "mean_throughput_items_per_second": self.mean_throughput_items_per_second,
            "mean_throughput_frames_per_second": self.mean_throughput_frames_per_second,
            "max_warm_latency_cv": self.max_warm_latency_cv,
            "peak_process_rss_mib": self.peak_process_rss_mib,
            "peak_process_gpu_mib": self.peak_process_gpu_mib,
            "max_mean_abs_diff": self.max_mean_abs_diff,
            "max_cosine_distance": self.max_cosine_distance,
            "passed": self.passed,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class BackendBenchmarkSummary:
    passed: bool
    backend_count: int
    successful_backend_count: int
    workload_count: int
    successful_workload_count: int
    batch_sizes: tuple[int, ...]
    max_mean_abs_diff: float | None
    max_cosine_distance: float | None
    lowest_mean_warm_latency_backend: str | None
    highest_mean_throughput_backend: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "backend_count": self.backend_count,
            "successful_backend_count": self.successful_backend_count,
            "workload_count": self.workload_count,
            "successful_workload_count": self.successful_workload_count,
            "batch_sizes": list(self.batch_sizes),
            "max_mean_abs_diff": self.max_mean_abs_diff,
            "max_cosine_distance": self.max_cosine_distance,
            "lowest_mean_warm_latency_backend": self.lowest_mean_warm_latency_backend,
            "highest_mean_throughput_backend": self.highest_mean_throughput_backend,
        }


@dataclass(frozen=True, slots=True)
class BackendBenchmarkReport:
    title: str
    report_id: str
    summary_text: str
    generated_at_utc: str
    project_root: str
    output_root: str
    source_config_path: str | None
    source_config_sha256: str | None
    model_version: str | None
    metadata_path: str
    source_checkpoint_path: str
    onnx_model_path: str
    tensorrt_report_path: str
    tensorrt_engine_path: str
    onnxruntime_provider_order: tuple[str, ...]
    validated_backends: dict[str, bool]
    evaluation: dict[str, object]
    workloads: tuple[dict[str, object], ...]
    backend_summaries: tuple[BackendBenchmarkBackendSummary, ...]
    workload_results: tuple[BackendBenchmarkWorkloadResult, ...]
    plot_assets: tuple[BackendBenchmarkPlotAsset, ...]
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]
    summary: BackendBenchmarkSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "summary_text": self.summary_text,
            "generated_at_utc": self.generated_at_utc,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "source_config_path": self.source_config_path,
            "source_config_sha256": self.source_config_sha256,
            "model_version": self.model_version,
            "metadata_path": self.metadata_path,
            "source_checkpoint_path": self.source_checkpoint_path,
            "onnx_model_path": self.onnx_model_path,
            "tensorrt_report_path": self.tensorrt_report_path,
            "tensorrt_engine_path": self.tensorrt_engine_path,
            "onnxruntime_provider_order": list(self.onnxruntime_provider_order),
            "validated_backends": dict(self.validated_backends),
            "evaluation": dict(self.evaluation),
            "workloads": [dict(workload) for workload in self.workloads],
            "backend_summaries": [summary.to_dict() for summary in self.backend_summaries],
            "workload_results": [result.to_dict() for result in self.workload_results],
            "plot_assets": [asset.to_dict() for asset in self.plot_assets],
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenBackendBenchmarkReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    workload_rows_path: str
    plot_paths: tuple[str, ...]
    source_config_copy_path: str | None
    summary: BackendBenchmarkSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "workload_rows_path": self.workload_rows_path,
            "plot_paths": list(self.plot_paths),
            "source_config_copy_path": self.source_config_copy_path,
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "BACKEND_BENCHMARK_REPORT_JSON_NAME",
    "BACKEND_BENCHMARK_REPORT_MARKDOWN_NAME",
    "BACKEND_BENCHMARK_WORKLOAD_ROWS_NAME",
    "BackendBenchmarkBackendSummary",
    "BackendBenchmarkPlotAsset",
    "BackendBenchmarkReport",
    "BackendBenchmarkSummary",
    "BackendBenchmarkWorkloadResult",
    "WrittenBackendBenchmarkReport",
]
