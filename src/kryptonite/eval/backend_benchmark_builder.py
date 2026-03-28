"""Builder for reproducible backend benchmark reports."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

from kryptonite.tracking import utc_now

from .backend_benchmark_config import BackendBenchmarkConfig
from .backend_benchmark_models import (
    BackendBenchmarkBackendSummary,
    BackendBenchmarkPlotAsset,
    BackendBenchmarkReport,
    BackendBenchmarkSummary,
    BackendBenchmarkWorkloadResult,
)
from .backend_benchmark_runtime import (
    build_backend_benchmark_summaries,
    resolve_backend_benchmark_artifacts,
    run_backend_benchmark,
)


def build_backend_benchmark_report(
    config: BackendBenchmarkConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> BackendBenchmarkReport:
    artifacts = resolve_backend_benchmark_artifacts(config=config, project_root=project_root)
    workload_results = run_backend_benchmark(config=config, artifacts=artifacts)
    backend_summaries = build_backend_benchmark_summaries(
        config=config,
        workload_results=workload_results,
    )
    plot_assets = _build_plot_assets(workload_results)
    source_config_path = None if config_path is None else str(Path(config_path).resolve())
    source_config_sha256 = None if source_config_path is None else _sha256(Path(source_config_path))
    return BackendBenchmarkReport(
        title=config.title,
        report_id=config.report_id,
        summary_text=config.summary,
        generated_at_utc=utc_now(),
        project_root=str(artifacts.project_root),
        output_root=str(artifacts.project_root / config.output_root),
        source_config_path=source_config_path,
        source_config_sha256=source_config_sha256,
        model_version=artifacts.model_version,
        metadata_path=str(artifacts.metadata_path),
        source_checkpoint_path=str(artifacts.source_checkpoint_path),
        onnx_model_path=str(artifacts.onnx_model_path),
        tensorrt_report_path=str(artifacts.tensorrt_report_path),
        tensorrt_engine_path=str(artifacts.tensorrt_engine_path),
        onnxruntime_provider_order=artifacts.onnxruntime_provider_order,
        validated_backends=dict(artifacts.validated_backends),
        evaluation=config.evaluation.to_dict(),
        workloads=tuple(workload.to_dict() for workload in config.workloads),
        backend_summaries=backend_summaries,
        workload_results=workload_results,
        plot_assets=plot_assets,
        validation_commands=config.validation_commands,
        notes=config.notes,
        summary=_build_summary(
            backend_summaries=backend_summaries,
            workload_results=workload_results,
        ),
    )


def _build_plot_assets(
    workload_results: tuple[BackendBenchmarkWorkloadResult, ...],
) -> tuple[BackendBenchmarkPlotAsset, ...]:
    batch_sizes = sorted(
        {result.batch_size for result in workload_results if result.status == "passed"}
    )
    assets: list[BackendBenchmarkPlotAsset] = []
    for batch_size in batch_sizes:
        point_count = sum(
            1
            for result in workload_results
            if result.status == "passed"
            and result.batch_size == batch_size
            and result.warm_mean_latency_ms is not None
        )
        if point_count == 0:
            continue
        assets.append(
            BackendBenchmarkPlotAsset(
                batch_size=batch_size,
                title=f"Warm latency by frame count (batch={batch_size})",
                path=f"backend_benchmark_latency_batch{batch_size}.svg",
                point_count=point_count,
            )
        )
    return tuple(assets)


def _build_summary(
    *,
    backend_summaries: tuple[BackendBenchmarkBackendSummary, ...],
    workload_results: tuple[BackendBenchmarkWorkloadResult, ...],
) -> BackendBenchmarkSummary:
    successful_results = [result for result in workload_results if result.status == "passed"]
    lowest_latency_backend = _pick_backend_with_extreme(
        summaries=backend_summaries,
        key=lambda summary: summary.mean_warm_latency_ms,
        lowest=True,
    )
    highest_throughput_backend = _pick_backend_with_extreme(
        summaries=backend_summaries,
        key=lambda summary: summary.mean_throughput_items_per_second,
        lowest=False,
    )
    max_mean_abs_diff = _max_defined(result.mean_abs_diff for result in successful_results)
    max_cosine_distance = _max_defined(result.cosine_distance for result in successful_results)
    return BackendBenchmarkSummary(
        passed=bool(backend_summaries) and all(summary.passed for summary in backend_summaries),
        backend_count=len(backend_summaries),
        successful_backend_count=sum(1 for summary in backend_summaries if summary.passed),
        workload_count=len(workload_results),
        successful_workload_count=len(successful_results),
        batch_sizes=tuple(sorted({result.batch_size for result in workload_results})),
        max_mean_abs_diff=max_mean_abs_diff,
        max_cosine_distance=max_cosine_distance,
        lowest_mean_warm_latency_backend=lowest_latency_backend,
        highest_mean_throughput_backend=highest_throughput_backend,
    )


def _pick_backend_with_extreme(
    *,
    summaries: tuple[BackendBenchmarkBackendSummary, ...],
    key: Callable[[BackendBenchmarkBackendSummary], float | None],
    lowest: bool,
) -> str | None:
    candidates: list[tuple[float, BackendBenchmarkBackendSummary]] = []
    for summary in summaries:
        metric = key(summary)
        if metric is not None:
            candidates.append((metric, summary))
    if not candidates:
        return None
    selected = (
        min(candidates, key=lambda item: item[0])
        if lowest
        else max(candidates, key=lambda item: item[0])
    )
    return selected[1].backend


def _max_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return round(max(defined), 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = ["build_backend_benchmark_report"]
