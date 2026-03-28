"""Runtime helpers for reproducible backend benchmark reports."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

import numpy as np

from kryptonite.deployment import resolve_project_path
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.serve.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.serve.inference_package import load_inference_package_from_model_metadata
from kryptonite.serve.tensorrt_engine_models import TensorRTFP16Profile
from kryptonite.serve.tensorrt_engine_runtime import TensorRTEngineRunner

from .backend_benchmark_config import BackendBenchmarkConfig, BackendBenchmarkWorkloadConfig
from .backend_benchmark_models import BackendBenchmarkBackendSummary, BackendBenchmarkWorkloadResult


@dataclass(frozen=True, slots=True)
class ResolvedBackendBenchmarkArtifacts:
    project_root: Path
    metadata_path: Path
    source_checkpoint_path: Path
    onnx_model_path: Path
    tensorrt_report_path: Path
    tensorrt_engine_path: Path
    model_version: str | None
    input_name: str
    output_name: str
    feature_dim: int
    embedding_dim: int
    onnxruntime_provider_order: tuple[str, ...]
    validated_backends: dict[str, bool]
    tensorrt_profiles: tuple[TensorRTFP16Profile, ...]


@dataclass(frozen=True, slots=True)
class _BackendIdentity:
    backend: str
    provider: str | None
    implementation: str
    version: str | None
    device: str


class _BenchmarkRunner(Protocol):
    def run(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class _TorchRunner:
    torch: Any
    model: Any
    device: Any

    def run(self, features: np.ndarray) -> np.ndarray:
        input_tensor = self.torch.as_tensor(features, dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            output = self.model(input_tensor).detach()
        return np.asarray(
            output.to(device="cpu", dtype=self.torch.float32).numpy(),
            dtype=np.float32,
        )


@dataclass(slots=True)
class _OnnxRuntimeRunner:
    session: Any
    input_name: str
    output_name: str

    def run(self, features: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: np.asarray(features, dtype=np.float32)},
        )
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("ONNX Runtime did not return the expected single output tensor.")
        return np.asarray(outputs[0], dtype=np.float32)


@dataclass(slots=True)
class _TensorRTRunner:
    torch: Any
    runner: TensorRTEngineRunner

    def run(self, features: np.ndarray) -> np.ndarray:
        input_tensor = self.torch.as_tensor(features, dtype=self.torch.float32, device="cuda")
        output = self.runner.run(input_tensor).detach()
        return np.asarray(
            output.to(device="cpu", dtype=self.torch.float32).numpy(),
            dtype=np.float32,
        )


def resolve_backend_benchmark_artifacts(
    *,
    config: BackendBenchmarkConfig,
    project_root: Path | str | None = None,
) -> ResolvedBackendBenchmarkArtifacts:
    resolved_project_root = resolve_project_path(str(project_root or config.project_root), ".")
    metadata_path = _resolve_required_file(
        project_root=resolved_project_root,
        raw_path=config.artifacts.model_bundle_metadata_path,
        field_name="artifacts.model_bundle_metadata_path",
    )
    metadata = _load_json_object(metadata_path)
    contract = load_export_boundary_from_model_metadata(metadata)
    inference_package = load_inference_package_from_model_metadata(metadata)

    onnx_model_path = _resolve_required_file(
        project_root=resolved_project_root,
        raw_path=(
            config.artifacts.onnx_model_path_override
            or inference_package.artifacts.onnx_model_file
            or _coerce_string(metadata.get("model_file"))
        ),
        field_name="onnx model artifact",
    )
    tensorrt_report_path = _resolve_required_file(
        project_root=resolved_project_root,
        raw_path=config.artifacts.tensorrt_report_path,
        field_name="artifacts.tensorrt_report_path",
    )
    tensorrt_payload = _load_json_object(tensorrt_report_path)
    tensorrt_engine_path = _resolve_required_file(
        project_root=resolved_project_root,
        raw_path=(
            config.artifacts.tensorrt_engine_path_override
            or _coerce_string(tensorrt_payload.get("engine_path"))
            or inference_package.artifacts.tensorrt_engine_file
            or _coerce_string(metadata.get("tensorrt_engine_file"))
        ),
        field_name="TensorRT engine artifact",
    )
    source_checkpoint_path = _resolve_required_file(
        project_root=resolved_project_root,
        raw_path=(
            config.artifacts.source_checkpoint_path_override
            or _coerce_string(tensorrt_payload.get("source_checkpoint_path"))
            or _coerce_string(metadata.get("source_checkpoint_path"))
        ),
        field_name="source checkpoint artifact",
    )
    feature_dim = _require_static_axis_size(
        contract.input_tensor.axes[-1].size,
        axis_name="feature_dim",
    )
    embedding_dim = _require_static_axis_size(
        contract.output_tensor.axes[-1].size,
        axis_name="embedding_dim",
    )
    return ResolvedBackendBenchmarkArtifacts(
        project_root=resolved_project_root,
        metadata_path=metadata_path,
        source_checkpoint_path=source_checkpoint_path,
        onnx_model_path=onnx_model_path,
        tensorrt_report_path=tensorrt_report_path,
        tensorrt_engine_path=tensorrt_engine_path,
        model_version=_coerce_string(metadata.get("model_version")),
        input_name=_coerce_string(tensorrt_payload.get("input_name")) or contract.input_tensor.name,
        output_name=(
            _coerce_string(tensorrt_payload.get("output_name")) or contract.output_tensor.name
        ),
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        onnxruntime_provider_order=inference_package.onnxruntime_provider_order,
        validated_backends=dict(inference_package.validated_backends),
        tensorrt_profiles=_parse_tensorrt_profiles(tensorrt_payload),
    )


def run_backend_benchmark(
    *,
    config: BackendBenchmarkConfig,
    artifacts: ResolvedBackendBenchmarkArtifacts,
) -> tuple[BackendBenchmarkWorkloadResult, ...]:
    reference_outputs, reference_error = _build_reference_outputs(
        config=config, artifacts=artifacts
    )
    results: list[BackendBenchmarkWorkloadResult] = []
    for backend in config.evaluation.backends:
        for workload_index, workload in enumerate(config.workloads):
            features = _build_input_tensor(
                workload=workload,
                feature_dim=artifacts.feature_dim,
                seed=config.evaluation.seed + workload_index,
            )
            result = _benchmark_backend_workload(
                backend=backend,
                workload=workload,
                features=features,
                config=config,
                artifacts=artifacts,
                reference_output=reference_outputs.get(workload.workload_id),
                reference_error=reference_error,
            )
            results.append(result)
    return tuple(results)


def build_backend_benchmark_summaries(
    *,
    config: BackendBenchmarkConfig,
    workload_results: tuple[BackendBenchmarkWorkloadResult, ...],
) -> tuple[BackendBenchmarkBackendSummary, ...]:
    summaries: list[BackendBenchmarkBackendSummary] = []
    for backend in config.evaluation.backends:
        backend_results = [result for result in workload_results if result.backend == backend]
        successful = [result for result in backend_results if result.status == "passed"]
        exemplar = next(iter(successful or backend_results), None)
        errors = tuple(
            dict.fromkeys(
                result.error
                for result in backend_results
                if result.error is not None and result.error
            )
        )
        summaries.append(
            BackendBenchmarkBackendSummary(
                backend=backend,
                provider=None if exemplar is None else exemplar.provider,
                implementation="unknown" if exemplar is None else exemplar.implementation,
                version=None if exemplar is None else exemplar.version,
                device=config.evaluation.device if exemplar is None else exemplar.device,
                configured_workload_count=len(backend_results),
                successful_workload_count=len(successful),
                mean_initialization_seconds=_mean_defined(
                    result.initialization_seconds for result in successful
                ),
                mean_cold_start_seconds=_mean_defined(
                    result.cold_start_seconds for result in successful
                ),
                mean_warm_latency_ms=_mean_defined(
                    result.warm_mean_latency_ms for result in successful
                ),
                mean_throughput_items_per_second=_mean_defined(
                    result.throughput_items_per_second for result in successful
                ),
                mean_throughput_frames_per_second=_mean_defined(
                    result.throughput_frames_per_second for result in successful
                ),
                max_warm_latency_cv=_max_defined(result.warm_latency_cv for result in successful),
                peak_process_rss_mib=_max_defined(
                    result.process_rss_peak_mib for result in successful
                ),
                peak_process_gpu_mib=_max_defined(
                    result.process_gpu_peak_mib for result in successful
                ),
                max_mean_abs_diff=_max_defined(result.mean_abs_diff for result in successful),
                max_cosine_distance=_max_defined(result.cosine_distance for result in successful),
                passed=bool(successful) and len(successful) == len(backend_results) and not errors,
                errors=errors,
            )
        )
    return tuple(summaries)


def _build_reference_outputs(
    *,
    config: BackendBenchmarkConfig,
    artifacts: ResolvedBackendBenchmarkArtifacts,
) -> tuple[dict[str, np.ndarray], str | None]:
    try:
        _, runner, _ = _build_torch_runner(config=config, artifacts=artifacts)
    except Exception as exc:
        return {}, f"torch reference unavailable: {exc}"

    outputs: dict[str, np.ndarray] = {}
    for workload_index, workload in enumerate(config.workloads):
        features = _build_input_tensor(
            workload=workload,
            feature_dim=artifacts.feature_dim,
            seed=config.evaluation.seed + workload_index,
        )
        output = runner.run(features)
        _validate_output_shape(
            output=output, workload=workload, embedding_dim=artifacts.embedding_dim
        )
        outputs[workload.workload_id] = output
    return outputs, None


def _benchmark_backend_workload(
    *,
    backend: str,
    workload: BackendBenchmarkWorkloadConfig,
    features: np.ndarray,
    config: BackendBenchmarkConfig,
    artifacts: ResolvedBackendBenchmarkArtifacts,
    reference_output: np.ndarray | None,
    reference_error: str | None,
) -> BackendBenchmarkWorkloadResult:
    rss_before = _capture_current_process_rss_mib()
    gpu_before = _capture_current_process_gpu_memory_mib()
    observed_rss: list[float | None] = [rss_before]
    observed_gpu: list[float | None] = [gpu_before]
    try:
        identity, runner, initialization_seconds = _build_runner(
            backend=backend,
            config=config,
            artifacts=artifacts,
        )
        observed_rss.append(_capture_current_process_rss_mib())
        observed_gpu.append(_capture_current_process_gpu_memory_mib())

        started_at = time.perf_counter()
        cold_output = runner.run(features)
        cold_elapsed = time.perf_counter() - started_at
        _validate_output_shape(
            output=cold_output,
            workload=workload,
            embedding_dim=artifacts.embedding_dim,
        )
        observed_rss.append(_capture_current_process_rss_mib())
        observed_gpu.append(_capture_current_process_gpu_memory_mib())

        for _ in range(config.evaluation.warmup_iterations):
            runner.run(features)

        latencies_ms: list[float] = []
        warm_output = cold_output
        for _ in range(config.evaluation.benchmark_iterations):
            iteration_started = time.perf_counter()
            warm_output = runner.run(features)
            latencies_ms.append((time.perf_counter() - iteration_started) * 1_000.0)
        observed_rss.append(_capture_current_process_rss_mib())
        observed_gpu.append(_capture_current_process_gpu_memory_mib())

        mean_abs_diff, max_abs_diff, cosine_distance, quality_passed = _quality_metrics(
            backend=backend,
            reference_output=reference_output,
            candidate_output=warm_output,
            max_mean_abs_diff=config.evaluation.max_mean_abs_diff,
            max_cosine_distance=config.evaluation.max_cosine_distance,
        )
        error = reference_error if quality_passed is None else None
        if quality_passed is False:
            error = (
                "quality thresholds exceeded: "
                f"mean_abs_diff={mean_abs_diff:.8f}, cosine_distance={cosine_distance:.8f}"
            )
        return BackendBenchmarkWorkloadResult(
            backend=identity.backend,
            provider=identity.provider,
            implementation=identity.implementation,
            version=identity.version,
            device=identity.device,
            workload_id=workload.workload_id,
            description=workload.description,
            batch_size=workload.batch_size,
            frame_count=workload.frame_count,
            status="passed" if error is None else "failed",
            initialization_seconds=_round_optional(initialization_seconds, digits=6),
            cold_start_seconds=_round_optional(initialization_seconds + cold_elapsed, digits=6),
            warm_mean_latency_ms=_round_optional(fmean(latencies_ms), digits=6),
            warm_median_latency_ms=_round_optional(_percentile(latencies_ms, 50.0), digits=6),
            warm_p95_latency_ms=_round_optional(_percentile(latencies_ms, 95.0), digits=6),
            warm_stddev_latency_ms=_round_optional(_stddev(latencies_ms), digits=6),
            warm_latency_cv=_round_optional(_latency_cv(latencies_ms), digits=6),
            throughput_items_per_second=_round_optional(
                _throughput(value=workload.batch_size, latency_ms=fmean(latencies_ms)),
                digits=6,
            ),
            throughput_frames_per_second=_round_optional(
                _throughput(
                    value=workload.batch_size * workload.frame_count,
                    latency_ms=fmean(latencies_ms),
                ),
                digits=6,
            ),
            process_rss_peak_mib=_round_optional(_max_defined(observed_rss), digits=3),
            process_rss_delta_mib=_round_optional(
                _delta_peak(before=rss_before, observed=observed_rss),
                digits=3,
            ),
            process_gpu_peak_mib=_round_optional(_max_defined(observed_gpu), digits=3),
            process_gpu_delta_mib=_round_optional(
                _delta_peak(before=gpu_before, observed=observed_gpu),
                digits=3,
            ),
            mean_abs_diff=_round_optional(mean_abs_diff, digits=8),
            max_abs_diff=_round_optional(max_abs_diff, digits=8),
            cosine_distance=_round_optional(cosine_distance, digits=8),
            quality_passed=quality_passed,
            error=error,
        )
    except Exception as exc:
        implementation = {
            "torch": "campp_encoder",
            "onnxruntime": "onnxruntime_session",
            "tensorrt": "tensorrt_plan",
        }.get(backend, backend)
        return BackendBenchmarkWorkloadResult(
            backend=backend,
            provider=None,
            implementation=implementation,
            version=None,
            device=config.evaluation.device,
            workload_id=workload.workload_id,
            description=workload.description,
            batch_size=workload.batch_size,
            frame_count=workload.frame_count,
            status="failed",
            initialization_seconds=None,
            cold_start_seconds=None,
            warm_mean_latency_ms=None,
            warm_median_latency_ms=None,
            warm_p95_latency_ms=None,
            warm_stddev_latency_ms=None,
            warm_latency_cv=None,
            throughput_items_per_second=None,
            throughput_frames_per_second=None,
            process_rss_peak_mib=_round_optional(_max_defined(observed_rss), digits=3),
            process_rss_delta_mib=_round_optional(
                _delta_peak(before=rss_before, observed=observed_rss),
                digits=3,
            ),
            process_gpu_peak_mib=_round_optional(_max_defined(observed_gpu), digits=3),
            process_gpu_delta_mib=_round_optional(
                _delta_peak(before=gpu_before, observed=observed_gpu),
                digits=3,
            ),
            mean_abs_diff=None,
            max_abs_diff=None,
            cosine_distance=None,
            quality_passed=None,
            error=str(exc),
        )


def _build_runner(
    *,
    backend: str,
    config: BackendBenchmarkConfig,
    artifacts: ResolvedBackendBenchmarkArtifacts,
) -> tuple[_BackendIdentity, _BenchmarkRunner, float]:
    if backend == "torch":
        return _build_torch_runner(config=config, artifacts=artifacts)
    if backend == "onnxruntime":
        return _build_onnxruntime_runner(config=config, artifacts=artifacts)
    if backend == "tensorrt":
        return _build_tensorrt_runner(config=config, artifacts=artifacts)
    raise ValueError(f"Unsupported backend {backend!r}.")


def _build_torch_runner(
    *,
    config: BackendBenchmarkConfig,
    artifacts: ResolvedBackendBenchmarkArtifacts,
) -> tuple[_BackendIdentity, _TorchRunner, float]:
    started_at = time.perf_counter()
    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch=torch, device=config.evaluation.device)
    _, model_config, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=artifacts.source_checkpoint_path,
        project_root=artifacts.project_root,
    )
    if int(model_config.feat_dim) != artifacts.feature_dim:
        raise ValueError(
            "Checkpoint feature dimension does not match the exported boundary: "
            f"{model_config.feat_dim} != {artifacts.feature_dim}."
        )
    model = model.to(device=resolved_device, dtype=torch.float32)
    model.eval()
    return (
        _BackendIdentity(
            backend="torch",
            provider=None,
            implementation="campp_encoder",
            version=str(getattr(torch, "__version__", "unknown")),
            device=str(resolved_device),
        ),
        _TorchRunner(torch=torch, model=model, device=resolved_device),
        time.perf_counter() - started_at,
    )


def _build_onnxruntime_runner(
    *,
    config: BackendBenchmarkConfig,
    artifacts: ResolvedBackendBenchmarkArtifacts,
) -> tuple[_BackendIdentity, _OnnxRuntimeRunner, float]:
    started_at = time.perf_counter()
    onnxruntime = importlib.import_module("onnxruntime")
    selected_provider, providers = _select_onnxruntime_provider_chain(
        onnxruntime=onnxruntime,
        requested_provider=config.evaluation.onnxruntime_provider,
        provider_order=artifacts.onnxruntime_provider_order,
    )
    session = onnxruntime.InferenceSession(
        artifacts.onnx_model_path.as_posix(),
        providers=list(providers),
    )
    return (
        _BackendIdentity(
            backend="onnxruntime",
            provider=selected_provider,
            implementation="onnxruntime_session",
            version=str(getattr(onnxruntime, "__version__", "unknown")),
            device="cuda" if selected_provider != "CPUExecutionProvider" else "cpu",
        ),
        _OnnxRuntimeRunner(
            session=session,
            input_name=artifacts.input_name,
            output_name=artifacts.output_name,
        ),
        time.perf_counter() - started_at,
    )


def _build_tensorrt_runner(
    *,
    config: BackendBenchmarkConfig,
    artifacts: ResolvedBackendBenchmarkArtifacts,
) -> tuple[_BackendIdentity, _TensorRTRunner, float]:
    started_at = time.perf_counter()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT benchmark requires a CUDA-capable torch runtime.")
    runner = TensorRTEngineRunner(
        engine_path=artifacts.tensorrt_engine_path,
        input_name=artifacts.input_name,
        output_name=artifacts.output_name,
        profiles=artifacts.tensorrt_profiles,
    )
    return (
        _BackendIdentity(
            backend="tensorrt",
            provider=None,
            implementation="tensorrt_plan",
            version=_distribution_version_or_none("tensorrt-cu12"),
            device="cuda",
        ),
        _TensorRTRunner(torch=torch, runner=runner),
        time.perf_counter() - started_at,
    )


def _resolve_torch_device(*, torch: Any, device: str) -> Any:
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested, but torch.cuda.is_available() is false.")
    return torch.device(normalized)


def _select_onnxruntime_provider_chain(
    *,
    onnxruntime: Any,
    requested_provider: str,
    provider_order: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    available = tuple(str(provider) for provider in onnxruntime.get_available_providers())
    if requested_provider == "cpu":
        candidates = ("CPUExecutionProvider",)
    elif requested_provider == "cuda":
        candidates = ("CUDAExecutionProvider", "CPUExecutionProvider")
    else:
        preferred = tuple(
            provider
            for provider in provider_order
            if provider in {"CUDAExecutionProvider", "CPUExecutionProvider"}
        )
        if not preferred:
            preferred = ("CUDAExecutionProvider", "CPUExecutionProvider")
        elif "CPUExecutionProvider" not in preferred:
            preferred = (*preferred, "CPUExecutionProvider")
        candidates = preferred
    selected = next((provider for provider in candidates if provider in available), None)
    if selected is None:
        raise RuntimeError(
            "ONNX Runtime is installed but none of the benchmark provider candidates are "
            f"available: {list(candidates)} vs {list(available)}."
        )
    if selected == "CPUExecutionProvider" or "CPUExecutionProvider" not in available:
        return selected, (selected,)
    return selected, (selected, "CPUExecutionProvider")


def _build_input_tensor(
    *,
    workload: BackendBenchmarkWorkloadConfig,
    feature_dim: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.standard_normal(
        size=(workload.batch_size, workload.frame_count, feature_dim),
        dtype=np.float32,
    )


def _quality_metrics(
    *,
    backend: str,
    reference_output: np.ndarray | None,
    candidate_output: np.ndarray,
    max_mean_abs_diff: float,
    max_cosine_distance: float,
) -> tuple[float | None, float | None, float | None, bool | None]:
    if backend == "torch":
        return 0.0, 0.0, 0.0, True
    if reference_output is None:
        return None, None, None, None
    absolute_diff = np.abs(np.asarray(reference_output) - np.asarray(candidate_output))
    mean_abs_diff = float(absolute_diff.mean()) if absolute_diff.size else 0.0
    max_abs_diff = float(absolute_diff.max()) if absolute_diff.size else 0.0
    cosine_distance = _cosine_distance(reference_output, candidate_output)
    quality_passed = mean_abs_diff <= max_mean_abs_diff and cosine_distance <= max_cosine_distance
    return mean_abs_diff, max_abs_diff, cosine_distance, quality_passed


def _validate_output_shape(
    *,
    output: np.ndarray,
    workload: BackendBenchmarkWorkloadConfig,
    embedding_dim: int,
) -> None:
    observed_shape = tuple(int(value) for value in output.shape)
    expected_shape = (workload.batch_size, embedding_dim)
    if observed_shape != expected_shape:
        raise RuntimeError(
            "Benchmark backend returned an unexpected output shape: "
            f"{observed_shape} != {expected_shape}."
        )


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return {str(key): value for key, value in payload.items()}


def _parse_tensorrt_profiles(payload: dict[str, object]) -> tuple[TensorRTFP16Profile, ...]:
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("TensorRT report must define a non-empty `profiles` list.")
    profiles: list[TensorRTFP16Profile] = []
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"profiles[{index}] in the TensorRT report must be an object.")
        profile = {str(key): value for key, value in raw_profile.items()}
        profiles.append(
            TensorRTFP16Profile(
                profile_id=_require_string(
                    profile.get("profile_id"), f"profiles[{index}].profile_id"
                ),
                min_shape=_require_int_tuple(
                    profile.get("min_shape"), f"profiles[{index}].min_shape"
                ),
                opt_shape=_require_int_tuple(
                    profile.get("opt_shape"), f"profiles[{index}].opt_shape"
                ),
                max_shape=_require_int_tuple(
                    profile.get("max_shape"), f"profiles[{index}].max_shape"
                ),
            )
        )
    return tuple(profiles)


def _resolve_required_file(
    *,
    project_root: Path,
    raw_path: str | None,
    field_name: str,
) -> Path:
    if raw_path is None or not raw_path.strip():
        raise ValueError(f"{field_name} must resolve to a non-empty file path.")
    resolved = resolve_project_path(str(project_root), raw_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} does not resolve to a file: {resolved}.")
    return resolved


def _require_static_axis_size(raw: object, *, axis_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"Export boundary axis `{axis_name}` must be a static integer size.")
    if raw <= 0:
        raise ValueError(f"Export boundary axis `{axis_name}` must be positive.")
    return raw


def _require_string(raw: object, field_name: str) -> str:
    value = _coerce_string(raw)
    if value is None:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_int_tuple(raw: object, field_name: str) -> tuple[int, int, int]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{field_name} must be a three-element integer array.")
    values: list[int] = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field_name}[{index}] must be an integer.")
        values.append(item)
    return (values[0], values[1], values[2])


def _capture_current_process_rss_mib() -> float | None:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return round(float(parts[1]) / 1024.0, 3)
        return None
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if not output or not output.isdigit():
        return None
    return round(float(output) / 1024.0, 3)


def _capture_current_process_gpu_memory_mib() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    total_mib = 0.0
    matched = False
    for line in completed.stdout.splitlines():
        pid_text, _, memory_text = line.partition(",")
        if not pid_text.strip().isdigit():
            continue
        if int(pid_text.strip()) != os.getpid():
            continue
        try:
            total_mib += float(memory_text.strip())
        except ValueError:
            continue
        matched = True
    if not matched:
        return 0.0
    return round(total_mib, 3)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _stddev(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _latency_cv(values: list[float]) -> float | None:
    if not values:
        return None
    mean_value = fmean(values)
    if mean_value <= 0.0:
        return 0.0
    std_value = _stddev(values)
    if std_value is None:
        return None
    return float(std_value / mean_value)


def _throughput(*, value: int, latency_ms: float) -> float | None:
    if latency_ms <= 0.0:
        return None
    return float(value / (latency_ms / 1_000.0))


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float32).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    cosine = float(np.dot(left_flat, right_flat) / (left_norm * right_norm))
    cosine = max(min(cosine, 1.0), -1.0)
    return float(1.0 - cosine)


def _delta_peak(*, before: float | None, observed: list[float | None]) -> float | None:
    peak_value = _max_defined(observed)
    if before is None or peak_value is None:
        return None
    return max(peak_value - before, 0.0)


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return float(fmean(defined))


def _max_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return float(max(defined))


def _round_optional(value: float | None, *, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _coerce_string(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _distribution_version_or_none(distribution: str) -> str | None:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None


__all__ = [
    "ResolvedBackendBenchmarkArtifacts",
    "build_backend_benchmark_summaries",
    "resolve_backend_benchmark_artifacts",
    "run_backend_benchmark",
]
