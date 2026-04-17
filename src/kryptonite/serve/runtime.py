"""Inference-runtime probes, backend resolution, and serving metadata."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from kryptonite.config import ProjectConfig
from kryptonite.deployment import ArtifactReport, resolve_project_path

from .inference_package import (
    InferencePackageContract,
    load_inference_package_from_model_metadata,
    normalize_requested_inference_backend,
)

_SUPPORTED_RUNTIME_IMPLEMENTATIONS = frozenset({"torch"})


@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: str
    distribution: str
    module: str
    enabled: bool
    required: bool


@dataclass(slots=True)
class BackendProbe:
    name: str
    distribution: str
    module: str
    enabled: bool
    required: bool
    available: bool
    version: str | None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "distribution": self.distribution,
            "module": self.module,
            "enabled": self.enabled,
            "required": self.required,
            "available": self.available,
            "version": self.version,
            "details": dict(self.details),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BackendSelectionStep:
    candidate: str
    backend: str
    provider: str | None
    selected: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "backend": self.backend,
            "provider": self.provider,
            "selected": self.selected,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BackendSelection:
    requested_backend: str
    selected_backend: str | None
    selected_provider: str | None
    reason: str
    error: str | None
    trace: tuple[BackendSelectionStep, ...]


@dataclass(slots=True)
class ServeRuntimeReport:
    python_version: str
    platform: str
    requested_backend: str
    selected_backend: str | None
    selected_provider: str | None
    selection_reason: str
    selection_error: str | None
    probes: list[BackendProbe]
    selection_trace: tuple[BackendSelectionStep, ...]

    @property
    def missing_required(self) -> list[str]:
        return [
            probe.distribution for probe in self.probes if probe.required and not probe.available
        ]

    @property
    def passed(self) -> bool:
        return self.selection_error is None and not self.missing_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "selected_provider": self.selected_provider,
            "selection_reason": self.selection_reason,
            "selection_error": self.selection_error,
            "passed": self.passed,
            "missing_required": self.missing_required,
            "probes": [probe.to_dict() for probe in self.probes],
            "selection_trace": [step.to_dict() for step in self.selection_trace],
        }


def build_serve_runtime_report(
    *,
    config: ProjectConfig,
    model_metadata: dict[str, object] | None = None,
) -> ServeRuntimeReport:
    requested_backend = normalize_requested_inference_backend(config.backends.inference)
    specs = (
        BackendSpec(
            name="torch",
            distribution="torch",
            module="torch",
            enabled=config.backends.allow_torch,
            required=requested_backend == "torch",
        ),
        BackendSpec(
            name="onnxruntime",
            distribution="onnxruntime",
            module="onnxruntime",
            enabled=config.backends.allow_onnx,
            required=requested_backend == "onnxruntime",
        ),
        BackendSpec(
            name="tensorrt",
            distribution="tensorrt-cu12",
            module="tensorrt",
            enabled=config.backends.allow_tensorrt,
            required=requested_backend == "tensorrt",
        ),
    )

    probes = [_probe_backend(spec) for spec in specs]
    inference_package = load_inference_package_from_model_metadata(model_metadata)
    selection = _select_backend(
        requested_backend=requested_backend,
        probes=probes,
        inference_package=inference_package,
        project_root=config.paths.project_root,
    )
    return ServeRuntimeReport(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()}-{platform.machine()}",
        requested_backend=requested_backend,
        selected_backend=selection.selected_backend,
        selected_provider=selection.selected_provider,
        selection_reason=selection.reason,
        selection_error=selection.error,
        probes=probes,
        selection_trace=selection.trace,
    )


def render_serve_runtime_report(report: ServeRuntimeReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"Serve runtime smoke: {status}",
        f"Python: {report.python_version}",
        f"Platform: {report.platform}",
        f"Requested backend: {report.requested_backend}",
        f"Selected backend: {report.selected_backend or '-'}",
        f"Selected provider: {report.selected_provider or '-'}",
        f"Selection reason: {report.selection_reason}",
    ]
    for probe in report.probes:
        if not probe.enabled:
            package_status = "disabled"
        else:
            package_status = "ok" if probe.available else "missing"
        version_text = probe.version or "-"
        line = (
            f"- {probe.name}: {package_status} "
            "("
            f"distribution={probe.distribution}, "
            f"version={version_text}, "
            f"required={probe.required}"
            ")"
        )
        if probe.details:
            detail_text = ", ".join(
                f"{key}={value}" for key, value in sorted(probe.details.items())
            )
            line = f"{line}; {detail_text}"
        if probe.error:
            line = f"{line}; error={probe.error}"
        lines.append(line)

    for step in report.selection_trace:
        provider = f"/{step.provider}" if step.provider else ""
        prefix = "selected" if step.selected else "skipped"
        lines.append(f"- selection {step.candidate}{provider}: {prefix}; {step.reason}")

    if report.passed:
        lines.append("Selected inference backend is available.")
    else:
        if report.selection_error is not None:
            lines.append(f"Selection failed: {report.selection_error}")
        if report.missing_required:
            missing = ", ".join(report.missing_required)
            lines.append(f"Missing required backends: {missing}")
    return "\n".join(lines)


def build_service_metadata(
    *,
    config: ProjectConfig,
    report: ServeRuntimeReport,
    artifact_report: ArtifactReport,
    enrollment_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "service": "kryptonite-infer",
        "status": "ok" if report.passed and artifact_report.passed else "degraded",
        "requested_backend": report.requested_backend,
        "selected_backend": report.selected_backend,
        "selected_provider": report.selected_provider,
        "selection": {
            "reason": report.selection_reason,
            "error": report.selection_error,
            "trace": [step.to_dict() for step in report.selection_trace],
        },
        "runtime": {
            "python_version": report.python_version,
            "platform": report.platform,
            "device": config.runtime.device,
            "log_level": config.runtime.log_level,
        },
        "config": {
            "tracking_enabled": config.tracking.enabled,
            "artifacts_root": config.paths.artifacts_root,
            "manifests_root": config.paths.manifests_root,
        },
        "artifacts": artifact_report.to_dict(),
        "backends": [probe.to_dict() for probe in report.probes],
        "enrollment_cache": dict(enrollment_cache or {}),
    }


def _select_backend(
    *,
    requested_backend: str,
    probes: list[BackendProbe],
    inference_package: InferencePackageContract,
    project_root: str,
) -> BackendSelection:
    if requested_backend == "auto":
        return _select_auto_backend(
            probes=probes,
            inference_package=inference_package,
            project_root=project_root,
        )
    if requested_backend == "torch":
        step = _evaluate_torch_candidate(probes=probes, inference_package=inference_package)
        return _selection_from_steps(
            requested_backend=requested_backend,
            steps=(step,),
            failure_message="Requested torch backend is not available.",
        )
    if requested_backend == "onnxruntime":
        step = _evaluate_onnxruntime_candidate(
            probes=probes,
            inference_package=inference_package,
            project_root=project_root,
            allow_cpu_provider=True,
        )
        return _selection_from_steps(
            requested_backend=requested_backend,
            steps=(step,),
            failure_message="Requested onnxruntime backend is not available.",
        )
    step = _evaluate_tensorrt_candidate(
        probes=probes,
        inference_package=inference_package,
        project_root=project_root,
    )
    return _selection_from_steps(
        requested_backend=requested_backend,
        steps=(step,),
        failure_message="Requested tensorrt backend is not available.",
    )


def _select_auto_backend(
    *,
    probes: list[BackendProbe],
    inference_package: InferencePackageContract,
    project_root: str,
) -> BackendSelection:
    steps = (
        _evaluate_tensorrt_candidate(
            probes=probes,
            inference_package=inference_package,
            project_root=project_root,
        ),
        _evaluate_onnxruntime_candidate(
            probes=probes,
            inference_package=inference_package,
            project_root=project_root,
            allow_cpu_provider=False,
        ),
        _evaluate_torch_candidate(probes=probes, inference_package=inference_package),
    )
    return _selection_from_steps(
        requested_backend="auto",
        steps=steps,
        failure_message=(
            "No eligible inference backend candidates were available in the configured "
            "fallback chain."
        ),
    )


def _selection_from_steps(
    *,
    requested_backend: str,
    steps: tuple[BackendSelectionStep, ...],
    failure_message: str,
) -> BackendSelection:
    for step in steps:
        if step.selected:
            provider = step.provider if step.backend == "onnxruntime" else None
            return BackendSelection(
                requested_backend=requested_backend,
                selected_backend=step.backend,
                selected_provider=provider,
                reason=step.reason,
                error=None,
                trace=steps,
            )
    last_reason = steps[-1].reason if steps else failure_message
    return BackendSelection(
        requested_backend=requested_backend,
        selected_backend=None,
        selected_provider=None,
        reason=last_reason,
        error=failure_message,
        trace=steps,
    )


def _evaluate_tensorrt_candidate(
    *,
    probes: list[BackendProbe],
    inference_package: InferencePackageContract,
    project_root: str,
) -> BackendSelectionStep:
    candidate = "tensorrt"
    probe = _probe_by_name(probes, "tensorrt")
    engine_path = inference_package.artifacts.tensorrt_engine_file
    if not probe.enabled:
        return BackendSelectionStep(
            candidate=candidate,
            backend="tensorrt",
            provider=None,
            selected=False,
            reason="disabled by config.allow_tensorrt=false.",
        )
    if not inference_package.backend_validated("tensorrt"):
        return BackendSelectionStep(
            candidate=candidate,
            backend="tensorrt",
            provider=None,
            selected=False,
            reason="not validated in model metadata inference_package contract.",
        )
    if "tensorrt" not in _SUPPORTED_RUNTIME_IMPLEMENTATIONS:
        return BackendSelectionStep(
            candidate=candidate,
            backend="tensorrt",
            provider=None,
            selected=False,
            reason="runtime backend is not implemented in this repository yet.",
        )
    if engine_path is None:
        return BackendSelectionStep(
            candidate=candidate,
            backend="tensorrt",
            provider=None,
            selected=False,
            reason="model metadata does not define a TensorRT engine artifact.",
        )
    if not _artifact_exists(project_root=project_root, configured_path=engine_path):
        return BackendSelectionStep(
            candidate=candidate,
            backend="tensorrt",
            provider=None,
            selected=False,
            reason=f"TensorRT engine artifact is missing: {engine_path}.",
        )
    if not probe.available:
        return BackendSelectionStep(
            candidate=candidate,
            backend="tensorrt",
            provider=None,
            selected=False,
            reason=f"runtime probe failed: {probe.error or 'distribution unavailable'}.",
        )
    return BackendSelectionStep(
        candidate=candidate,
        backend="tensorrt",
        provider=None,
        selected=True,
        reason="selected highest-priority direct TensorRT backend.",
    )


def _evaluate_onnxruntime_candidate(
    *,
    probes: list[BackendProbe],
    inference_package: InferencePackageContract,
    project_root: str,
    allow_cpu_provider: bool,
) -> BackendSelectionStep:
    probe = _probe_by_name(probes, "onnxruntime")
    candidate = "onnxruntime"
    onnx_model_file = inference_package.artifacts.onnx_model_file
    if not probe.enabled:
        return BackendSelectionStep(
            candidate=candidate,
            backend="onnxruntime",
            provider=None,
            selected=False,
            reason="disabled by config.allow_onnx=false.",
        )
    if not inference_package.backend_validated("onnxruntime"):
        return BackendSelectionStep(
            candidate=candidate,
            backend="onnxruntime",
            provider=None,
            selected=False,
            reason="not validated in model metadata inference_package contract.",
        )
    if "onnxruntime" not in _SUPPORTED_RUNTIME_IMPLEMENTATIONS:
        return BackendSelectionStep(
            candidate=candidate,
            backend="onnxruntime",
            provider=None,
            selected=False,
            reason="runtime backend is not implemented in this repository yet.",
        )
    if onnx_model_file is None:
        return BackendSelectionStep(
            candidate=candidate,
            backend="onnxruntime",
            provider=None,
            selected=False,
            reason="model metadata does not define an ONNX model artifact.",
        )
    if not _artifact_exists(project_root=project_root, configured_path=onnx_model_file):
        return BackendSelectionStep(
            candidate=candidate,
            backend="onnxruntime",
            provider=None,
            selected=False,
            reason=f"ONNX model artifact is missing: {onnx_model_file}.",
        )
    if not probe.available:
        return BackendSelectionStep(
            candidate=candidate,
            backend="onnxruntime",
            provider=None,
            selected=False,
            reason=f"runtime probe failed: {probe.error or 'distribution unavailable'}.",
        )
    available_providers = _available_onnxruntime_providers(probe)
    provider_order = inference_package.onnxruntime_provider_order
    if not allow_cpu_provider:
        provider_order = tuple(
            provider for provider in provider_order if provider != "CPUExecutionProvider"
        )
    for provider in provider_order:
        if provider in available_providers:
            selection_reason = (
                f"selected ONNX Runtime provider from the validated provider chain: {provider}."
            )
            return BackendSelectionStep(
                candidate=candidate,
                backend="onnxruntime",
                provider=provider,
                selected=True,
                reason=selection_reason,
            )
    allowed = ", ".join(provider_order)
    return BackendSelectionStep(
        candidate=candidate,
        backend="onnxruntime",
        provider=None,
        selected=False,
        reason=f"available providers do not include any of: {allowed}.",
    )


def _evaluate_torch_candidate(
    *,
    probes: list[BackendProbe],
    inference_package: InferencePackageContract,
) -> BackendSelectionStep:
    candidate = "torch"
    probe = _probe_by_name(probes, "torch")
    if not probe.enabled:
        return BackendSelectionStep(
            candidate=candidate,
            backend="torch",
            provider=None,
            selected=False,
            reason="disabled by config.allow_torch=false.",
        )
    if not inference_package.backend_validated("torch"):
        return BackendSelectionStep(
            candidate=candidate,
            backend="torch",
            provider=None,
            selected=False,
            reason="not validated in model metadata inference_package contract.",
        )
    if not probe.available:
        return BackendSelectionStep(
            candidate=candidate,
            backend="torch",
            provider=None,
            selected=False,
            reason=f"runtime probe failed: {probe.error or 'distribution unavailable'}.",
        )
    return BackendSelectionStep(
        candidate=candidate,
        backend="torch",
        provider=None,
        selected=True,
        reason="selected proven PyTorch fallback backend.",
    )


def _probe_backend(spec: BackendSpec) -> BackendProbe:
    if not spec.enabled:
        error = "backend disabled in config" if spec.required else None
        return BackendProbe(
            name=spec.name,
            distribution=spec.distribution,
            module=spec.module,
            enabled=spec.enabled,
            required=spec.required,
            available=False,
            version=_distribution_version(spec.distribution),
            error=error,
        )

    try:
        module = _load_module(spec.module)
    except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
        return BackendProbe(
            name=spec.name,
            distribution=spec.distribution,
            module=spec.module,
            enabled=spec.enabled,
            required=spec.required,
            available=False,
            version=_distribution_version(spec.distribution),
            error=f"{type(exc).__name__}: {exc}",
        )

    return BackendProbe(
        name=spec.name,
        distribution=spec.distribution,
        module=spec.module,
        enabled=spec.enabled,
        required=spec.required,
        available=True,
        version=_distribution_version(spec.distribution) or getattr(module, "__version__", None),
        details=_build_probe_details(spec.module, module),
    )


def _probe_by_name(probes: list[BackendProbe], name: str) -> BackendProbe:
    for probe in probes:
        if probe.name == name:
            return probe
    raise KeyError(f"Missing backend probe for {name!r}.")


def _available_onnxruntime_providers(probe: BackendProbe) -> tuple[str, ...]:
    providers = probe.details.get("providers")
    if not isinstance(providers, list):
        return ()
    return tuple(provider for provider in providers if isinstance(provider, str))


def _artifact_exists(*, project_root: str, configured_path: str) -> bool:
    return resolve_project_path(project_root, configured_path).exists()


def _load_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _build_probe_details(module_name: str, module: Any) -> dict[str, Any]:
    if module_name == "torch":
        details: dict[str, Any] = {}
        cuda = getattr(module, "cuda", None)
        if cuda is not None and hasattr(cuda, "is_available"):
            details["cuda_available"] = bool(cuda.is_available())
        torch_version = getattr(module, "version", None)
        if torch_version is not None and hasattr(torch_version, "cuda"):
            details["cuda_version"] = torch_version.cuda
        return details

    if module_name == "onnxruntime":
        providers = getattr(module, "get_available_providers", None)
        if callable(providers):
            return {"providers": providers()}
        return {}

    if module_name == "tensorrt":
        return {"logger_available": hasattr(module, "Logger")}

    return {}


__all__ = [
    "BackendProbe",
    "BackendSelection",
    "BackendSelectionStep",
    "BackendSpec",
    "ServeRuntimeReport",
    "build_service_metadata",
    "build_serve_runtime_report",
    "render_serve_runtime_report",
]
