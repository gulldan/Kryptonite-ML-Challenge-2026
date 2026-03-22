"""Inference-runtime probes and serving metadata for thin API entrypoints."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from kryptonite.config import ProjectConfig


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


@dataclass(slots=True)
class ServeRuntimeReport:
    python_version: str
    platform: str
    selected_backend: str
    probes: list[BackendProbe]

    @property
    def missing_required(self) -> list[str]:
        return [
            probe.distribution for probe in self.probes if probe.required and not probe.available
        ]

    @property
    def passed(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "selected_backend": self.selected_backend,
            "passed": self.passed,
            "missing_required": self.missing_required,
            "probes": [probe.to_dict() for probe in self.probes],
        }


def build_serve_runtime_report(*, config: ProjectConfig) -> ServeRuntimeReport:
    specs = (
        BackendSpec(
            name="torch",
            distribution="torch",
            module="torch",
            enabled=config.backends.allow_torch,
            required=config.backends.inference == "torch",
        ),
        BackendSpec(
            name="onnxruntime",
            distribution="onnxruntime",
            module="onnxruntime",
            enabled=config.backends.allow_onnx,
            required=config.backends.inference in {"onnx", "onnxruntime"},
        ),
        BackendSpec(
            name="tensorrt",
            distribution="tensorrt-cu12",
            module="tensorrt",
            enabled=config.backends.allow_tensorrt,
            required=config.backends.inference == "tensorrt",
        ),
    )

    probes = [_probe_backend(spec) for spec in specs]
    return ServeRuntimeReport(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()}-{platform.machine()}",
        selected_backend=config.backends.inference,
        probes=probes,
    )


def render_serve_runtime_report(report: ServeRuntimeReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"Serve runtime smoke: {status}",
        f"Python: {report.python_version}",
        f"Platform: {report.platform}",
        f"Selected backend: {report.selected_backend}",
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

    if report.passed:
        lines.append("Selected inference backend is available.")
    else:
        missing = ", ".join(report.missing_required)
        lines.append(f"Missing required backends: {missing}")
    return "\n".join(lines)


def build_service_metadata(
    *,
    config: ProjectConfig,
    report: ServeRuntimeReport,
) -> dict[str, Any]:
    return {
        "service": "kryptonite-infer",
        "status": "ok" if report.passed else "degraded",
        "selected_backend": report.selected_backend,
        "runtime": {
            "python_version": report.python_version,
            "platform": report.platform,
            "device": config.runtime.device,
            "log_level": config.runtime.log_level,
        },
        "config": {
            "tracking_enabled": config.tracking.enabled,
            "artifacts_root": config.paths.artifacts_root,
        },
        "backends": [probe.to_dict() for probe in report.probes],
    }


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
