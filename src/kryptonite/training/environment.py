"""Training-environment probes for smoke checks and machine setup validation."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any


@dataclass(frozen=True, slots=True)
class PackageSpec:
    group: str
    distribution: str
    module: str
    required: bool = True


@dataclass(slots=True)
class PackageProbe:
    group: str
    distribution: str
    module: str
    required: bool
    available: bool
    version: str | None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "distribution": self.distribution,
            "module": self.module,
            "required": self.required,
            "available": self.available,
            "version": self.version,
            "details": dict(self.details),
            "error": self.error,
        }


@dataclass(slots=True)
class EnvironmentReport:
    python_version: str
    platform: str
    require_gpu: bool
    packages: list[PackageProbe]

    @property
    def missing_required(self) -> list[str]:
        return [
            probe.distribution for probe in self.packages if probe.required and not probe.available
        ]

    @property
    def passed(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "require_gpu": self.require_gpu,
            "passed": self.passed,
            "missing_required": self.missing_required,
            "packages": [probe.to_dict() for probe in self.packages],
        }


CORE_PACKAGE_SPECS: tuple[PackageSpec, ...] = (
    PackageSpec(group="train", distribution="torch", module="torch"),
    PackageSpec(group="train", distribution="torchaudio", module="torchaudio"),
    PackageSpec(group="train", distribution="onnx", module="onnx"),
    PackageSpec(group="train", distribution="onnxruntime", module="onnxruntime"),
    PackageSpec(group="train", distribution="hydra-core", module="hydra"),
    PackageSpec(group="train", distribution="typer", module="typer"),
    PackageSpec(group="tracking", distribution="mlflow", module="mlflow"),
    PackageSpec(group="tracking", distribution="wandb", module="wandb"),
)

GPU_PACKAGE_SPECS: tuple[PackageSpec, ...] = (
    PackageSpec(group="gpu", distribution="tensorrt-cu12", module="tensorrt"),
)


def build_training_environment_report(*, require_gpu: bool = False) -> EnvironmentReport:
    specs = list(CORE_PACKAGE_SPECS)
    if require_gpu:
        specs.extend(GPU_PACKAGE_SPECS)

    probes = [_probe_package(spec) for spec in specs]
    return EnvironmentReport(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()}-{platform.machine()}",
        require_gpu=require_gpu,
        packages=probes,
    )


def render_training_environment_report(report: EnvironmentReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"Training environment smoke: {status}",
        f"Python: {report.python_version}",
        f"Platform: {report.platform}",
        f"GPU stack required: {'yes' if report.require_gpu else 'no'}",
    ]
    for probe in report.packages:
        package_status = "ok" if probe.available else "missing"
        version_text = probe.version or "-"
        line = (
            f"- [{probe.group}] {probe.distribution} -> {probe.module}: "
            f"{package_status} ({version_text})"
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
        lines.append("All required imports succeeded.")
    else:
        missing = ", ".join(report.missing_required)
        lines.append(f"Missing required packages: {missing}")
    return "\n".join(lines)


def _probe_package(spec: PackageSpec) -> PackageProbe:
    try:
        module = _load_module(spec.module)
    except Exception as exc:  # pragma: no cover - exercised in unit tests via monkeypatch
        return PackageProbe(
            group=spec.group,
            distribution=spec.distribution,
            module=spec.module,
            required=spec.required,
            available=False,
            version=_distribution_version(spec.distribution),
            error=f"{type(exc).__name__}: {exc}",
        )

    return PackageProbe(
        group=spec.group,
        distribution=spec.distribution,
        module=spec.module,
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
        backends = getattr(module, "backends", None)
        if backends is not None:
            mps = getattr(backends, "mps", None)
            if mps is not None and hasattr(mps, "is_available"):
                details["mps_available"] = bool(mps.is_available())
        return details

    if module_name == "onnxruntime":
        providers = getattr(module, "get_available_providers", None)
        if callable(providers):
            return {"providers": providers()}
        return {}

    if module_name == "tensorrt":
        return {"logger_available": hasattr(module, "Logger")}

    return {}
