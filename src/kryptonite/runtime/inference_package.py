"""Inference-package metadata and backend selection defaults for serving bundles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

INFERENCE_PACKAGE_FORMAT_VERSION = "kryptonite.runtime.inference_package.v1"
SUPPORTED_REQUESTED_INFERENCE_BACKENDS = frozenset(
    {"auto", "torch", "onnx", "onnxruntime", "tensorrt"}
)
DEFAULT_VALIDATED_BACKENDS: dict[str, bool] = {
    "torch": True,
    "onnxruntime": False,
    "tensorrt": False,
}
DEFAULT_AUTO_BACKEND_CHAIN: tuple[str, ...] = ("tensorrt", "onnxruntime", "torch")
DEFAULT_ONNXRUNTIME_PROVIDER_ORDER: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


@dataclass(frozen=True, slots=True)
class InferencePackageArtifacts:
    onnx_model_file: str | None
    tensorrt_engine_file: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "onnx_model_file": self.onnx_model_file,
            "tensorrt_engine_file": self.tensorrt_engine_file,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InferencePackageArtifacts:
        return cls(
            onnx_model_file=_coerce_string(payload.get("onnx_model_file")),
            tensorrt_engine_file=_coerce_string(payload.get("tensorrt_engine_file")),
        )


@dataclass(frozen=True, slots=True)
class InferencePackageContract:
    format_version: str
    backend_chain: tuple[str, ...]
    onnxruntime_provider_order: tuple[str, ...]
    validated_backends: dict[str, bool]
    artifacts: InferencePackageArtifacts

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "backend_chain": list(self.backend_chain),
            "onnxruntime_provider_order": list(self.onnxruntime_provider_order),
            "validated_backends": dict(self.validated_backends),
            "artifacts": self.artifacts.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InferencePackageContract:
        format_version = payload.get("format_version")
        if format_version != INFERENCE_PACKAGE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported inference package format version: "
                f"{format_version!r}; expected {INFERENCE_PACKAGE_FORMAT_VERSION!r}."
            )
        raw_backend_chain = payload.get("backend_chain")
        raw_provider_order = payload.get("onnxruntime_provider_order")
        raw_validated = payload.get("validated_backends")
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_backend_chain, list) or not raw_backend_chain:
            raise ValueError("Inference package metadata must define non-empty `backend_chain`.")
        if not isinstance(raw_provider_order, list) or not raw_provider_order:
            raise ValueError(
                "Inference package metadata must define non-empty `onnxruntime_provider_order`."
            )
        if not isinstance(raw_validated, Mapping):
            raise ValueError("Inference package metadata must define object `validated_backends`.")
        if not isinstance(raw_artifacts, Mapping):
            raise ValueError("Inference package metadata must define object `artifacts`.")
        backend_chain = tuple(_normalize_runtime_backend_name(item) for item in raw_backend_chain)
        provider_order = tuple(_coerce_non_empty_str_list(cast(list[object], raw_provider_order)))
        validated = _normalized_validated_backends(cast(Mapping[str, object], raw_validated))
        return cls(
            format_version=cast(str, format_version),
            backend_chain=backend_chain,
            onnxruntime_provider_order=provider_order,
            validated_backends=validated,
            artifacts=InferencePackageArtifacts.from_dict(
                cast(Mapping[str, object], raw_artifacts)
            ),
        )

    def backend_validated(self, backend: str) -> bool:
        normalized = _normalize_runtime_backend_name(backend)
        return bool(self.validated_backends.get(normalized, False))


def build_inference_package_contract(
    *,
    onnx_model_file: str | None,
    tensorrt_engine_file: str | None = None,
    validated_backends: Mapping[str, bool] | None = None,
) -> InferencePackageContract:
    return InferencePackageContract(
        format_version=INFERENCE_PACKAGE_FORMAT_VERSION,
        backend_chain=DEFAULT_AUTO_BACKEND_CHAIN,
        onnxruntime_provider_order=DEFAULT_ONNXRUNTIME_PROVIDER_ORDER,
        validated_backends=_normalized_validated_backends(validated_backends or {}),
        artifacts=InferencePackageArtifacts(
            onnx_model_file=_coerce_string(onnx_model_file),
            tensorrt_engine_file=_coerce_string(tensorrt_engine_file),
        ),
    )


def load_inference_package_from_model_metadata(
    model_metadata: Mapping[str, object] | None,
) -> InferencePackageContract:
    if model_metadata is None:
        return build_inference_package_contract(onnx_model_file=None)
    raw_package = model_metadata.get("inference_package")
    if isinstance(raw_package, Mapping):
        return InferencePackageContract.from_dict(cast(Mapping[str, object], raw_package))
    return build_inference_package_contract(
        onnx_model_file=_coerce_string(model_metadata.get("model_file")),
        tensorrt_engine_file=_coerce_string(model_metadata.get("tensorrt_engine_file")),
    )


def normalize_requested_inference_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "onnx":
        return "onnxruntime"
    if normalized not in SUPPORTED_REQUESTED_INFERENCE_BACKENDS:
        raise ValueError(
            "Unsupported backends.inference value "
            f"{value!r}; expected one of {sorted(SUPPORTED_REQUESTED_INFERENCE_BACKENDS)}."
        )
    return normalized


def _coerce_non_empty_str_list(values: list[object]) -> list[str]:
    coerced: list[str] = []
    for item in values:
        value = _coerce_string(item)
        if value is None:
            raise ValueError("Inference package metadata lists must contain non-empty strings.")
        coerced.append(value)
    return coerced


def _normalized_validated_backends(
    values: Mapping[str, object] | Mapping[str, bool],
) -> dict[str, bool]:
    validated = dict(DEFAULT_VALIDATED_BACKENDS)
    for backend_name, raw_value in values.items():
        normalized_name = _normalize_runtime_backend_name(backend_name)
        if not isinstance(raw_value, bool):
            raise ValueError("Inference package `validated_backends` values must be booleans.")
        validated[normalized_name] = raw_value
    return validated


def _normalize_runtime_backend_name(value: object) -> str:
    normalized = _coerce_string(value)
    if normalized is None:
        raise ValueError("Inference package backend names must be non-empty strings.")
    normalized = normalized.lower()
    if normalized == "onnx":
        return "onnxruntime"
    if normalized not in {"torch", "onnxruntime", "tensorrt"}:
        raise ValueError(
            "Inference package backend names must be one of ['onnxruntime', 'tensorrt', 'torch']."
        )
    return normalized


def _coerce_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "DEFAULT_AUTO_BACKEND_CHAIN",
    "DEFAULT_ONNXRUNTIME_PROVIDER_ORDER",
    "DEFAULT_VALIDATED_BACKENDS",
    "INFERENCE_PACKAGE_FORMAT_VERSION",
    "InferencePackageArtifacts",
    "InferencePackageContract",
    "SUPPORTED_REQUESTED_INFERENCE_BACKENDS",
    "build_inference_package_contract",
    "load_inference_package_from_model_metadata",
    "normalize_requested_inference_backend",
]
