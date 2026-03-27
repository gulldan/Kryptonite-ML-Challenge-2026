"""Build Triton model repositories from the repo's encoder-boundary artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from kryptonite.config import ProjectConfig
from kryptonite.deployment import (
    ArtifactReport,
    ArtifactSpec,
    build_artifact_report,
    render_artifact_report,
    resolve_project_path,
)

from .enrollment_cache import MODEL_BUNDLE_METADATA_NAME, load_model_bundle_metadata
from .export_boundary import ExportBoundaryContract, load_export_boundary_from_model_metadata

DEFAULT_TRITON_MODEL_NAME = "kryptonite_encoder"
DEFAULT_TRITON_REPOSITORY_ROOT = "artifacts/triton-model-repository"
DEFAULT_TRITON_SAMPLE_FRAME_COUNT = 12
TRITON_METADATA_COPY_NAME = "metadata.json"
TRITON_ONNX_MODEL_FILENAME = "model.onnx"
TRITON_TENSORRT_MODEL_FILENAME = "model.plan"
SUPPORTED_TRITON_BACKEND_MODES = frozenset({"onnx", "tensorrt"})


@dataclass(frozen=True, slots=True)
class TritonDynamicBatchingConfig:
    preferred_batch_sizes: tuple[int, ...] = (1, 4, 8)
    max_queue_delay_microseconds: int = 1_000

    def __post_init__(self) -> None:
        if self.max_queue_delay_microseconds < 0:
            raise ValueError("max_queue_delay_microseconds must be non-negative")
        if any(size <= 0 for size in self.preferred_batch_sizes):
            raise ValueError("preferred_batch_sizes must contain only positive integers")


@dataclass(frozen=True, slots=True)
class TritonRepositoryRequest:
    output_root: str = DEFAULT_TRITON_REPOSITORY_ROOT
    model_name: str = DEFAULT_TRITON_MODEL_NAME
    backend_mode: str = "onnx"
    engine_path: str | None = None
    version: int = 1
    max_batch_size: int = 8
    instance_group_count: int = 1
    sample_frame_count: int = DEFAULT_TRITON_SAMPLE_FRAME_COUNT
    dynamic_batching: TritonDynamicBatchingConfig = field(
        default_factory=TritonDynamicBatchingConfig
    )

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if self.version <= 0:
            raise ValueError("version must be positive.")
        if self.max_batch_size < 0:
            raise ValueError("max_batch_size must be non-negative.")
        if self.instance_group_count <= 0:
            raise ValueError("instance_group_count must be positive.")
        if self.sample_frame_count <= 0:
            raise ValueError("sample_frame_count must be positive.")


@dataclass(frozen=True, slots=True)
class BuiltTritonModelRepository:
    repository_root: str
    model_root: str
    version_root: str
    backend_mode: str
    platform: str
    instance_kind: str
    model_name: str
    model_path: str
    config_path: str
    metadata_path: str
    smoke_request_path: str
    readme_path: str
    input_name: str
    output_name: str
    input_dims: tuple[int, ...]
    output_dims: tuple[int, ...]

    @property
    def sample_curl_command(self) -> str:
        smoke_path = Path(self.smoke_request_path)
        return (
            "curl -s -X POST "
            f'"${{TRITON_SERVER_URL:-http://127.0.0.1:8000}}/v2/models/{self.model_name}/infer" '
            '-H "Content-Type: application/json" '
            f'--data @"{smoke_path}"'
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": self.repository_root,
            "model_root": self.model_root,
            "version_root": self.version_root,
            "backend_mode": self.backend_mode,
            "platform": self.platform,
            "instance_kind": self.instance_kind,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "metadata_path": self.metadata_path,
            "smoke_request_path": self.smoke_request_path,
            "readme_path": self.readme_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_dims": list(self.input_dims),
            "output_dims": list(self.output_dims),
            "sample_curl_command": self.sample_curl_command,
        }


def normalize_triton_backend_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_TRITON_BACKEND_MODES:
        raise ValueError(
            f"backend_mode must be one of {sorted(SUPPORTED_TRITON_BACKEND_MODES)}, got {value!r}."
        )
    return normalized


def build_triton_repository_source_report(
    *,
    config: ProjectConfig,
    request: TritonRepositoryRequest,
) -> ArtifactReport:
    backend_mode = normalize_triton_backend_mode(request.backend_mode)
    metadata_path = Path(config.deployment.model_bundle_root) / MODEL_BUNDLE_METADATA_NAME
    specs = [
        ArtifactSpec(
            name="model_bundle_root",
            configured_path=config.deployment.model_bundle_root,
            path_type="dir",
            require_non_empty=True,
            description="Source model bundle that seeds the Triton model repository.",
        ),
        ArtifactSpec(
            name="model_bundle_metadata_file",
            configured_path=str(metadata_path),
            path_type="file",
            require_non_empty=True,
            description="Model bundle metadata that carries the encoder-boundary contract.",
        ),
    ]
    if backend_mode == "onnx":
        specs.append(
            ArtifactSpec(
                name="source_model_file",
                configured_path=_configured_onnx_model_path(config=config),
                path_type="file",
                require_non_empty=True,
                description="ONNX model copied into Triton's versioned model directory.",
            )
        )
    else:
        specs.append(
            ArtifactSpec(
                name="source_engine_file",
                configured_path=_configured_engine_path(config=config, request=request),
                path_type="file",
                require_non_empty=True,
                description="TensorRT engine plan copied into Triton's versioned model directory.",
            )
        )
    return build_artifact_report(
        scope="triton",
        strict=True,
        project_root=config.paths.project_root,
        specs=specs,
    )


def build_triton_model_repository(
    *,
    config: ProjectConfig,
    request: TritonRepositoryRequest | None = None,
) -> BuiltTritonModelRepository:
    resolved_request = TritonRepositoryRequest() if request is None else request
    source_report = build_triton_repository_source_report(
        config=config,
        request=resolved_request,
    )
    if not source_report.passed:
        raise RuntimeError(render_artifact_report(source_report))

    backend_mode = normalize_triton_backend_mode(resolved_request.backend_mode)
    metadata_path = resolve_project_path(
        config.paths.project_root,
        f"{config.deployment.model_bundle_root}/{MODEL_BUNDLE_METADATA_NAME}",
    )
    metadata = load_model_bundle_metadata(metadata_path)
    contract = load_export_boundary_from_model_metadata(metadata)

    source_model_path = _resolve_source_model_path(
        config=config,
        request=resolved_request,
        backend_mode=backend_mode,
    )
    repository_root = resolve_project_path(config.paths.project_root, resolved_request.output_root)
    model_root = repository_root / resolved_request.model_name
    version_root = model_root / str(resolved_request.version)
    smoke_root = repository_root / "smoke"

    if model_root.exists():
        shutil.rmtree(model_root)
    version_root.mkdir(parents=True, exist_ok=True)
    smoke_root.mkdir(parents=True, exist_ok=True)

    output_model_path = version_root / _output_model_filename(backend_mode)
    shutil.copy2(source_model_path, output_model_path)

    metadata_copy_path = model_root / TRITON_METADATA_COPY_NAME
    shutil.copy2(metadata_path, metadata_copy_path)

    input_dims = _triton_tensor_dims(contract.input_tensor.axes)
    output_dims = _triton_tensor_dims(contract.output_tensor.axes)
    platform = _platform_for_backend(backend_mode)
    instance_kind = _instance_kind_for_backend(config=config, backend_mode=backend_mode)

    config_path = model_root / "config.pbtxt"
    config_path.write_text(
        render_triton_model_config(
            contract=contract,
            request=resolved_request,
            platform=platform,
            instance_kind=instance_kind,
            input_dims=input_dims,
            output_dims=output_dims,
        ),
        encoding="utf-8",
    )

    smoke_request_path = smoke_root / f"{resolved_request.model_name}_infer_request.json"
    smoke_request_path.write_text(
        json.dumps(
            build_triton_smoke_request(
                contract=contract,
                model_name=resolved_request.model_name,
                sample_frame_count=resolved_request.sample_frame_count,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readme_path = repository_root / "README.md"
    built = BuiltTritonModelRepository(
        repository_root=str(repository_root),
        model_root=str(model_root),
        version_root=str(version_root),
        backend_mode=backend_mode,
        platform=platform,
        instance_kind=instance_kind,
        model_name=resolved_request.model_name,
        model_path=str(output_model_path),
        config_path=str(config_path),
        metadata_path=str(metadata_copy_path),
        smoke_request_path=str(smoke_request_path),
        readme_path=str(readme_path),
        input_name=contract.input_tensor.name,
        output_name=contract.output_tensor.name,
        input_dims=input_dims,
        output_dims=output_dims,
    )
    readme_path.write_text(render_triton_repository_readme(built), encoding="utf-8")
    return built


def build_triton_smoke_request(
    *,
    contract: ExportBoundaryContract,
    model_name: str,
    sample_frame_count: int,
) -> dict[str, object]:
    if sample_frame_count <= 0:
        raise ValueError("sample_frame_count must be positive.")
    if not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    mel_bins = contract.input_tensor.axes[-1].size
    if not isinstance(mel_bins, int):
        raise ValueError("Triton smoke request requires a fixed mel-bin dimension.")

    data = [
        [
            [
                round(float((frame_index + 1) * (bin_index + 1) / 10_000.0), 8)
                for bin_index in range(mel_bins)
            ]
            for frame_index in range(sample_frame_count)
        ]
    ]
    return {
        "id": f"{model_name}-smoke",
        "inputs": [
            {
                "name": contract.input_tensor.name,
                "shape": [1, sample_frame_count, mel_bins],
                "datatype": _infer_datatype(contract.input_tensor.dtype),
                "data": data,
            }
        ],
        "outputs": [{"name": contract.output_tensor.name}],
    }


def render_triton_model_config(
    *,
    contract: ExportBoundaryContract,
    request: TritonRepositoryRequest,
    platform: str,
    instance_kind: str,
    input_dims: tuple[int, ...],
    output_dims: tuple[int, ...],
) -> str:
    lines = [
        f'name: "{request.model_name}"',
        f'platform: "{platform}"',
        f"max_batch_size: {request.max_batch_size}",
        "input [",
        "  {",
        f'    name: "{contract.input_tensor.name}"',
        f"    data_type: {_triton_dtype(contract.input_tensor.dtype)}",
        f"    dims: {_render_dims(input_dims)}",
        "  }",
        "]",
        "output [",
        "  {",
        f'    name: "{contract.output_tensor.name}"',
        f"    data_type: {_triton_dtype(contract.output_tensor.dtype)}",
        f"    dims: {_render_dims(output_dims)}",
        "  }",
        "]",
        "instance_group [",
        "  {",
        f"    kind: {instance_kind}",
        f"    count: {request.instance_group_count}",
        "  }",
        "]",
    ]
    if request.dynamic_batching.preferred_batch_sizes:
        preferred = ", ".join(str(size) for size in request.dynamic_batching.preferred_batch_sizes)
        lines.extend(
            [
                "dynamic_batching {",
                f"  preferred_batch_size: [{preferred}]",
                (
                    "  max_queue_delay_microseconds: "
                    f"{request.dynamic_batching.max_queue_delay_microseconds}"
                ),
                "}",
            ]
        )
    lines.extend(
        [
            'parameters: { key: "encoder_boundary" value: { string_value: "encoder_only" } }',
            ('parameters: { key: "frontend_location" value: { string_value: "runtime" } }'),
            (
                f'parameters: {{ key: "embedding_stage" value: '
                f'{{ string_value: "{contract.embedding_stage}" }} }}'
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def render_triton_repository_readme(built: BuiltTritonModelRepository) -> str:
    return (
        "\n".join(
            [
                "# Triton Model Repository",
                "",
                (
                    "This repository packages the repo's `encoder_input -> embedding` "
                    "export boundary for"
                ),
                "Triton. The raw-audio frontend still lives outside Triton:",
                "",
                "- decode / resample / loudness normalization",
                "- optional VAD trimming",
                "- waveform chunking",
                "- log-Mel / Fbank extraction",
                "",
                (
                    "Those steps stay in the Kryptonite runtime by design. Triton serves "
                    "the encoder-only"
                ),
                (
                    "graph, which keeps the deployment contract aligned with "
                    "`docs/export-boundary.md`."
                ),
                "",
                "## Packaged Model",
                "",
                f"- model name: `{built.model_name}`",
                f"- backend mode: `{built.backend_mode}`",
                f"- platform: `{built.platform}`",
                f"- instance kind: `{built.instance_kind}`",
                f"- input: `{built.input_name}` dims `{list(built.input_dims)}`",
                f"- output: `{built.output_name}` dims `{list(built.output_dims)}`",
                "",
                "## Layout",
                "",
                f"- config: `{Path(built.config_path).relative_to(built.repository_root)}`",
                f"- model: `{Path(built.model_path).relative_to(built.repository_root)}`",
                (
                    "- metadata copy: "
                    f"`{Path(built.metadata_path).relative_to(built.repository_root)}`"
                ),
                (
                    "- sample request: "
                    f"`{Path(built.smoke_request_path).relative_to(built.repository_root)}`"
                ),
                "",
                "## Smoke",
                "",
                "Assuming Triton is already running and this repository is mounted as the model",
                "repository, the sample infer request is:",
                "",
                "```bash",
                built.sample_curl_command,
                "```",
                "",
                "For a repository-local smoke helper, run:",
                "",
                "```bash",
                "uv run python scripts/triton_infer_smoke.py "
                f'--repository-root "{built.repository_root}" --model-name "{built.model_name}"',
                "```",
            ]
        )
        + "\n"
    )


def _configured_onnx_model_path(*, config: ProjectConfig) -> str:
    metadata_path = resolve_project_path(
        config.paths.project_root,
        f"{config.deployment.model_bundle_root}/{MODEL_BUNDLE_METADATA_NAME}",
    )
    if metadata_path.exists():
        try:
            metadata = load_model_bundle_metadata(metadata_path)
        except ValueError:
            pass
        else:
            model_file = metadata.get("model_file")
            if isinstance(model_file, str) and model_file.strip():
                return model_file
    return f"{config.deployment.model_bundle_root}/{TRITON_ONNX_MODEL_FILENAME}"


def _configured_engine_path(
    *,
    config: ProjectConfig,
    request: TritonRepositoryRequest,
) -> str:
    if request.engine_path is not None and request.engine_path.strip():
        return request.engine_path
    return f"{config.deployment.model_bundle_root}/{TRITON_TENSORRT_MODEL_FILENAME}"


def _resolve_source_model_path(
    *,
    config: ProjectConfig,
    request: TritonRepositoryRequest,
    backend_mode: str,
) -> Path:
    if backend_mode == "onnx":
        return resolve_project_path(
            config.paths.project_root,
            _configured_onnx_model_path(config=config),
        )
    return resolve_project_path(
        config.paths.project_root,
        _configured_engine_path(config=config, request=request),
    )


def _output_model_filename(backend_mode: str) -> str:
    if backend_mode == "onnx":
        return TRITON_ONNX_MODEL_FILENAME
    return TRITON_TENSORRT_MODEL_FILENAME


def _platform_for_backend(backend_mode: str) -> str:
    if backend_mode == "onnx":
        return "onnxruntime_onnx"
    return "tensorrt_plan"


def _instance_kind_for_backend(*, config: ProjectConfig, backend_mode: str) -> str:
    if backend_mode == "tensorrt":
        return "KIND_GPU"
    if config.runtime.device.lower().startswith("cuda"):
        return "KIND_GPU"
    return "KIND_CPU"


def _triton_tensor_dims(axes: tuple[object, ...]) -> tuple[int, ...]:
    dimensions: list[int] = []
    for axis in axes[1:]:
        size = getattr(axis, "size", None)
        if isinstance(size, int):
            dimensions.append(size)
        elif isinstance(size, str):
            dimensions.append(-1)
        else:
            raise ValueError(f"Unsupported tensor axis size: {size!r}")
    return tuple(dimensions)


def _triton_dtype(dtype: str) -> str:
    normalized = dtype.strip().lower()
    mapping = {
        "float16": "TYPE_FP16",
        "float32": "TYPE_FP32",
        "float64": "TYPE_FP64",
        "int32": "TYPE_INT32",
        "int64": "TYPE_INT64",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported Triton dtype mapping for {dtype!r}.")
    return mapping[normalized]


def _infer_datatype(dtype: str) -> str:
    normalized = dtype.strip().lower()
    mapping = {
        "float16": "FP16",
        "float32": "FP32",
        "float64": "FP64",
        "int32": "INT32",
        "int64": "INT64",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported Triton infer datatype mapping for {dtype!r}.")
    return mapping[normalized]


def _render_dims(dims: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in dims) + "]"


__all__ = [
    "DEFAULT_TRITON_MODEL_NAME",
    "DEFAULT_TRITON_REPOSITORY_ROOT",
    "DEFAULT_TRITON_SAMPLE_FRAME_COUNT",
    "BuiltTritonModelRepository",
    "SUPPORTED_TRITON_BACKEND_MODES",
    "TritonDynamicBatchingConfig",
    "TritonRepositoryRequest",
    "build_triton_model_repository",
    "build_triton_repository_source_report",
    "build_triton_smoke_request",
    "normalize_triton_backend_mode",
    "render_triton_model_config",
    "render_triton_repository_readme",
]
