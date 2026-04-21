"""Generic TensorRT helpers for ONNX graphs with one or more inputs."""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Shape = tuple[int, ...]
TENSORRT_ENGINE_METADATA_SUFFIX = ".metadata.json"


@dataclass(frozen=True, slots=True)
class TensorRTInputProfile:
    min_shape: Shape
    opt_shape: Shape
    max_shape: Shape

    def contains(self, shape: Shape) -> bool:
        return all(
            low <= value <= high
            for value, low, high in zip(
                shape,
                self.min_shape,
                self.max_shape,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class TensorRTMultiInputProfile:
    profile_id: str
    inputs: dict[str, TensorRTInputProfile]

    def contains(self, shapes: Mapping[str, Shape]) -> bool:
        return all(
            name in shapes and profile.contains(tuple(int(dim) for dim in shapes[name]))
            for name, profile in self.inputs.items()
        )


def build_serialized_tensorrt_engine(
    *,
    onnx_model_path: Path,
    profiles: tuple[TensorRTMultiInputProfile, ...],
    workspace_size_mib: int,
    fp16: bool = True,
    builder_optimization_level: int = 3,
    version_compatible: bool = False,
    hardware_compatibility: str = "",
) -> bytes:
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if fp16 and not getattr(builder, "platform_has_fast_fp16", False):
        raise RuntimeError("TensorRT builder does not report fast FP16 support.")

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    parsed = (
        bool(parser.parse_from_file(onnx_model_path.as_posix()))
        if hasattr(parser, "parse_from_file")
        else bool(parser.parse(onnx_model_path.read_bytes()))
    )
    if not parsed:
        error_count = int(getattr(parser, "num_errors", 0))
        errors = [str(parser.get_error(index)) for index in range(error_count)]
        detail = "; ".join(errors) if errors else "unknown ONNX parser error"
        raise RuntimeError(f"TensorRT ONNX parser failed for {onnx_model_path}: {detail}")

    network_input_names = {
        str(network.get_input(index).name) for index in range(int(network.num_inputs))
    }
    required_names = {name for profile in profiles for name in profile.inputs}
    missing = sorted(required_names - network_input_names)
    if missing:
        raise ValueError(f"TensorRT profiles reference unknown network inputs: {missing}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_size_mib * 1024 * 1024,
    )
    _configure_build_detail(
        trt=trt,
        build_config=build_config,
        builder_optimization_level=builder_optimization_level,
        version_compatible=version_compatible,
        hardware_compatibility=hardware_compatibility,
    )
    if fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)

    for profile in profiles:
        optimization_profile = builder.create_optimization_profile()
        for input_name, input_profile in profile.inputs.items():
            optimization_profile.set_shape(
                input_name,
                input_profile.min_shape,
                input_profile.opt_shape,
                input_profile.max_shape,
            )
        build_config.add_optimization_profile(optimization_profile)

    serialized_engine = builder.build_serialized_network(network, build_config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT builder returned an empty serialized engine.")
    return bytes(serialized_engine)


def _configure_build_detail(
    *,
    trt: Any,
    build_config: Any,
    builder_optimization_level: int,
    version_compatible: bool,
    hardware_compatibility: str,
) -> None:
    if hasattr(build_config, "builder_optimization_level"):
        build_config.builder_optimization_level = int(builder_optimization_level)
    set_optimization_level = getattr(build_config, "set_builder_optimization_level", None)
    if callable(set_optimization_level):
        set_optimization_level(int(builder_optimization_level))
    verbosity = getattr(trt, "ProfilingVerbosity", None)
    detailed = None if verbosity is None else getattr(verbosity, "DETAILED", None)
    if detailed is not None and hasattr(build_config, "profiling_verbosity"):
        build_config.profiling_verbosity = detailed
    if version_compatible:
        builder_flag = getattr(trt, "BuilderFlag", None)
        flag = None if builder_flag is None else getattr(builder_flag, "VERSION_COMPATIBLE", None)
        if flag is None:
            raise RuntimeError("TensorRT runtime does not expose BuilderFlag.VERSION_COMPATIBLE.")
        build_config.set_flag(flag)
    _set_hardware_compatibility(
        trt=trt,
        build_config=build_config,
        hardware_compatibility=hardware_compatibility,
    )


def _set_hardware_compatibility(
    *,
    trt: Any,
    build_config: Any,
    hardware_compatibility: str,
) -> None:
    normalized = normalize_hardware_compatibility(hardware_compatibility)
    if not normalized:
        return
    levels = getattr(trt, "HardwareCompatibilityLevel", None)
    if levels is None:
        raise RuntimeError("TensorRT runtime does not expose HardwareCompatibilityLevel.")
    mapping = {
        "same_compute_capability": getattr(levels, "SAME_COMPUTE_CAPABILITY", None),
        "ampere_plus": getattr(levels, "AMPERE_PLUS", None),
    }
    level = mapping.get(normalized)
    if level is None:
        raise RuntimeError(
            f"Unsupported TensorRT hardware compatibility mode: {hardware_compatibility!r}"
        )
    if hasattr(build_config, "hardware_compatibility_level"):
        build_config.hardware_compatibility_level = level
        return
    setter = getattr(build_config, "set_hardware_compatibility_level", None)
    if callable(setter):
        setter(level)
        return
    raise RuntimeError("TensorRT build config does not expose hardware compatibility controls.")


def normalize_hardware_compatibility(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    return "" if normalized in {"", "none"} else normalized


def tensorrt_engine_metadata_path(engine_path: Path) -> Path:
    return engine_path.with_name(f"{engine_path.name}{TENSORRT_ENGINE_METADATA_SUFFIX}")


def build_tensorrt_engine_metadata(
    *,
    trt: Any,
    torch: Any,
    builder_image: str,
    version_compatible: bool,
    hardware_compatibility: str,
) -> dict[str, Any]:
    compute_capability: str | None = None
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        compute_capability = f"{int(capability[0])}.{int(capability[1])}"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "builder_image": builder_image,
        "tensorrt_version": str(getattr(trt, "__version__", "")),
        "cuda_version": str(getattr(torch.version, "cuda", "")),
        "version_compatible": bool(version_compatible),
        "hardware_compatibility": normalize_hardware_compatibility(hardware_compatibility),
        "compute_capability": compute_capability,
    }


def write_tensorrt_engine_metadata(*, engine_path: Path, metadata: Mapping[str, Any]) -> Path:
    metadata_path = tensorrt_engine_metadata_path(engine_path)
    metadata_path.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def load_tensorrt_engine_metadata(engine_path: Path) -> dict[str, Any] | None:
    metadata_path = tensorrt_engine_metadata_path(engine_path)
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"TensorRT engine metadata must be a JSON object: {metadata_path}")
    return payload


def validate_tensorrt_engine_metadata(
    *,
    engine_path: Path,
    metadata: Mapping[str, Any],
    runtime_tensorrt_version: str,
    device_compute_capability: str | None,
) -> None:
    expected_version = str(metadata.get("tensorrt_version", "")).strip()
    version_compatible = bool(metadata.get("version_compatible", False))
    if expected_version:
        _validate_runtime_tensorrt_version(
            engine_path=engine_path,
            expected_version=expected_version,
            runtime_version=runtime_tensorrt_version,
            version_compatible=version_compatible,
        )
    expected_capability = str(metadata.get("compute_capability", "")).strip()
    hardware_compatibility = normalize_hardware_compatibility(
        str(metadata.get("hardware_compatibility", ""))
    )
    if expected_capability and device_compute_capability:
        if hardware_compatibility == "same_compute_capability":
            if device_compute_capability != expected_capability:
                raise RuntimeError(
                    "TensorRT engine metadata compute capability mismatch for "
                    f"{engine_path}: built for {expected_capability}, "
                    f"runtime device is {device_compute_capability}."
                )
        elif hardware_compatibility == "ampere_plus":
            runtime_major = _major_compute_capability(device_compute_capability)
            if runtime_major < 8:
                raise RuntimeError(
                    "TensorRT engine metadata requires an Ampere-or-newer GPU for "
                    f"{engine_path}, got compute capability {device_compute_capability}."
                )


def _validate_runtime_tensorrt_version(
    *,
    engine_path: Path,
    expected_version: str,
    runtime_version: str,
    version_compatible: bool,
) -> None:
    expected = _parse_numeric_version(expected_version)
    runtime = _parse_numeric_version(runtime_version)
    if not expected or not runtime:
        return
    if version_compatible:
        if runtime[0] != expected[0] or runtime < expected:
            raise RuntimeError(
                "TensorRT engine metadata version mismatch for "
                f"{engine_path}: built with {expected_version}, runtime has "
                f"{runtime_version}, and version-compatible engines only run on "
                "same-major newer TensorRT runtimes."
            )
        return
    if runtime != expected:
        raise RuntimeError(
            "TensorRT engine metadata version mismatch for "
            f"{engine_path}: built with {expected_version}, runtime has {runtime_version}."
        )


def _parse_numeric_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in value.split("."):
        digits = "".join(character for character in chunk if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _major_compute_capability(value: str) -> int:
    head = value.split(".", maxsplit=1)[0].strip()
    return int(head) if head.isdigit() else -1


class MultiInputTensorRTEngineRunner:
    def __init__(
        self,
        *,
        engine_path: Path,
        output_name: str,
    ) -> None:
        self._trt = _import_tensorrt()
        self._torch = _import_torch()
        metadata = load_tensorrt_engine_metadata(engine_path)
        if metadata is not None:
            validate_tensorrt_engine_metadata(
                engine_path=engine_path,
                metadata=metadata,
                runtime_tensorrt_version=str(getattr(self._trt, "__version__", "")),
                device_compute_capability=_runtime_device_compute_capability(self._torch),
            )
        logger = self._trt.Logger(self._trt.Logger.WARNING)
        runtime = self._trt.Runtime(logger)
        _enable_engine_host_code(runtime=runtime, metadata=metadata)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT failed to create an execution context.")
        self._engine = engine
        self._context = context
        self._output_name = output_name
        self._active_profile_index: int | None = None
        self._streams: dict[int, Any] = {}

    def run(self, inputs: Mapping[str, Any], *, profile_index: int = 0) -> Any:
        torch = self._torch
        if not inputs:
            raise ValueError("TensorRT execution expects at least one input tensor.")
        prepared: dict[str, Any] = {}
        device = None
        for name, tensor in inputs.items():
            if not bool(tensor.is_cuda):
                raise ValueError(f"TensorRT input {name!r} must be a CUDA tensor.")
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise ValueError("All TensorRT inputs must be on the same CUDA device.")
            prepared[name] = tensor.contiguous()
        assert device is not None

        current_stream = torch.cuda.current_stream(device)
        execution_stream = self._get_execution_stream(device)
        execution_stream.wait_stream(current_stream)
        with torch.cuda.stream(execution_stream):
            stream_handle = int(execution_stream.cuda_stream)
            self._set_optimization_profile(profile_index, stream_handle)
            for name, tensor in prepared.items():
                self._set_input_shape(name, tuple(int(dim) for dim in tensor.shape))
            output_shape = tuple(
                int(dim) for dim in self._context.get_tensor_shape(self._output_name)
            )
            output_dtype = _torch_dtype_from_trt(
                trt=self._trt,
                torch=torch,
                dtype=self._engine.get_tensor_dtype(self._output_name),
            )
            output_tensor = torch.empty(output_shape, device=device, dtype=output_dtype)
            for name, tensor in prepared.items():
                self._set_tensor_address(name, int(tensor.data_ptr()))
            self._set_tensor_address(self._output_name, int(output_tensor.data_ptr()))
            ok = bool(self._context.execute_async_v3(stream_handle=stream_handle))
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
        current_stream.wait_stream(execution_stream)
        return output_tensor

    def _get_execution_stream(self, device: Any) -> Any:
        torch = self._torch
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream = self._streams.get(int(device_index))
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
            self._streams[int(device_index)] = stream
        return stream

    def _set_input_shape(self, tensor_name: str, shape: Shape) -> None:
        setter = getattr(self._context, "set_input_shape", None)
        if not callable(setter):
            raise RuntimeError("TensorRT execution context does not expose set_input_shape.")
        ok = bool(setter(tensor_name, shape))
        if not ok:
            raise RuntimeError(
                f"TensorRT execution context rejected shape {shape} for {tensor_name!r}."
            )

    def _set_optimization_profile(self, profile_index: int, stream_handle: int) -> None:
        if self._active_profile_index == profile_index:
            return
        setter = getattr(self._context, "set_optimization_profile_async", None)
        if callable(setter):
            ok = bool(setter(profile_index, stream_handle))
            if not ok:
                raise RuntimeError(
                    f"TensorRT execution context rejected optimization profile {profile_index}."
                )
            self._active_profile_index = profile_index
            return
        if profile_index != 0:
            raise RuntimeError(
                "TensorRT execution context does not expose set_optimization_profile_async."
            )
        self._active_profile_index = profile_index

    def _set_tensor_address(self, tensor_name: str, address: int) -> None:
        setter = getattr(self._context, "set_tensor_address", None)
        if not callable(setter):
            raise RuntimeError("TensorRT execution context does not expose set_tensor_address.")
        setter(tensor_name, address)


def select_profile(
    profiles: tuple[TensorRTMultiInputProfile, ...],
    *,
    shapes: Mapping[str, Shape],
) -> TensorRTMultiInputProfile:
    normalized = {name: tuple(int(dim) for dim in shape) for name, shape in shapes.items()}
    covering = [profile for profile in profiles if profile.contains(normalized)]
    if not covering:
        raise ValueError(f"Input shapes {normalized} are not covered by any TensorRT profile.")
    return min(covering, key=_profile_volume)


def benchmark_cuda_callable(
    *,
    torch: Any,
    function: Any,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> float:
    for _ in range(warmup_iterations):
        function()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    for _ in range(benchmark_iterations):
        function()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started_at) * 1_000.0
    return elapsed_ms / float(benchmark_iterations)


def _profile_volume(profile: TensorRTMultiInputProfile) -> int:
    volume = 0
    for input_profile in profile.inputs.values():
        current = 1
        for dim in input_profile.max_shape:
            current *= int(dim)
        volume += current
    return volume


def _torch_dtype_from_trt(*, trt: Any, torch: Any, dtype: object) -> Any:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    bf16 = getattr(trt.DataType, "BF16", None)
    if bf16 is not None:
        mapping[bf16] = torch.bfloat16
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype!r}")
    return mapping[dtype]


def _import_tensorrt() -> Any:
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorRT engine builds require the `tensorrt` Python package. "
            "Install the CUDA-specific TensorRT wheel in the repo-local `.venv`."
        ) from exc


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorRT execution requires torch in the repo-local `.venv`.") from exc


def _runtime_device_compute_capability(torch: Any) -> str | None:
    if not torch.cuda.is_available():
        return None
    capability = torch.cuda.get_device_capability()
    return f"{int(capability[0])}.{int(capability[1])}"


def _enable_engine_host_code(*, runtime: Any, metadata: Mapping[str, Any] | None) -> None:
    if hasattr(runtime, "engine_host_code_allowed"):
        runtime.engine_host_code_allowed = True
        return
    setter = getattr(runtime, "set_engine_host_code_allowed", None)
    if callable(setter):
        setter(True)


__all__ = [
    "MultiInputTensorRTEngineRunner",
    "TensorRTInputProfile",
    "TensorRTMultiInputProfile",
    "benchmark_cuda_callable",
    "build_tensorrt_engine_metadata",
    "build_serialized_tensorrt_engine",
    "load_tensorrt_engine_metadata",
    "normalize_hardware_compatibility",
    "select_profile",
    "tensorrt_engine_metadata_path",
    "validate_tensorrt_engine_metadata",
    "write_tensorrt_engine_metadata",
]
