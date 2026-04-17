"""Generic TensorRT helpers for ONNX graphs with one or more inputs."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Shape = tuple[int, ...]


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


class MultiInputTensorRTEngineRunner:
    def __init__(
        self,
        *,
        engine_path: Path,
        output_name: str,
    ) -> None:
        self._trt = _import_tensorrt()
        self._torch = _import_torch()
        logger = self._trt.Logger(self._trt.Logger.WARNING)
        runtime = self._trt.Runtime(logger)
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


__all__ = [
    "MultiInputTensorRTEngineRunner",
    "TensorRTInputProfile",
    "TensorRTMultiInputProfile",
    "benchmark_cuda_callable",
    "build_serialized_tensorrt_engine",
    "select_profile",
]
