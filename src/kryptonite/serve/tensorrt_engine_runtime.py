"""TensorRT engine build and validation runtime helpers."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint

from .tensorrt_engine_config import TensorRTFP16SampleConfig
from .tensorrt_engine_models import TensorRTFP16Profile, TensorRTFP16SampleResult


def build_serialized_tensorrt_engine(
    *,
    onnx_model_path: Path,
    input_name: str,
    profiles: tuple[TensorRTFP16Profile, ...],
    workspace_size_mib: int,
    builder_optimization_level: int = 3,
) -> bytes:
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if not getattr(builder, "platform_has_fast_fp16", False):
        raise RuntimeError("TensorRT builder does not report fast FP16 support on this machine.")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    parsed = False
    if hasattr(parser, "parse_from_file"):
        parsed = bool(parser.parse_from_file(onnx_model_path.as_posix()))
    else:  # pragma: no cover - compatibility fallback
        parsed = bool(parser.parse(onnx_model_path.read_bytes()))
    if not parsed:
        error_count = int(getattr(parser, "num_errors", 0))
        errors = [str(parser.get_error(index)) for index in range(error_count)]
        detail = "; ".join(errors) if errors else "unknown ONNX parser error"
        raise RuntimeError(f"TensorRT ONNX parser failed for {onnx_model_path}: {detail}")

    if network.num_inputs != 1:
        raise ValueError(
            f"Expected exactly one TensorRT network input, found {network.num_inputs}."
        )
    network_input = network.get_input(0)
    if network_input.name != input_name:
        raise ValueError(
            f"TensorRT parser input name mismatch: {network_input.name!r} != {input_name!r}."
        )

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
    build_config.set_flag(trt.BuilderFlag.FP16)

    for profile in profiles:
        optimization_profile = builder.create_optimization_profile()
        optimization_profile.set_shape(
            input_name,
            profile.min_shape,
            profile.opt_shape,
            profile.max_shape,
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


def validate_tensorrt_engine(
    *,
    engine_path: Path,
    source_checkpoint_path: Path,
    project_root: Path,
    input_name: str,
    output_name: str,
    feature_dim: int,
    embedding_dim: int,
    profiles: tuple[TensorRTFP16Profile, ...],
    samples: tuple[TensorRTFP16SampleConfig, ...],
    seed: int,
    warmup_iterations: int,
    benchmark_iterations: int,
    max_mean_abs_diff: float,
    max_cosine_distance: float,
    min_speedup_ratio: float,
) -> tuple[TensorRTFP16SampleResult, ...]:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT FP16 validation requires a CUDA-capable torch runtime.")
    _, model_config, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=source_checkpoint_path,
        project_root=project_root,
    )
    if model_config.feat_dim != feature_dim:
        raise ValueError(
            "TensorRT validation feature dimension does not match the source checkpoint: "
            f"{feature_dim} != {model_config.feat_dim}."
        )
    model = model.to(device="cuda", dtype=torch.float32)
    model.eval()

    runner = _TensorRTEngineRunner(
        engine_path=engine_path,
        input_name=input_name,
        output_name=output_name,
    )
    results: list[TensorRTFP16SampleResult] = []
    for index, sample in enumerate(samples):
        profile = _select_profile(
            profiles,
            shape=(sample.batch_size, sample.frame_count, feature_dim),
        )
        profile_index = profiles.index(profile)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + index)
        sample_input = torch.randn(
            sample.batch_size,
            sample.frame_count,
            feature_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(device="cuda", dtype=torch.float32)

        with torch.inference_mode():
            reference_output = model(sample_input).detach()
        torch_latency_ms = _benchmark_cuda_callable(
            torch=torch,
            function=lambda sample_input=sample_input: _run_torch_encoder(
                torch=torch,
                model=model,
                input_tensor=sample_input,
            ),
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )
        tensorrt_latency_ms = _benchmark_cuda_callable(
            torch=torch,
            function=lambda sample_input=sample_input, profile_index=profile_index: runner.run(
                sample_input,
                profile_index=profile_index,
            ),
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )
        tensorrt_output = (
            runner.run(sample_input, profile_index=profile_index).detach().to(dtype=torch.float32)
        )

        if tuple(int(dim) for dim in tensorrt_output.shape) != (sample.batch_size, embedding_dim):
            raise RuntimeError(
                "TensorRT engine returned an unexpected output shape: "
                f"{tuple(int(dim) for dim in tensorrt_output.shape)} != "
                f"({sample.batch_size}, {embedding_dim})."
            )

        reference_np = reference_output.to(device="cpu", dtype=torch.float32).numpy()
        tensorrt_np = tensorrt_output.to(device="cpu", dtype=torch.float32).numpy()
        absolute_diff = np.abs(reference_np - tensorrt_np)
        max_abs_diff = float(absolute_diff.max()) if absolute_diff.size else 0.0
        mean_abs_diff = float(absolute_diff.mean()) if absolute_diff.size else 0.0
        cosine_distance = _cosine_distance(reference_np, tensorrt_np)
        speedup_ratio = (
            float(torch_latency_ms / tensorrt_latency_ms)
            if tensorrt_latency_ms > 0.0
            else float("inf")
        )
        results.append(
            TensorRTFP16SampleResult(
                sample_id=sample.sample_id,
                profile_id=profile.profile_id,
                batch_size=sample.batch_size,
                frame_count=sample.frame_count,
                output_shape=(sample.batch_size, embedding_dim),
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                cosine_distance=cosine_distance,
                torch_latency_ms=torch_latency_ms,
                tensorrt_latency_ms=tensorrt_latency_ms,
                speedup_ratio=speedup_ratio,
                passed_quality=(
                    mean_abs_diff <= max_mean_abs_diff and cosine_distance <= max_cosine_distance
                ),
                passed_speedup=speedup_ratio >= min_speedup_ratio,
            )
        )
    return tuple(results)


class _TensorRTEngineRunner:
    def __init__(self, *, engine_path: Path, input_name: str, output_name: str) -> None:
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
        self._input_name = input_name
        self._output_name = output_name
        self._active_profile_index: int | None = None
        self._streams: dict[int, Any] = {}

    def run(self, input_tensor: Any, *, profile_index: int = 0) -> Any:
        torch = self._torch
        if not bool(input_tensor.is_cuda):
            raise ValueError("TensorRT execution expects CUDA tensors.")
        prepared_input = input_tensor.contiguous()
        current_stream = torch.cuda.current_stream(prepared_input.device)
        execution_stream = self._get_execution_stream(prepared_input.device)
        execution_stream.wait_stream(current_stream)
        with torch.cuda.stream(execution_stream):
            self._set_optimization_profile(profile_index, int(execution_stream.cuda_stream))
            self._set_input_shape(tuple(int(dim) for dim in prepared_input.shape))
            output_shape = tuple(
                int(dim) for dim in self._context.get_tensor_shape(self._output_name)
            )
            output_dtype = _torch_dtype_from_trt(
                trt=self._trt,
                torch=torch,
                dtype=self._engine.get_tensor_dtype(self._output_name),
            )
            output_tensor = torch.empty(
                output_shape,
                device=prepared_input.device,
                dtype=output_dtype,
            )
            self._set_tensor_address(self._input_name, int(prepared_input.data_ptr()))
            self._set_tensor_address(self._output_name, int(output_tensor.data_ptr()))
            ok = bool(
                self._context.execute_async_v3(stream_handle=int(execution_stream.cuda_stream))
            )
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

    def _set_input_shape(self, shape: tuple[int, ...]) -> None:
        setter = getattr(self._context, "set_input_shape", None)
        if callable(setter):
            ok = bool(setter(self._input_name, shape))
            if not ok:
                raise RuntimeError("TensorRT execution context rejected the input shape.")
            return
        raise RuntimeError("TensorRT execution context does not expose set_input_shape.")

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


def _benchmark_cuda_callable(
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


def _run_torch_encoder(*, torch: Any, model: Any, input_tensor: Any) -> Any:
    with torch.inference_mode():
        return model(input_tensor)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left.reshape(-1).astype(np.float64, copy=False)
    right_vector = right.reshape(-1).astype(np.float64, copy=False)
    denominator = (np.linalg.norm(left_vector) * np.linalg.norm(right_vector)) + 1e-12
    cosine_similarity = float(np.dot(left_vector, right_vector) / denominator)
    return float(1.0 - cosine_similarity)


def _select_profile(
    profiles: tuple[TensorRTFP16Profile, ...],
    *,
    shape: tuple[int, int, int],
) -> TensorRTFP16Profile:
    covering = [
        profile
        for profile in profiles
        if all(
            low <= value <= high
            for value, low, high in zip(shape, profile.min_shape, profile.max_shape, strict=True)
        )
    ]
    if not covering:
        raise ValueError(f"Input shape {shape} is not covered by any TensorRT profile.")
    return min(covering, key=lambda profile: _profile_volume(profile.max_shape))


def _profile_volume(shape: tuple[int, int, int]) -> int:
    return int(shape[0] * shape[1] * shape[2])


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
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TensorRT FP16 engine builds require the `tensorrt` Python package. "
            "Install the GPU extras into the repo-local `.venv` before running this workflow."
        ) from exc


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TensorRT FP16 engine validation requires torch in the repo-local `.venv`."
        ) from exc
    return torch


__all__ = ["build_serialized_tensorrt_engine", "validate_tensorrt_engine"]
