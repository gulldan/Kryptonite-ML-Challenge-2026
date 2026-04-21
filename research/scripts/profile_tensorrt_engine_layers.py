"""Profile TensorRT engine layers with the same Python runtime used for inference."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class _LayerProfiler:
    def __init__(self, trt: Any) -> None:
        class Profiler(trt.IProfiler):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                super().__init__()
                self.records: list[tuple[str, float]] = []

            def report_layer_time(self, layer_name: str, ms: float) -> None:
                self.records.append((str(layer_name), float(ms)))

        self.instance = Profiler()


def main() -> None:
    args = _parse_args()
    output_json = Path(args.output_json)
    output_log = Path(args.output_log)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    shapes = _parse_shapes(args.shape)
    profile = _profile_python(
        engine_path=Path(args.engine),
        onnx_path=Path(args.onnx) if args.onnx else None,
        shapes=shapes,
        profile_index=args.profile_index,
        warmup_ms=args.warmup_ms,
        iterations=args.avg_runs,
    )
    output_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_log.write_text(_render_log(profile), encoding="utf-8")
    print(f"wrote {output_json}")
    print(f"wrote {output_log}")


def _profile_python(
    *,
    engine_path: Path,
    onnx_path: Path | None,
    shapes: dict[str, tuple[int, ...]],
    profile_index: int,
    warmup_ms: int,
    iterations: int,
) -> dict[str, Any]:
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT failed to create execution context.")

    stream = torch.cuda.current_stream()
    _set_profile(context=context, profile_index=profile_index, stream_handle=stream.cuda_stream)
    tensor_names = [
        str(engine.get_tensor_name(index)) for index in range(int(engine.num_io_tensors))
    ]
    inputs = _make_inputs(
        trt=trt, torch=torch, engine=engine, tensor_names=tensor_names, shapes=shapes
    )
    for name, tensor in inputs.items():
        _set_input_shape(context=context, name=name, shape=tuple(int(dim) for dim in tensor.shape))
    outputs = _make_outputs(
        trt=trt, torch=torch, engine=engine, context=context, tensor_names=tensor_names
    )
    for name, tensor in inputs.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))
    for name, tensor in outputs.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))

    layer_profiler = _LayerProfiler(trt)
    context.profiler = layer_profiler.instance
    if hasattr(context, "enqueue_emits_profile"):
        context.enqueue_emits_profile = True

    warmup_deadline = time.perf_counter() + max(warmup_ms, 0) / 1000.0
    while time.perf_counter() < warmup_deadline:
        _execute(context=context, stream=stream)
    layer_profiler.instance.records.clear()

    iteration_count = max(1, iterations)
    started = time.perf_counter()
    for _ in range(iteration_count):
        before = len(layer_profiler.instance.records)
        _execute(context=context, stream=stream)
        if len(layer_profiler.instance.records) == before and hasattr(
            context, "report_to_profiler"
        ):
            context.report_to_profiler()
    elapsed_s = time.perf_counter() - started

    layers = _summarize_layers(layer_profiler.instance.records, iteration_count=iteration_count)
    total_layer_mean_ms = sum(layer["mean_ms"] for layer in layers)
    for layer in layers:
        layer["share"] = (
            float(layer["mean_ms"]) / total_layer_mean_ms if total_layer_mean_ms > 0.0 else 0.0
        )
    engine_summary = _engine_summary(engine_path=engine_path, engine=engine)
    onnx_size_profile = _profile_onnx_initializer_sizes(onnx_path) if onnx_path else None
    return {
        "backend": "python_tensorrt",
        "engine": str(engine_path),
        "onnx": None if onnx_path is None else str(onnx_path),
        "elapsed_s": round(elapsed_s, 6),
        "iterations": iteration_count,
        "profile_index": profile_index,
        "shapes": {name: list(shape) for name, shape in shapes.items()},
        "engine_summary": engine_summary,
        "onnx_size_profile": onnx_size_profile,
        "total_layer_mean_ms": round(total_layer_mean_ms, 6),
        "layers": layers,
    }


def _make_inputs(
    *,
    trt: Any,
    torch: Any,
    engine: Any,
    tensor_names: list[str],
    shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    inputs = {}
    for name in tensor_names:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if name not in shapes:
            raise ValueError(f"Missing --shape for TensorRT input {name!r}.")
        dtype = _torch_dtype_from_trt(trt=trt, torch=torch, dtype=engine.get_tensor_dtype(name))
        shape = shapes[name]
        if dtype.is_floating_point:
            tensor = torch.randn(shape, device="cuda", dtype=dtype)
        elif dtype == torch.bool:
            tensor = torch.ones(shape, device="cuda", dtype=dtype)
        else:
            tensor = torch.ones(shape, device="cuda", dtype=dtype)
        inputs[name] = tensor.contiguous()
    return inputs


def _make_outputs(
    *,
    trt: Any,
    torch: Any,
    engine: Any,
    context: Any,
    tensor_names: list[str],
) -> dict[str, Any]:
    outputs = {}
    for name in tensor_names:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
        if any(dim <= 0 for dim in shape):
            raise RuntimeError(f"TensorRT output {name!r} has unresolved shape {shape}.")
        dtype = _torch_dtype_from_trt(trt=trt, torch=torch, dtype=engine.get_tensor_dtype(name))
        outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)
    return outputs


def _summarize_layers(
    records: list[tuple[str, float]], *, iteration_count: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for name, ms in records:
        grouped[name].append(ms)
    layers = [
        {
            "name": name,
            "calls": len(values),
            "mean_ms": round(sum(values) / max(1, iteration_count), 6),
            "mean_reported_ms": round(sum(values) / len(values), 6),
        }
        for name, values in grouped.items()
    ]
    return sorted(layers, key=lambda item: float(item["mean_ms"]), reverse=True)


def _engine_summary(*, engine_path: Path, engine: Any) -> dict[str, Any]:
    engine_size_bytes = engine_path.stat().st_size
    return {
        "engine_size_bytes": engine_size_bytes,
        "engine_size_mib": round(engine_size_bytes / (1024 * 1024), 6),
        "device_memory_size_bytes": int(getattr(engine, "device_memory_size", 0) or 0),
        "device_memory_size_v2_bytes": int(getattr(engine, "device_memory_size_v2", 0) or 0),
        "num_io_tensors": int(getattr(engine, "num_io_tensors", 0) or 0),
        "num_layers": int(getattr(engine, "num_layers", 0) or 0),
        "num_aux_streams": int(getattr(engine, "num_aux_streams", 0) or 0),
    }


def _profile_onnx_initializer_sizes(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load_model(path, load_external_data=False)
    initializer_sizes = {
        initializer.name: _initializer_nbytes(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        layer_name = node.name or f"{node.op_type}_{index}"
        for input_name in node.input:
            if input_name in initializer_sizes:
                consumers[input_name].append((layer_name, node.op_type))

    by_layer: dict[str, dict[str, Any]] = {}
    for initializer_name, nbytes in initializer_sizes.items():
        used_by = consumers.get(initializer_name, [])
        layer_name, op_type = used_by[0] if used_by else ("<unconsumed>", "<unconsumed>")
        row = by_layer.setdefault(
            layer_name,
            {
                "name": layer_name,
                "op_type": op_type,
                "parameter_bytes": 0,
                "initializer_count": 0,
                "shared_initializer_count": 0,
            },
        )
        row["parameter_bytes"] += nbytes
        row["initializer_count"] += 1
        if len(used_by) > 1:
            row["shared_initializer_count"] += 1

    layers = sorted(
        by_layer.values(),
        key=lambda item: int(item["parameter_bytes"]),
        reverse=True,
    )
    total_bytes = sum(initializer_sizes.values())
    by_op: dict[str, int] = defaultdict(int)
    for row in layers:
        bytes_for_row = int(row["parameter_bytes"])
        by_op[str(row["op_type"])] += bytes_for_row
        row["parameter_mib"] = round(bytes_for_row / (1024 * 1024), 6)
        row["share"] = bytes_for_row / total_bytes if total_bytes else 0.0
    op_type_totals = [
        {
            "op_type": op_type,
            "parameter_bytes": nbytes,
            "parameter_mib": round(nbytes / (1024 * 1024), 6),
            "share": nbytes / total_bytes if total_bytes else 0.0,
        }
        for op_type, nbytes in sorted(by_op.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "onnx_path": str(path),
        "initializer_total_bytes": total_bytes,
        "initializer_total_mib": round(total_bytes / (1024 * 1024), 6),
        "initializer_count": len(initializer_sizes),
        "layers": layers,
        "op_type_totals": op_type_totals,
    }


def _initializer_nbytes(initializer: Any) -> int:
    if initializer.raw_data:
        return len(initializer.raw_data)
    tensor_length = int(np.prod(tuple(int(dim) for dim in initializer.dims), dtype=np.int64))
    return tensor_length * _onnx_itemsize(initializer.data_type)


def _onnx_itemsize(data_type: int) -> int:
    item_sizes = {
        1: 4,  # FLOAT
        2: 1,  # UINT8
        3: 1,  # INT8
        4: 2,  # UINT16
        5: 2,  # INT16
        6: 4,  # INT32
        7: 8,  # INT64
        9: 1,  # BOOL
        10: 2,  # FLOAT16
        11: 8,  # DOUBLE
        12: 4,  # UINT32
        13: 8,  # UINT64
        14: 8,  # COMPLEX64
        15: 16,  # COMPLEX128
        16: 2,  # BFLOAT16
        17: 1,  # FLOAT8E4M3FN
        18: 1,  # FLOAT8E4M3FNUZ
        19: 1,  # FLOAT8E5M2
        20: 1,  # FLOAT8E5M2FNUZ
        21: 1,  # UINT4, conservative unpacked approximation
        22: 1,  # INT4, conservative unpacked approximation
    }
    return item_sizes.get(int(data_type), 0)


def _execute(*, context: Any, stream: Any) -> None:
    ok = bool(context.execute_async_v3(stream_handle=int(stream.cuda_stream)))
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 returned false.")
    stream.synchronize()


def _set_profile(*, context: Any, profile_index: int, stream_handle: int) -> None:
    setter = getattr(context, "set_optimization_profile_async", None)
    if callable(setter):
        ok = bool(setter(profile_index, int(stream_handle)))
        if not ok:
            raise RuntimeError(f"TensorRT rejected optimization profile {profile_index}.")
    elif profile_index != 0:
        raise RuntimeError("This TensorRT context does not support non-zero profiles.")


def _set_input_shape(*, context: Any, name: str, shape: tuple[int, ...]) -> None:
    setter = getattr(context, "set_input_shape", None)
    if not callable(setter):
        raise RuntimeError("TensorRT context does not expose set_input_shape.")
    ok = bool(setter(name, shape))
    if not ok:
        raise RuntimeError(f"TensorRT rejected shape {shape} for input {name!r}.")


def _torch_dtype_from_trt(*, trt: Any, torch: Any, dtype: Any) -> Any:
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.bfloat16:
        return torch.bfloat16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype!r}")


def _parse_shapes(values: list[str]) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for value in values:
        name, _, raw_shape = value.partition(":")
        if not name or not raw_shape:
            raise ValueError(f"Invalid --shape value {value!r}; expected name:1x2x3.")
        shapes[name] = tuple(int(dim) for dim in raw_shape.lower().split("x") if dim)
    return shapes


def _render_log(profile: dict[str, Any]) -> str:
    lines = [
        f"backend={profile['backend']}",
        f"engine={profile['engine']}",
        f"iterations={profile['iterations']}",
        f"total_layer_mean_ms={profile['total_layer_mean_ms']}",
    ]
    engine_summary = profile.get("engine_summary") or {}
    if engine_summary:
        lines.extend(
            [
                f"engine_size_mib={engine_summary.get('engine_size_mib')}",
                f"device_memory_size_v2_bytes={engine_summary.get('device_memory_size_v2_bytes')}",
            ]
        )
    lines.extend(["", "Top layers:"])
    for layer in profile["layers"][:40]:
        lines.append(
            f"{layer['mean_ms']:>10.4f} ms {100.0 * layer.get('share', 0.0):>6.2f}% {layer['name']}"
        )
    size_profile = profile.get("onnx_size_profile") or {}
    if size_profile:
        lines.extend(["", "Top ONNX initializer layers:"])
        for layer in size_profile.get("layers", [])[:40]:
            lines.append(
                f"{layer['parameter_mib']:>10.4f} MiB "
                f"{100.0 * layer.get('share', 0.0):>6.2f}% "
                f"{layer['op_type']} {layer['name']}"
            )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="TensorRT serialized engine path.")
    parser.add_argument("--onnx", default="", help="Optional ONNX path for initializer sizes.")
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        help="Input shape in name:dimxdim form. Repeat per input.",
    )
    parser.add_argument("--output-json", required=True, help="Layer profile JSON path.")
    parser.add_argument("--output-log", required=True, help="Human-readable layer profile log.")
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--duration-s", type=int, default=5, help="Accepted for CLI parity.")
    parser.add_argument("--avg-runs", type=int, default=10)
    parser.add_argument("--trtexec", default="trtexec", help="Accepted for old command parity.")
    parser.add_argument("--no-data-transfers", action="store_true", help="Accepted for parity.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
