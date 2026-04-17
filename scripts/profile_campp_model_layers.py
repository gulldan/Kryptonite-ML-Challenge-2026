"""Profile CAM++ encoder modules, CUDA kernels, and TensorRT layers."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.deployment import resolve_project_path
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.serve.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.serve.tensorrt_engine_config import load_tensorrt_fp16_config
from kryptonite.serve.tensorrt_engine_models import TensorRTFP16Profile
from kryptonite.serve.tensorrt_engine_runtime import _select_profile, _TensorRTEngineRunner


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    torch = _import_torch()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    _, _, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=args.checkpoint_path,
    )
    model = model.to(device=device, dtype=torch.float32).eval()
    sample_input = torch.randn(
        args.batch_size,
        args.frame_count,
        args.feature_dim,
        device=device,
        dtype=torch.float32,
    )

    group_rows = _profile_modules(
        torch=torch,
        model=model,
        sample_input=sample_input,
        modules=_named_group_modules(model),
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
    )
    leaf_rows = _profile_modules(
        torch=torch,
        model=model,
        sample_input=sample_input,
        modules=_named_leaf_modules(model),
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
    )
    kernel_rows = _profile_torch_kernels(
        torch=torch,
        model=model,
        sample_input=sample_input,
        warmup_iterations=args.warmup_iterations,
        profile_iterations=args.profile_iterations,
        top_n=args.top_kernels,
    )
    trt_rows = _profile_tensorrt_layers(
        args=args,
        torch=torch,
        sample_input=sample_input,
    )
    payload = {
        "config": {
            "checkpoint_path": args.checkpoint_path,
            "tensorrt_config": args.tensorrt_config,
            "batch_size": args.batch_size,
            "frame_count": args.frame_count,
            "feature_dim": args.feature_dim,
            "device": args.device,
            "warmup_iterations": args.warmup_iterations,
            "benchmark_iterations": args.benchmark_iterations,
            "profile_iterations": args.profile_iterations,
        },
        "group_modules": group_rows,
        "leaf_modules": leaf_rows,
        "torch_kernels": kernel_rows,
        "tensorrt_layers": trt_rows,
        "summary": _summarize(group_rows, leaf_rows, kernel_rows, trt_rows),
    }
    (output_root / "campp_model_layer_profile.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "campp_model_layer_profile.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    _write_csv(output_root / "group_modules.csv", group_rows)
    _write_csv(output_root / "leaf_modules.csv", leaf_rows)
    _write_csv(output_root / "torch_kernels.csv", kernel_rows)
    _write_csv(output_root / "tensorrt_layers.csv", trt_rows)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"[layer-profile] wrote {output_root / 'campp_model_layer_profile.json'}")


def _named_group_modules(model: Any) -> list[tuple[str, Any]]:
    names = [
        "head",
        "xvector.tdnn",
        "xvector.block1",
        "xvector.transit1",
        "xvector.block2",
        "xvector.transit2",
        "xvector.block3",
        "xvector.transit3",
        "xvector.out_nonlinear",
        "xvector.stats",
        "xvector.dense",
    ]
    return [(name, model.get_submodule(name)) for name in names]


def _named_leaf_modules(model: Any) -> list[tuple[str, Any]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and not any(module.children())
    ]


def _profile_modules(
    *,
    torch: Any,
    model: Any,
    sample_input: Any,
    modules: list[tuple[str, Any]],
    warmup_iterations: int,
    benchmark_iterations: int,
) -> list[dict[str, Any]]:
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(sample_input)
    torch.cuda.synchronize()

    records: dict[str, list[float]] = defaultdict(list)
    events: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    hooks = []

    def pre_hook(name: str) -> Callable[[Any, tuple[Any, ...]], None]:
        def hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            events[name].append((start, end))

        return hook

    def post_hook(name: str) -> Callable[[Any, tuple[Any, ...], Any], None]:
        def hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            events[name][-1][1].record()

        return hook

    for name, module in modules:
        hooks.append(module.register_forward_pre_hook(pre_hook(name)))
        hooks.append(module.register_forward_hook(post_hook(name)))
    total_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(benchmark_iterations):
            model(sample_input)
    torch.cuda.synchronize()
    total_wall_ms = (time.perf_counter() - total_started) * 1_000.0
    for hook in hooks:
        hook.remove()

    for name, pairs in events.items():
        for start, end in pairs:
            records[name].append(float(start.elapsed_time(end)))
    rows = []
    for name, values in records.items():
        module = dict(modules)[name]
        total_ms = float(np.sum(values))
        rows.append(
            {
                "name": name,
                "type": type(module).__name__,
                "parameter_bytes": _module_tensor_bytes(module, include_buffers=False),
                "buffer_bytes": _module_buffer_bytes(module),
                "calls": len(values),
                "total_ms": total_ms,
                "avg_ms": float(np.mean(values)),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "share_of_profile_wall": total_ms / total_wall_ms if total_wall_ms > 0 else 0.0,
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows


def _profile_torch_kernels(
    *,
    torch: Any,
    model: Any,
    sample_input: Any,
    warmup_iterations: int,
    profile_iterations: int,
    top_n: int,
) -> list[dict[str, Any]]:
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(sample_input)
    torch.cuda.synchronize()
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as profiler:
            for _ in range(profile_iterations):
                model(sample_input)
                torch.cuda.synchronize()
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        device_time_us = float(
            getattr(event, "device_time_total", 0.0)
            or getattr(event, "cuda_time_total", 0.0)
            or 0.0
        )
        if device_time_us <= 0:
            continue
        rows.append(
            {
                "name": str(event.key),
                "calls": int(event.count),
                "device_time_ms": device_time_us / 1_000.0,
                "avg_device_us": device_time_us / max(int(event.count), 1),
                "input_shapes": str(getattr(event, "input_shapes", "")),
            }
        )
    total = sum(row["device_time_ms"] for row in rows)
    for row in rows:
        row["share"] = row["device_time_ms"] / total if total > 0 else 0.0
    rows.sort(key=lambda row: row["device_time_ms"], reverse=True)
    return rows[:top_n]


def _profile_tensorrt_layers(
    *,
    args: argparse.Namespace,
    torch: Any,
    sample_input: Any,
) -> list[dict[str, Any]]:
    if not args.tensorrt_config:
        return []
    runner, profiles, feature_dim = _build_tensorrt_runner(args)
    profile = _select_profile(
        profiles,
        shape=(args.batch_size, args.frame_count, feature_dim),
    )
    profile_index = profiles.index(profile)
    with torch.inference_mode():
        for _ in range(args.warmup_iterations):
            runner.run(sample_input, profile_index=profile_index)
    torch.cuda.synchronize()
    profiler = _make_tensorrt_profiler(runner._trt)
    runner._context.profiler = profiler
    if hasattr(runner._context, "enqueue_emits_profile"):
        runner._context.enqueue_emits_profile = True
    with torch.inference_mode():
        for _ in range(args.profile_iterations):
            runner.run(sample_input, profile_index=profile_index)
            torch.cuda.synchronize()
            reporter = getattr(runner._context, "report_to_profiler", None)
            if callable(reporter):
                reporter()
    rows_by_name: dict[str, list[float]] = defaultdict(list)
    for name, milliseconds in profiler.rows:
        rows_by_name[name].append(milliseconds)
    rows = []
    for name, values in rows_by_name.items():
        total_ms = float(np.sum(values))
        rows.append(
            {
                "name": name,
                "calls": len(values),
                "total_ms": total_ms,
                "avg_ms": float(np.mean(values)),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
            }
        )
    total = sum(row["total_ms"] for row in rows)
    for row in rows:
        row["share"] = row["total_ms"] / total if total > 0 else 0.0
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows


def _make_tensorrt_profiler(trt: Any) -> Any:
    class PythonProfiler(trt.IProfiler):
        def __init__(self) -> None:
            trt.IProfiler.__init__(self)
            self.rows: list[tuple[str, float]] = []

        def report_layer_time(self, layer_name: str, ms: float) -> None:
            self.rows.append((layer_name, float(ms)))

    return PythonProfiler()


def _build_tensorrt_runner(
    args: argparse.Namespace,
) -> tuple[_TensorRTEngineRunner, tuple[TensorRTFP16Profile, ...], int]:
    config = load_tensorrt_fp16_config(config_path=args.tensorrt_config)
    project_root = resolve_project_path(config.project_root, ".")
    metadata_path = resolve_project_path(
        str(project_root),
        config.artifacts.model_bundle_metadata_path,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = load_export_boundary_from_model_metadata(metadata)
    feature_dim = _require_int(contract.input_tensor.axes[-1].size, "feature_dim")
    profiles = tuple(
        TensorRTFP16Profile(
            profile_id=profile.profile_id,
            min_shape=(profile.min_batch_size, profile.min_frame_count, feature_dim),
            opt_shape=(profile.opt_batch_size, profile.opt_frame_count, feature_dim),
            max_shape=(profile.max_batch_size, profile.max_frame_count, feature_dim),
        )
        for profile in config.build.profiles
    )
    engine_path = resolve_project_path(str(project_root), config.artifacts.engine_output_path)
    runner = _TensorRTEngineRunner(
        engine_path=engine_path,
        input_name=contract.input_tensor.name,
        output_name=contract.output_tensor.name,
    )
    return runner, profiles, feature_dim


def _summarize(
    group_rows: list[dict[str, Any]],
    leaf_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    trt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "top_group_modules": group_rows[:10],
        "top_leaf_modules": leaf_rows[:20],
        "top_torch_kernels": kernel_rows[:20],
        "top_tensorrt_layers": trt_rows[:20],
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# CAM++ Model Layer Profile", ""]
    for title, key, time_key in (
        ("Group Modules", "group_modules", "total_ms"),
        ("Leaf Modules", "leaf_modules", "total_ms"),
        ("Torch CUDA Kernels", "torch_kernels", "device_time_ms"),
        ("TensorRT Layers", "tensorrt_layers", "total_ms"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Rank | Name | Type/Calls | ms | Share | Params MiB | Buffers MiB |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for index, row in enumerate(payload[key][:30], start=1):
            type_or_calls = row.get("type", f"calls={row.get('calls', 0)}")
            share = row.get("share", row.get("share_of_profile_wall", 0.0))
            parameter_mib = float(row.get("parameter_bytes", 0)) / (1024 * 1024)
            buffer_mib = float(row.get("buffer_bytes", 0)) / (1024 * 1024)
            lines.append(
                f"| {index} | `{row['name']}` | `{type_or_calls}` | "
                f"{row[time_key]:.6f} | {share:.4f} | {parameter_mib:.4f} | "
                f"{buffer_mib:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _module_tensor_bytes(module: Any, *, include_buffers: bool) -> int:
    tensors = list(module.parameters(recurse=True))
    if include_buffers:
        tensors.extend(module.buffers(recurse=True))
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


def _module_buffer_bytes(module: Any) -> int:
    return sum(
        int(tensor.numel() * tensor.element_size()) for tensor in module.buffers(recurse=True)
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_int(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be static.")
    return value


def _import_torch() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CAM++ layer profiling requires CUDA.")
    return torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tensorrt-config", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frame-count", type=int, default=600)
    parser.add_argument("--feature-dim", type=int, default=80)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--top-kernels", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
