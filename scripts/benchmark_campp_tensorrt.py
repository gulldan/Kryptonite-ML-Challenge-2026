"""Benchmark a built CAM++ TensorRT engine against the source PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from kryptonite.deployment import resolve_project_path
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.serve.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.serve.tensorrt_engine_config import load_tensorrt_fp16_config
from kryptonite.serve.tensorrt_engine_models import TensorRTFP16Profile
from kryptonite.serve.tensorrt_engine_runtime import (
    _benchmark_cuda_callable,
    _run_torch_encoder,
    _select_profile,
    _TensorRTEngineRunner,
)


class BenchmarkRow(TypedDict):
    batch_size: int
    frame_count: int
    profile_id: str
    torch_latency_ms: float
    tensorrt_latency_ms: float
    speedup_ratio: float
    torch_items_per_second: float
    tensorrt_items_per_second: float
    max_abs_diff: float
    mean_abs_diff: float
    cosine_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/benchmarks/campp-tensorrt",
        help="Directory for benchmark.json and benchmark.md.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        action="append",
        default=None,
        help="Frame count to benchmark. Can be passed multiple times.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="append",
        default=None,
        help="Batch size to benchmark. Can be passed multiple times.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_tensorrt_fp16_config(config_path=args.config)
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CAM++ TensorRT benchmark requires a CUDA-capable torch runtime.")

    project_root = resolve_project_path(config.project_root, ".")
    metadata_path = resolve_project_path(
        str(project_root),
        config.artifacts.model_bundle_metadata_path,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = load_export_boundary_from_model_metadata(metadata)
    feature_dim = _require_static_axis_size(contract.input_tensor.axes[-1].size, "mel_bins")
    embedding_dim = _require_static_axis_size(
        contract.output_tensor.axes[-1].size,
        "embedding_dim",
    )
    checkpoint_path = resolve_project_path(
        str(project_root),
        _require_string(metadata.get("source_checkpoint_path"), "source_checkpoint_path"),
    )
    engine_path = resolve_project_path(str(project_root), config.artifacts.engine_output_path)
    profiles = tuple(
        TensorRTFP16Profile(
            profile_id=profile.profile_id,
            min_shape=(profile.min_batch_size, profile.min_frame_count, feature_dim),
            opt_shape=(profile.opt_batch_size, profile.opt_frame_count, feature_dim),
            max_shape=(profile.max_batch_size, profile.max_frame_count, feature_dim),
        )
        for profile in config.build.profiles
    )

    _, model_config, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=checkpoint_path,
        project_root=project_root,
    )
    if model_config.feat_dim != feature_dim:
        raise ValueError(f"Checkpoint feat_dim {model_config.feat_dim} != export {feature_dim}.")
    model = model.to(device="cuda", dtype=torch.float32)
    model.eval()
    runner = _TensorRTEngineRunner(
        engine_path=engine_path,
        input_name=contract.input_tensor.name,
        output_name=contract.output_tensor.name,
    )

    frame_counts = tuple(args.frame_count or [100, 240, 600])
    batch_sizes = tuple(args.batch_size or [1, 8, 16, 32, 64])
    samples = _build_sample_grid(
        profiles=profiles,
        feature_dim=feature_dim,
        batch_sizes=batch_sizes,
        frame_counts=frame_counts,
    )
    rows: list[BenchmarkRow] = []
    for index, (batch_size, frame_count, profile) in enumerate(samples):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + index)
        sample_input = torch.randn(
            batch_size,
            frame_count,
            feature_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(device="cuda", dtype=torch.float32)
        profile_index = profiles.index(profile)
        with torch.inference_mode():
            reference_output = model(sample_input).detach()
        torch_latency_ms = _benchmark_cuda_callable(
            torch=torch,
            function=lambda sample_input=sample_input: _run_torch_encoder(
                torch=torch,
                model=model,
                input_tensor=sample_input,
            ),
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
        tensorrt_latency_ms = _benchmark_cuda_callable(
            torch=torch,
            function=lambda sample_input=sample_input, profile_index=profile_index: runner.run(
                sample_input,
                profile_index=profile_index,
            ),
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
        tensorrt_output = runner.run(sample_input, profile_index=profile_index).detach().float()
        if tuple(int(dim) for dim in tensorrt_output.shape) != (batch_size, embedding_dim):
            raise RuntimeError(
                "Unexpected TensorRT output shape: "
                f"{tuple(int(dim) for dim in tensorrt_output.shape)}"
            )
        row = _build_result_row(
            reference_output=reference_output,
            tensorrt_output=tensorrt_output,
            batch_size=batch_size,
            frame_count=frame_count,
            profile_id=profile.profile_id,
            torch_latency_ms=torch_latency_ms,
            tensorrt_latency_ms=tensorrt_latency_ms,
        )
        rows.append(row)
        if args.output == "text":
            print(_render_console_row(row))

    output_root = resolve_project_path(str(project_root), args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": "campp-tensorrt-batch-sweep",
        "device": torch.cuda.get_device_name(0),
        "config_path": str(Path(args.config).resolve()),
        "engine_path": str(engine_path),
        "checkpoint_path": str(checkpoint_path),
        "warmup_iterations": args.warmup_iterations,
        "benchmark_iterations": args.benchmark_iterations,
        "samples": rows,
    }
    benchmark_json_path = output_root / "benchmark.json"
    benchmark_md_path = output_root / "benchmark.md"
    benchmark_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    benchmark_md_path.write_text(
        _render_markdown(
            device=torch.cuda.get_device_name(0),
            engine_path=str(engine_path),
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
            rows=rows,
        ),
        encoding="utf-8",
    )
    written = {
        "benchmark_json_path": str(benchmark_json_path),
        "benchmark_markdown_path": str(benchmark_md_path),
    }
    if args.output == "json":
        print(json.dumps({"written": written, "summary": _summarize(rows)}, indent=2))
    else:
        print(f"Benchmark JSON: {benchmark_json_path}")
        print(f"Benchmark Markdown: {benchmark_md_path}")


def _build_sample_grid(
    *,
    profiles: tuple[TensorRTFP16Profile, ...],
    feature_dim: int,
    batch_sizes: tuple[int, ...],
    frame_counts: tuple[int, ...],
) -> tuple[tuple[int, int, TensorRTFP16Profile], ...]:
    samples = []
    for frame_count in frame_counts:
        for batch_size in batch_sizes:
            try:
                profile = _select_profile(
                    profiles,
                    shape=(batch_size, frame_count, feature_dim),
                )
            except ValueError:
                continue
            samples.append((batch_size, frame_count, profile))
    return tuple(samples)


def _build_result_row(
    *,
    reference_output: Any,
    tensorrt_output: Any,
    batch_size: int,
    frame_count: int,
    profile_id: str,
    torch_latency_ms: float,
    tensorrt_latency_ms: float,
) -> BenchmarkRow:
    reference_np = reference_output.detach().cpu().float().numpy()
    tensorrt_np = tensorrt_output.detach().cpu().float().numpy()
    absolute_diff = np.abs(reference_np - tensorrt_np)
    speedup_ratio = torch_latency_ms / tensorrt_latency_ms
    return {
        "batch_size": batch_size,
        "frame_count": frame_count,
        "profile_id": profile_id,
        "torch_latency_ms": torch_latency_ms,
        "tensorrt_latency_ms": tensorrt_latency_ms,
        "speedup_ratio": speedup_ratio,
        "torch_items_per_second": batch_size * 1_000.0 / torch_latency_ms,
        "tensorrt_items_per_second": batch_size * 1_000.0 / tensorrt_latency_ms,
        "max_abs_diff": float(absolute_diff.max()) if absolute_diff.size else 0.0,
        "mean_abs_diff": float(absolute_diff.mean()) if absolute_diff.size else 0.0,
        "cosine_distance": _cosine_distance(reference_np, tensorrt_np),
    }


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left.reshape(-1).astype(np.float64, copy=False)
    right_vector = right.reshape(-1).astype(np.float64, copy=False)
    denominator = (np.linalg.norm(left_vector) * np.linalg.norm(right_vector)) + 1e-12
    cosine_similarity = float(np.dot(left_vector, right_vector) / denominator)
    return float(1.0 - cosine_similarity)


def _render_console_row(row: BenchmarkRow) -> str:
    return (
        f"B={row['batch_size']:>2} T={row['frame_count']:>3} "
        f"{row['profile_id']:<14} "
        f"torch={row['torch_latency_ms']:.4f}ms "
        f"trt={row['tensorrt_latency_ms']:.4f}ms "
        f"speedup={row['speedup_ratio']:.3f}x "
        f"trt_ips={row['tensorrt_items_per_second']:.1f}"
    )


def _render_markdown(
    *,
    device: str,
    engine_path: str,
    warmup_iterations: int,
    benchmark_iterations: int,
    rows: Sequence[BenchmarkRow],
) -> str:
    lines = [
        "# CAM++ TensorRT Batch Benchmark",
        "",
        f"- Device: `{device}`",
        f"- Engine: `{engine_path}`",
        f"- Warmup iterations: `{warmup_iterations}`",
        f"- Benchmark iterations: `{benchmark_iterations}`",
        "",
        (
            "| Batch | Frames | Profile | Torch ms | TensorRT ms | Speedup | "
            "TensorRT items/s | mean abs diff | cosine distance |"
        ),
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['frame_count']} | `{row['profile_id']}` | "
            f"{row['torch_latency_ms']:.4f} | {row['tensorrt_latency_ms']:.4f} | "
            f"{row['speedup_ratio']:.4f} | {row['tensorrt_items_per_second']:.2f} | "
            f"{row['mean_abs_diff']:.8f} | {row['cosine_distance']:.8f} |"
        )
    summary = _summarize(rows)
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Max speedup: `{summary['max_speedup_ratio']:.4f}`",
            f"- Min speedup: `{summary['min_speedup_ratio']:.4f}`",
            f"- Max TensorRT items/s: `{summary['max_tensorrt_items_per_second']:.2f}`",
            f"- Max mean abs diff: `{summary['max_mean_abs_diff']:.8f}`",
            f"- Max cosine distance: `{summary['max_cosine_distance']:.8f}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _summarize(rows: Sequence[BenchmarkRow]) -> dict[str, float]:
    return {
        "max_speedup_ratio": max(row["speedup_ratio"] for row in rows),
        "min_speedup_ratio": min(row["speedup_ratio"] for row in rows),
        "max_tensorrt_items_per_second": max(row["tensorrt_items_per_second"] for row in rows),
        "max_mean_abs_diff": max(row["mean_abs_diff"] for row in rows),
        "max_cosine_distance": max(row["cosine_distance"] for row in rows),
    }


def _require_static_axis_size(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be static for TensorRT benchmarking.")
    return value


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model metadata must define {field_name}.")
    return value


def _import_torch() -> Any:
    import torch

    return torch


if __name__ == "__main__":
    main()
