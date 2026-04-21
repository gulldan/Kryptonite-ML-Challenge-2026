"""Build and benchmark a TensorRT engine for a generic ONNX encoder graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.runtime.tensorrt_generic import (
    MultiInputTensorRTEngineRunner,
    TensorRTInputProfile,
    TensorRTMultiInputProfile,
    benchmark_cuda_callable,
    build_serialized_tensorrt_engine,
    build_tensorrt_engine_metadata,
    select_profile,
    write_tensorrt_engine_metadata,
)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    config = _load_config(config_path)
    project_root = Path(config.get("project_root", ".")).resolve()
    onnx_path = _resolve(project_root, str(config["onnx_path"]))
    engine_path = _resolve(project_root, str(config["engine_path"]))
    output_root = _resolve(project_root, str(config.get("output_root", engine_path.parent)))
    output_root.mkdir(parents=True, exist_ok=True)

    profiles = _parse_profiles(config)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    version_compatible = bool(config.get("version_compatible", True))
    hardware_compatibility = str(config.get("hardware_compatibility", "same_compute_capability"))
    engine_bytes = build_serialized_tensorrt_engine(
        onnx_model_path=onnx_path,
        profiles=profiles,
        workspace_size_mib=int(config.get("workspace_size_mib", 4096)),
        fp16=bool(config.get("fp16", True)),
        builder_optimization_level=int(config.get("builder_optimization_level", 3)),
        version_compatible=version_compatible,
        hardware_compatibility=hardware_compatibility,
    )
    engine_path.write_bytes(engine_bytes)
    if engine_path.stat().st_size <= 0:
        raise RuntimeError(f"TensorRT engine write produced an empty file: {engine_path}")
    import tensorrt as trt
    import torch

    metadata = build_tensorrt_engine_metadata(
        trt=trt,
        torch=torch,
        builder_image=str(
            config.get("builder_image", os.environ.get("KRYPTONITE_TRT_BUILDER_IMAGE", ""))
        ),
        version_compatible=version_compatible,
        hardware_compatibility=hardware_compatibility,
    )
    metadata_path = write_tensorrt_engine_metadata(engine_path=engine_path, metadata=metadata)

    report = {
        "title": str(config.get("title", "Generic TensorRT engine")),
        "config_path": str(config_path),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "engine_metadata_path": str(metadata_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "workspace_size_mib": int(config.get("workspace_size_mib", 4096)),
        "builder_optimization_level": int(config.get("builder_optimization_level", 3)),
        "fp16": bool(config.get("fp16", True)),
        "version_compatible": version_compatible,
        "hardware_compatibility": hardware_compatibility,
        "engine_metadata": metadata,
        "profiles": [_profile_to_dict(profile) for profile in profiles],
        "validation": [],
        "benchmarks": [],
    }

    input_dtypes = _load_onnx_input_dtypes(onnx_path)
    output_name = str(config.get("output_name", "embedding"))
    validation = config.get("validation", {})
    if bool(validation.get("enabled", True)):
        report["validation"] = _run_validation(
            config=validation,
            onnx_path=onnx_path,
            engine_path=engine_path,
            output_name=output_name,
            profiles=profiles,
            input_dtypes=input_dtypes,
        )
    report["benchmarks"] = _run_benchmarks(
        config=config.get("benchmark", {}),
        engine_path=engine_path,
        output_name=output_name,
        profiles=profiles,
        input_dtypes=input_dtypes,
    )
    report["summary"] = _summarize(report)

    json_path = output_root / "generic_tensorrt_engine_report.json"
    markdown_path = output_root / "generic_tensorrt_engine_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    if args.output == "json":
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
    else:
        print(
            "\n".join(
                [
                    "Generic TensorRT engine workflow complete",
                    f"Status: {report['summary']['status']}",
                    f"Engine: {engine_path}",
                    f"Report JSON: {json_path}",
                    f"Report Markdown: {markdown_path}",
                ]
            )
        )


def _run_validation(
    *,
    config: dict[str, Any],
    onnx_path: Path,
    engine_path: Path,
    output_name: str,
    profiles: tuple[TensorRTMultiInputProfile, ...],
    input_dtypes: dict[str, np.dtype[Any]],
) -> list[dict[str, Any]]:
    import onnxruntime as ort
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT validation requires CUDA.")
    runner = MultiInputTensorRTEngineRunner(engine_path=engine_path, output_name=output_name)
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    session = ort.InferenceSession(onnx_path.as_posix(), providers=providers)
    rows = []
    for sample in _parse_samples(config.get("samples", []), profiles=profiles):
        np_inputs = _make_numpy_inputs(sample["shapes"], input_dtypes=input_dtypes)
        torch_inputs = {
            name: torch.from_numpy(value).to(device="cuda") for name, value in np_inputs.items()
        }
        profile = select_profile(profiles, shapes=sample["shapes"])
        profile_index = profiles.index(profile)
        ort_output = np.asarray(
            session.run([output_name], {name: value for name, value in np_inputs.items()})[0],
            dtype=np.float32,
        )
        with torch.inference_mode():
            trt_output = runner.run(torch_inputs, profile_index=profile_index)
        trt_np = trt_output.detach().cpu().float().numpy()
        diff = np.abs(ort_output - trt_np)
        max_abs_diff = float(diff.max()) if diff.size else 0.0
        mean_abs_diff = float(diff.mean()) if diff.size else 0.0
        cosine_distance = _cosine_distance(ort_output, trt_np)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "profile_id": profile.profile_id,
                "shapes": {name: list(shape) for name, shape in sample["shapes"].items()},
                "onnxruntime_provider": session.get_providers()[0],
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "cosine_distance": cosine_distance,
                "passed": (
                    mean_abs_diff <= float(config.get("max_mean_abs_diff", 0.05))
                    and cosine_distance <= float(config.get("max_cosine_distance", 0.01))
                ),
            }
        )
    return rows


def _run_benchmarks(
    *,
    config: dict[str, Any],
    engine_path: Path,
    output_name: str,
    profiles: tuple[TensorRTMultiInputProfile, ...],
    input_dtypes: dict[str, np.dtype[Any]],
) -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT benchmark requires CUDA.")
    runner = MultiInputTensorRTEngineRunner(engine_path=engine_path, output_name=output_name)
    rows = []
    for sample in _parse_samples(config.get("samples", []), profiles=profiles):
        np_inputs = _make_numpy_inputs(sample["shapes"], input_dtypes=input_dtypes)
        torch_inputs = {
            name: torch.from_numpy(value).to(device="cuda") for name, value in np_inputs.items()
        }
        profile = select_profile(profiles, shapes=sample["shapes"])
        profile_index = profiles.index(profile)
        latency_ms = benchmark_cuda_callable(
            torch=torch,
            function=lambda torch_inputs=torch_inputs, profile_index=profile_index: runner.run(
                torch_inputs,
                profile_index=profile_index,
            ),
            warmup_iterations=int(config.get("warmup_iterations", 10)),
            benchmark_iterations=int(config.get("benchmark_iterations", 50)),
        )
        batch_size = _infer_batch_size(sample["shapes"])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "profile_id": profile.profile_id,
                "shapes": {name: list(shape) for name, shape in sample["shapes"].items()},
                "latency_ms": latency_ms,
                "items_per_second": batch_size / (latency_ms / 1_000.0),
            }
        )
    return rows


def _parse_samples(
    raw_samples: Any,
    *,
    profiles: tuple[TensorRTMultiInputProfile, ...],
) -> list[dict[str, Any]]:
    if raw_samples:
        samples = []
        for index, sample in enumerate(raw_samples):
            if not isinstance(sample, dict):
                raise ValueError("Each sample must be a TOML table.")
            shapes_raw = sample.get("shapes")
            if not isinstance(shapes_raw, dict):
                raise ValueError("Each sample must define a shapes table.")
            samples.append(
                {
                    "sample_id": str(sample.get("sample_id", f"sample{index + 1}")),
                    "shapes": {
                        name: tuple(int(dim) for dim in shape) for name, shape in shapes_raw.items()
                    },
                }
            )
        return samples
    return [
        {
            "sample_id": profile.profile_id,
            "shapes": {
                name: input_profile.opt_shape for name, input_profile in profile.inputs.items()
            },
        }
        for profile in profiles
    ]


def _make_numpy_inputs(
    shapes: dict[str, tuple[int, ...]],
    *,
    input_dtypes: dict[str, np.dtype[Any]],
) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(20260416)
    for name, shape in shapes.items():
        dtype = input_dtypes.get(name, np.dtype("float32"))
        if np.issubdtype(dtype, np.floating):
            inputs[name] = rng.standard_normal(shape).astype(dtype, copy=False)
        elif np.issubdtype(dtype, np.integer):
            inputs[name] = np.ones(shape, dtype=dtype)
        elif dtype == np.dtype("bool"):
            inputs[name] = np.ones(shape, dtype=dtype)
        else:
            raise ValueError(f"Unsupported input dtype for {name}: {dtype}")
    return inputs


def _load_onnx_input_dtypes(path: Path) -> dict[str, np.dtype[Any]]:
    import onnx

    model = onnx.load(path)
    initializer_names = {item.name for item in model.graph.initializer}
    dtypes: dict[str, np.dtype[Any]] = {}
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        elem_type = value.type.tensor_type.elem_type
        dtypes[value.name] = _numpy_dtype_from_onnx(onnx, elem_type)
    return dtypes


def _numpy_dtype_from_onnx(onnx: Any, elem_type: int) -> np.dtype[Any]:
    mapping = {
        onnx.TensorProto.FLOAT: np.dtype("float32"),
        onnx.TensorProto.FLOAT16: np.dtype("float16"),
        onnx.TensorProto.BFLOAT16: np.dtype("float32"),
        onnx.TensorProto.DOUBLE: np.dtype("float64"),
        onnx.TensorProto.INT64: np.dtype("int64"),
        onnx.TensorProto.INT32: np.dtype("int32"),
        onnx.TensorProto.INT8: np.dtype("int8"),
        onnx.TensorProto.UINT8: np.dtype("uint8"),
        onnx.TensorProto.BOOL: np.dtype("bool"),
    }
    if elem_type not in mapping:
        raise ValueError(f"Unsupported ONNX input elem_type={elem_type}")
    return mapping[elem_type]


def _parse_profiles(config: dict[str, Any]) -> tuple[TensorRTMultiInputProfile, ...]:
    profiles = []
    raw_profiles = config.get("profiles", [])
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("TensorRT config must define at least one [[profiles]] table.")
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise ValueError("Each TensorRT profile must be a table.")
        raw_inputs = raw_profile.get("inputs")
        if not isinstance(raw_inputs, dict) or not raw_inputs:
            raise ValueError("Each TensorRT profile must define [profiles.inputs.<name>] tables.")
        inputs = {}
        for name, values in raw_inputs.items():
            if not isinstance(values, dict):
                raise ValueError(f"Profile input {name!r} must be a table.")
            inputs[name] = TensorRTInputProfile(
                min_shape=_parse_shape(values.get("min"), field_name=f"{name}.min"),
                opt_shape=_parse_shape(values.get("opt"), field_name=f"{name}.opt"),
                max_shape=_parse_shape(values.get("max"), field_name=f"{name}.max"),
            )
        profiles.append(
            TensorRTMultiInputProfile(
                profile_id=str(raw_profile.get("profile_id", f"profile{len(profiles) + 1}")),
                inputs=inputs,
            )
        )
    return tuple(profiles)


def _profile_to_dict(profile: TensorRTMultiInputProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "inputs": {
            name: {
                "min": list(input_profile.min_shape),
                "opt": list(input_profile.opt_shape),
                "max": list(input_profile.max_shape),
            }
            for name, input_profile in profile.inputs.items()
        },
    }


def _parse_shape(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty integer list.")
    return tuple(int(item) for item in value)


def _infer_batch_size(shapes: dict[str, tuple[int, ...]]) -> int:
    first_shape = next(iter(shapes.values()))
    return int(first_shape[0]) if first_shape else 1


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    validation_rows = list(report["validation"])
    benchmark_rows = list(report["benchmarks"])
    validation_passed = all(bool(row["passed"]) for row in validation_rows)
    max_items_per_second = (
        max(float(row["items_per_second"]) for row in benchmark_rows)
        if benchmark_rows
        else math.nan
    )
    return {
        "status": "pass" if validation_passed else "fail",
        "validation_count": len(validation_rows),
        "validation_passed_count": sum(1 for row in validation_rows if bool(row["passed"])),
        "benchmark_count": len(benchmark_rows),
        "max_items_per_second": max_items_per_second,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['title']}",
        "",
        f"- ONNX: `{report['onnx_path']}`",
        f"- TensorRT engine: `{report['engine_path']}`",
        f"- FP16: `{str(report['fp16']).lower()}`",
        f"- Workspace MiB: `{report['workspace_size_mib']}`",
        f"- Builder optimization level: `{report['builder_optimization_level']}`",
        f"- Status: `{report['summary']['status']}`",
        "",
        "## Validation",
        "",
        "| Sample | Profile | mean abs diff | cosine distance | Passed |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in report["validation"]:
        lines.append(
            f"| `{row['sample_id']}` | `{row['profile_id']}` | "
            f"{row['mean_abs_diff']:.8f} | {row['cosine_distance']:.8f} | "
            f"`{str(row['passed']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## Benchmarks",
            "",
            "| Sample | Profile | Latency ms | Items/s |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in report["benchmarks"]:
        lines.append(
            f"| `{row['sample_id']}` | `{row['profile_id']}` | "
            f"{row['latency_ms']:.6f} | {row['items_per_second']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left.reshape(-1).astype(np.float64, copy=False)
    right_vector = right.reshape(-1).astype(np.float64, copy=False)
    denominator = (np.linalg.norm(left_vector) * np.linalg.norm(right_vector)) + 1e-12
    cosine_similarity = float(np.dot(left_vector, right_vector) / denominator)
    return float(1.0 - cosine_similarity)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must contain a TOML object.")
    return payload


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to generic TensorRT TOML config.")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser.parse_args()


if __name__ == "__main__":
    main()
