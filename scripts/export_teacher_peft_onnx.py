"""Export a Teacher-PEFT speaker encoder checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.training.teacher_peft import (
    load_teacher_peft_encoder_from_checkpoint,
    merge_teacher_lora_backbone,
)


class _TeacherPeftONNXWrapper:
    def __init__(self, *, torch: Any, encoder: Any, input_names: Sequence[str]) -> None:
        self._torch = torch
        self._encoder = encoder
        self._input_names = tuple(input_names)

    def module(self) -> Any:
        torch = self._torch
        encoder = self._encoder
        input_names = self._input_names

        class ExportableTeacherPeft(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder

            def forward(self, *values: Any) -> Any:
                model_inputs = {
                    name: value for name, value in zip(input_names, values, strict=True)
                }
                return self.encoder(**model_inputs)

        return ExportableTeacherPeft().eval()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    torch = _import_torch()
    onnx = _import_onnx()
    ort = _import_onnxruntime()

    token = os.environ.get(args.hf_token_env) or None
    checkpoint_dir, metadata, feature_extractor, encoder = (
        load_teacher_peft_encoder_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            token=token,
            trainable=False,
        )
    )
    if args.merge_lora:
        encoder = merge_teacher_lora_backbone(encoder)
    encoder.eval()
    sample_inputs_dict = _build_sample_inputs(
        feature_extractor=feature_extractor,
        sample_batch_size=args.sample_batch_size,
        sample_seconds=args.sample_seconds,
        sample_rate_hz=args.sample_rate_hz,
    )
    input_names = tuple(sample_inputs_dict)
    sample_inputs = tuple(sample_inputs_dict[name] for name in input_names)
    wrapper = _TeacherPeftONNXWrapper(
        torch=torch,
        encoder=encoder,
        input_names=input_names,
    ).module()
    with torch.inference_mode():
        reference = wrapper(*sample_inputs).detach().cpu().numpy()

    model_path = output_root / "model.onnx"
    dynamic_axes: dict[str, dict[int, str]] = {name: {0: "batch"} for name in input_names}
    dynamic_axes[args.output_name] = {0: "batch"}
    if args.dynamic_frames:
        for name, tensor in sample_inputs_dict.items():
            if tensor.ndim >= 2:
                dynamic_axes[name][1] = "frames"

    _export_onnx(
        torch=torch,
        model=wrapper,
        inputs=sample_inputs,
        model_path=model_path,
        input_names=list(input_names),
        output_name=args.output_name,
        opset=args.opset,
        dynamic_axes=dynamic_axes,
        external_data=args.external_data,
    )
    onnx.checker.check_model(model_path.as_posix())
    validation = _run_onnxruntime_smoke(
        ort=ort,
        model_path=model_path,
        input_names=input_names,
        output_name=args.output_name,
        sample_inputs=sample_inputs_dict,
        reference=reference,
    )
    model_config = metadata.get("model", {}) if isinstance(metadata, dict) else {}
    model_version = args.model_version or f"teacher-peft-onnx-{checkpoint_dir.parent.name}"
    payload = {
        "status": "pass",
        "model_family": "w2vbert2_teacher_peft",
        "model_version": model_version,
        "source_checkpoint_path": str(checkpoint_dir),
        "model_path": str(model_path),
        "input_names": list(input_names),
        "output_name": args.output_name,
        "sample_input_shapes": {
            name: list(tensor.shape) for name, tensor in sample_inputs_dict.items()
        },
        "output_shape": list(reference.shape),
        "opset": args.opset,
        "dynamic_axes": dynamic_axes,
        "external_data": bool(args.external_data),
        "merge_lora": bool(args.merge_lora),
        "source_model_config": model_config,
        "validation": validation,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "export_report.md").write_text(_render_report(payload), encoding="utf-8")
    if args.output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "\n".join(
                [
                    "Teacher-PEFT ONNX export: PASS",
                    f"Checkpoint: {checkpoint_dir}",
                    f"ONNX: {model_path}",
                    f"Inputs: {', '.join(input_names)}",
                    f"Mean abs diff: {validation['mean_abs_diff']:.8f}",
                ]
            )
        )


def _build_sample_inputs(
    *,
    feature_extractor: Any,
    sample_batch_size: int,
    sample_seconds: float,
    sample_rate_hz: int,
) -> dict[str, Any]:
    import torch

    sample_count = int(round(sample_seconds * sample_rate_hz))
    waveforms = [np.zeros(sample_count, dtype=np.float32) for _ in range(sample_batch_size)]
    encoded = feature_extractor(
        waveforms,
        sampling_rate=sample_rate_hz,
        padding=True,
        return_tensors="pt",
    )
    inputs = {}
    for key, value in encoded.items():
        if key == "attention_mask":
            inputs[key] = value.to(dtype=torch.int32)
        else:
            inputs[key] = value.to(dtype=torch.float32)
    return inputs


def _export_onnx(
    *,
    torch: Any,
    model: Any,
    inputs: tuple[Any, ...],
    model_path: Path,
    input_names: list[str],
    output_name: str,
    opset: int,
    dynamic_axes: dict[str, dict[int, str]],
    external_data: bool,
) -> None:
    kwargs = {
        "input_names": input_names,
        "output_names": [output_name],
        "opset_version": opset,
        "dynamic_axes": dynamic_axes,
        "do_constant_folding": True,
        "dynamo": False,
    }
    if external_data:
        kwargs["external_data"] = True
    try:
        torch.onnx.export(model, inputs, model_path.as_posix(), **kwargs)
    except TypeError as exc:
        if "external_data" not in str(exc) and "dynamo" not in str(exc):
            raise
        kwargs.pop("external_data", None)
        kwargs.pop("dynamo", None)
        torch.onnx.export(model, inputs, model_path.as_posix(), **kwargs)


def _run_onnxruntime_smoke(
    *,
    ort: Any,
    model_path: Path,
    input_names: Sequence[str],
    output_name: str,
    sample_inputs: dict[str, Any],
    reference: np.ndarray,
) -> dict[str, Any]:
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path.as_posix(), providers=providers)
    feeds = {name: sample_inputs[name].detach().cpu().numpy() for name in input_names}
    output = np.asarray(session.run([output_name], feeds)[0], dtype=np.float32)
    diff = np.abs(output - reference.astype(np.float32, copy=False))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    cosine_distance = _cosine_distance(output, reference)
    return {
        "onnxruntime_provider": session.get_providers()[0],
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_distance": cosine_distance,
        "passed": bool(mean_abs_diff <= 1e-3 and cosine_distance <= 1e-4),
    }


def _render_report(payload: dict[str, Any]) -> str:
    validation = payload["validation"]
    lines = [
        "# Teacher-PEFT ONNX Export",
        "",
        f"- status: `{payload['status']}`",
        f"- model version: `{payload['model_version']}`",
        f"- checkpoint: `{payload['source_checkpoint_path']}`",
        f"- ONNX: `{payload['model_path']}`",
        f"- inputs: `{', '.join(payload['input_names'])}`",
        f"- output: `{payload['output_name']}`",
        f"- sample shapes: `{payload['sample_input_shapes']}`",
        "",
        "## Validation",
        "",
        f"- ONNX Runtime provider: `{validation['onnxruntime_provider']}`",
        f"- max abs diff: `{validation['max_abs_diff']:.8f}`",
        f"- mean abs diff: `{validation['mean_abs_diff']:.8f}`",
        f"- cosine distance: `{validation['cosine_distance']:.8f}`",
        f"- passed: `{str(validation['passed']).lower()}`",
    ]
    return "\n".join(lines) + "\n"


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left.reshape(-1).astype(np.float64, copy=False)
    right_vector = right.reshape(-1).astype(np.float64, copy=False)
    denominator = (np.linalg.norm(left_vector) * np.linalg.norm(right_vector)) + 1e-12
    cosine_similarity = float(np.dot(left_vector, right_vector) / denominator)
    return float(1.0 - cosine_similarity)


def _import_torch() -> Any:
    import torch

    return torch


def _import_onnx() -> Any:
    import onnx

    return onnx


def _import_onnxruntime() -> Any:
    import onnxruntime

    return onnxruntime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--hf-token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--output-name", default="embedding")
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--sample-seconds", type=float, default=6.0)
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--dynamic-frames", action="store_true")
    parser.add_argument("--external-data", dest="external_data", action="store_true", default=True)
    parser.add_argument("--no-external-data", dest="external_data", action="store_false")
    parser.add_argument("--merge-lora", dest="merge_lora", action="store_true", default=True)
    parser.add_argument("--no-merge-lora", dest="merge_lora", action="store_false")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser.parse_args()


if __name__ == "__main__":
    main()
