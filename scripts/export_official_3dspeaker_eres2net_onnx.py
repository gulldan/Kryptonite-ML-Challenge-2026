"""Export an official 3D-Speaker ERes2Net checkpoint to ONNX."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    torch = _import_torch()
    onnx = _import_onnx()
    ort = _import_onnxruntime()
    checkpoint_path, model = _load_official_eres2net_encoder(
        torch=torch,
        checkpoint_path=Path(args.checkpoint_path),
        speakerlab_root=Path(args.speakerlab_root),
        feat_dim=args.feat_dim,
        embedding_size=args.embedding_size,
        m_channels=args.m_channels,
    )
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    sample_input = torch.randn(
        args.sample_batch_size,
        args.sample_frame_count,
        args.feat_dim,
        generator=generator,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        reference = model(sample_input).detach().cpu().numpy()

    model_path = output_root / "model.onnx"
    dynamic_axes: dict[str, dict[int, str]] = {
        args.input_name: {0: "batch"},
        args.output_name: {0: "batch"},
    }
    if args.dynamic_frames:
        dynamic_axes[args.input_name][1] = "frames"
    _export_onnx(
        torch=torch,
        model=model,
        sample_input=sample_input,
        model_path=model_path,
        input_name=args.input_name,
        output_name=args.output_name,
        opset=args.opset,
        dynamic_axes=dynamic_axes,
        external_data=args.external_data,
    )
    onnx.checker.check_model(model_path.as_posix())
    validation = _run_onnxruntime_smoke(
        ort=ort,
        model_path=model_path,
        input_name=args.input_name,
        output_name=args.output_name,
        sample_input=sample_input.detach().cpu().numpy(),
        reference=reference,
    )
    model_version = (
        args.model_version or f"official-3dspeaker-eres2net-onnx-{checkpoint_path.parent.name}"
    )
    payload = {
        "status": "pass",
        "model_family": "official_3dspeaker_eres2net",
        "model_version": model_version,
        "source_checkpoint_path": str(checkpoint_path),
        "speakerlab_root": str(Path(args.speakerlab_root)),
        "model_path": str(model_path),
        "input_name": args.input_name,
        "output_name": args.output_name,
        "sample_input_shape": list(sample_input.shape),
        "output_shape": list(reference.shape),
        "opset": args.opset,
        "dynamic_axes": dynamic_axes,
        "external_data": bool(args.external_data),
        "source_model_config": {
            "feat_dim": args.feat_dim,
            "embedding_size": args.embedding_size,
            "m_channels": args.m_channels,
        },
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
                    "Official ERes2Net ONNX export: PASS",
                    f"Checkpoint: {checkpoint_path}",
                    f"ONNX: {model_path}",
                    f"Mean abs diff: {validation['mean_abs_diff']:.8f}",
                ]
            )
        )


def _load_official_eres2net_encoder(
    *,
    torch: Any,
    checkpoint_path: Path,
    speakerlab_root: Path,
    feat_dim: int,
    embedding_size: int,
    m_channels: int,
) -> tuple[Path, Any]:
    if str(speakerlab_root) not in sys.path:
        sys.path.insert(0, str(speakerlab_root))
    eres_module = importlib.import_module("speakerlab.models.eres2net.ERes2Net")
    model = eres_module.ERes2Net(
        feat_dim=feat_dim,
        embedding_size=embedding_size,
        m_channels=m_channels,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_model_state_dict(payload)
    model.load_state_dict(state_dict)
    return checkpoint_path, model


def _extract_model_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict payload, got {type(state_dict)!r}.")
    return state_dict


def _export_onnx(
    *,
    torch: Any,
    model: Any,
    sample_input: Any,
    model_path: Path,
    input_name: str,
    output_name: str,
    opset: int,
    dynamic_axes: dict[str, dict[int, str]],
    external_data: bool,
) -> None:
    kwargs = {
        "input_names": [input_name],
        "output_names": [output_name],
        "opset_version": opset,
        "dynamic_axes": dynamic_axes,
        "do_constant_folding": True,
        "dynamo": False,
    }
    if external_data:
        kwargs["external_data"] = True
    try:
        torch.onnx.export(model, (sample_input,), model_path.as_posix(), **kwargs)
    except TypeError as exc:
        if "external_data" not in str(exc) and "dynamo" not in str(exc):
            raise
        kwargs.pop("external_data", None)
        kwargs.pop("dynamo", None)
        torch.onnx.export(model, (sample_input,), model_path.as_posix(), **kwargs)


def _run_onnxruntime_smoke(
    *,
    ort: Any,
    model_path: Path,
    input_name: str,
    output_name: str,
    sample_input: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    output = np.asarray(
        session.run([output_name], {input_name: sample_input.astype(np.float32, copy=False)})[0],
        dtype=np.float32,
    )
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
    return "\n".join(
        [
            "# Official 3D-Speaker ERes2Net ONNX Export",
            "",
            f"- status: `{payload['status']}`",
            f"- model version: `{payload['model_version']}`",
            f"- checkpoint: `{payload['source_checkpoint_path']}`",
            f"- speakerlab root: `{payload['speakerlab_root']}`",
            f"- ONNX: `{payload['model_path']}`",
            f"- input: `{payload['input_name']}`",
            f"- output: `{payload['output_name']}`",
            f"- sample input shape: `{payload['sample_input_shape']}`",
            "",
            "## Validation",
            "",
            f"- ONNX Runtime provider: `{validation['onnxruntime_provider']}`",
            f"- max abs diff: `{validation['max_abs_diff']:.8f}`",
            f"- mean abs diff: `{validation['mean_abs_diff']:.8f}`",
            f"- cosine distance: `{validation['cosine_distance']:.8f}`",
            f"- passed: `{str(validation['passed']).lower()}`",
            "",
        ]
    )


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
    parser.add_argument("--speakerlab-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--input-name", default="features")
    parser.add_argument("--output-name", default="embedding")
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--sample-frame-count", type=int, default=600)
    parser.add_argument("--feat-dim", type=int, default=80)
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--m-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--dynamic-frames", action="store_true", default=True)
    parser.add_argument("--static-frames", dest="dynamic_frames", action="store_false")
    parser.add_argument("--external-data", dest="external_data", action="store_true", default=True)
    parser.add_argument("--no-external-data", dest="external_data", action="store_false")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser.parse_args()


if __name__ == "__main__":
    main()
