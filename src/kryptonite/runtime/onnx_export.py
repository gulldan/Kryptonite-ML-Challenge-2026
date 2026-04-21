"""Export CAM++ checkpoints to encoder-only ONNX model bundles."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.config import ProjectConfig
from kryptonite.deployment import resolve_project_path
from kryptonite.features import SUPPORTED_CHUNKING_STAGES
from kryptonite.models import CAMPPlusConfig
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.runtime.export_boundary import (
    ExportBoundaryContract,
    build_export_boundary_contract,
    render_export_boundary_markdown,
)
from kryptonite.runtime.inference_package import build_inference_package_contract
from kryptonite.tracking import utc_now

DEFAULT_CAMPP_ONNX_BUNDLE_ROOT = "artifacts/model-bundle-campp-onnx"
EXPORT_BOUNDARY_JSON_NAME = "export_boundary.json"
EXPORT_REPORT_JSON_NAME = "export_report.json"
EXPORT_REPORT_MARKDOWN_NAME = "export_report.md"


@dataclass(frozen=True, slots=True)
class CAMPPONNXExportRequest:
    checkpoint_path: str
    output_root: str = DEFAULT_CAMPP_ONNX_BUNDLE_ROOT
    model_version: str | None = None
    sample_batch_size: int = 1
    sample_frame_count: int = 200
    embedding_stage: str = "eval"

    def __post_init__(self) -> None:
        if not self.checkpoint_path.strip():
            raise ValueError("checkpoint_path must be a non-empty string.")
        if not self.output_root.strip():
            raise ValueError("output_root must be a non-empty string.")
        if self.sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive.")
        if self.sample_frame_count <= 0:
            raise ValueError("sample_frame_count must be positive.")
        if self.embedding_stage.lower() not in SUPPORTED_CHUNKING_STAGES:
            raise ValueError(
                "embedding_stage must be one of "
                f"{sorted(SUPPORTED_CHUNKING_STAGES)}, got {self.embedding_stage!r}."
            )


@dataclass(frozen=True, slots=True)
class ONNXSmokeValidation:
    checker_passed: bool
    onnxruntime_smoke_passed: bool
    sample_input_shape: tuple[int, int, int]
    sample_output_shape: tuple[int, int]
    max_abs_diff: float
    mean_abs_diff: float

    def to_dict(self) -> dict[str, object]:
        return {
            "checker_passed": self.checker_passed,
            "onnxruntime_smoke_passed": self.onnxruntime_smoke_passed,
            "sample_input_shape": list(self.sample_input_shape),
            "sample_output_shape": list(self.sample_output_shape),
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
        }


@dataclass(frozen=True, slots=True)
class ExportedCAMPPONNXBundle:
    output_root: str
    source_checkpoint_path: str
    model_path: str
    metadata_path: str
    export_boundary_path: str
    report_json_path: str
    report_markdown_path: str
    model_version: str
    input_name: str
    output_name: str
    embedding_dim: int
    validation: ONNXSmokeValidation

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "source_checkpoint_path": self.source_checkpoint_path,
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "export_boundary_path": self.export_boundary_path,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "model_version": self.model_version,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "embedding_dim": self.embedding_dim,
            "validation": self.validation.to_dict(),
        }


def export_campp_checkpoint_to_onnx(
    *,
    config: ProjectConfig,
    request: CAMPPONNXExportRequest,
) -> ExportedCAMPPONNXBundle:
    torch = _import_torch()
    onnx = _import_onnx()
    onnxruntime = _import_onnxruntime()

    project_root = resolve_project_path(config.paths.project_root, ".")
    checkpoint_path, model_config, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=request.checkpoint_path,
        project_root=project_root,
    )
    output_root = resolve_project_path(str(project_root), request.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.runtime.seed)
    example_input = torch.randn(
        request.sample_batch_size,
        request.sample_frame_count,
        model_config.feat_dim,
        generator=generator,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        reference_output = model(example_input).detach().cpu().numpy()

    model_path = output_root / "model.onnx"
    torch.onnx.export(
        model,
        (example_input,),
        model_path.as_posix(),
        input_names=[config.export.input_name],
        output_names=[config.export.output_name],
        opset_version=config.export.opset,
        dynamo=True,
        dynamic_shapes=_build_dynamic_shapes(
            torch=torch,
            config=config,
            request=request,
        ),
    )

    onnx.checker.check_model(onnx.load(model_path))
    validation = _run_onnxruntime_smoke(
        onnxruntime=onnxruntime,
        model_path=model_path,
        input_name=config.export.input_name,
        output_name=config.export.output_name,
        example_input=example_input.detach().cpu().numpy(),
        reference_output=reference_output,
    )

    contract = build_export_boundary_contract(
        config=config,
        inferencer_backend="campp_encoder",
        embedding_stage=request.embedding_stage.lower(),
        embedding_mode=None,
        embedding_dim=model_config.embedding_size,
    )
    model_version = request.model_version or _default_model_version(checkpoint_path)
    relative_model_path = _relative_to_project(model_path, project_root)
    relative_checkpoint_path = _relative_to_project(checkpoint_path, project_root)

    metadata = _build_model_bundle_metadata(
        config=config,
        contract=contract,
        relative_model_path=relative_model_path,
        relative_checkpoint_path=relative_checkpoint_path,
        model_config=model_config,
        model_version=model_version,
        validation=validation,
        exported_with={
            "torch": str(torch.__version__),
            "onnx": str(onnx.__version__),
            "onnxruntime": str(onnxruntime.__version__),
        },
    )

    metadata_path = output_root / "metadata.json"
    export_boundary_path = output_root / EXPORT_BOUNDARY_JSON_NAME
    report_json_path = output_root / EXPORT_REPORT_JSON_NAME
    report_markdown_path = output_root / EXPORT_REPORT_MARKDOWN_NAME

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    export_boundary_path.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bundle = ExportedCAMPPONNXBundle(
        output_root=_relative_to_project(output_root, project_root),
        source_checkpoint_path=relative_checkpoint_path,
        model_path=relative_model_path,
        metadata_path=_relative_to_project(metadata_path, project_root),
        export_boundary_path=_relative_to_project(export_boundary_path, project_root),
        report_json_path=_relative_to_project(report_json_path, project_root),
        report_markdown_path=_relative_to_project(report_markdown_path, project_root),
        model_version=model_version,
        input_name=contract.input_tensor.name,
        output_name=contract.output_tensor.name,
        embedding_dim=model_config.embedding_size,
        validation=validation,
    )
    report_json_path.write_text(
        json.dumps(
            _build_export_report_payload(
                bundle=bundle,
                config=config,
                contract=contract,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_campp_onnx_export_markdown(
            bundle=bundle,
            contract=contract,
            config=config,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle


def render_campp_onnx_export_markdown(
    *,
    bundle: ExportedCAMPPONNXBundle,
    contract: ExportBoundaryContract,
    config: ProjectConfig,
) -> str:
    validation = bundle.validation
    return "\n".join(
        [
            "# CAM++ ONNX Export",
            "",
            "## Outcome",
            "",
            "- status: `pass`",
            f"- model version: `{bundle.model_version}`",
            f"- source checkpoint: `{bundle.source_checkpoint_path}`",
            f"- output root: `{bundle.output_root}`",
            f"- ONNX model: `{bundle.model_path}`",
            f"- metadata: `{bundle.metadata_path}`",
            f"- export boundary: `{bundle.export_boundary_path}`",
            "",
            "## Export Contract",
            "",
            f"- opset: `{config.export.opset}`",
            f"- input name: `{bundle.input_name}`",
            f"- output name: `{bundle.output_name}`",
            f"- dynamic time axis: `{str(contract.dynamic_time_axis).lower()}`",
            f"- output embedding dim: `{bundle.embedding_dim}`",
            "",
            "## Validation",
            "",
            f"- ONNX checker: `{str(validation.checker_passed).lower()}`",
            "- ONNX Runtime single-sample smoke: "
            f"`{str(validation.onnxruntime_smoke_passed).lower()}`",
            f"- sample input shape: `{list(validation.sample_input_shape)}`",
            f"- sample output shape: `{list(validation.sample_output_shape)}`",
            f"- max abs diff: `{validation.max_abs_diff:.8f}`",
            f"- mean abs diff: `{validation.mean_abs_diff:.8f}`",
            "",
            "## Limits",
            "",
            "- This bundle validates graph materialization and one encoder-level smoke input only.",
            "- Raw-audio decode, VAD, chunking, and Fbank extraction remain outside the graph.",
            "- Runtime backend promotion stays blocked until the broader parity work is complete.",
            "",
            "## Boundary Summary",
            "",
            render_export_boundary_markdown(contract),
        ]
    )


def _build_model_bundle_metadata(
    *,
    config: ProjectConfig,
    contract: ExportBoundaryContract,
    relative_model_path: str,
    relative_checkpoint_path: str,
    model_config: CAMPPlusConfig,
    model_version: str,
    validation: ONNXSmokeValidation,
    exported_with: dict[str, str],
) -> dict[str, object]:
    return {
        "model_file": relative_model_path,
        "model_version": model_version,
        "input_name": contract.input_tensor.name,
        "output_name": contract.output_tensor.name,
        "inferencer_backend": "campp_encoder",
        "embedding_stage": contract.embedding_stage,
        "embedding_mode": contract.embedding_mode,
        "enrollment_cache_compatibility_id": f"{model_version}-encoder-only-cache-v1",
        "description": (
            "Exported CAM++ encoder-only ONNX bundle. The raw-audio frontend remains runtime-owned "
            "and runtime backends stay unvalidated here until the dedicated parity task lands."
        ),
        "export_boundary": contract.to_dict(),
        "inference_package": build_inference_package_contract(
            onnx_model_file=relative_model_path,
            validated_backends={
                "torch": False,
                "onnxruntime": False,
                "tensorrt": False,
            },
        ).to_dict(),
        "source_checkpoint_path": relative_checkpoint_path,
        "source_model_family": "campp",
        "source_model_config": asdict(model_config),
        "export_validation": {
            **validation.to_dict(),
            "validated_at_utc": utc_now(),
            "runtime_backends_promoted": False,
            "runtime_backends_promotion_blocker": "Awaiting broader ONNX Runtime parity coverage.",
        },
        "exported_with": exported_with,
        "export_profile": config.export.profile,
    }


def _build_export_report_payload(
    *,
    bundle: ExportedCAMPPONNXBundle,
    config: ProjectConfig,
    contract: ExportBoundaryContract,
) -> dict[str, object]:
    return {
        "status": "pass",
        "generated_at_utc": utc_now(),
        "bundle": bundle.to_dict(),
        "export": {
            "opset": config.export.opset,
            "dynamic_axes": config.export.dynamic_axes,
            "profile": config.export.profile,
        },
        "contract": contract.summary_dict(),
        "limits": [
            "graph_materialization_only",
            "single_sample_onnxruntime_smoke_only",
            "runtime_parity_not_promoted",
        ],
    }


def _default_model_version(checkpoint_path: Path) -> str:
    run_name = checkpoint_path.parent.name.strip() or checkpoint_path.stem
    return f"campp-onnx-{run_name}"


def _build_dynamic_shapes(
    *,
    torch: Any,
    config: ProjectConfig,
    request: CAMPPONNXExportRequest,
) -> dict[str, dict[int, object]] | None:
    if not config.export.dynamic_axes:
        return None
    max_frame_seconds = max(
        config.chunking.eval_max_full_utterance_seconds,
        config.chunking.eval_chunk_seconds,
        config.chunking.demo_max_full_utterance_seconds,
        config.chunking.demo_chunk_seconds,
    )
    frame_shift_ms = max(config.features.frame_shift_ms, 1e-6)
    estimated_max_frames = max(
        request.sample_frame_count,
        int(math.ceil((max_frame_seconds * 1_000.0) / frame_shift_ms)),
    )
    return {
        "features": {
            0: torch.export.Dim("batch", min=1, max=max(8, request.sample_batch_size)),
            1: torch.export.Dim("frames", min=1, max=max(estimated_max_frames, 100)),
        }
    }


def _run_onnxruntime_smoke(
    *,
    onnxruntime: Any,
    model_path: Path,
    input_name: str,
    output_name: str,
    example_input: np.ndarray,
    reference_output: np.ndarray,
) -> ONNXSmokeValidation:
    # The export smoke is only meant to catch obviously broken graphs. The
    # broader numeric gate now lives in the dedicated ONNX parity suite, so keep
    # this assertion tolerant enough to avoid false failures on valid CPU EP
    # exports while still rejecting large regressions.
    smoke_rtol = 1e-3
    smoke_atol = 1e-3
    session = onnxruntime.InferenceSession(
        model_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    session_inputs = tuple(item.name for item in session.get_inputs())
    session_outputs = tuple(item.name for item in session.get_outputs())
    if input_name not in session_inputs:
        raise ValueError(
            f"Exported ONNX graph does not expose input {input_name!r}; got {session_inputs}."
        )
    if output_name not in session_outputs:
        raise ValueError(
            f"Exported ONNX graph does not expose output {output_name!r}; got {session_outputs}."
        )
    output = np.asarray(
        session.run([output_name], {input_name: example_input.astype(np.float32, copy=False)})[0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(output, reference_output, rtol=smoke_rtol, atol=smoke_atol)
    absolute_diff = np.abs(output - reference_output)
    max_abs_diff = float(absolute_diff.max()) if absolute_diff.size else 0.0
    mean_abs_diff = float(absolute_diff.mean()) if absolute_diff.size else 0.0
    return ONNXSmokeValidation(
        checker_passed=True,
        onnxruntime_smoke_passed=True,
        sample_input_shape=(
            int(example_input.shape[0]),
            int(example_input.shape[1]),
            int(example_input.shape[2]),
        ),
        sample_output_shape=(int(output.shape[0]), int(output.shape[1])),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
    )


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _import_onnx() -> Any:
    import onnx

    return onnx


def _import_onnxruntime() -> Any:
    import onnxruntime

    return onnxruntime


def _import_torch() -> Any:
    import torch

    return torch


__all__ = [
    "DEFAULT_CAMPP_ONNX_BUNDLE_ROOT",
    "EXPORT_BOUNDARY_JSON_NAME",
    "EXPORT_REPORT_JSON_NAME",
    "EXPORT_REPORT_MARKDOWN_NAME",
    "CAMPPONNXExportRequest",
    "ONNXSmokeValidation",
    "ExportedCAMPPONNXBundle",
    "export_campp_checkpoint_to_onnx",
    "render_campp_onnx_export_markdown",
]
