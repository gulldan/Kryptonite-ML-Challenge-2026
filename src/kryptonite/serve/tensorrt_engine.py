"""TensorRT FP16 engine orchestration and report writing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from kryptonite.deployment import resolve_project_path
from kryptonite.serve.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.serve.inference_package import load_inference_package_from_model_metadata
from kryptonite.tracking import utc_now

from .tensorrt_engine_config import TensorRTFP16Config
from .tensorrt_engine_models import (
    TENSORRT_FP16_REPORT_JSON_NAME,
    TENSORRT_FP16_REPORT_MARKDOWN_NAME,
    TensorRTFP16Profile,
    TensorRTFP16PromotionState,
    TensorRTFP16Report,
    TensorRTFP16SampleResult,
    TensorRTFP16Summary,
    WrittenTensorRTFP16Report,
)
from .tensorrt_engine_runtime import (
    build_serialized_tensorrt_engine as _build_serialized_tensorrt_engine,
)
from .tensorrt_engine_runtime import validate_tensorrt_engine as _validate_tensorrt_engine


def build_tensorrt_fp16_report(
    config: TensorRTFP16Config,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> TensorRTFP16Report:
    resolved_project_root = _resolve_project_root(project_root, config.project_root)
    metadata_path = resolve_project_path(
        str(resolved_project_root),
        config.artifacts.model_bundle_metadata_path,
    )
    metadata = _load_json_object(metadata_path)
    inference_package = load_inference_package_from_model_metadata(metadata)
    if config.build.require_onnxruntime_parity and not inference_package.backend_validated(
        "onnxruntime"
    ):
        raise RuntimeError(
            "TensorRT FP16 build requires a parity-promoted ONNX bundle. "
            "Run the ONNX Runtime parity workflow first so "
            "`inference_package.validated_backends.onnxruntime=true`."
        )

    onnx_model_rel = inference_package.artifacts.onnx_model_file or _coerce_optional_string(
        metadata.get("model_file")
    )
    if onnx_model_rel is None:
        raise ValueError("Model metadata does not define an ONNX model artifact.")

    source_checkpoint_rel = _coerce_optional_string(metadata.get("source_checkpoint_path"))
    if source_checkpoint_rel is None:
        raise ValueError("Model metadata does not define source_checkpoint_path.")

    onnx_model_path = resolve_project_path(str(resolved_project_root), onnx_model_rel)
    source_checkpoint_path = resolve_project_path(
        str(resolved_project_root),
        source_checkpoint_rel,
    )
    engine_path = resolve_project_path(
        str(resolved_project_root),
        config.artifacts.engine_output_path,
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    contract = load_export_boundary_from_model_metadata(metadata)
    input_name = contract.input_tensor.name
    output_name = contract.output_tensor.name
    mel_bins = _require_static_axis_size(contract.input_tensor.axes[-1].size, axis_name="mel_bins")
    embedding_dim = _require_static_axis_size(
        contract.output_tensor.axes[-1].size,
        axis_name="embedding_dim",
    )
    profiles = tuple(
        TensorRTFP16Profile(
            profile_id=profile_config.profile_id,
            min_shape=(
                profile_config.min_batch_size,
                profile_config.min_frame_count,
                mel_bins,
            ),
            opt_shape=(
                profile_config.opt_batch_size,
                profile_config.opt_frame_count,
                mel_bins,
            ),
            max_shape=(
                profile_config.max_batch_size,
                profile_config.max_frame_count,
                mel_bins,
            ),
        )
        for profile_config in config.build.profiles
    )

    engine_bytes = _build_serialized_tensorrt_engine(
        onnx_model_path=onnx_model_path,
        input_name=input_name,
        profiles=profiles,
        workspace_size_mib=config.build.workspace_size_mib,
    )
    engine_path.write_bytes(engine_bytes)
    if engine_path.stat().st_size <= 0:
        raise RuntimeError(f"TensorRT engine write produced an empty file: {engine_path}")

    sample_results = _validate_tensorrt_engine(
        engine_path=engine_path,
        source_checkpoint_path=source_checkpoint_path,
        project_root=resolved_project_root,
        input_name=input_name,
        output_name=output_name,
        feature_dim=mel_bins,
        embedding_dim=embedding_dim,
        profiles=profiles,
        samples=config.evaluation.samples,
        seed=config.evaluation.seed,
        warmup_iterations=config.evaluation.warmup_iterations,
        benchmark_iterations=config.evaluation.benchmark_iterations,
        max_mean_abs_diff=config.evaluation.max_mean_abs_diff,
        max_cosine_distance=config.evaluation.max_cosine_distance,
        min_speedup_ratio=config.evaluation.min_speedup_ratio,
    )
    summary = _build_summary(
        sample_results=sample_results,
        target_min_speedup_ratio=config.evaluation.min_speedup_ratio,
    )

    output_root = resolve_project_path(str(resolved_project_root), config.output_root)
    report_json_path = output_root / TENSORRT_FP16_REPORT_JSON_NAME
    return TensorRTFP16Report(
        title=config.title,
        report_id=config.report_id,
        summary_text=config.summary,
        generated_at_utc=utc_now(),
        project_root=str(resolved_project_root),
        output_root=str(output_root),
        source_config_path=None if config_path is None else str(Path(config_path).resolve()),
        model_version=_coerce_optional_string(metadata.get("model_version")),
        metadata_path=str(metadata_path),
        onnx_model_path=str(onnx_model_path),
        engine_path=str(engine_path),
        source_checkpoint_path=str(source_checkpoint_path),
        input_name=input_name,
        output_name=output_name,
        embedding_dim=embedding_dim,
        engine_size_bytes=engine_path.stat().st_size,
        workspace_size_mib=config.build.workspace_size_mib,
        profiles=profiles,
        samples=sample_results,
        promotion=TensorRTFP16PromotionState(
            requested=config.build.promote_validated_backend,
            ready=summary.passed,
            applied=False,
            target_metadata_path=str(metadata_path),
            report_json_path=str(report_json_path),
        ),
        validation_commands=config.validation_commands,
        notes=config.notes,
        summary=summary,
    )


def write_tensorrt_fp16_report(report: TensorRTFP16Report) -> WrittenTensorRTFP16Report:
    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report_json_path = output_root / TENSORRT_FP16_REPORT_JSON_NAME
    report_markdown_path = output_root / TENSORRT_FP16_REPORT_MARKDOWN_NAME
    promotion = _apply_metadata_update(report=report, report_json_path=report_json_path)
    updated_report = replace(report, promotion=promotion)

    report_json_path.write_text(
        json.dumps(updated_report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_tensorrt_fp16_markdown(updated_report),
        encoding="utf-8",
    )

    source_config_copy_path = None
    if updated_report.source_config_path is not None:
        destination = output_root / "sources" / "tensorrt_fp16_config.toml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            Path(updated_report.source_config_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        source_config_copy_path = str(destination)

    return WrittenTensorRTFP16Report(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        source_config_copy_path=source_config_copy_path,
        promotion=updated_report.promotion,
        summary=updated_report.summary,
    )


def render_tensorrt_fp16_markdown(report: TensorRTFP16Report) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Report id: `{report.report_id}`",
        f"- Generated at: `{report.generated_at_utc}`",
        f"- Model version: `{report.model_version or 'unknown'}`",
        f"- Metadata: `{report.metadata_path}`",
        f"- ONNX model: `{report.onnx_model_path}`",
        f"- TensorRT engine: `{report.engine_path}`",
        f"- Source checkpoint: `{report.source_checkpoint_path}`",
        f"- Overall status: `{'pass' if report.summary.passed else 'fail'}`",
        "",
        "## Optimization Profiles",
        "",
        f"- Input name: `{report.input_name}`",
        f"- Output name: `{report.output_name}`",
        f"- Embedding dim: `{report.embedding_dim}`",
        f"- Workspace size MiB: `{report.workspace_size_mib}`",
        f"- Engine size bytes: `{report.engine_size_bytes}`",
        "",
    ]
    for profile in report.profiles:
        lines.append(
            f"- `{profile.profile_id}`: min `{list(profile.min_shape)}`, "
            f"opt `{list(profile.opt_shape)}`, max `{list(profile.max_shape)}`"
        )
    lines.extend(
        [
            "",
            "## Validation Samples",
            "",
            (
                "| Sample | Profile | Shape | max abs diff | mean abs diff | cosine distance | "
                "Torch ms | TensorRT ms | speedup | quality | speed |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in report.samples:
        lines.append(_render_sample_row(sample))

    lines.extend(
        [
            "",
            "## Summary",
            "",
            report.summary_text.strip(),
            "",
            (
                "- Samples passed quality: "
                f"`{report.summary.passed_quality_count}/{report.summary.sample_count}`"
            ),
            (
                "- Samples passed speed gate: "
                f"`{report.summary.passed_speed_count}/{report.summary.sample_count}`"
            ),
            f"- Max abs diff: `{report.summary.max_abs_diff:.8f}`",
            f"- Max mean abs diff: `{report.summary.max_mean_abs_diff:.8f}`",
            f"- Max cosine distance: `{report.summary.max_cosine_distance:.8f}`",
            f"- Min observed speedup: `{report.summary.min_speedup_ratio_observed:.4f}`",
            f"- Target minimum speedup: `{report.summary.target_min_speedup_ratio:.4f}`",
            "",
            "## Metadata Promotion",
            "",
            f"- Requested: `{str(report.promotion.requested).lower()}`",
            f"- Ready: `{str(report.promotion.ready).lower()}`",
            f"- Applied: `{str(report.promotion.applied).lower()}`",
            f"- Target metadata: `{report.promotion.target_metadata_path}`",
            f"- Report JSON path: `{report.promotion.report_json_path}`",
        ]
    )
    if report.promotion.error:
        lines.append(f"- Error: `{report.promotion.error}`")
    if report.validation_commands:
        lines.extend(["", "## Validation Commands", ""])
        lines.extend(f"- `{command}`" for command in report.validation_commands)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    lines.extend(
        [
            "",
            "## Limits",
            "",
            (
                "- This workflow validates the encoder-only feature-tensor boundary, "
                "not raw-audio runtime parity."
            ),
            (
                "- The TensorRT runtime backend is still selected separately by "
                "the serving runtime contract."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _render_sample_row(sample: TensorRTFP16SampleResult) -> str:
    return (
        "| "
        f"{sample.sample_id} | "
        f"`{sample.profile_id}` | "
        f"`[{sample.batch_size}, {sample.frame_count}, -]` | "
        f"`{sample.max_abs_diff:.8f}` | "
        f"`{sample.mean_abs_diff:.8f}` | "
        f"`{sample.cosine_distance:.8f}` | "
        f"`{sample.torch_latency_ms:.4f}` | "
        f"`{sample.tensorrt_latency_ms:.4f}` | "
        f"`{sample.speedup_ratio:.4f}` | "
        f"`{str(sample.passed_quality).lower()}` | "
        f"`{str(sample.passed_speedup).lower()}` |"
    )


def _build_summary(
    *,
    sample_results: tuple[TensorRTFP16SampleResult, ...],
    target_min_speedup_ratio: float,
) -> TensorRTFP16Summary:
    if not sample_results:
        return TensorRTFP16Summary(
            passed=False,
            sample_count=0,
            passed_quality_count=0,
            passed_speed_count=0,
            max_abs_diff=0.0,
            max_mean_abs_diff=0.0,
            max_cosine_distance=0.0,
            min_speedup_ratio_observed=0.0,
            target_min_speedup_ratio=target_min_speedup_ratio,
        )
    return TensorRTFP16Summary(
        passed=all(sample.passed_quality and sample.passed_speedup for sample in sample_results),
        sample_count=len(sample_results),
        passed_quality_count=sum(1 for sample in sample_results if sample.passed_quality),
        passed_speed_count=sum(1 for sample in sample_results if sample.passed_speedup),
        max_abs_diff=max(sample.max_abs_diff for sample in sample_results),
        max_mean_abs_diff=max(sample.mean_abs_diff for sample in sample_results),
        max_cosine_distance=max(sample.cosine_distance for sample in sample_results),
        min_speedup_ratio_observed=min(sample.speedup_ratio for sample in sample_results),
        target_min_speedup_ratio=target_min_speedup_ratio,
    )


def _apply_metadata_update(
    *,
    report: TensorRTFP16Report,
    report_json_path: Path,
) -> TensorRTFP16PromotionState:
    metadata_path = Path(report.promotion.target_metadata_path)
    try:
        metadata = _load_json_object(metadata_path)
        inference_package = load_inference_package_from_model_metadata(metadata).to_dict()
        artifacts = _coerce_json_object(inference_package.get("artifacts"))
        artifacts["tensorrt_engine_file"] = _relative_to_project(
            Path(report.engine_path),
            Path(report.project_root),
        )
        inference_package["artifacts"] = artifacts

        validated_backends = _coerce_bool_mapping(
            inference_package.get("validated_backends"),
            field_name="validated_backends",
        )
        if report.promotion.requested:
            validated_backends["tensorrt"] = report.summary.passed
        inference_package["validated_backends"] = validated_backends
        metadata["inference_package"] = inference_package
        metadata["tensorrt_engine_file"] = artifacts["tensorrt_engine_file"]

        export_validation = _coerce_json_object(metadata.get("export_validation"))
        export_validation["tensorrt_fp16_engine_path"] = artifacts["tensorrt_engine_file"]
        export_validation["tensorrt_fp16_report_path"] = _relative_to_project(
            report_json_path,
            Path(report.project_root),
        )
        export_validation["tensorrt_fp16_built_at_utc"] = report.generated_at_utc
        export_validation["tensorrt_fp16_validated"] = report.summary.passed
        if report.summary.passed:
            export_validation["tensorrt_fp16_validated_at_utc"] = utc_now()
        metadata["export_validation"] = export_validation
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return replace(
            report.promotion,
            applied=report.promotion.requested and report.summary.passed,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive serialization path
        return replace(
            report.promotion,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return {str(key): value for key, value in payload.items()}


def _resolve_project_root(project_root: Path | str | None, configured_root: str) -> Path:
    if project_root is not None:
        candidate = Path(project_root)
        return candidate if candidate.is_absolute() else candidate.resolve()
    return resolve_project_path(".", configured_root)


def _require_static_axis_size(size: object, *, axis_name: str) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"Expected a static integer size for {axis_name}, got {size!r}.")
    if size <= 0:
        raise ValueError(f"{axis_name} must be positive.")
    return size


def _coerce_json_object(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def _coerce_bool_mapping(raw: object, *, field_name: str) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected `{field_name}` to be a JSON object.")
    values: dict[str, bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, bool):
            raise ValueError(f"`{field_name}` must contain string -> bool entries.")
        values[key] = value
    return values


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _coerce_optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "TENSORRT_FP16_REPORT_JSON_NAME",
    "TENSORRT_FP16_REPORT_MARKDOWN_NAME",
    "TensorRTFP16Profile",
    "TensorRTFP16PromotionState",
    "TensorRTFP16Report",
    "TensorRTFP16SampleResult",
    "TensorRTFP16Summary",
    "WrittenTensorRTFP16Report",
    "build_tensorrt_fp16_report",
    "render_tensorrt_fp16_markdown",
    "write_tensorrt_fp16_report",
]
