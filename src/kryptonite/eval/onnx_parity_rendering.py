"""Rendering helpers for reproducible ONNX Runtime parity reports."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from kryptonite.serve.inference_package import load_inference_package_from_model_metadata
from kryptonite.tracking import utc_now

from .onnx_parity_models import (
    ONNX_PARITY_AUDIO_ROWS_NAME,
    ONNX_PARITY_REPORT_JSON_NAME,
    ONNX_PARITY_REPORT_MARKDOWN_NAME,
    ONNX_PARITY_TRIAL_ROWS_NAME,
    ONNXParityPromotionState,
    ONNXParityReport,
    ONNXParityVariantSummary,
    WrittenONNXParityReport,
)


def render_onnx_parity_markdown(report: ONNXParityReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Report id: `{report.report_id}`",
        f"- Generated at: `{report.generated_at_utc}`",
        f"- Model version: `{report.model_version or 'unknown'}`",
        f"- Embedding stage: `{report.embedding_stage}`",
        f"- Overall status: `{'pass' if report.summary.passed else 'fail'}`",
        f"- Output root: `{report.output_root}`",
        f"- Metadata: `{report.metadata_path}`",
        f"- ONNX model: `{report.onnx_model_path}`",
        f"- Source checkpoint: `{report.source_checkpoint_path}`",
        "",
        "## Summary",
        "",
        report.summary_text.strip(),
        "",
        (
            "- Variants passed: "
            f"`{report.summary.passed_variant_count}/{report.summary.variant_count}`"
        ),
        f"- Audio rows: `{report.summary.audio_record_count}`",
        f"- Trial rows: `{report.summary.trial_record_count}`",
        f"- Max chunk abs diff: `{report.summary.max_chunk_max_abs_diff:.8f}`",
        f"- Max pooled abs diff: `{report.summary.max_pooled_max_abs_diff:.8f}`",
        f"- Max pooled cosine distance: `{report.summary.max_pooled_cosine_distance:.8f}`",
        f"- Max score abs diff: `{report.summary.max_score_abs_diff:.8f}`",
        f"- Max EER delta: `{report.summary.max_eer_delta:.8f}`",
        f"- Max minDCF delta: `{report.summary.max_min_dcf_delta:.8f}`",
        "",
        "## Variant Breakdown",
        "",
        (
            "| Variant | Status | max chunk diff | max pooled diff | max cosine dist | "
            "max score diff | EER delta | minDCF delta |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant in report.variants:
        lines.append(_render_variant_row(variant))
    for variant in report.variants:
        if not variant.failure_reasons:
            continue
        lines.extend(["", f"Failure reasons for `{variant.variant_id}`:"])
        lines.extend(f"- {reason}" for reason in variant.failure_reasons)

    lines.extend(
        [
            "",
            "## Runtime Promotion",
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
    return "\n".join(lines) + "\n"


def write_onnx_parity_report(report: ONNXParityReport) -> WrittenONNXParityReport:
    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    updated_report = report
    if report.promotion.requested:
        updated_report = replace(report, promotion=_apply_metadata_promotion(report))

    report_json_path = output_root / ONNX_PARITY_REPORT_JSON_NAME
    report_json_path.write_text(
        json.dumps(updated_report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path = output_root / ONNX_PARITY_REPORT_MARKDOWN_NAME
    report_markdown_path.write_text(
        render_onnx_parity_markdown(updated_report),
        encoding="utf-8",
    )
    audio_rows_path = output_root / ONNX_PARITY_AUDIO_ROWS_NAME
    audio_rows_path.write_text(
        "".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n"
            for record in updated_report.audio_records
        ),
        encoding="utf-8",
    )
    trial_rows_path = output_root / ONNX_PARITY_TRIAL_ROWS_NAME
    trial_rows_path.write_text(
        "".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n"
            for record in updated_report.trial_records
        ),
        encoding="utf-8",
    )

    source_config_copy_path = None
    if updated_report.source_config_path is not None:
        source_config_copy = output_root / "sources" / "onnx_parity_config.toml"
        source_config_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(updated_report.source_config_path, source_config_copy)
        source_config_copy_path = str(source_config_copy)

    return WrittenONNXParityReport(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        audio_rows_path=str(audio_rows_path),
        trial_rows_path=str(trial_rows_path),
        source_config_copy_path=source_config_copy_path,
        promotion=updated_report.promotion,
        summary=updated_report.summary,
    )


def _render_variant_row(variant: ONNXParityVariantSummary) -> str:
    status = "pass" if variant.passed else "fail"
    return (
        f"| {variant.variant_id} | `{status}` | "
        f"`{variant.max_chunk_max_abs_diff:.8f}` | "
        f"`{variant.max_pooled_max_abs_diff:.8f}` | "
        f"`{variant.max_pooled_cosine_distance:.8f}` | "
        f"`{variant.max_score_abs_diff:.8f}` | "
        f"`{variant.eer_delta:.8f}` | "
        f"`{variant.min_dcf_delta:.8f}` |"
    )


def _apply_metadata_promotion(report: ONNXParityReport) -> ONNXParityPromotionState:
    if not report.promotion.ready:
        return replace(
            report.promotion,
            applied=False,
            error="Parity report did not pass tolerance gates; metadata promotion skipped.",
        )
    metadata_path = Path(report.promotion.target_metadata_path)
    try:
        metadata = _load_json_object(metadata_path)
        inference_package = load_inference_package_from_model_metadata(metadata).to_dict()
        validated_backends = _coerce_bool_mapping(
            inference_package.get("validated_backends"),
            field_name="validated_backends",
        )
        validated_backends["onnxruntime"] = True
        inference_package["validated_backends"] = validated_backends
        metadata["inference_package"] = inference_package

        export_validation = _coerce_json_object(metadata.get("export_validation"))
        export_validation["runtime_backends_promoted"] = True
        export_validation["runtime_backends_promotion_blocker"] = None
        export_validation["onnx_parity_report_path"] = _relative_to_project(
            Path(report.promotion.report_json_path),
            Path(report.project_root),
        )
        export_validation["onnxruntime_parity_validated_at_utc"] = utc_now()
        metadata["export_validation"] = export_validation
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return replace(report.promotion, applied=True, error=None)
    except Exception as exc:  # pragma: no cover - defensive serialization path
        return replace(
            report.promotion,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )


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


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object metadata payload in {path}.")
    return {str(key): value for key, value in payload.items()}


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


__all__ = [
    "render_onnx_parity_markdown",
    "write_onnx_parity_report",
]
