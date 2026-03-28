"""Builder for reproducible ONNX Runtime parity reports."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.features import FbankExtractionRequest, FbankExtractor, UtteranceChunkingRequest
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.repro import fingerprint_path
from kryptonite.serve.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.tracking import utc_now

from .onnx_parity_config import ONNXParityConfig, ONNXParityVariantConfig
from .onnx_parity_models import (
    ONNX_PARITY_REPORT_JSON_NAME,
    ONNXParityAudioRecord,
    ONNXParityPromotionState,
    ONNXParityReport,
    ONNXParitySummary,
    ONNXParityTrialRecord,
    ONNXParityVariantSummary,
)
from .onnx_parity_runtime import (
    build_variant_audio_records,
    build_variant_trial_records,
    coerce_optional_string,
    mean,
    resolve_trials_and_audio_items,
)
from .verification_data import (
    build_trial_item_index,
    load_verification_metadata_rows,
    load_verification_trial_rows,
)
from .verification_metrics import compute_verification_metrics


def build_onnx_parity_report(
    config: ONNXParityConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> ONNXParityReport:
    resolved_project_root = _resolve_project_root(project_root or config.project_root)
    metadata_path = resolve_project_path(
        str(resolved_project_root),
        config.artifacts.model_bundle_metadata_path,
    )
    model_metadata = _load_json_object(metadata_path)
    export_boundary = load_export_boundary_from_model_metadata(model_metadata)
    if export_boundary.inferencer_backend != "campp_encoder":
        raise ValueError(
            "ONNX parity currently supports only `campp_encoder` export bundles, got "
            f"{export_boundary.inferencer_backend!r}."
        )

    resolved_onnx_path = _require_relative_artifact(
        model_metadata.get("model_file"),
        field_name="model_file",
        project_root=resolved_project_root,
    )
    resolved_checkpoint_path = _require_relative_artifact(
        model_metadata.get("source_checkpoint_path"),
        field_name="source_checkpoint_path",
        project_root=resolved_project_root,
    )
    trial_rows_path = resolve_project_path(
        str(resolved_project_root),
        config.artifacts.trial_rows_path,
    )
    metadata_rows_path = resolve_project_path(
        str(resolved_project_root),
        config.artifacts.metadata_rows_path,
    )

    torch = importlib.import_module("torch")
    onnxruntime = importlib.import_module("onnxruntime")
    _, _, torch_model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=resolved_checkpoint_path,
        project_root=resolved_project_root,
    )
    onnx_session = onnxruntime.InferenceSession(
        resolved_onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )

    audio_request, fbank_request, chunking_request = _runtime_frontend_requests(model_metadata)
    extractor = FbankExtractor(request=fbank_request)
    trial_rows = load_verification_trial_rows(trial_rows_path)
    if config.evaluation.max_trial_count is not None:
        trial_rows = trial_rows[: config.evaluation.max_trial_count]
    metadata_rows = load_verification_metadata_rows(metadata_rows_path)
    metadata_index = build_trial_item_index(metadata_rows)
    resolved_trials, audio_items = resolve_trials_and_audio_items(
        trial_rows=trial_rows,
        metadata_index=metadata_index,
        prefer_demo_subset=config.evaluation.prefer_demo_subset,
    )

    variant_summaries: list[ONNXParityVariantSummary] = []
    all_audio_records: list[ONNXParityAudioRecord] = []
    all_trial_records: list[ONNXParityTrialRecord] = []
    for variant in config.variants:
        embeddings_by_item, audio_records = build_variant_audio_records(
            variant=variant,
            audio_items=audio_items,
            project_root=resolved_project_root,
            audio_request=audio_request,
            extractor=extractor,
            chunking_request=chunking_request,
            embedding_stage=export_boundary.embedding_stage,
            input_name=export_boundary.input_tensor.name,
            output_name=export_boundary.output_tensor.name,
            torch=torch,
            torch_model=torch_model,
            onnx_session=onnx_session,
            seed=config.evaluation.seed,
        )
        trial_records = build_variant_trial_records(
            variant=variant,
            trials=resolved_trials,
            audio_items=audio_items,
            embeddings_by_item=embeddings_by_item,
            normalize_scores=config.evaluation.score_normalize,
        )
        variant_summaries.append(
            _build_variant_summary(
                variant=variant,
                audio_records=audio_records,
                trial_records=trial_records,
                config=config,
            )
        )
        all_audio_records.extend(audio_records)
        all_trial_records.extend(trial_records)

    summary = ONNXParitySummary(
        passed=all(variant_summary.passed for variant_summary in variant_summaries),
        variant_count=len(variant_summaries),
        passed_variant_count=sum(
            1 for variant_summary in variant_summaries if variant_summary.passed
        ),
        audio_record_count=len(all_audio_records),
        trial_record_count=len(all_trial_records),
        max_chunk_max_abs_diff=max(
            variant_summary.max_chunk_max_abs_diff for variant_summary in variant_summaries
        ),
        max_pooled_max_abs_diff=max(
            variant_summary.max_pooled_max_abs_diff for variant_summary in variant_summaries
        ),
        max_pooled_cosine_distance=max(
            variant_summary.max_pooled_cosine_distance for variant_summary in variant_summaries
        ),
        max_score_abs_diff=max(
            variant_summary.max_score_abs_diff for variant_summary in variant_summaries
        ),
        max_eer_delta=max(variant_summary.eer_delta for variant_summary in variant_summaries),
        max_min_dcf_delta=max(
            variant_summary.min_dcf_delta for variant_summary in variant_summaries
        ),
    )

    source_config_file = None if config_path is None else Path(config_path).resolve()
    source_config_sha256 = None
    if source_config_file is not None:
        source_fingerprint = fingerprint_path(source_config_file)
        source_config_sha256 = (
            None if not bool(source_fingerprint["exists"]) else str(source_fingerprint["sha256"])
        )

    output_root = resolve_project_path(str(resolved_project_root), config.output_root)
    promotion = ONNXParityPromotionState(
        requested=config.evaluation.promote_validated_backend,
        ready=summary.passed,
        applied=False,
        target_metadata_path=str(metadata_path),
        report_json_path=str(output_root / ONNX_PARITY_REPORT_JSON_NAME),
        error=None,
    )
    return ONNXParityReport(
        title=config.title,
        report_id=config.report_id,
        summary_text=config.summary,
        generated_at_utc=utc_now(),
        project_root=str(resolved_project_root),
        output_root=str(output_root),
        source_config_path=None if source_config_file is None else str(source_config_file),
        source_config_sha256=source_config_sha256,
        model_version=coerce_optional_string(model_metadata.get("model_version")),
        embedding_stage=export_boundary.embedding_stage,
        input_name=export_boundary.input_tensor.name,
        output_name=export_boundary.output_tensor.name,
        metadata_path=str(metadata_path),
        onnx_model_path=str(resolved_onnx_path),
        source_checkpoint_path=str(resolved_checkpoint_path),
        trial_rows_path=str(trial_rows_path),
        metadata_rows_path=str(metadata_rows_path),
        evaluation=config.evaluation.to_dict(),
        tolerances=config.tolerances.to_dict(),
        variants=tuple(variant_summaries),
        audio_records=tuple(all_audio_records),
        trial_records=tuple(all_trial_records),
        promotion=promotion,
        validation_commands=config.validation_commands,
        notes=config.notes,
        summary=summary,
    )


def _resolve_project_root(project_root: Path | str) -> Path:
    return resolve_project_path(str(project_root), ".")


def _require_relative_artifact(
    raw_path: object,
    *,
    field_name: str,
    project_root: Path,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Model bundle metadata must define a non-empty `{field_name}`.")
    resolved = resolve_project_path(str(project_root), raw_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} does not resolve to a file: {resolved}.")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object metadata payload in {path}.")
    return {str(key): value for key, value in payload.items()}


def _runtime_frontend_requests(
    model_metadata: dict[str, Any],
) -> tuple[AudioLoadRequest, FbankExtractionRequest, UtteranceChunkingRequest]:
    export_boundary = load_export_boundary_from_model_metadata(model_metadata)
    runtime_frontend = export_boundary.runtime_frontend
    audio_payload = runtime_frontend.get("audio_load_request")
    features_payload = runtime_frontend.get("features")
    chunking_payload = runtime_frontend.get("chunking")
    if not isinstance(audio_payload, dict):
        raise ValueError("export_boundary.runtime_frontend.audio_load_request must be an object.")
    if not isinstance(features_payload, dict):
        raise ValueError("export_boundary.runtime_frontend.features must be an object.")
    if not isinstance(chunking_payload, dict):
        raise ValueError("export_boundary.runtime_frontend.chunking must be an object.")
    return (
        AudioLoadRequest(**audio_payload),
        FbankExtractionRequest(**features_payload),
        UtteranceChunkingRequest(**chunking_payload),
    )


def _build_variant_summary(
    *,
    variant: ONNXParityVariantConfig,
    audio_records: list[ONNXParityAudioRecord],
    trial_records: list[ONNXParityTrialRecord],
    config: ONNXParityConfig,
) -> ONNXParityVariantSummary:
    torch_metrics = compute_verification_metrics(
        [{"label": record.label, "score": record.torch_score} for record in trial_records]
    )
    onnx_metrics = compute_verification_metrics(
        [{"label": record.label, "score": record.onnx_score} for record in trial_records]
    )
    max_chunk_max_abs_diff = max(record.max_chunk_max_abs_diff for record in audio_records)
    max_pooled_max_abs_diff = max(record.pooled_max_abs_diff for record in audio_records)
    max_pooled_cosine_distance = max(record.pooled_cosine_distance for record in audio_records)
    max_score_abs_diff = max(record.score_abs_diff for record in trial_records)
    eer_delta = round(abs(torch_metrics.eer - onnx_metrics.eer), 8)
    min_dcf_delta = round(abs(torch_metrics.min_dcf - onnx_metrics.min_dcf), 8)
    tolerances = config.tolerances
    failure_reasons: list[str] = []
    if max_chunk_max_abs_diff > tolerances.max_chunk_max_abs_diff:
        failure_reasons.append(
            "chunk max abs diff exceeded "
            f"{tolerances.max_chunk_max_abs_diff:.8f} (got {max_chunk_max_abs_diff:.8f})"
        )
    if max_pooled_max_abs_diff > tolerances.max_pooled_max_abs_diff:
        failure_reasons.append(
            "pooled max abs diff exceeded "
            f"{tolerances.max_pooled_max_abs_diff:.8f} (got {max_pooled_max_abs_diff:.8f})"
        )
    if max_pooled_cosine_distance > tolerances.max_pooled_cosine_distance:
        failure_reasons.append(
            "pooled cosine distance exceeded "
            f"{tolerances.max_pooled_cosine_distance:.8f} (got {max_pooled_cosine_distance:.8f})"
        )
    if max_score_abs_diff > tolerances.max_score_abs_diff:
        failure_reasons.append(
            "score abs diff exceeded "
            f"{tolerances.max_score_abs_diff:.8f} (got {max_score_abs_diff:.8f})"
        )
    if eer_delta > tolerances.max_eer_delta:
        failure_reasons.append(
            f"EER delta exceeded {tolerances.max_eer_delta:.8f} (got {eer_delta:.8f})"
        )
    if min_dcf_delta > tolerances.max_min_dcf_delta:
        failure_reasons.append(
            f"minDCF delta exceeded {tolerances.max_min_dcf_delta:.8f} (got {min_dcf_delta:.8f})"
        )
    return ONNXParityVariantSummary(
        variant_id=variant.variant_id,
        kind=variant.kind,
        description=variant.description,
        audio_record_count=len(audio_records),
        trial_count=len(trial_records),
        positive_count=sum(1 for record in trial_records if record.label == 1),
        negative_count=sum(1 for record in trial_records if record.label == 0),
        max_chunk_max_abs_diff=round(max_chunk_max_abs_diff, 8),
        mean_chunk_mean_abs_diff=round(
            mean(record.mean_chunk_mean_abs_diff for record in audio_records),
            8,
        ),
        max_pooled_max_abs_diff=round(max_pooled_max_abs_diff, 8),
        mean_pooled_mean_abs_diff=round(
            mean(record.pooled_mean_abs_diff for record in audio_records),
            8,
        ),
        max_pooled_cosine_distance=round(max_pooled_cosine_distance, 8),
        max_score_abs_diff=round(max_score_abs_diff, 8),
        mean_score_abs_diff=round(mean(record.score_abs_diff for record in trial_records), 8),
        torch_metrics=torch_metrics,
        onnx_metrics=onnx_metrics,
        eer_delta=eer_delta,
        min_dcf_delta=min_dcf_delta,
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
    )


__all__ = [
    "build_onnx_parity_report",
]
