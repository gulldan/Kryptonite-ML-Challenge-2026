"""Datamodels for reproducible ONNX Runtime parity reports."""

from __future__ import annotations

from dataclasses import dataclass

from .verification_metrics import VerificationMetricsSummary

ONNX_PARITY_REPORT_JSON_NAME = "onnx_parity_report.json"
ONNX_PARITY_REPORT_MARKDOWN_NAME = "onnx_parity_report.md"
ONNX_PARITY_AUDIO_ROWS_NAME = "onnx_parity_audio_rows.jsonl"
ONNX_PARITY_TRIAL_ROWS_NAME = "onnx_parity_trial_rows.jsonl"


@dataclass(frozen=True, slots=True)
class ONNXParityAudioRecord:
    variant_id: str
    variant_kind: str
    item_id: str
    speaker_id: str | None
    role: str | None
    source_audio_path: str
    audio_path: str
    corruption_applied: bool
    sample_rate_hz: int
    duration_seconds: float
    chunk_count: int
    max_chunk_max_abs_diff: float
    mean_chunk_mean_abs_diff: float
    pooled_max_abs_diff: float
    pooled_mean_abs_diff: float
    pooled_cosine_distance: float
    torch_embedding_norm: float
    onnx_embedding_norm: float

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "variant_kind": self.variant_kind,
            "item_id": self.item_id,
            "speaker_id": self.speaker_id,
            "role": self.role,
            "source_audio_path": self.source_audio_path,
            "audio_path": self.audio_path,
            "corruption_applied": self.corruption_applied,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_seconds": self.duration_seconds,
            "chunk_count": self.chunk_count,
            "max_chunk_max_abs_diff": self.max_chunk_max_abs_diff,
            "mean_chunk_mean_abs_diff": self.mean_chunk_mean_abs_diff,
            "pooled_max_abs_diff": self.pooled_max_abs_diff,
            "pooled_mean_abs_diff": self.pooled_mean_abs_diff,
            "pooled_cosine_distance": self.pooled_cosine_distance,
            "torch_embedding_norm": self.torch_embedding_norm,
            "onnx_embedding_norm": self.onnx_embedding_norm,
        }


@dataclass(frozen=True, slots=True)
class ONNXParityTrialRecord:
    variant_id: str
    variant_kind: str
    label: int
    left_id: str
    right_id: str
    left_audio_path: str
    right_audio_path: str
    torch_score: float
    onnx_score: float
    score_abs_diff: float

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "variant_kind": self.variant_kind,
            "label": self.label,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_audio_path": self.left_audio_path,
            "right_audio_path": self.right_audio_path,
            "torch_score": self.torch_score,
            "onnx_score": self.onnx_score,
            "score_abs_diff": self.score_abs_diff,
        }


@dataclass(frozen=True, slots=True)
class ONNXParityVariantSummary:
    variant_id: str
    kind: str
    description: str
    audio_record_count: int
    trial_count: int
    positive_count: int
    negative_count: int
    max_chunk_max_abs_diff: float
    mean_chunk_mean_abs_diff: float
    max_pooled_max_abs_diff: float
    mean_pooled_mean_abs_diff: float
    max_pooled_cosine_distance: float
    max_score_abs_diff: float
    mean_score_abs_diff: float
    torch_metrics: VerificationMetricsSummary
    onnx_metrics: VerificationMetricsSummary
    eer_delta: float
    min_dcf_delta: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "kind": self.kind,
            "description": self.description,
            "audio_record_count": self.audio_record_count,
            "trial_count": self.trial_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "max_chunk_max_abs_diff": self.max_chunk_max_abs_diff,
            "mean_chunk_mean_abs_diff": self.mean_chunk_mean_abs_diff,
            "max_pooled_max_abs_diff": self.max_pooled_max_abs_diff,
            "mean_pooled_mean_abs_diff": self.mean_pooled_mean_abs_diff,
            "max_pooled_cosine_distance": self.max_pooled_cosine_distance,
            "max_score_abs_diff": self.max_score_abs_diff,
            "mean_score_abs_diff": self.mean_score_abs_diff,
            "torch_metrics": self.torch_metrics.to_dict(),
            "onnx_metrics": self.onnx_metrics.to_dict(),
            "eer_delta": self.eer_delta,
            "min_dcf_delta": self.min_dcf_delta,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class ONNXParityPromotionState:
    requested: bool
    ready: bool
    applied: bool
    target_metadata_path: str
    report_json_path: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "ready": self.ready,
            "applied": self.applied,
            "target_metadata_path": self.target_metadata_path,
            "report_json_path": self.report_json_path,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ONNXParitySummary:
    passed: bool
    variant_count: int
    passed_variant_count: int
    audio_record_count: int
    trial_record_count: int
    max_chunk_max_abs_diff: float
    max_pooled_max_abs_diff: float
    max_pooled_cosine_distance: float
    max_score_abs_diff: float
    max_eer_delta: float
    max_min_dcf_delta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "variant_count": self.variant_count,
            "passed_variant_count": self.passed_variant_count,
            "audio_record_count": self.audio_record_count,
            "trial_record_count": self.trial_record_count,
            "max_chunk_max_abs_diff": self.max_chunk_max_abs_diff,
            "max_pooled_max_abs_diff": self.max_pooled_max_abs_diff,
            "max_pooled_cosine_distance": self.max_pooled_cosine_distance,
            "max_score_abs_diff": self.max_score_abs_diff,
            "max_eer_delta": self.max_eer_delta,
            "max_min_dcf_delta": self.max_min_dcf_delta,
        }


@dataclass(frozen=True, slots=True)
class ONNXParityReport:
    title: str
    report_id: str
    summary_text: str
    generated_at_utc: str
    project_root: str
    output_root: str
    source_config_path: str | None
    source_config_sha256: str | None
    model_version: str | None
    embedding_stage: str
    input_name: str
    output_name: str
    metadata_path: str
    onnx_model_path: str
    source_checkpoint_path: str
    trial_rows_path: str
    metadata_rows_path: str
    evaluation: dict[str, object]
    tolerances: dict[str, object]
    variants: tuple[ONNXParityVariantSummary, ...]
    audio_records: tuple[ONNXParityAudioRecord, ...]
    trial_records: tuple[ONNXParityTrialRecord, ...]
    promotion: ONNXParityPromotionState
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]
    summary: ONNXParitySummary

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "summary_text": self.summary_text,
            "generated_at_utc": self.generated_at_utc,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "source_config_path": self.source_config_path,
            "source_config_sha256": self.source_config_sha256,
            "model_version": self.model_version,
            "embedding_stage": self.embedding_stage,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "metadata_path": self.metadata_path,
            "onnx_model_path": self.onnx_model_path,
            "source_checkpoint_path": self.source_checkpoint_path,
            "trial_rows_path": self.trial_rows_path,
            "metadata_rows_path": self.metadata_rows_path,
            "evaluation": dict(self.evaluation),
            "tolerances": dict(self.tolerances),
            "variants": [variant.to_dict() for variant in self.variants],
            "audio_records": [record.to_dict() for record in self.audio_records],
            "trial_records": [record.to_dict() for record in self.trial_records],
            "promotion": self.promotion.to_dict(),
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenONNXParityReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    audio_rows_path: str
    trial_rows_path: str
    source_config_copy_path: str | None
    promotion: ONNXParityPromotionState
    summary: ONNXParitySummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "audio_rows_path": self.audio_rows_path,
            "trial_rows_path": self.trial_rows_path,
            "source_config_copy_path": self.source_config_copy_path,
            "promotion": self.promotion.to_dict(),
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "ONNX_PARITY_AUDIO_ROWS_NAME",
    "ONNX_PARITY_REPORT_JSON_NAME",
    "ONNX_PARITY_REPORT_MARKDOWN_NAME",
    "ONNX_PARITY_TRIAL_ROWS_NAME",
    "ONNXParityAudioRecord",
    "ONNXParityPromotionState",
    "ONNXParityReport",
    "ONNXParitySummary",
    "ONNXParityTrialRecord",
    "ONNXParityVariantSummary",
    "WrittenONNXParityReport",
]
