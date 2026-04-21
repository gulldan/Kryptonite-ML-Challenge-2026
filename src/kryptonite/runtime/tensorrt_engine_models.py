"""Datamodels for TensorRT FP16 engine reports."""

from __future__ import annotations

from dataclasses import dataclass

TENSORRT_FP16_REPORT_JSON_NAME = "tensorrt_fp16_engine_report.json"
TENSORRT_FP16_REPORT_MARKDOWN_NAME = "tensorrt_fp16_engine_report.md"


@dataclass(frozen=True, slots=True)
class TensorRTFP16Profile:
    profile_id: str
    min_shape: tuple[int, int, int]
    opt_shape: tuple[int, int, int]
    max_shape: tuple[int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "min_shape": list(self.min_shape),
            "opt_shape": list(self.opt_shape),
            "max_shape": list(self.max_shape),
        }


@dataclass(frozen=True, slots=True)
class TensorRTFP16SampleResult:
    sample_id: str
    profile_id: str
    batch_size: int
    frame_count: int
    output_shape: tuple[int, int]
    max_abs_diff: float
    mean_abs_diff: float
    cosine_distance: float
    torch_latency_ms: float
    tensorrt_latency_ms: float
    speedup_ratio: float
    passed_quality: bool
    passed_speedup: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "profile_id": self.profile_id,
            "batch_size": self.batch_size,
            "frame_count": self.frame_count,
            "output_shape": list(self.output_shape),
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "cosine_distance": self.cosine_distance,
            "torch_latency_ms": self.torch_latency_ms,
            "tensorrt_latency_ms": self.tensorrt_latency_ms,
            "speedup_ratio": self.speedup_ratio,
            "passed_quality": self.passed_quality,
            "passed_speedup": self.passed_speedup,
        }


@dataclass(frozen=True, slots=True)
class TensorRTFP16PromotionState:
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
class TensorRTFP16Summary:
    passed: bool
    sample_count: int
    passed_quality_count: int
    passed_speed_count: int
    max_abs_diff: float
    max_mean_abs_diff: float
    max_cosine_distance: float
    min_speedup_ratio_observed: float
    target_min_speedup_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "sample_count": self.sample_count,
            "passed_quality_count": self.passed_quality_count,
            "passed_speed_count": self.passed_speed_count,
            "max_abs_diff": self.max_abs_diff,
            "max_mean_abs_diff": self.max_mean_abs_diff,
            "max_cosine_distance": self.max_cosine_distance,
            "min_speedup_ratio_observed": self.min_speedup_ratio_observed,
            "target_min_speedup_ratio": self.target_min_speedup_ratio,
        }


@dataclass(frozen=True, slots=True)
class TensorRTFP16Report:
    title: str
    report_id: str
    summary_text: str
    generated_at_utc: str
    project_root: str
    output_root: str
    source_config_path: str | None
    model_version: str | None
    metadata_path: str
    onnx_model_path: str
    engine_path: str
    source_checkpoint_path: str
    input_name: str
    output_name: str
    embedding_dim: int
    engine_size_bytes: int
    workspace_size_mib: int
    builder_optimization_level: int
    profiles: tuple[TensorRTFP16Profile, ...]
    samples: tuple[TensorRTFP16SampleResult, ...]
    promotion: TensorRTFP16PromotionState
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]
    summary: TensorRTFP16Summary

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "summary_text": self.summary_text,
            "generated_at_utc": self.generated_at_utc,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "source_config_path": self.source_config_path,
            "model_version": self.model_version,
            "metadata_path": self.metadata_path,
            "onnx_model_path": self.onnx_model_path,
            "engine_path": self.engine_path,
            "source_checkpoint_path": self.source_checkpoint_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "embedding_dim": self.embedding_dim,
            "engine_size_bytes": self.engine_size_bytes,
            "workspace_size_mib": self.workspace_size_mib,
            "builder_optimization_level": self.builder_optimization_level,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "samples": [sample.to_dict() for sample in self.samples],
            "promotion": self.promotion.to_dict(),
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenTensorRTFP16Report:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    source_config_copy_path: str | None
    promotion: TensorRTFP16PromotionState
    summary: TensorRTFP16Summary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "source_config_copy_path": self.source_config_copy_path,
            "promotion": self.promotion.to_dict(),
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "TENSORRT_FP16_REPORT_JSON_NAME",
    "TENSORRT_FP16_REPORT_MARKDOWN_NAME",
    "TensorRTFP16Profile",
    "TensorRTFP16PromotionState",
    "TensorRTFP16Report",
    "TensorRTFP16SampleResult",
    "TensorRTFP16Summary",
    "WrittenTensorRTFP16Report",
]
