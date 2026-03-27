"""Datamodels for reproducible INT8 feasibility reports."""

from __future__ import annotations

from dataclasses import dataclass

INT8_FEASIBILITY_JSON_NAME = "int8_feasibility.json"
INT8_FEASIBILITY_MARKDOWN_NAME = "int8_feasibility.md"


@dataclass(frozen=True, slots=True)
class Int8FeasibilityArtifactRef:
    label: str
    configured_path: str
    resolved_path: str
    required: bool
    exists: bool
    kind: str
    sha256: str | None
    file_count: int
    description: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "configured_path": self.configured_path,
            "resolved_path": self.resolved_path,
            "required": self.required,
            "exists": self.exists,
            "kind": self.kind,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "description": self.description,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CalibrationCatalogEntry:
    scenario_id: str
    category: str
    audio_path: str
    duration_seconds: float | None
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CalibrationSetSummary:
    source_catalog_path: str
    catalog_exists: bool
    include_categories: tuple[str, ...]
    exclude_scenarios: tuple[str, ...]
    selected_inputs: tuple[CalibrationCatalogEntry, ...]
    selected_input_count: int
    total_duration_seconds: float | None
    category_counts: dict[str, int]
    short_count: int
    mid_count: int
    long_count: int
    unknown_duration_count: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_catalog_path": self.source_catalog_path,
            "catalog_exists": self.catalog_exists,
            "include_categories": list(self.include_categories),
            "exclude_scenarios": list(self.exclude_scenarios),
            "selected_inputs": [item.to_dict() for item in self.selected_inputs],
            "selected_input_count": self.selected_input_count,
            "total_duration_seconds": self.total_duration_seconds,
            "category_counts": dict(self.category_counts),
            "short_count": self.short_count,
            "mid_count": self.mid_count,
            "long_count": self.long_count,
            "unknown_duration_count": self.unknown_duration_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class BackendMeasurementSummary:
    label: str
    verification_report_path: str | None
    stress_report_path: str | None
    metrics_available: bool
    trial_count: int | None
    eer: float | None
    min_dcf: float | None
    mean_ms_per_audio_at_largest_batch: float | None
    peak_process_rss_mib: float | None
    peak_cuda_allocated_mib: float | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "verification_report_path": self.verification_report_path,
            "stress_report_path": self.stress_report_path,
            "metrics_available": self.metrics_available,
            "trial_count": self.trial_count,
            "eer": self.eer,
            "min_dcf": self.min_dcf,
            "mean_ms_per_audio_at_largest_batch": self.mean_ms_per_audio_at_largest_batch,
            "peak_process_rss_mib": self.peak_process_rss_mib,
            "peak_cuda_allocated_mib": self.peak_cuda_allocated_mib,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class Int8DeltaSummary:
    eer_delta: float | None
    min_dcf_delta: float | None
    latency_speedup_ratio: float | None
    process_rss_delta_mib: float | None
    cuda_allocated_delta_mib: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "eer_delta": self.eer_delta,
            "min_dcf_delta": self.min_dcf_delta,
            "latency_speedup_ratio": self.latency_speedup_ratio,
            "process_rss_delta_mib": self.process_rss_delta_mib,
            "cuda_allocated_delta_mib": self.cuda_allocated_delta_mib,
        }


@dataclass(frozen=True, slots=True)
class Int8FeasibilityCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class Int8FeasibilitySummary:
    decision: str
    passed_check_count: int
    failed_check_count: int
    selected_input_count: int
    blocker_count: int
    key_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "passed_check_count": self.passed_check_count,
            "failed_check_count": self.failed_check_count,
            "selected_input_count": self.selected_input_count,
            "blocker_count": self.blocker_count,
            "key_blockers": list(self.key_blockers),
        }


@dataclass(frozen=True, slots=True)
class Int8FeasibilityReport:
    title: str
    report_id: str
    candidate_label: str
    summary_text: str
    output_root: str
    source_config_path: str | None
    source_config_sha256: str | None
    model_version: str | None
    structural_stub: bool | None
    export_profile: str | None
    calibration_set: CalibrationSetSummary
    fp16: BackendMeasurementSummary
    int8: BackendMeasurementSummary
    deltas: Int8DeltaSummary
    artifacts: tuple[Int8FeasibilityArtifactRef, ...]
    checks: tuple[Int8FeasibilityCheck, ...]
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]
    summary: Int8FeasibilitySummary

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "candidate_label": self.candidate_label,
            "summary_text": self.summary_text,
            "output_root": self.output_root,
            "source_config_path": self.source_config_path,
            "source_config_sha256": self.source_config_sha256,
            "model_version": self.model_version,
            "structural_stub": self.structural_stub,
            "export_profile": self.export_profile,
            "calibration_set": self.calibration_set.to_dict(),
            "fp16": self.fp16.to_dict(),
            "int8": self.int8.to_dict(),
            "deltas": self.deltas.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "checks": [check.to_dict() for check in self.checks],
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenInt8FeasibilityReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    source_config_copy_path: str | None
    summary: Int8FeasibilitySummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "source_config_copy_path": self.source_config_copy_path,
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "CalibrationCatalogEntry",
    "CalibrationSetSummary",
    "BackendMeasurementSummary",
    "INT8_FEASIBILITY_JSON_NAME",
    "INT8_FEASIBILITY_MARKDOWN_NAME",
    "Int8DeltaSummary",
    "Int8FeasibilityArtifactRef",
    "Int8FeasibilityCheck",
    "Int8FeasibilityReport",
    "Int8FeasibilitySummary",
    "WrittenInt8FeasibilityReport",
]
