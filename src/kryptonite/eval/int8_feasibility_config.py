"""Typed config loader for reproducible INT8 feasibility reports."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from kryptonite.common.parsing import (
    coerce_optional_float as _coerce_optional_float,
    coerce_optional_string as _coerce_optional_string,
    coerce_required_float as _coerce_required_float,
    coerce_string_list as _coerce_string_list,
    coerce_table as _coerce_table,
)


@dataclass(frozen=True, slots=True)
class Int8FeasibilityArtifactsConfig:
    model_bundle_metadata_path: str | None
    onnx_model_path: str | None
    fp16_engine_path: str | None
    int8_engine_path: str | None
    onnx_parity_report_path: str | None
    fp16_verification_report_path: str | None
    fp16_stress_report_path: str | None
    int8_verification_report_path: str | None
    int8_stress_report_path: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Int8CalibrationSetConfig:
    source_catalog_path: str
    include_categories: tuple[str, ...]
    exclude_scenarios: tuple[str, ...]
    short_max_duration_seconds: float
    mid_max_duration_seconds: float
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_catalog_path.strip():
            raise ValueError("calibration_set.source_catalog_path must not be empty.")
        if not self.include_categories:
            raise ValueError("calibration_set.include_categories must not be empty.")
        if self.short_max_duration_seconds <= 0:
            raise ValueError("calibration_set.short_max_duration_seconds must be positive.")
        if self.mid_max_duration_seconds <= self.short_max_duration_seconds:
            raise ValueError(
                "calibration_set.mid_max_duration_seconds must be larger than "
                "short_max_duration_seconds."
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["include_categories"] = list(self.include_categories)
        payload["exclude_scenarios"] = list(self.exclude_scenarios)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class Int8FeasibilityGatesConfig:
    require_non_stub_model: bool
    require_fp16_engine: bool
    require_onnx_parity_report: bool
    require_int8_engine: bool
    max_eer_delta: float | None
    max_min_dcf_delta: float | None
    min_latency_speedup_ratio: float | None
    max_process_rss_delta_mib: float | None
    max_cuda_allocated_delta_mib: float | None

    def __post_init__(self) -> None:
        _validate_non_negative(self.max_eer_delta, "gates.max_eer_delta")
        _validate_non_negative(self.max_min_dcf_delta, "gates.max_min_dcf_delta")
        _validate_positive(self.min_latency_speedup_ratio, "gates.min_latency_speedup_ratio")
        _validate_non_negative(
            self.max_process_rss_delta_mib,
            "gates.max_process_rss_delta_mib",
        )
        _validate_non_negative(
            self.max_cuda_allocated_delta_mib,
            "gates.max_cuda_allocated_delta_mib",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Int8FeasibilityConfig:
    title: str
    report_id: str
    candidate_label: str
    summary: str
    output_root: str
    artifacts: Int8FeasibilityArtifactsConfig
    calibration_set: Int8CalibrationSetConfig
    gates: Int8FeasibilityGatesConfig
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty.")
        if not self.report_id.strip():
            raise ValueError("report_id must not be empty.")
        if not self.candidate_label.strip():
            raise ValueError("candidate_label must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "candidate_label": self.candidate_label,
            "summary": self.summary,
            "output_root": self.output_root,
            "artifacts": self.artifacts.to_dict(),
            "calibration_set": self.calibration_set.to_dict(),
            "gates": self.gates.to_dict(),
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
        }


def load_int8_feasibility_config(*, config_path: Path | str) -> Int8FeasibilityConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    report_id = str(raw.get("report_id", "")).strip()
    output_root = str(raw.get("output_root", "")).strip() or (
        f"artifacts/release-decisions/{report_id}"
    )
    artifacts = _coerce_table(raw.get("artifacts"), "artifacts")
    calibration_set = _coerce_table(raw.get("calibration_set"), "calibration_set")
    gates = _coerce_table(raw.get("gates"), "gates")
    return Int8FeasibilityConfig(
        title=str(raw.get("title", "")).strip(),
        report_id=report_id,
        candidate_label=str(raw.get("candidate_label", "")).strip(),
        summary=str(raw.get("summary", "")).strip(),
        output_root=output_root,
        artifacts=Int8FeasibilityArtifactsConfig(
            model_bundle_metadata_path=_coerce_optional_string(
                artifacts.get("model_bundle_metadata_path")
            ),
            onnx_model_path=_coerce_optional_string(artifacts.get("onnx_model_path")),
            fp16_engine_path=_coerce_optional_string(artifacts.get("fp16_engine_path")),
            int8_engine_path=_coerce_optional_string(artifacts.get("int8_engine_path")),
            onnx_parity_report_path=_coerce_optional_string(
                artifacts.get("onnx_parity_report_path")
            ),
            fp16_verification_report_path=_coerce_optional_string(
                artifacts.get("fp16_verification_report_path")
            ),
            fp16_stress_report_path=_coerce_optional_string(
                artifacts.get("fp16_stress_report_path")
            ),
            int8_verification_report_path=_coerce_optional_string(
                artifacts.get("int8_verification_report_path")
            ),
            int8_stress_report_path=_coerce_optional_string(
                artifacts.get("int8_stress_report_path")
            ),
        ),
        calibration_set=Int8CalibrationSetConfig(
            source_catalog_path=str(calibration_set.get("source_catalog_path", "")).strip(),
            include_categories=tuple(
                _coerce_string_list(
                    calibration_set.get("include_categories"),
                    "calibration_set.include_categories",
                )
            ),
            exclude_scenarios=tuple(
                _coerce_string_list(
                    calibration_set.get("exclude_scenarios", []),
                    "calibration_set.exclude_scenarios",
                )
            ),
            short_max_duration_seconds=_coerce_required_float(
                calibration_set.get("short_max_duration_seconds", 1.0),
                "calibration_set.short_max_duration_seconds",
            ),
            mid_max_duration_seconds=_coerce_required_float(
                calibration_set.get("mid_max_duration_seconds", 4.0),
                "calibration_set.mid_max_duration_seconds",
            ),
            notes=tuple(
                _coerce_string_list(calibration_set.get("notes", []), "calibration_set.notes")
            ),
        ),
        gates=Int8FeasibilityGatesConfig(
            require_non_stub_model=bool(gates.get("require_non_stub_model", True)),
            require_fp16_engine=bool(gates.get("require_fp16_engine", True)),
            require_onnx_parity_report=bool(gates.get("require_onnx_parity_report", True)),
            require_int8_engine=bool(gates.get("require_int8_engine", True)),
            max_eer_delta=_coerce_optional_float(gates.get("max_eer_delta")),
            max_min_dcf_delta=_coerce_optional_float(gates.get("max_min_dcf_delta")),
            min_latency_speedup_ratio=_coerce_optional_float(
                gates.get("min_latency_speedup_ratio")
            ),
            max_process_rss_delta_mib=_coerce_optional_float(
                gates.get("max_process_rss_delta_mib")
            ),
            max_cuda_allocated_delta_mib=_coerce_optional_float(
                gates.get("max_cuda_allocated_delta_mib")
            ),
        ),
        validation_commands=tuple(
            _coerce_string_list(raw.get("validation_commands", []), "validation_commands")
        ),
        notes=tuple(_coerce_string_list(raw.get("notes", []), "notes")),
    )
def _validate_non_negative(value: float | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


def _validate_positive(value: float | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be positive when provided.")


__all__ = [
    "Int8CalibrationSetConfig",
    "Int8FeasibilityArtifactsConfig",
    "Int8FeasibilityConfig",
    "Int8FeasibilityGatesConfig",
    "load_int8_feasibility_config",
]
