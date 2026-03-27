"""Builder for reproducible INT8 feasibility decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kryptonite.deployment import resolve_project_path
from kryptonite.project import get_project_layout
from kryptonite.repro import fingerprint_path

from .int8_feasibility_config import Int8FeasibilityConfig
from .int8_feasibility_models import (
    BackendMeasurementSummary,
    CalibrationCatalogEntry,
    CalibrationSetSummary,
    Int8DeltaSummary,
    Int8FeasibilityArtifactRef,
    Int8FeasibilityCheck,
    Int8FeasibilityReport,
    Int8FeasibilitySummary,
)

_ARTIFACT_DESCRIPTIONS = {
    "calibration_catalog": (
        "Tracked calibration-set catalog used to seed representative INT8 samples."
    ),
    "model_bundle_metadata": (
        "Model-bundle metadata used to detect structural stubs and export profile."
    ),
    "onnx_model": "Promoted ONNX export that INT8 calibration would quantize from.",
    "fp16_engine": "Reference FP16 TensorRT engine that INT8 must justify replacing.",
    "int8_engine": "Candidate INT8 TensorRT engine built from the same promoted ONNX export.",
    "onnx_parity_report": (
        "Parity evidence proving the ONNX path matches the active PyTorch reference."
    ),
    "fp16_verification_report": "Reference FP16 verification quality report.",
    "fp16_stress_report": "Reference FP16 latency and memory stress report.",
    "int8_verification_report": "Candidate INT8 verification quality report.",
    "int8_stress_report": "Candidate INT8 latency and memory stress report.",
}


def build_int8_feasibility_report(
    config: Int8FeasibilityConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> Int8FeasibilityReport:
    resolved_project_root = _resolve_project_root(project_root)
    calibration_set = _build_calibration_set_summary(
        config=config,
        project_root=resolved_project_root,
    )
    artifacts = _build_artifact_refs(config=config, project_root=resolved_project_root)
    artifacts_by_label = {artifact.label: artifact for artifact in artifacts}
    model_metadata = _load_optional_json_object(
        artifacts_by_label["model_bundle_metadata"],
        expected_label="model_bundle_metadata",
    )
    model_version = _maybe_str(model_metadata.get("model_version")) if model_metadata else None
    structural_stub = _coerce_optional_bool(
        None if model_metadata is None else model_metadata.get("structural_stub")
    )
    export_profile = _extract_export_profile(model_metadata)

    fp16 = _build_backend_measurement_summary(
        label="fp16",
        verification_artifact=artifacts_by_label["fp16_verification_report"],
        stress_artifact=artifacts_by_label["fp16_stress_report"],
    )
    int8 = _build_backend_measurement_summary(
        label="int8",
        verification_artifact=artifacts_by_label["int8_verification_report"],
        stress_artifact=artifacts_by_label["int8_stress_report"],
    )
    deltas = _build_delta_summary(fp16=fp16, int8=int8)
    checks = _build_checks(
        config=config,
        artifacts_by_label=artifacts_by_label,
        calibration_set=calibration_set,
        structural_stub=structural_stub,
        fp16=fp16,
        int8=int8,
        deltas=deltas,
    )
    failed_checks = tuple(check for check in checks if not check.passed)

    resolved_output_root = resolve_project_path(str(resolved_project_root), config.output_root)
    source_config_file = None if config_path is None else Path(config_path).resolve()
    source_config_sha256 = None
    if source_config_file is not None:
        source_fingerprint = fingerprint_path(source_config_file)
        source_config_sha256 = (
            None if not bool(source_fingerprint["exists"]) else str(source_fingerprint["sha256"])
        )

    summary = Int8FeasibilitySummary(
        decision="go" if not failed_checks else "no_go",
        passed_check_count=sum(1 for check in checks if check.passed),
        failed_check_count=len(failed_checks),
        selected_input_count=calibration_set.selected_input_count,
        blocker_count=len(failed_checks),
        key_blockers=tuple(check.detail for check in failed_checks[:5]),
    )
    return Int8FeasibilityReport(
        title=config.title,
        report_id=config.report_id,
        candidate_label=config.candidate_label,
        summary_text=config.summary,
        output_root=str(resolved_output_root),
        source_config_path=None if source_config_file is None else str(source_config_file),
        source_config_sha256=source_config_sha256,
        model_version=model_version,
        structural_stub=structural_stub,
        export_profile=export_profile,
        calibration_set=calibration_set,
        fp16=fp16,
        int8=int8,
        deltas=deltas,
        artifacts=artifacts,
        checks=checks,
        validation_commands=config.validation_commands,
        notes=config.notes,
        summary=summary,
    )


def _build_calibration_set_summary(
    *,
    config: Int8FeasibilityConfig,
    project_root: Path,
) -> CalibrationSetSummary:
    catalog_path = resolve_project_path(
        str(project_root),
        config.calibration_set.source_catalog_path,
    )
    if not catalog_path.exists():
        warning = f"Calibration catalog is missing: {config.calibration_set.source_catalog_path}"
        return CalibrationSetSummary(
            source_catalog_path=config.calibration_set.source_catalog_path,
            catalog_exists=False,
            include_categories=config.calibration_set.include_categories,
            exclude_scenarios=config.calibration_set.exclude_scenarios,
            selected_inputs=(),
            selected_input_count=0,
            total_duration_seconds=None,
            category_counts={},
            short_count=0,
            mid_count=0,
            long_count=0,
            unknown_duration_count=0,
            warnings=(warning,),
        )

    warnings: list[str] = []
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        raw_inputs = payload.get("inputs", [])
        if not isinstance(raw_inputs, list):
            raise ValueError("catalog.inputs must be a list.")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warning = f"Failed to read calibration catalog: {type(exc).__name__}: {exc}"
        return CalibrationSetSummary(
            source_catalog_path=config.calibration_set.source_catalog_path,
            catalog_exists=True,
            include_categories=config.calibration_set.include_categories,
            exclude_scenarios=config.calibration_set.exclude_scenarios,
            selected_inputs=(),
            selected_input_count=0,
            total_duration_seconds=None,
            category_counts={},
            short_count=0,
            mid_count=0,
            long_count=0,
            unknown_duration_count=0,
            warnings=(warning,),
        )

    include_categories = set(config.calibration_set.include_categories)
    exclude_scenarios = set(config.calibration_set.exclude_scenarios)
    selected_inputs: list[CalibrationCatalogEntry] = []
    category_counts: dict[str, int] = {}
    short_count = 0
    mid_count = 0
    long_count = 0
    unknown_duration_count = 0
    total_duration_seconds = 0.0
    seen_categories = set()

    for item in raw_inputs:
        if not isinstance(item, dict):
            warnings.append("Ignored non-object calibration catalog entry.")
            continue
        scenario_id = str(item.get("scenario_id", "")).strip()
        category = str(item.get("category", "")).strip()
        audio_path = str(item.get("audio_path", "")).strip()
        notes = str(item.get("notes", "")).strip()
        duration_seconds = _coerce_optional_float(item.get("duration_seconds"))
        if not scenario_id or not category or not audio_path:
            warnings.append("Ignored incomplete calibration catalog entry.")
            continue
        if category not in include_categories:
            continue
        if scenario_id in exclude_scenarios:
            continue

        selected_inputs.append(
            CalibrationCatalogEntry(
                scenario_id=scenario_id,
                category=category,
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                notes=notes,
            )
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        seen_categories.add(category)
        if duration_seconds is None:
            unknown_duration_count += 1
            continue
        total_duration_seconds += duration_seconds
        if duration_seconds <= config.calibration_set.short_max_duration_seconds:
            short_count += 1
        elif duration_seconds <= config.calibration_set.mid_max_duration_seconds:
            mid_count += 1
        else:
            long_count += 1

    missing_categories = include_categories - seen_categories
    if missing_categories:
        warnings.append(
            "Calibration catalog does not cover required categories: "
            + ", ".join(sorted(missing_categories))
        )
    if not selected_inputs:
        warnings.append("Calibration selection is empty after include/exclude filters.")

    return CalibrationSetSummary(
        source_catalog_path=config.calibration_set.source_catalog_path,
        catalog_exists=True,
        include_categories=config.calibration_set.include_categories,
        exclude_scenarios=config.calibration_set.exclude_scenarios,
        selected_inputs=tuple(selected_inputs),
        selected_input_count=len(selected_inputs),
        total_duration_seconds=round(total_duration_seconds, 6) if selected_inputs else None,
        category_counts=category_counts,
        short_count=short_count,
        mid_count=mid_count,
        long_count=long_count,
        unknown_duration_count=unknown_duration_count,
        warnings=tuple(warnings),
    )


def _build_artifact_refs(
    *,
    config: Int8FeasibilityConfig,
    project_root: Path,
) -> tuple[Int8FeasibilityArtifactRef, ...]:
    required_for_measurements = any(
        value is not None
        for value in (
            config.gates.max_eer_delta,
            config.gates.max_min_dcf_delta,
            config.gates.min_latency_speedup_ratio,
            config.gates.max_process_rss_delta_mib,
            config.gates.max_cuda_allocated_delta_mib,
        )
    )
    artifact_specs = (
        (
            "calibration_catalog",
            config.calibration_set.source_catalog_path,
            True,
        ),
        (
            "model_bundle_metadata",
            config.artifacts.model_bundle_metadata_path,
            config.gates.require_non_stub_model,
        ),
        ("onnx_model", config.artifacts.onnx_model_path, True),
        ("fp16_engine", config.artifacts.fp16_engine_path, config.gates.require_fp16_engine),
        ("int8_engine", config.artifacts.int8_engine_path, config.gates.require_int8_engine),
        (
            "onnx_parity_report",
            config.artifacts.onnx_parity_report_path,
            config.gates.require_onnx_parity_report,
        ),
        (
            "fp16_verification_report",
            config.artifacts.fp16_verification_report_path,
            required_for_measurements,
        ),
        (
            "fp16_stress_report",
            config.artifacts.fp16_stress_report_path,
            required_for_measurements,
        ),
        (
            "int8_verification_report",
            config.artifacts.int8_verification_report_path,
            required_for_measurements,
        ),
        (
            "int8_stress_report",
            config.artifacts.int8_stress_report_path,
            required_for_measurements,
        ),
    )
    refs: list[Int8FeasibilityArtifactRef] = []
    for label, configured_path, required in artifact_specs:
        refs.append(
            _build_artifact_ref(
                label=label,
                configured_path=configured_path,
                required=required,
                project_root=project_root,
            )
        )
    return tuple(refs)


def _build_artifact_ref(
    *,
    label: str,
    configured_path: str | None,
    required: bool,
    project_root: Path,
) -> Int8FeasibilityArtifactRef:
    configured = configured_path or ""
    resolved = resolve_project_path(str(project_root), configured) if configured else project_root
    if not configured:
        return Int8FeasibilityArtifactRef(
            label=label,
            configured_path="",
            resolved_path=str(resolved),
            required=required,
            exists=False,
            kind="missing",
            sha256=None,
            file_count=0,
            description=_ARTIFACT_DESCRIPTIONS[label],
            error="path not configured",
        )
    fingerprint = fingerprint_path(resolved)
    return Int8FeasibilityArtifactRef(
        label=label,
        configured_path=configured,
        resolved_path=str(resolved),
        required=required,
        exists=bool(fingerprint["exists"]),
        kind=str(fingerprint["kind"]),
        sha256=None if fingerprint["sha256"] is None else str(fingerprint["sha256"]),
        file_count=int(fingerprint["file_count"]),
        description=_ARTIFACT_DESCRIPTIONS[label],
    )


def _build_backend_measurement_summary(
    *,
    label: str,
    verification_artifact: Int8FeasibilityArtifactRef,
    stress_artifact: Int8FeasibilityArtifactRef,
) -> BackendMeasurementSummary:
    warnings: list[str] = []
    verification_payload = _load_optional_json_object(
        verification_artifact,
        expected_label=verification_artifact.label,
    )
    stress_payload = _load_optional_json_object(
        stress_artifact, expected_label=stress_artifact.label
    )

    trial_count = None
    eer = None
    min_dcf = None
    mean_ms_per_audio_at_largest_batch = None
    peak_process_rss_mib = None
    peak_cuda_allocated_mib = None

    if verification_payload is None:
        warnings.append(f"{label} verification metrics are unavailable.")
    else:
        try:
            trial_count, eer, min_dcf = _extract_quality_metrics(verification_payload)
        except ValueError as exc:
            warnings.append(f"{label} verification report is invalid: {exc}")

    if stress_payload is None:
        warnings.append(f"{label} stress metrics are unavailable.")
    else:
        try:
            (
                mean_ms_per_audio_at_largest_batch,
                peak_process_rss_mib,
                peak_cuda_allocated_mib,
            ) = _extract_stress_metrics(stress_payload)
        except ValueError as exc:
            warnings.append(f"{label} stress report is invalid: {exc}")

    metrics_available = (
        trial_count is not None
        and eer is not None
        and min_dcf is not None
        and mean_ms_per_audio_at_largest_batch is not None
    )
    return BackendMeasurementSummary(
        label=label,
        verification_report_path=verification_artifact.configured_path or None,
        stress_report_path=stress_artifact.configured_path or None,
        metrics_available=metrics_available,
        trial_count=trial_count,
        eer=eer,
        min_dcf=min_dcf,
        mean_ms_per_audio_at_largest_batch=mean_ms_per_audio_at_largest_batch,
        peak_process_rss_mib=peak_process_rss_mib,
        peak_cuda_allocated_mib=peak_cuda_allocated_mib,
        warnings=tuple(warnings),
    )


def _build_delta_summary(
    *,
    fp16: BackendMeasurementSummary,
    int8: BackendMeasurementSummary,
) -> Int8DeltaSummary:
    return Int8DeltaSummary(
        eer_delta=_delta(fp16.eer, int8.eer),
        min_dcf_delta=_delta(fp16.min_dcf, int8.min_dcf),
        latency_speedup_ratio=_speedup_ratio(
            baseline=fp16.mean_ms_per_audio_at_largest_batch,
            candidate=int8.mean_ms_per_audio_at_largest_batch,
        ),
        process_rss_delta_mib=_delta(fp16.peak_process_rss_mib, int8.peak_process_rss_mib),
        cuda_allocated_delta_mib=_delta(
            fp16.peak_cuda_allocated_mib,
            int8.peak_cuda_allocated_mib,
        ),
    )


def _build_checks(
    *,
    config: Int8FeasibilityConfig,
    artifacts_by_label: dict[str, Int8FeasibilityArtifactRef],
    calibration_set: CalibrationSetSummary,
    structural_stub: bool | None,
    fp16: BackendMeasurementSummary,
    int8: BackendMeasurementSummary,
    deltas: Int8DeltaSummary,
) -> tuple[Int8FeasibilityCheck, ...]:
    checks: list[Int8FeasibilityCheck] = []
    checks.append(
        _build_boolean_check(
            name="calibration catalog selection is non-empty",
            passed=calibration_set.selected_input_count > 0,
            success_detail=(
                f"Selected {calibration_set.selected_input_count} calibration inputs across "
                f"{len(calibration_set.category_counts)} categories."
            ),
            failure_detail="Representative calibration-set selection is empty.",
        )
    )
    checks.append(
        _build_boolean_check(
            name="calibration catalog covers requested categories",
            passed=not any(
                "does not cover required categories" in warning
                for warning in calibration_set.warnings
            ),
            success_detail="Calibration catalog covers every requested category.",
            failure_detail="Calibration catalog is missing one or more requested categories.",
        )
    )

    if config.gates.require_non_stub_model:
        checks.append(
            _build_boolean_check(
                name="model bundle is not a structural stub",
                passed=structural_stub is False,
                success_detail="Model bundle metadata marks the export surface as non-stub.",
                failure_detail=(
                    "Model bundle is still structural-stub or missing metadata, so INT8 would "
                    "be quantizing a non-production surface."
                ),
            )
        )
    if config.gates.require_fp16_engine:
        checks.append(
            _build_artifact_presence_check(
                name="reference FP16 engine exists",
                artifact=artifacts_by_label["fp16_engine"],
            )
        )
    if config.gates.require_onnx_parity_report:
        checks.append(
            _build_artifact_presence_check(
                name="ONNX parity report exists",
                artifact=artifacts_by_label["onnx_parity_report"],
            )
        )
    if config.gates.require_int8_engine:
        checks.append(
            _build_artifact_presence_check(
                name="candidate INT8 engine exists",
                artifact=artifacts_by_label["int8_engine"],
            )
        )

    checks.append(
        _build_boolean_check(
            name="FP16 metrics are available",
            passed=fp16.metrics_available,
            success_detail="FP16 verification and stress metrics are available.",
            failure_detail="FP16 verification/stress metrics are missing or invalid.",
        )
    )
    checks.append(
        _build_boolean_check(
            name="INT8 metrics are available",
            passed=int8.metrics_available,
            success_detail="INT8 verification and stress metrics are available.",
            failure_detail="INT8 verification/stress metrics are missing or invalid.",
        )
    )

    if config.gates.max_eer_delta is not None:
        checks.append(
            _build_optional_threshold_check(
                name="INT8 EER degradation stays within gate",
                observed=deltas.eer_delta,
                threshold=config.gates.max_eer_delta,
                comparator=lambda observed, threshold: observed <= threshold,
                format_observed=lambda value: f"EER delta={value:.6f}",
                gate_label=f"max {config.gates.max_eer_delta:.6f}",
            )
        )
    if config.gates.max_min_dcf_delta is not None:
        checks.append(
            _build_optional_threshold_check(
                name="INT8 minDCF degradation stays within gate",
                observed=deltas.min_dcf_delta,
                threshold=config.gates.max_min_dcf_delta,
                comparator=lambda observed, threshold: observed <= threshold,
                format_observed=lambda value: f"minDCF delta={value:.6f}",
                gate_label=f"max {config.gates.max_min_dcf_delta:.6f}",
            )
        )
    if config.gates.min_latency_speedup_ratio is not None:
        checks.append(
            _build_optional_threshold_check(
                name="INT8 latency speedup justifies calibration cost",
                observed=deltas.latency_speedup_ratio,
                threshold=config.gates.min_latency_speedup_ratio,
                comparator=lambda observed, threshold: observed >= threshold,
                format_observed=lambda value: f"speedup={value:.3f}x",
                gate_label=f"min {config.gates.min_latency_speedup_ratio:.3f}x",
            )
        )
    if config.gates.max_process_rss_delta_mib is not None:
        checks.append(
            _build_optional_threshold_check(
                name="INT8 process RSS increase stays within gate",
                observed=deltas.process_rss_delta_mib,
                threshold=config.gates.max_process_rss_delta_mib,
                comparator=lambda observed, threshold: observed <= threshold,
                format_observed=lambda value: f"RSS delta={value:.3f} MiB",
                gate_label=f"max {config.gates.max_process_rss_delta_mib:.3f} MiB",
            )
        )
    if config.gates.max_cuda_allocated_delta_mib is not None:
        checks.append(
            _build_optional_threshold_check(
                name="INT8 CUDA allocation increase stays within gate",
                observed=deltas.cuda_allocated_delta_mib,
                threshold=config.gates.max_cuda_allocated_delta_mib,
                comparator=lambda observed, threshold: observed <= threshold,
                format_observed=lambda value: f"CUDA delta={value:.3f} MiB",
                gate_label=f"max {config.gates.max_cuda_allocated_delta_mib:.3f} MiB",
            )
        )
    return tuple(checks)


def _build_artifact_presence_check(
    *,
    name: str,
    artifact: Int8FeasibilityArtifactRef,
) -> Int8FeasibilityCheck:
    return _build_boolean_check(
        name=name,
        passed=artifact.exists,
        success_detail=f"Artifact exists at {artifact.configured_path}.",
        failure_detail=(f"{name} is missing: {artifact.configured_path or '<not configured>'}."),
    )


def _build_optional_threshold_check(
    *,
    name: str,
    observed: float | None,
    threshold: float,
    comparator: Any,
    format_observed: Any,
    gate_label: str,
) -> Int8FeasibilityCheck:
    if observed is None:
        return Int8FeasibilityCheck(
            name=name,
            passed=False,
            detail=f"Observed value is unavailable; gate requires {gate_label}.",
        )
    passed = bool(comparator(observed, threshold))
    detail = f"{format_observed(observed)} against gate {gate_label}."
    return Int8FeasibilityCheck(name=name, passed=passed, detail=detail)


def _build_boolean_check(
    *,
    name: str,
    passed: bool,
    success_detail: str,
    failure_detail: str,
) -> Int8FeasibilityCheck:
    return Int8FeasibilityCheck(
        name=name,
        passed=passed,
        detail=success_detail if passed else failure_detail,
    )


def _load_optional_json_object(
    artifact: Int8FeasibilityArtifactRef,
    *,
    expected_label: str,
) -> dict[str, Any] | None:
    if not artifact.exists:
        return None
    path = Path(artifact.resolved_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{expected_label} must be a JSON object.")
    return {str(key): value for key, value in payload.items()}


def _extract_quality_metrics(payload: dict[str, Any]) -> tuple[int, float, float]:
    summary = _require_mapping(payload.get("summary"), "verification.summary")
    metrics = _require_mapping(summary.get("metrics"), "verification.summary.metrics")
    return (
        int(metrics["trial_count"]),
        float(metrics["eer"]),
        float(metrics["min_dcf"]),
    )


def _extract_stress_metrics(payload: dict[str, Any]) -> tuple[float, float | None, float | None]:
    batch_bursts = payload.get("batch_bursts", [])
    if not isinstance(batch_bursts, list):
        raise ValueError("stress.batch_bursts must be a list.")
    successful_bursts = [
        burst
        for burst in batch_bursts
        if isinstance(burst, dict) and str(burst.get("status", "")).lower() == "passed"
    ]
    largest_burst = max(
        successful_bursts,
        key=lambda burst: int(burst.get("batch_size", 0)),
        default=None,
    )
    if largest_burst is None:
        raise ValueError("stress report does not contain a passed batch burst.")
    memory_payload = payload.get("memory", {})
    memory = _require_mapping(memory_payload, "stress.memory")
    return (
        float(largest_burst["mean_ms_per_audio"]),
        _coerce_optional_float(memory.get("peak_process_rss_mib")),
        _coerce_optional_float(memory.get("peak_cuda_allocated_mib")),
    )


def _extract_export_profile(model_metadata: dict[str, Any] | None) -> str | None:
    if model_metadata is None:
        return None
    embedded_boundary = model_metadata.get("export_boundary")
    if not isinstance(embedded_boundary, dict):
        return None
    return _maybe_str(embedded_boundary.get("export_profile"))


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(right - left, 6)


def _speedup_ratio(*, baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate <= 0:
        return None
    return round(baseline / candidate, 6)


def _require_mapping(raw: object, field_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return {str(key): value for key, value in raw.items()}


def _coerce_optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("numeric values must not be booleans.")
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _coerce_optional_bool(raw: object) -> bool | None:
    if raw is None:
        return None
    if not isinstance(raw, bool):
        return None
    return raw


def _maybe_str(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    return stripped or None


def _resolve_project_root(project_root: Path | str | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return get_project_layout().root


__all__ = ["build_int8_feasibility_report"]
