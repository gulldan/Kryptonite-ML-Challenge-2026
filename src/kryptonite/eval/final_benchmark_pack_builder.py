"""Builder for the self-contained final benchmark pack."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any, cast

from kryptonite.deployment import resolve_project_path
from kryptonite.project import get_project_layout

from .final_benchmark_pack_config import (
    FinalBenchmarkCandidateConfig,
    FinalBenchmarkPackConfig,
)
from .final_benchmark_pack_models import (
    BenchmarkPackArtifactRef,
    CandidateBenchmarkSummary,
    CandidateBundleSummary,
    CandidateOperationalSummary,
    CandidateQualitySummary,
    FinalBenchmarkPackReport,
    FinalBenchmarkPackSummary,
    PairwiseBenchmarkComparison,
)


def build_final_benchmark_pack(
    config: FinalBenchmarkPackConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> FinalBenchmarkPackReport:
    """Build one self-contained release benchmark pack and stage source copies."""

    resolved_project_root = _resolve_project_root(project_root)
    output_root = resolve_project_path(str(resolved_project_root), config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_config_artifact = None
    if config_path is not None:
        source_config_artifact = _copy_artifact(
            source_path=Path(config_path).resolve(),
            destination_path=output_root / "sources" / "final_benchmark_pack_config.toml",
            output_root=output_root,
            kind="pack_config",
        )

    candidates = tuple(
        _build_candidate_summary(
            candidate_config=candidate_config,
            project_root=resolved_project_root,
            output_root=output_root,
        )
        for candidate_config in config.candidates
    )
    pairwise_comparisons = _build_pairwise_comparisons(candidates)
    summary = _build_pack_summary(candidates=candidates, pairwise_comparisons=pairwise_comparisons)

    return FinalBenchmarkPackReport(
        title=config.title,
        summary_text=config.summary,
        output_root=str(output_root),
        source_config_artifact=source_config_artifact,
        notes=config.notes,
        summary=summary,
        candidates=candidates,
        pairwise_comparisons=pairwise_comparisons,
    )


def _build_candidate_summary(
    *,
    candidate_config: FinalBenchmarkCandidateConfig,
    project_root: Path,
    output_root: Path,
) -> CandidateBenchmarkSummary:
    candidate_source_root = output_root / "candidates" / candidate_config.candidate_id / "sources"
    candidate_source_root.mkdir(parents=True, exist_ok=True)

    verification_report_path = _resolve_path(
        project_root, candidate_config.verification_report_path
    )
    threshold_calibration_path = (
        None
        if candidate_config.threshold_calibration_path is None
        else _resolve_path(project_root, candidate_config.threshold_calibration_path)
    )
    stress_report_path = _resolve_path(project_root, candidate_config.stress_report_path)
    model_bundle_metadata_path = _resolve_path(
        project_root,
        candidate_config.model_bundle_metadata_path,
    )
    export_boundary_path = (
        None
        if candidate_config.export_boundary_path is None
        else _resolve_path(project_root, candidate_config.export_boundary_path)
    )

    verification_report_payload = _load_json_object(verification_report_path)
    threshold_calibration_payload = (
        None
        if threshold_calibration_path is None
        else _load_json_object(threshold_calibration_path)
    )
    stress_report_payload = _load_json_object(stress_report_path)
    model_bundle_metadata_payload = _load_json_object(model_bundle_metadata_path)
    export_boundary_payload = (
        None if export_boundary_path is None else _load_json_object(export_boundary_path)
    )

    source_artifacts: list[BenchmarkPackArtifactRef] = [
        _copy_artifact(
            source_path=verification_report_path,
            destination_path=candidate_source_root / "verification_eval_report.json",
            output_root=output_root,
            kind="verification_report",
        ),
        _copy_artifact(
            source_path=stress_report_path,
            destination_path=candidate_source_root / "inference_stress_report.json",
            output_root=output_root,
            kind="stress_report",
        ),
        _copy_artifact(
            source_path=model_bundle_metadata_path,
            destination_path=candidate_source_root / "model_bundle_metadata.json",
            output_root=output_root,
            kind="model_bundle_metadata",
        ),
    ]
    if threshold_calibration_path is not None:
        source_artifacts.append(
            _copy_artifact(
                source_path=threshold_calibration_path,
                destination_path=candidate_source_root / "verification_threshold_calibration.json",
                output_root=output_root,
                kind="threshold_calibration",
            )
        )
    if export_boundary_path is not None:
        source_artifacts.append(
            _copy_artifact(
                source_path=export_boundary_path,
                destination_path=candidate_source_root / "export_boundary.json",
                output_root=output_root,
                kind="export_boundary",
            )
        )
    for index, configured_path in enumerate(candidate_config.config_paths, start=1):
        source_path = _resolve_path(project_root, configured_path)
        source_artifacts.append(
            _copy_artifact(
                source_path=source_path,
                destination_path=candidate_source_root / f"config_{index:02d}_{source_path.name}",
                output_root=output_root,
                kind="config",
            )
        )
    for index, configured_path in enumerate(candidate_config.supporting_paths, start=1):
        source_path = _resolve_path(project_root, configured_path)
        source_artifacts.append(
            _copy_artifact(
                source_path=source_path,
                destination_path=candidate_source_root
                / f"supporting_{index:02d}_{source_path.name}",
                output_root=output_root,
                kind="supporting",
            )
        )

    return CandidateBenchmarkSummary(
        candidate_id=candidate_config.candidate_id,
        label=candidate_config.label,
        family=candidate_config.family,
        quality=_extract_quality_summary(
            verification_report_payload,
            threshold_calibration_payload=threshold_calibration_payload,
        ),
        operational=_extract_operational_summary(stress_report_payload),
        bundle=_extract_bundle_summary(
            model_bundle_metadata_payload,
            export_boundary_payload=export_boundary_payload,
        ),
        source_artifacts=tuple(source_artifacts),
        notes=candidate_config.notes,
    )


def _extract_quality_summary(
    verification_report_payload: dict[str, Any],
    *,
    threshold_calibration_payload: dict[str, Any] | None,
) -> CandidateQualitySummary:
    summary_payload = _require_mapping(
        verification_report_payload.get("summary"), "verification.summary"
    )
    metrics = _require_mapping(summary_payload.get("metrics"), "verification.summary.metrics")
    score_statistics = _require_mapping(
        summary_payload.get("score_statistics"),
        "verification.summary.score_statistics",
    )
    thresholds = _extract_threshold_map(threshold_calibration_payload)
    return CandidateQualitySummary(
        trial_count=int(metrics["trial_count"]),
        positive_count=int(metrics["positive_count"]),
        negative_count=int(metrics["negative_count"]),
        eer=float(metrics["eer"]),
        eer_threshold=float(metrics["eer_threshold"]),
        min_dcf=float(metrics["min_dcf"]),
        min_dcf_threshold=float(metrics["min_dcf_threshold"]),
        mean_positive_score=_maybe_float(score_statistics.get("mean_positive_score")),
        mean_negative_score=_maybe_float(score_statistics.get("mean_negative_score")),
        score_gap=_maybe_float(score_statistics.get("score_gap")),
        balanced_threshold=thresholds.get("balanced"),
        demo_threshold=thresholds.get("demo"),
        production_threshold=thresholds.get("production"),
    )


def _extract_operational_summary(
    stress_report_payload: dict[str, Any],
) -> CandidateOperationalSummary:
    summary_payload = _require_mapping(stress_report_payload.get("summary"), "stress.summary")
    hard_limits = _require_mapping(stress_report_payload.get("hard_limits"), "stress.hard_limits")
    memory_payload = stress_report_payload.get("memory")
    memory = {} if memory_payload is None else _require_mapping(memory_payload, "stress.memory")

    batch_bursts = stress_report_payload.get("batch_bursts", [])
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

    return CandidateOperationalSummary(
        stress_passed=bool(summary_payload.get("passed", False)),
        validated_stage=str(hard_limits.get("validated_stage", "")),
        largest_validated_batch_size=int(hard_limits.get("largest_validated_batch_size", 0)),
        mean_ms_per_audio_at_largest_batch=(
            None if largest_burst is None else _maybe_float(largest_burst.get("mean_ms_per_audio"))
        ),
        max_validated_duration_seconds=_maybe_float(
            hard_limits.get("max_validated_duration_seconds")
        ),
        largest_validated_total_chunk_count=int(
            hard_limits.get("largest_validated_total_chunk_count", 0)
        ),
        peak_process_rss_mib=_maybe_float(memory.get("peak_process_rss_mib")),
        peak_process_rss_delta_mib=_maybe_float(memory.get("peak_process_rss_delta_mib")),
        peak_cuda_allocated_mib=_maybe_float(memory.get("peak_cuda_allocated_mib")),
        peak_cuda_reserved_mib=_maybe_float(memory.get("peak_cuda_reserved_mib")),
    )


def _extract_bundle_summary(
    model_bundle_metadata_payload: dict[str, Any],
    *,
    export_boundary_payload: dict[str, Any] | None,
) -> CandidateBundleSummary:
    embedded_boundary = model_bundle_metadata_payload.get("export_boundary")
    resolved_boundary = (
        _require_mapping(embedded_boundary, "model_bundle.export_boundary")
        if isinstance(embedded_boundary, dict)
        else export_boundary_payload
    )
    return CandidateBundleSummary(
        model_version=str(model_bundle_metadata_payload.get("model_version", "unknown")),
        model_file=_maybe_str(model_bundle_metadata_payload.get("model_file")),
        input_name=_maybe_str(model_bundle_metadata_payload.get("input_name")),
        output_name=_maybe_str(model_bundle_metadata_payload.get("output_name")),
        sample_rate_hz=_maybe_int(model_bundle_metadata_payload.get("sample_rate_hz")),
        enrollment_cache_compatibility_id=_maybe_str(
            model_bundle_metadata_payload.get("enrollment_cache_compatibility_id")
        ),
        export_boundary=(
            None if resolved_boundary is None else _maybe_str(resolved_boundary.get("boundary"))
        ),
        export_profile=(
            None
            if resolved_boundary is None
            else _maybe_str(resolved_boundary.get("export_profile"))
        ),
        frontend_location=(
            None
            if resolved_boundary is None
            else _maybe_str(resolved_boundary.get("frontend_location"))
        ),
    )


def _extract_threshold_map(
    threshold_calibration_payload: dict[str, Any] | None,
) -> dict[str, float | None]:
    if threshold_calibration_payload is None:
        return {}
    profiles = threshold_calibration_payload.get("global_profiles", [])
    if not isinstance(profiles, list):
        raise ValueError("threshold_calibration.global_profiles must be a list.")
    threshold_map: dict[str, float | None] = {}
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        name = str(profile.get("name", "")).strip()
        if not name:
            continue
        threshold_map[name] = _maybe_float(profile.get("threshold"))
    return threshold_map


def _build_pairwise_comparisons(
    candidates: tuple[CandidateBenchmarkSummary, ...],
) -> tuple[PairwiseBenchmarkComparison, ...]:
    comparisons: list[PairwiseBenchmarkComparison] = []
    for left, right in combinations(candidates, 2):
        comparisons.append(
            PairwiseBenchmarkComparison(
                left_candidate_id=left.candidate_id,
                right_candidate_id=right.candidate_id,
                eer_delta_right_minus_left=round(right.quality.eer - left.quality.eer, 6),
                min_dcf_delta_right_minus_left=round(
                    right.quality.min_dcf - left.quality.min_dcf,
                    6,
                ),
                latency_delta_ms_per_audio_right_minus_left=_delta(
                    left.operational.mean_ms_per_audio_at_largest_batch,
                    right.operational.mean_ms_per_audio_at_largest_batch,
                    digits=6,
                ),
                process_rss_delta_mib_right_minus_left=_delta(
                    left.operational.peak_process_rss_mib,
                    right.operational.peak_process_rss_mib,
                    digits=3,
                ),
                cuda_allocated_delta_mib_right_minus_left=_delta(
                    left.operational.peak_cuda_allocated_mib,
                    right.operational.peak_cuda_allocated_mib,
                    digits=3,
                ),
                better_quality_candidate_id=_pick_lower(
                    left.candidate_id,
                    left.quality.eer,
                    right.candidate_id,
                    right.quality.eer,
                    tie_break_left=left.quality.min_dcf,
                    tie_break_right=right.quality.min_dcf,
                ),
                lower_latency_candidate_id=_pick_lower(
                    left.candidate_id,
                    left.operational.mean_ms_per_audio_at_largest_batch,
                    right.candidate_id,
                    right.operational.mean_ms_per_audio_at_largest_batch,
                ),
                lower_process_rss_candidate_id=_pick_lower(
                    left.candidate_id,
                    left.operational.peak_process_rss_mib,
                    right.candidate_id,
                    right.operational.peak_process_rss_mib,
                ),
                lower_cuda_allocated_candidate_id=_pick_lower(
                    left.candidate_id,
                    left.operational.peak_cuda_allocated_mib,
                    right.candidate_id,
                    right.operational.peak_cuda_allocated_mib,
                ),
            )
        )
    return tuple(comparisons)


def _build_pack_summary(
    *,
    candidates: tuple[CandidateBenchmarkSummary, ...],
    pairwise_comparisons: tuple[PairwiseBenchmarkComparison, ...],
) -> FinalBenchmarkPackSummary:
    return FinalBenchmarkPackSummary(
        candidate_count=len(candidates),
        pairwise_comparison_count=len(pairwise_comparisons),
        best_eer_candidate_id=_best_candidate_id(
            candidates, lambda candidate: candidate.quality.eer
        ),
        best_min_dcf_candidate_id=_best_candidate_id(
            candidates,
            lambda candidate: candidate.quality.min_dcf,
        ),
        lowest_latency_candidate_id=_best_candidate_id(
            candidates,
            lambda candidate: candidate.operational.mean_ms_per_audio_at_largest_batch,
        ),
        lowest_process_rss_candidate_id=_best_candidate_id(
            candidates,
            lambda candidate: candidate.operational.peak_process_rss_mib,
        ),
        lowest_cuda_allocated_candidate_id=_best_candidate_id(
            candidates,
            lambda candidate: candidate.operational.peak_cuda_allocated_mib,
        ),
    )


def _best_candidate_id(
    candidates: tuple[CandidateBenchmarkSummary, ...],
    metric_getter: Callable[[CandidateBenchmarkSummary], float | None],
) -> str | None:
    ranked = [
        (metric_getter(candidate), candidate.candidate_id)
        for candidate in candidates
        if metric_getter(candidate) is not None
    ]
    if not ranked:
        return None
    return min(ranked, key=lambda item: (item[0], item[1]))[1]


def _resolve_project_root(project_root: Path | str | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return get_project_layout().root.resolve()


def _resolve_path(project_root: Path, configured_path: str) -> Path:
    resolved = resolve_project_path(str(project_root), configured_path)
    if not resolved.is_file():
        raise ValueError(f"Expected file at {resolved}, but it does not exist.")
    return resolved


def _copy_artifact(
    *,
    source_path: Path,
    destination_path: Path,
    output_root: Path,
    kind: str,
) -> BenchmarkPackArtifactRef:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    return BenchmarkPackArtifactRef(
        kind=kind,
        original_path=str(source_path),
        copied_path=_relative_to(destination_path, output_root),
        sha256=_sha256_file(source_path),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return payload


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(dict[str, Any], value)


def _pick_lower(
    left_candidate_id: str,
    left_value: float | None,
    right_candidate_id: str,
    right_value: float | None,
    *,
    tie_break_left: float | None = None,
    tie_break_right: float | None = None,
) -> str | None:
    if left_value is None and right_value is None:
        return None
    if left_value is None:
        return right_candidate_id
    if right_value is None:
        return left_candidate_id
    if left_value < right_value:
        return left_candidate_id
    if right_value < left_value:
        return right_candidate_id
    if tie_break_left is None or tie_break_right is None:
        return None
    if tie_break_left < tie_break_right:
        return left_candidate_id
    if tie_break_right < tie_break_left:
        return right_candidate_id
    return None


def _delta(left: float | None, right: float | None, *, digits: int) -> float | None:
    if left is None or right is None:
        return None
    return round(right - left, digits)


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(cast(Any, value))


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    return int(cast(Any, value))


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


__all__ = ["build_final_benchmark_pack"]
