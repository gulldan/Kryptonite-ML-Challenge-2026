"""Teacher-vs-student robust-dev comparison builder."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .teacher_student_robust_dev_config import (
    TeacherStudentRobustDevCandidateConfig,
    TeacherStudentRobustDevConfig,
)
from .teacher_student_robust_dev_models import (
    CandidateEvidence,
    TeacherStudentRobustDevCandidateReport,
    TeacherStudentRobustDevPairwiseComparison,
    TeacherStudentRobustDevReport,
    TeacherStudentRobustDevSuiteDelta,
    TeacherStudentRobustDevSuiteEvaluation,
    TeacherStudentRobustDevSummary,
)
from .teacher_student_robust_dev_rendering import (
    render_teacher_student_robust_dev_markdown,
    write_teacher_student_robust_dev_report,
)
from .teacher_student_robust_dev_runtime import evaluate_candidate, load_corrupted_suites


def build_teacher_student_robust_dev_report(
    config: TeacherStudentRobustDevConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str = ".",
) -> TeacherStudentRobustDevReport:
    resolved_root = Path(project_root)
    suites = load_corrupted_suites(
        project_root=resolved_root,
        catalog_path=config.corrupted_suites.catalog_path,
        suite_ids=config.corrupted_suites.suite_ids,
    )
    runtimes = tuple(
        evaluate_candidate(
            candidate=candidate,
            config=config,
            suites=suites,
            project_root=resolved_root,
        )
        for candidate in config.candidates
    )
    candidates = _rank_candidates(
        runtimes,
        clean_enabled=config.corrupted_suites.run_clean_dev,
        config=config,
    )
    teacher = next(candidate for candidate in candidates if candidate.role == "teacher")
    pairwise = tuple(
        _build_pairwise_comparison(teacher=teacher, student=student)
        for student in candidates
        if student.role == "student"
    )
    best_quality_candidate = min(candidates, key=lambda item: item.selection_score)
    return TeacherStudentRobustDevReport(
        title=config.title,
        ticket_id=config.ticket_id,
        report_id=config.report_id,
        summary=TeacherStudentRobustDevSummary(
            generated_at=datetime.now(UTC).isoformat(),
            config_path=None if config_path is None else str(Path(config_path)),
            output_root=config.output_root,
            teacher_candidate_id=config.teacher_candidate_id,
            corrupted_suite_ids=tuple(suite.suite_id for suite in suites),
            clean_weight=config.selection.clean_weight,
            corrupted_weight=config.selection.corrupted_weight,
            eer_weight=config.selection.eer_weight,
            min_dcf_weight=config.selection.min_dcf_weight,
            best_quality_candidate_id=best_quality_candidate.candidate_id,
        ),
        candidates=candidates,
        pairwise=pairwise,
        notes=config.notes,
    )


def _rank_candidates(
    runtimes: tuple[
        tuple[
            TeacherStudentRobustDevCandidateConfig,
            CandidateEvidence,
            tuple[TeacherStudentRobustDevSuiteEvaluation, ...],
        ],
        ...,
    ],
    *,
    clean_enabled: bool,
    config: TeacherStudentRobustDevConfig,
) -> tuple[TeacherStudentRobustDevCandidateReport, ...]:
    clean_weight, corrupted_weight = _normalized_weights(
        config.selection.clean_weight if clean_enabled else 0.0,
        config.selection.corrupted_weight,
    )
    eer_weight, min_dcf_weight = _normalized_weights(
        config.selection.eer_weight,
        config.selection.min_dcf_weight,
    )
    reports: list[TeacherStudentRobustDevCandidateReport] = []
    for candidate, evidence, suites in runtimes:
        robust_eer = _mean_metric(tuple(suite.eer for suite in suites))
        robust_min_dcf = _mean_metric(tuple(suite.min_dcf for suite in suites))
        robust_score_gap = _mean_optional_metric(tuple(suite.score_gap for suite in suites))
        weighted_eer = clean_weight * evidence.clean_eer + corrupted_weight * robust_eer
        weighted_min_dcf = clean_weight * evidence.clean_min_dcf + corrupted_weight * robust_min_dcf
        selection_score = eer_weight * weighted_eer + min_dcf_weight * weighted_min_dcf
        reports.append(
            TeacherStudentRobustDevCandidateReport(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                role=candidate.role,
                family=candidate.family,
                rank=0,
                run_root=str(evidence.run_root),
                clean_report_markdown_path=evidence.clean_report_markdown_path,
                clean_trial_count=evidence.clean_trial_count,
                clean_eer=round(evidence.clean_eer, 6),
                clean_min_dcf=round(evidence.clean_min_dcf, 6),
                clean_score_gap=_round_optional(evidence.clean_score_gap),
                robust_eer=round(robust_eer, 6),
                robust_min_dcf=round(robust_min_dcf, 6),
                robust_score_gap=_round_optional(robust_score_gap),
                weighted_eer=round(weighted_eer, 6),
                weighted_min_dcf=round(weighted_min_dcf, 6),
                selection_score=round(selection_score, 6),
                cost=evidence.cost,
                suites=suites,
                notes=candidate.notes,
            )
        )
    ranked = sorted(
        reports,
        key=lambda item: (item.selection_score, item.robust_eer, item.clean_eer, item.label),
    )
    return tuple(replace(candidate, rank=index) for index, candidate in enumerate(ranked, start=1))


def _build_pairwise_comparison(
    *,
    teacher: TeacherStudentRobustDevCandidateReport,
    student: TeacherStudentRobustDevCandidateReport,
) -> TeacherStudentRobustDevPairwiseComparison:
    student_suites = {suite.suite_id: suite for suite in student.suites}
    suite_deltas = tuple(
        TeacherStudentRobustDevSuiteDelta(
            suite_id=teacher_suite.suite_id,
            family=teacher_suite.family,
            teacher_eer=teacher_suite.eer,
            student_eer=student_suites[teacher_suite.suite_id].eer,
            eer_delta=round(student_suites[teacher_suite.suite_id].eer - teacher_suite.eer, 6),
            teacher_min_dcf=teacher_suite.min_dcf,
            student_min_dcf=student_suites[teacher_suite.suite_id].min_dcf,
            min_dcf_delta=round(
                student_suites[teacher_suite.suite_id].min_dcf - teacher_suite.min_dcf,
                6,
            ),
        )
        for teacher_suite in teacher.suites
        if teacher_suite.suite_id in student_suites
    )
    return TeacherStudentRobustDevPairwiseComparison(
        teacher_candidate_id=teacher.candidate_id,
        student_candidate_id=student.candidate_id,
        student_label=student.label,
        clean_eer_delta=round(student.clean_eer - teacher.clean_eer, 6),
        robust_eer_delta=round(student.robust_eer - teacher.robust_eer, 6),
        weighted_eer_delta=round(student.weighted_eer - teacher.weighted_eer, 6),
        clean_min_dcf_delta=round(student.clean_min_dcf - teacher.clean_min_dcf, 6),
        robust_min_dcf_delta=round(student.robust_min_dcf - teacher.robust_min_dcf, 6),
        weighted_min_dcf_delta=round(student.weighted_min_dcf - teacher.weighted_min_dcf, 6),
        total_parameter_ratio=_safe_ratio(
            teacher.cost.total_parameters,
            student.cost.total_parameters,
        ),
        checkpoint_size_ratio=_safe_ratio(
            teacher.cost.checkpoint_size_bytes,
            student.cost.checkpoint_size_bytes,
        ),
        suite_deltas=suite_deltas,
    )


def _normalized_weights(primary: float, secondary: float) -> tuple[float, float]:
    total = primary + secondary
    if total <= 0.0:
        raise ValueError("Weight sum must be positive.")
    return primary / total, secondary / total


def _mean_metric(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _mean_optional_metric(values: tuple[float | None, ...]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 6)


__all__ = [
    "TeacherStudentRobustDevReport",
    "build_teacher_student_robust_dev_report",
    "render_teacher_student_robust_dev_markdown",
    "write_teacher_student_robust_dev_report",
]
