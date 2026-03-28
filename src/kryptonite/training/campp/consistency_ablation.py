"""Robust-dev ablation report for CAM++ baseline vs consistency checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from kryptonite.eval.teacher_student_robust_dev_config import TeacherStudentRobustDevCandidateConfig
from kryptonite.eval.teacher_student_robust_dev_models import (
    CorruptedSuiteEntry,
    TeacherStudentRobustDevSuiteEvaluation,
)
from kryptonite.eval.teacher_student_robust_dev_runtime import (
    evaluate_candidate_suites,
    load_candidate_artifacts,
    load_corrupted_suites,
)

ROBUST_DEV_ABLATION_JSON_NAME = "robust_dev_ablation.json"
ROBUST_DEV_ABLATION_MARKDOWN_NAME = "robust_dev_ablation.md"


@dataclass(frozen=True, slots=True)
class ConsistencyAblationCandidate:
    candidate_id: str
    label: str
    checkpoint_path: str
    clean_report_markdown_path: str
    clean_eer: float
    clean_min_dcf: float
    clean_score_gap: float | None
    robust_eer: float
    robust_min_dcf: float
    robust_score_gap: float | None
    weighted_eer: float
    weighted_min_dcf: float
    suites: tuple[TeacherStudentRobustDevSuiteEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["suites"] = [suite.to_dict() for suite in self.suites]
        return payload


@dataclass(frozen=True, slots=True)
class ConsistencyAblationSuiteDelta:
    suite_id: str
    family: str
    baseline_eer: float
    consistency_eer: float
    eer_delta: float
    baseline_min_dcf: float
    consistency_min_dcf: float
    min_dcf_delta: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConsistencyAblationSummary:
    generated_at: str
    output_root: str
    clean_weight: float
    corrupted_weight: float
    corrupted_suite_ids: tuple[str, ...]
    winner_candidate_id: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["corrupted_suite_ids"] = list(self.corrupted_suite_ids)
        return payload


@dataclass(frozen=True, slots=True)
class ConsistencyAblationReport:
    title: str
    ticket_id: str
    summary: ConsistencyAblationSummary
    baseline: ConsistencyAblationCandidate
    consistency: ConsistencyAblationCandidate
    suite_deltas: tuple[ConsistencyAblationSuiteDelta, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "ticket_id": self.ticket_id,
            "summary": self.summary.to_dict(),
            "baseline": self.baseline.to_dict(),
            "consistency": self.consistency.to_dict(),
            "suite_deltas": [delta.to_dict() for delta in self.suite_deltas],
        }


@dataclass(frozen=True, slots=True)
class WrittenConsistencyAblationReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    report: ConsistencyAblationReport

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "report": self.report.to_dict(),
        }


def write_consistency_ablation_report(
    *,
    title: str,
    ticket_id: str,
    output_root: Path,
    project_root: Path,
    device: str,
    catalog_path: str,
    suite_ids: tuple[str, ...],
    clean_weight: float,
    corrupted_weight: float,
    baseline_checkpoint_path: str,
    baseline_label: str,
    baseline_clean_report_markdown_path: str,
    baseline_clean_eer: float,
    baseline_clean_min_dcf: float,
    baseline_clean_score_gap: float | None,
    consistency_checkpoint_path: str,
    consistency_label: str,
    consistency_clean_report_markdown_path: str,
    consistency_clean_eer: float,
    consistency_clean_min_dcf: float,
    consistency_clean_score_gap: float | None,
) -> WrittenConsistencyAblationReport:
    output_root.mkdir(parents=True, exist_ok=True)
    clean_weight, corrupted_weight = _normalized_weights(clean_weight, corrupted_weight)
    suites = load_corrupted_suites(
        project_root=project_root,
        catalog_path=catalog_path,
        suite_ids=suite_ids,
    )
    baseline = _evaluate_candidate(
        candidate_id="stage3_baseline",
        label=baseline_label,
        checkpoint_path=baseline_checkpoint_path,
        clean_report_markdown_path=baseline_clean_report_markdown_path,
        clean_eer=baseline_clean_eer,
        clean_min_dcf=baseline_clean_min_dcf,
        clean_score_gap=baseline_clean_score_gap,
        device=device,
        suites=suites,
        project_root=project_root,
        output_root=output_root / "baseline",
        clean_weight=clean_weight,
        corrupted_weight=corrupted_weight,
    )
    consistency = _evaluate_candidate(
        candidate_id="campp_consistency",
        label=consistency_label,
        checkpoint_path=consistency_checkpoint_path,
        clean_report_markdown_path=consistency_clean_report_markdown_path,
        clean_eer=consistency_clean_eer,
        clean_min_dcf=consistency_clean_min_dcf,
        clean_score_gap=consistency_clean_score_gap,
        device=device,
        suites=suites,
        project_root=project_root,
        output_root=output_root / "consistency",
        clean_weight=clean_weight,
        corrupted_weight=corrupted_weight,
    )
    baseline_by_suite = {suite.suite_id: suite for suite in baseline.suites}
    suite_deltas = tuple(
        ConsistencyAblationSuiteDelta(
            suite_id=suite.suite_id,
            family=suite.family,
            baseline_eer=baseline_by_suite[suite.suite_id].eer,
            consistency_eer=suite.eer,
            eer_delta=round(suite.eer - baseline_by_suite[suite.suite_id].eer, 6),
            baseline_min_dcf=baseline_by_suite[suite.suite_id].min_dcf,
            consistency_min_dcf=suite.min_dcf,
            min_dcf_delta=round(suite.min_dcf - baseline_by_suite[suite.suite_id].min_dcf, 6),
        )
        for suite in consistency.suites
        if suite.suite_id in baseline_by_suite
    )
    winner = min(
        (baseline, consistency),
        key=lambda candidate: (candidate.weighted_eer, candidate.weighted_min_dcf, candidate.label),
    )
    report = ConsistencyAblationReport(
        title=title,
        ticket_id=ticket_id,
        summary=ConsistencyAblationSummary(
            generated_at=datetime.now(UTC).isoformat(),
            output_root=str(output_root),
            clean_weight=clean_weight,
            corrupted_weight=corrupted_weight,
            corrupted_suite_ids=tuple(suite.suite_id for suite in suites),
            winner_candidate_id=winner.candidate_id,
        ),
        baseline=baseline,
        consistency=consistency,
        suite_deltas=suite_deltas,
    )
    report_json_path = output_root / ROBUST_DEV_ABLATION_JSON_NAME
    report_markdown_path = output_root / ROBUST_DEV_ABLATION_MARKDOWN_NAME
    report_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_consistency_ablation_markdown(report),
        encoding="utf-8",
    )
    return WrittenConsistencyAblationReport(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        report=report,
    )


def render_consistency_ablation_markdown(report: ConsistencyAblationReport) -> str:
    lines = [
        "# CAM++ Consistency Robust-Dev Ablation",
        "",
        f"- Ticket: `{report.ticket_id}`",
        f"- Winner: `{report.summary.winner_candidate_id}`",
        (
            "- Weighted objective: "
            f"clean={report.summary.clean_weight}, "
            f"corrupted={report.summary.corrupted_weight}"
        ),
        (
            "- Delta convention: negative means the consistency checkpoint improved because "
            "lower EER/minDCF is better."
        ),
        "",
        (
            "| Candidate | Clean EER | Robust EER | Weighted EER | "
            "Clean minDCF | Robust minDCF | Weighted minDCF |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {report.baseline.label} | {report.baseline.clean_eer:.6f} | "
            f"{report.baseline.robust_eer:.6f} | {report.baseline.weighted_eer:.6f} | "
            f"{report.baseline.clean_min_dcf:.6f} | {report.baseline.robust_min_dcf:.6f} | "
            f"{report.baseline.weighted_min_dcf:.6f} |"
        ),
        (
            f"| {report.consistency.label} | {report.consistency.clean_eer:.6f} | "
            f"{report.consistency.robust_eer:.6f} | {report.consistency.weighted_eer:.6f} | "
            f"{report.consistency.clean_min_dcf:.6f} | {report.consistency.robust_min_dcf:.6f} | "
            f"{report.consistency.weighted_min_dcf:.6f} |"
        ),
        "",
        "## Per-Suite Deltas",
        "",
        (
            "| Suite | Family | Baseline EER | Consistency EER | Delta | "
            "Baseline minDCF | Consistency minDCF | Delta |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for delta in report.suite_deltas:
        lines.append(
            f"| {delta.suite_id} | {delta.family} | {delta.baseline_eer:.6f} | "
            f"{delta.consistency_eer:.6f} | {delta.eer_delta:.6f} | "
            f"{delta.baseline_min_dcf:.6f} | {delta.consistency_min_dcf:.6f} | "
            f"{delta.min_dcf_delta:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _evaluate_candidate(
    *,
    candidate_id: str,
    label: str,
    checkpoint_path: str,
    clean_report_markdown_path: str,
    clean_eer: float,
    clean_min_dcf: float,
    clean_score_gap: float | None,
    device: str,
    suites: tuple[CorruptedSuiteEntry, ...],
    project_root: Path,
    output_root: Path,
    clean_weight: float,
    corrupted_weight: float,
) -> ConsistencyAblationCandidate:
    candidate = TeacherStudentRobustDevCandidateConfig(
        candidate_id=candidate_id,
        label=label,
        role="student",
        family="campp",
        run_root=checkpoint_path,
    )
    _, runtime, _ = load_candidate_artifacts(
        candidate=candidate,
        run_root=Path(checkpoint_path),
        project_root=project_root,
    )
    suite_results, _ = evaluate_candidate_suites(
        candidate=candidate,
        runtime=runtime,
        suites=suites,
        device=device,
        project_root=project_root,
        report_output_root=output_root,
    )
    robust_eer = _mean_metric(tuple(suite.eer for suite in suite_results))
    robust_min_dcf = _mean_metric(tuple(suite.min_dcf for suite in suite_results))
    robust_score_gap = _mean_optional_metric(tuple(suite.score_gap for suite in suite_results))
    return ConsistencyAblationCandidate(
        candidate_id=candidate_id,
        label=label,
        checkpoint_path=checkpoint_path,
        clean_report_markdown_path=clean_report_markdown_path,
        clean_eer=round(clean_eer, 6),
        clean_min_dcf=round(clean_min_dcf, 6),
        clean_score_gap=_round_optional(clean_score_gap),
        robust_eer=round(robust_eer, 6),
        robust_min_dcf=round(robust_min_dcf, 6),
        robust_score_gap=_round_optional(robust_score_gap),
        weighted_eer=round((clean_eer * clean_weight) + (robust_eer * corrupted_weight), 6),
        weighted_min_dcf=round(
            (clean_min_dcf * clean_weight) + (robust_min_dcf * corrupted_weight),
            6,
        ),
        suites=suite_results,
    )


def _normalized_weights(clean_weight: float, corrupted_weight: float) -> tuple[float, float]:
    total = clean_weight + corrupted_weight
    if total <= 0.0:
        raise ValueError("clean_weight and corrupted_weight must sum to a positive value.")
    return clean_weight / total, corrupted_weight / total


def _mean_metric(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _mean_optional_metric(values: tuple[float | None, ...]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


__all__ = [
    "ROBUST_DEV_ABLATION_JSON_NAME",
    "ROBUST_DEV_ABLATION_MARKDOWN_NAME",
    "ConsistencyAblationReport",
    "WrittenConsistencyAblationReport",
    "render_consistency_ablation_markdown",
    "write_consistency_ablation_report",
]
