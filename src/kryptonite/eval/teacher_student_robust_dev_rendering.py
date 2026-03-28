"""Rendering helpers for teacher-vs-student robust-dev reports."""

from __future__ import annotations

import json
from pathlib import Path

from .teacher_student_robust_dev_models import (
    TEACHER_STUDENT_ROBUST_DEV_JSON_NAME,
    TEACHER_STUDENT_ROBUST_DEV_MARKDOWN_NAME,
    TeacherStudentRobustDevReport,
    WrittenTeacherStudentRobustDevReport,
)


def render_teacher_student_robust_dev_markdown(report: TeacherStudentRobustDevReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Ticket: `{report.ticket_id}`",
        f"- Report id: `{report.report_id}`",
        f"- Generated at: `{report.summary.generated_at}`",
        f"- Teacher candidate: `{report.summary.teacher_candidate_id}`",
        f"- Best quality candidate: `{report.summary.best_quality_candidate_id}`",
        "",
        "## Quality Leaderboard",
        "",
        (
            "| Rank | Candidate | Role | Family | Clean EER | Robust EER | Weighted EER | "
            "Robust minDCF | Score |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in report.candidates:
        lines.append(
            "| "
            f"{candidate.rank} | {candidate.label} | {candidate.role} | {candidate.family} | "
            f"{candidate.clean_eer:.6f} | {candidate.robust_eer:.6f} | "
            f"{candidate.weighted_eer:.6f} | "
            f"{candidate.robust_min_dcf:.6f} | {candidate.selection_score:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Cost Snapshot",
            "",
            (
                "| Candidate | Params (M) | Trainable (M) | Checkpoint (MiB) | Precision | "
                "Batch | Effective batch | Epochs |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in report.candidates:
        lines.append(
            "| "
            f"{candidate.label} | {_format_optional_millions(candidate.cost.total_parameters)} | "
            f"{_format_optional_millions(candidate.cost.trainable_parameters)} | "
            f"{candidate.cost.checkpoint_size_bytes / (1024 * 1024):.2f} | "
            f"{candidate.cost.precision} | {candidate.cost.train_batch_size} | "
            f"{candidate.cost.effective_batch_size} | {candidate.cost.max_epochs} |"
        )

    if report.pairwise:
        lines.extend(["", "## Teacher Vs Students", ""])
        for comparison in report.pairwise:
            lines.extend(
                [
                    f"### Teacher vs {comparison.student_label}",
                    "",
                    (
                        "- Delta convention: positive means the teacher is better because it "
                        "lowers the metric."
                    ),
                    f"- Clean EER delta: `{comparison.clean_eer_delta:.6f}`",
                    f"- Robust EER delta: `{comparison.robust_eer_delta:.6f}`",
                    f"- Weighted EER delta: `{comparison.weighted_eer_delta:.6f}`",
                    f"- Clean minDCF delta: `{comparison.clean_min_dcf_delta:.6f}`",
                    f"- Robust minDCF delta: `{comparison.robust_min_dcf_delta:.6f}`",
                    f"- Weighted minDCF delta: `{comparison.weighted_min_dcf_delta:.6f}`",
                ]
            )
            if comparison.total_parameter_ratio is not None:
                lines.append(
                    f"- Parameter ratio teacher/student: `{comparison.total_parameter_ratio:.3f}x`"
                )
            if comparison.checkpoint_size_ratio is not None:
                lines.append(
                    f"- Checkpoint-size ratio teacher/student: "
                    f"`{comparison.checkpoint_size_ratio:.3f}x`"
                )
            lines.extend(
                [
                    "",
                    (
                        "| Suite | Family | Teacher EER | Student EER | Delta | Teacher minDCF | "
                        "Student minDCF | Delta |"
                    ),
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for delta in comparison.suite_deltas:
                lines.append(
                    "| "
                    f"{delta.suite_id} | {delta.family} | {delta.teacher_eer:.6f} | "
                    f"{delta.student_eer:.6f} | {delta.eer_delta:.6f} | "
                    f"{delta.teacher_min_dcf:.6f} | {delta.student_min_dcf:.6f} | "
                    f"{delta.min_dcf_delta:.6f} |"
                )
            lines.append("")

    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines).rstrip() + "\n"


def write_teacher_student_robust_dev_report(
    report: TeacherStudentRobustDevReport,
    *,
    project_root: Path | str = ".",
) -> WrittenTeacherStudentRobustDevReport:
    output_root = Path(project_root) / report.summary.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    report_json_path = output_root / TEACHER_STUDENT_ROBUST_DEV_JSON_NAME
    report_markdown_path = output_root / TEACHER_STUDENT_ROBUST_DEV_MARKDOWN_NAME
    report_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_teacher_student_robust_dev_markdown(report),
        encoding="utf-8",
    )
    return WrittenTeacherStudentRobustDevReport(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
    )


def _format_optional_millions(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1_000_000:.3f}"


__all__ = [
    "render_teacher_student_robust_dev_markdown",
    "write_teacher_student_robust_dev_report",
]
