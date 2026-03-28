"""Datamodels for teacher-vs-student robust-dev reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .teacher_student_robust_dev_config import CandidateFamily, CandidateRole

TEACHER_STUDENT_ROBUST_DEV_JSON_NAME = "teacher_student_robust_dev_report.json"
TEACHER_STUDENT_ROBUST_DEV_MARKDOWN_NAME = "teacher_student_robust_dev_report.md"
SUITE_TRIALS_FILE_NAME = "suite_trials.jsonl"


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevCostSummary:
    training_device: str
    precision: str
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    max_epochs: int
    train_row_count: int
    dev_row_count: int
    total_parameters: int | None
    trainable_parameters: int | None
    checkpoint_size_bytes: int
    embedding_dim: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevSuiteEvaluation:
    suite_id: str
    family: str
    manifest_path: str
    trials_path: str
    output_root: str
    report_markdown_path: str
    trial_count: int
    eer: float
    min_dcf: float
    score_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevCandidateReport:
    candidate_id: str
    label: str
    role: CandidateRole
    family: CandidateFamily
    rank: int
    run_root: str
    clean_report_markdown_path: str
    clean_trial_count: int
    clean_eer: float
    clean_min_dcf: float
    clean_score_gap: float | None
    robust_eer: float
    robust_min_dcf: float
    robust_score_gap: float | None
    weighted_eer: float
    weighted_min_dcf: float
    selection_score: float
    cost: TeacherStudentRobustDevCostSummary
    suites: tuple[TeacherStudentRobustDevSuiteEvaluation, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost"] = self.cost.to_dict()
        payload["suites"] = [suite.to_dict() for suite in self.suites]
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevSuiteDelta:
    suite_id: str
    family: str
    teacher_eer: float
    student_eer: float
    eer_delta: float
    teacher_min_dcf: float
    student_min_dcf: float
    min_dcf_delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevPairwiseComparison:
    teacher_candidate_id: str
    student_candidate_id: str
    student_label: str
    clean_eer_delta: float
    robust_eer_delta: float
    weighted_eer_delta: float
    clean_min_dcf_delta: float
    robust_min_dcf_delta: float
    weighted_min_dcf_delta: float
    total_parameter_ratio: float | None
    checkpoint_size_ratio: float | None
    suite_deltas: tuple[TeacherStudentRobustDevSuiteDelta, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suite_deltas"] = [item.to_dict() for item in self.suite_deltas]
        return payload


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevSummary:
    generated_at: str
    config_path: str | None
    output_root: str
    teacher_candidate_id: str
    corrupted_suite_ids: tuple[str, ...]
    clean_weight: float
    corrupted_weight: float
    eer_weight: float
    min_dcf_weight: float
    best_quality_candidate_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["corrupted_suite_ids"] = list(self.corrupted_suite_ids)
        return payload


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevReport:
    title: str
    ticket_id: str
    report_id: str
    summary: TeacherStudentRobustDevSummary
    candidates: tuple[TeacherStudentRobustDevCandidateReport, ...]
    pairwise: tuple[TeacherStudentRobustDevPairwiseComparison, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "ticket_id": self.ticket_id,
            "report_id": self.report_id,
            "summary": self.summary.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "pairwise": [comparison.to_dict() for comparison in self.pairwise],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class WrittenTeacherStudentRobustDevReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorruptedSuiteEntry:
    suite_id: str
    family: str
    description: str
    manifest_path: str
    trial_manifest_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    run_root: Path
    clean_report_markdown_path: str
    clean_trial_count: int
    clean_eer: float
    clean_min_dcf: float
    clean_score_gap: float | None
    cost: TeacherStudentRobustDevCostSummary
    train_manifest_path: str
    max_dev_rows: int | None


__all__ = [
    "CandidateEvidence",
    "CorruptedSuiteEntry",
    "SUITE_TRIALS_FILE_NAME",
    "TEACHER_STUDENT_ROBUST_DEV_JSON_NAME",
    "TEACHER_STUDENT_ROBUST_DEV_MARKDOWN_NAME",
    "TeacherStudentRobustDevCandidateReport",
    "TeacherStudentRobustDevCostSummary",
    "TeacherStudentRobustDevPairwiseComparison",
    "TeacherStudentRobustDevReport",
    "TeacherStudentRobustDevSuiteDelta",
    "TeacherStudentRobustDevSuiteEvaluation",
    "TeacherStudentRobustDevSummary",
    "WrittenTeacherStudentRobustDevReport",
]
