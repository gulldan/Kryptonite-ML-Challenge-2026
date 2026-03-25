"""Data models and constants for verification error analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

VERIFICATION_ERROR_ANALYSIS_JSON_NAME = "verification_error_analysis.json"
VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME = "verification_error_analysis.md"


@dataclass(frozen=True, slots=True)
class VerificationErrorAnalysisSummary:
    threshold_source: str
    decision_threshold: float
    trial_count: int
    positive_count: int
    negative_count: int
    false_accept_count: int
    false_reject_count: int
    total_error_count: int
    false_accept_rate: float
    false_reject_rate: float
    total_error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationErrorExample:
    error_type: str
    score: float
    label: int
    margin: float
    left_id: str | None
    right_id: str | None
    left_speaker_id: str | None
    right_speaker_id: str | None
    dataset: str | None
    channel: str | None
    role_pair: str | None
    duration_bucket: str | None
    noise_slice: str | None
    reverb_slice: str | None
    channel_slice: str | None
    distance_slice: str | None
    silence_slice: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationDomainFailure:
    field_name: str
    field_value: str
    trial_count: int
    error_count: int
    false_accept_count: int
    false_reject_count: int
    error_rate: float
    mean_error_margin: float
    mean_error_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationSpeakerConfusion:
    speaker_a: str
    speaker_b: str
    trial_count: int
    false_accept_count: int
    false_accept_rate: float
    mean_false_accept_score: float
    max_false_accept_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationSpeakerFailure:
    speaker_id: str
    positive_trial_count: int
    false_reject_count: int
    false_reject_rate: float
    mean_false_reject_score: float
    min_false_reject_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationPriorityFinding:
    finding_type: str
    title: str
    evidence: str
    trial_count: int
    error_count: int
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationErrorAnalysisReport:
    summary: VerificationErrorAnalysisSummary
    priority_findings: tuple[VerificationPriorityFinding, ...]
    hard_false_accepts: tuple[VerificationErrorExample, ...]
    hard_false_rejects: tuple[VerificationErrorExample, ...]
    domain_failures: tuple[VerificationDomainFailure, ...]
    speaker_confusions: tuple[VerificationSpeakerConfusion, ...]
    speaker_failures: tuple[VerificationSpeakerFailure, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "priority_findings": [item.to_dict() for item in self.priority_findings],
            "hard_false_accepts": [item.to_dict() for item in self.hard_false_accepts],
            "hard_false_rejects": [item.to_dict() for item in self.hard_false_rejects],
            "domain_failures": [item.to_dict() for item in self.domain_failures],
            "speaker_confusions": [item.to_dict() for item in self.speaker_confusions],
            "speaker_failures": [item.to_dict() for item in self.speaker_failures],
        }


@dataclass(frozen=True, slots=True)
class WrittenVerificationErrorAnalysis:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    summary: VerificationErrorAnalysisSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "VERIFICATION_ERROR_ANALYSIS_JSON_NAME",
    "VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME",
    "VerificationDomainFailure",
    "VerificationErrorAnalysisReport",
    "VerificationErrorAnalysisSummary",
    "VerificationErrorExample",
    "VerificationPriorityFinding",
    "VerificationSpeakerConfusion",
    "VerificationSpeakerFailure",
    "WrittenVerificationErrorAnalysis",
]
