"""Data models for the cleanup backlog."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

ACTION_ORDER: dict[str, int] = {
    "fix": 0,
    "quarantine": 1,
    "keep": 2,
    "document": 3,
}
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


@dataclass(frozen=True, slots=True)
class BacklogSource:
    name: str
    script_path: str
    artifact_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "script_path": self.script_path,
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True, slots=True)
class DataIssue:
    code: str
    severity: str
    action: str
    category: str
    title: str
    summary: str
    rationale: str
    stop_rule: str | None = None
    evidence: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "action": self.action,
            "category": self.category,
            "title": self.title,
            "summary": self.summary,
            "rationale": self.rationale,
            "stop_rule": self.stop_rule,
            "evidence": list(self.evidence),
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class WrittenDataIssuesBacklogReport:
    output_root: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


@dataclass(slots=True)
class DataIssuesBacklogReport:
    generated_at: str
    project_root: str
    manifests_root: str
    profile_manifest_count: int
    leakage_finding_count: int
    audio_pattern_count: int
    quarantine_manifest_count: int
    quarantine_row_count: int
    sources: list[BacklogSource]
    issues: list[DataIssue]
    stop_rules: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def issue_counts_by_action(self) -> dict[str, int]:
        return _ordered_counts(
            Counter(issue.action for issue in self.issues),
            ACTION_ORDER,
        )

    @property
    def issue_counts_by_severity(self) -> dict[str, int]:
        return _ordered_counts(
            Counter(issue.severity for issue in self.issues),
            SEVERITY_ORDER,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "manifests_root": self.manifests_root,
            "profile_manifest_count": self.profile_manifest_count,
            "leakage_finding_count": self.leakage_finding_count,
            "audio_pattern_count": self.audio_pattern_count,
            "quarantine_manifest_count": self.quarantine_manifest_count,
            "quarantine_row_count": self.quarantine_row_count,
            "issue_count": self.issue_count,
            "issue_counts_by_action": self.issue_counts_by_action,
            "issue_counts_by_severity": self.issue_counts_by_severity,
            "stop_rules": list(self.stop_rules),
            "warnings": list(self.warnings),
            "sources": [source.to_dict() for source in self.sources],
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class QuarantineManifestSummary:
    manifest_path: str
    row_count: int

    def to_dict(self) -> dict[str, object]:
        return {"manifest_path": self.manifest_path, "row_count": self.row_count}


@dataclass(frozen=True, slots=True)
class QuarantineSummary:
    manifests: tuple[QuarantineManifestSummary, ...]
    issue_counts: dict[str, int]
    invalid_line_count: int

    @property
    def manifest_count(self) -> int:
        return len(self.manifests)

    @property
    def row_count(self) -> int:
        return sum(manifest.row_count for manifest in self.manifests)


def _ordered_counts(
    counts: Counter[str],
    order: dict[str, int],
) -> dict[str, int]:
    ordered = sorted(counts.items(), key=lambda item: order.get(item[0], 999))
    return {name: count for name, count in ordered}
