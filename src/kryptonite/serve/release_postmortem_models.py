"""Datamodels for reproducible release postmortem and backlog reports."""

from __future__ import annotations

from dataclasses import dataclass

RELEASE_POSTMORTEM_JSON_NAME = "release_postmortem.json"
RELEASE_POSTMORTEM_MARKDOWN_NAME = "release_postmortem.md"


@dataclass(frozen=True, slots=True)
class ReleasePostmortemEvidenceRef:
    label: str
    kind: str
    path: str
    resolved_path: str
    description: str
    path_kind: str
    sha256: str | None
    file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "path": self.path,
            "resolved_path": self.resolved_path,
            "description": self.description,
            "path_kind": self.path_kind,
            "sha256": self.sha256,
            "file_count": self.file_count,
        }


@dataclass(frozen=True, slots=True)
class ReleasePostmortemFinding:
    area: str
    outcome: str
    title: str
    detail: str
    evidence_labels: tuple[str, ...]
    related_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "area": self.area,
            "outcome": self.outcome,
            "title": self.title,
            "detail": self.detail,
            "evidence_labels": list(self.evidence_labels),
            "related_issues": list(self.related_issues),
        }


@dataclass(frozen=True, slots=True)
class ReleaseBacklogItem:
    title: str
    priority: str
    disposition: str
    area: str
    rationale: str
    related_issue: str | None
    dependencies: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "priority": self.priority,
            "disposition": self.disposition,
            "area": self.area,
            "rationale": self.rationale,
            "related_issue": self.related_issue,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True, slots=True)
class ReleasePostmortemSummary:
    worked_count: int
    missed_count: int
    risk_count: int
    next_iteration_count: int
    de_scoped_count: int
    monitor_count: int
    highest_priority_next_items: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "worked_count": self.worked_count,
            "missed_count": self.missed_count,
            "risk_count": self.risk_count,
            "next_iteration_count": self.next_iteration_count,
            "de_scoped_count": self.de_scoped_count,
            "monitor_count": self.monitor_count,
            "highest_priority_next_items": list(self.highest_priority_next_items),
        }


@dataclass(frozen=True, slots=True)
class ReleasePostmortemReport:
    title: str
    release_id: str
    release_tag: str | None
    summary_text: str
    output_root: str
    source_config_path: str | None
    source_config_sha256: str | None
    evidence: tuple[ReleasePostmortemEvidenceRef, ...]
    findings: tuple[ReleasePostmortemFinding, ...]
    backlog_items: tuple[ReleaseBacklogItem, ...]
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]
    summary: ReleasePostmortemSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "release_id": self.release_id,
            "release_tag": self.release_tag,
            "summary_text": self.summary_text,
            "output_root": self.output_root,
            "source_config_path": self.source_config_path,
            "source_config_sha256": self.source_config_sha256,
            "evidence": [item.to_dict() for item in self.evidence],
            "findings": [item.to_dict() for item in self.findings],
            "backlog_items": [item.to_dict() for item in self.backlog_items],
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenReleasePostmortem:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    summary: ReleasePostmortemSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "RELEASE_POSTMORTEM_JSON_NAME",
    "RELEASE_POSTMORTEM_MARKDOWN_NAME",
    "ReleaseBacklogItem",
    "ReleasePostmortemEvidenceRef",
    "ReleasePostmortemFinding",
    "ReleasePostmortemReport",
    "ReleasePostmortemSummary",
    "WrittenReleasePostmortem",
]
