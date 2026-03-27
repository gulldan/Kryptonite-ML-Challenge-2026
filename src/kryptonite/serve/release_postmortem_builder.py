"""Builder for release postmortem and backlog reports."""

from __future__ import annotations

from pathlib import Path

from kryptonite.deployment import resolve_project_path
from kryptonite.project import get_project_layout
from kryptonite.repro import fingerprint_path

from .release_postmortem_config import (
    ReleasePostmortemConfig,
    ReleasePostmortemFindingConfig,
)
from .release_postmortem_models import (
    ReleaseBacklogItem,
    ReleasePostmortemEvidenceRef,
    ReleasePostmortemFinding,
    ReleasePostmortemReport,
    ReleasePostmortemSummary,
)

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_DISPOSITION_ORDER = {"next_iteration": 0, "monitor": 1, "de_scoped": 2}


def build_release_postmortem(
    config: ReleasePostmortemConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> ReleasePostmortemReport:
    resolved_project_root = _resolve_project_root(project_root)
    evidence = tuple(
        _build_evidence_ref(
            label=item.label,
            kind=item.kind,
            path=item.path,
            description=item.description,
            project_root=resolved_project_root,
        )
        for item in config.evidence
    )
    evidence_by_label = {item.label: item for item in evidence}
    _validate_finding_references(config.findings, evidence_by_label=evidence_by_label)

    findings = tuple(
        ReleasePostmortemFinding(
            area=item.area,
            outcome=item.outcome,
            title=item.title,
            detail=item.detail,
            evidence_labels=item.evidence_labels,
            related_issues=item.related_issues,
        )
        for item in config.findings
    )
    backlog_items = tuple(
        ReleaseBacklogItem(
            title=item.title,
            priority=item.priority,
            disposition=item.disposition,
            area=item.area,
            rationale=item.rationale,
            related_issue=item.related_issue,
            dependencies=item.dependencies,
        )
        for item in config.backlog_items
    )
    summary = _build_summary(findings=findings, backlog_items=backlog_items)

    resolved_output_root = resolve_project_path(str(resolved_project_root), config.output_root)
    source_config_file = None if config_path is None else Path(config_path).resolve()
    source_config_sha256 = None
    if source_config_file is not None:
        source_fingerprint = fingerprint_path(source_config_file)
        source_config_sha256 = (
            None if not source_fingerprint["exists"] else str(source_fingerprint["sha256"])
        )

    return ReleasePostmortemReport(
        title=config.title,
        release_id=config.release_id,
        release_tag=config.release_tag,
        summary_text=config.summary,
        output_root=str(resolved_output_root),
        source_config_path=None if source_config_file is None else str(source_config_file),
        source_config_sha256=source_config_sha256,
        evidence=evidence,
        findings=findings,
        backlog_items=backlog_items,
        validation_commands=config.validation_commands,
        notes=config.notes,
        summary=summary,
    )


def _build_evidence_ref(
    *,
    label: str,
    kind: str,
    path: str,
    description: str,
    project_root: Path,
) -> ReleasePostmortemEvidenceRef:
    resolved_path = resolve_project_path(str(project_root), path)
    fingerprint = fingerprint_path(resolved_path)
    if not bool(fingerprint["exists"]):
        raise ValueError(f"Evidence path {path!r} does not exist.")
    return ReleasePostmortemEvidenceRef(
        label=label,
        kind=kind,
        path=path,
        resolved_path=str(resolved_path),
        description=description,
        path_kind=str(fingerprint["kind"]),
        sha256=None if fingerprint["sha256"] is None else str(fingerprint["sha256"]),
        file_count=int(fingerprint["file_count"]),
    )


def _validate_finding_references(
    findings: tuple[ReleasePostmortemFindingConfig, ...],
    *,
    evidence_by_label: dict[str, ReleasePostmortemEvidenceRef],
) -> None:
    unknown_labels: set[str] = set()
    for finding in findings:
        for label in finding.evidence_labels:
            if label not in evidence_by_label:
                unknown_labels.add(label)
    if unknown_labels:
        joined = ", ".join(sorted(unknown_labels))
        raise ValueError(f"Unknown evidence labels referenced by findings: {joined}")


def _build_summary(
    *,
    findings: tuple[ReleasePostmortemFinding, ...],
    backlog_items: tuple[ReleaseBacklogItem, ...],
) -> ReleasePostmortemSummary:
    worked_count = sum(1 for item in findings if item.outcome == "worked")
    missed_count = sum(1 for item in findings if item.outcome == "missed")
    risk_count = sum(1 for item in findings if item.outcome == "risk")
    next_iteration = tuple(item for item in backlog_items if item.disposition == "next_iteration")
    ordered_next_iteration = sorted(
        next_iteration,
        key=lambda item: (_PRIORITY_ORDER[item.priority], item.area.lower(), item.title.lower()),
    )
    return ReleasePostmortemSummary(
        worked_count=worked_count,
        missed_count=missed_count,
        risk_count=risk_count,
        next_iteration_count=len(next_iteration),
        de_scoped_count=sum(1 for item in backlog_items if item.disposition == "de_scoped"),
        monitor_count=sum(1 for item in backlog_items if item.disposition == "monitor"),
        highest_priority_next_items=tuple(item.title for item in ordered_next_iteration[:3]),
    )


def sort_backlog_items(
    items: tuple[ReleaseBacklogItem, ...],
) -> tuple[ReleaseBacklogItem, ...]:
    return tuple(
        sorted(
            items,
            key=lambda item: (
                _PRIORITY_ORDER[item.priority],
                _DISPOSITION_ORDER[item.disposition],
                item.area.lower(),
                item.title.lower(),
            ),
        )
    )


def _resolve_project_root(project_root: Path | str | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return get_project_layout().root


__all__ = [
    "build_release_postmortem",
    "sort_backlog_items",
]
