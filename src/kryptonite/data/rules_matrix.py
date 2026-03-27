"""Competition-facing rules matrix and risk register for challenge policy decisions."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from kryptonite.deployment import resolve_project_path

RulesDecision = Literal["allow", "deny", "unknown"]
DecisionConfidence = Literal["high", "medium", "low"]
RiskSeverity = Literal["high", "medium", "low"]

ALLOWED_DECISIONS: tuple[RulesDecision, ...] = ("allow", "deny", "unknown")
ALLOWED_CONFIDENCE: tuple[DecisionConfidence, ...] = ("high", "medium", "low")
ALLOWED_RISK_SEVERITIES: tuple[RiskSeverity, ...] = ("high", "medium", "low")


@dataclass(frozen=True, slots=True)
class RulesMatrixSource:
    id: str
    title: str
    kind: str
    url: str
    reviewed_on: str
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RulesMatrixItem:
    id: str
    name: str
    category: str
    decision: RulesDecision
    confidence: DecisionConfidence
    repo_position: str
    reasoning: str
    owner: str
    clarification_channel: str
    next_checkpoint: str
    source_ids: list[str]
    repo_references: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RulesMatrixRisk:
    id: str
    title: str
    severity: RiskSeverity
    description: str
    owner: str
    clarification_channel: str
    mitigation: str
    related_item_ids: list[str]


@dataclass(frozen=True, slots=True)
class RulesMatrixPlan:
    title: str
    reviewed_on: str
    summary: list[str]
    sources: list[RulesMatrixSource]
    items: list[RulesMatrixItem]
    risks: list[RulesMatrixRisk]


@dataclass(frozen=True, slots=True)
class RulesMatrixPathCheck:
    configured_path: str
    resolved_path: str
    exists: bool
    path_type: str

    def to_dict(self) -> dict[str, object]:
        return {
            "configured_path": self.configured_path,
            "resolved_path": self.resolved_path,
            "exists": self.exists,
            "path_type": self.path_type,
        }


@dataclass(slots=True)
class RulesMatrixEntry:
    item: RulesMatrixItem
    sources: list[RulesMatrixSource]
    repo_reference_checks: list[RulesMatrixPathCheck]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.item.id,
            "name": self.item.name,
            "category": self.item.category,
            "decision": self.item.decision,
            "confidence": self.item.confidence,
            "repo_position": self.item.repo_position,
            "reasoning": self.item.reasoning,
            "owner": self.item.owner,
            "clarification_channel": self.item.clarification_channel,
            "next_checkpoint": self.item.next_checkpoint,
            "notes": list(self.item.notes),
            "sources": [source_to_dict(source) for source in self.sources],
            "repo_reference_checks": [check.to_dict() for check in self.repo_reference_checks],
        }


@dataclass(slots=True)
class RulesMatrixRiskEntry:
    risk: RulesMatrixRisk
    related_items: list[RulesMatrixItem]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.risk.id,
            "title": self.risk.title,
            "severity": self.risk.severity,
            "description": self.risk.description,
            "owner": self.risk.owner,
            "clarification_channel": self.risk.clarification_channel,
            "mitigation": self.risk.mitigation,
            "related_items": [
                {"id": item.id, "name": item.name, "decision": item.decision}
                for item in self.related_items
            ],
        }


@dataclass(slots=True)
class RulesMatrixReport:
    generated_at: str
    project_root: str
    plan_path: str | None
    title: str
    reviewed_on: str
    summary: list[str]
    sources: list[RulesMatrixSource]
    entries: list[RulesMatrixEntry]
    risks: list[RulesMatrixRiskEntry]

    @property
    def item_count(self) -> int:
        return len(self.entries)

    @property
    def decision_counts(self) -> dict[str, int]:
        counts = Counter(entry.item.decision for entry in self.entries)
        return {decision: counts.get(decision, 0) for decision in ALLOWED_DECISIONS}

    @property
    def confidence_counts(self) -> dict[str, int]:
        counts = Counter(entry.item.confidence for entry in self.entries)
        return {level: counts.get(level, 0) for level in ALLOWED_CONFIDENCE}

    @property
    def category_counts(self) -> dict[str, int]:
        counts = Counter(entry.item.category for entry in self.entries)
        return dict(sorted(counts.items()))

    @property
    def open_question_count(self) -> int:
        return self.decision_counts.get("unknown", 0)

    @property
    def risk_severity_counts(self) -> dict[str, int]:
        counts = Counter(entry.risk.severity for entry in self.risks)
        return {severity: counts.get(severity, 0) for severity in ALLOWED_RISK_SEVERITIES}

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "plan_path": self.plan_path,
            "title": self.title,
            "reviewed_on": self.reviewed_on,
            "summary": list(self.summary),
            "item_count": self.item_count,
            "decision_counts": self.decision_counts,
            "confidence_counts": self.confidence_counts,
            "category_counts": self.category_counts,
            "open_question_count": self.open_question_count,
            "risk_severity_counts": self.risk_severity_counts,
            "sources": [source_to_dict(source) for source in self.sources],
            "entries": [entry.to_dict() for entry in self.entries],
            "risks": [risk.to_dict() for risk in self.risks],
        }


@dataclass(frozen=True, slots=True)
class WrittenRulesMatrixReport:
    output_root: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def load_rules_matrix_plan(path: Path | str) -> RulesMatrixPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())
    title = _require_str(data, "title")
    reviewed_on = _require_str(data, "reviewed_on")
    source_tables = _require_table_list(data, "sources")
    item_tables = _require_table_list(data, "items")
    risk_tables = _require_table_list(data, "risks")

    sources = [
        RulesMatrixSource(
            id=_require_str(source_data, "id"),
            title=_require_str(source_data, "title"),
            kind=_require_str(source_data, "kind"),
            url=_require_str(source_data, "url"),
            reviewed_on=_require_str(source_data, "reviewed_on"),
            evidence=_optional_str_list(source_data, "evidence"),
        )
        for source_data in source_tables
    ]
    _ensure_unique_ids("source", [source.id for source in sources])
    source_ids = {source.id for source in sources}

    items = [
        RulesMatrixItem(
            id=_require_str(item_data, "id"),
            name=_require_str(item_data, "name"),
            category=_require_str(item_data, "category"),
            decision=_require_literal(item_data, "decision", ALLOWED_DECISIONS),
            confidence=_require_literal(item_data, "confidence", ALLOWED_CONFIDENCE),
            repo_position=_require_str(item_data, "repo_position"),
            reasoning=_require_str(item_data, "reasoning"),
            owner=_require_str(item_data, "owner"),
            clarification_channel=_require_str(item_data, "clarification_channel"),
            next_checkpoint=_require_str(item_data, "next_checkpoint"),
            source_ids=_require_str_list(item_data, "source_ids"),
            repo_references=_optional_str_list(item_data, "repo_references"),
            notes=_optional_str_list(item_data, "notes"),
        )
        for item_data in item_tables
    ]
    _ensure_unique_ids("item", [item.id for item in items])
    item_ids = {item.id for item in items}
    for item in items:
        missing_sources = sorted(set(item.source_ids) - source_ids)
        if missing_sources:
            missing_sources_text = ", ".join(missing_sources)
            raise ValueError(
                f"Rules item '{item.id}' references unknown source ids: {missing_sources_text}."
            )

    risks = [
        RulesMatrixRisk(
            id=_require_str(risk_data, "id"),
            title=_require_str(risk_data, "title"),
            severity=_require_literal(risk_data, "severity", ALLOWED_RISK_SEVERITIES),
            description=_require_str(risk_data, "description"),
            owner=_require_str(risk_data, "owner"),
            clarification_channel=_require_str(risk_data, "clarification_channel"),
            mitigation=_require_str(risk_data, "mitigation"),
            related_item_ids=_require_str_list(risk_data, "related_item_ids"),
        )
        for risk_data in risk_tables
    ]
    _ensure_unique_ids("risk", [risk.id for risk in risks])
    for risk in risks:
        missing_items = sorted(set(risk.related_item_ids) - item_ids)
        if missing_items:
            raise ValueError(
                f"Risk '{risk.id}' references unknown item ids: {', '.join(missing_items)}."
            )

    return RulesMatrixPlan(
        title=title,
        reviewed_on=reviewed_on,
        summary=_optional_str_list(data, "summary"),
        sources=sources,
        items=items,
        risks=risks,
    )


def build_rules_matrix_report(
    *,
    project_root: Path | str,
    plan: RulesMatrixPlan,
    plan_path: Path | str | None = None,
) -> RulesMatrixReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    resolved_plan_path = (
        str(resolve_project_path(str(project_root_path), str(plan_path)))
        if plan_path is not None
        else None
    )
    source_index = {source.id: source for source in plan.sources}
    item_index = {item.id: item for item in plan.items}

    entries = [
        RulesMatrixEntry(
            item=item,
            sources=[source_index[source_id] for source_id in item.source_ids],
            repo_reference_checks=[
                _build_path_check(project_root=project_root_path, configured_path=path)
                for path in item.repo_references
            ],
        )
        for item in plan.items
    ]
    risks = [
        RulesMatrixRiskEntry(
            risk=risk,
            related_items=[item_index[item_id] for item_id in risk.related_item_ids],
        )
        for risk in plan.risks
    ]
    return RulesMatrixReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        plan_path=resolved_plan_path,
        title=plan.title,
        reviewed_on=plan.reviewed_on,
        summary=list(plan.summary),
        sources=list(plan.sources),
        entries=entries,
        risks=risks,
    )


def render_rules_matrix_markdown(report: RulesMatrixReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Reviewed on: `{report.reviewed_on}`",
        f"- Project root: `{report.project_root}`",
        f"- Plan path: `{report.plan_path or '-'}`",
        "",
    ]

    if report.summary:
        lines.extend(["## Executive Summary", ""])
        lines.extend(f"- {summary_line}" for summary_line in report.summary)
        lines.append("")

    lines.extend(
        [
            "## Overview",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Tracked items", str(report.item_count)],
                    ["Allow", str(report.decision_counts.get("allow", 0))],
                    ["Deny", str(report.decision_counts.get("deny", 0))],
                    ["Unknown", str(report.decision_counts.get("unknown", 0))],
                    ["Open questions", str(report.open_question_count)],
                    ["High-confidence items", str(report.confidence_counts.get("high", 0))],
                    ["Medium-confidence items", str(report.confidence_counts.get("medium", 0))],
                    ["Low-confidence items", str(report.confidence_counts.get("low", 0))],
                    ["Categories", _format_counts(report.category_counts)],
                    ["Risk severities", _format_counts(report.risk_severity_counts)],
                ],
            ),
            "",
            "## Source Snapshot",
            "",
            _markdown_table(
                ["Source", "Kind", "Reviewed on", "URL"],
                [
                    [source.title, source.kind, source.reviewed_on, source.url]
                    for source in report.sources
                ],
            ),
            "",
            "## Source Details",
            "",
        ]
    )

    for source in report.sources:
        lines.extend(_render_source(source))

    lines.extend(
        [
            "## Rules Matrix",
            "",
            _markdown_table(
                ["Item", "Decision", "Confidence", "Category", "Owner", "Next checkpoint"],
                [
                    [
                        entry.item.name,
                        entry.item.decision,
                        entry.item.confidence,
                        entry.item.category,
                        entry.item.owner,
                        entry.item.next_checkpoint,
                    ]
                    for entry in report.entries
                ],
            ),
            "",
            "## Decision Details",
            "",
        ]
    )

    for entry in report.entries:
        lines.extend(_render_entry(entry))

    lines.extend(
        [
            "## Risk Register",
            "",
            _markdown_table(
                ["Risk", "Severity", "Owner", "Related items"],
                [
                    [
                        risk_entry.risk.title,
                        risk_entry.risk.severity,
                        risk_entry.risk.owner,
                        ", ".join(item.id for item in risk_entry.related_items),
                    ]
                    for risk_entry in report.risks
                ],
            ),
            "",
        ]
    )

    for risk_entry in report.risks:
        lines.extend(_render_risk(risk_entry))

    return "\n".join(lines).rstrip() + "\n"


def write_rules_matrix_report(
    *,
    report: RulesMatrixReport,
    output_root: Path | str,
) -> WrittenRulesMatrixReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "dataton_rules_matrix.json"
    markdown_path = output_root_path / "dataton_rules_matrix.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_rules_matrix_markdown(report))
    return WrittenRulesMatrixReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def source_to_dict(source: RulesMatrixSource) -> dict[str, object]:
    return {
        "id": source.id,
        "title": source.title,
        "kind": source.kind,
        "url": source.url,
        "reviewed_on": source.reviewed_on,
        "evidence": list(source.evidence),
    }


def _render_entry(entry: RulesMatrixEntry) -> list[str]:
    item = entry.item
    lines = [
        f"### {item.name}",
        "",
        f"- Id: `{item.id}`",
        f"- Category: `{item.category}`",
        f"- Decision: `{item.decision}`",
        f"- Confidence: `{item.confidence}`",
        f"- Repo position: {item.repo_position}",
        f"- Reasoning: {item.reasoning}",
        f"- Owner: {item.owner}",
        f"- Clarification channel: {item.clarification_channel}",
        f"- Next checkpoint: {item.next_checkpoint}",
    ]

    if entry.sources:
        lines.append("- Primary sources:")
        lines.extend(f"  - [{source.title}]({source.url})" for source in entry.sources)

    if entry.repo_reference_checks:
        lines.append("- Repo references:")
        lines.extend(
            "  - "
            f"`{check.configured_path}` -> `{check.path_type}` "
            f"({'present' if check.exists else 'missing'})"
            for check in entry.repo_reference_checks
        )

    if item.notes:
        lines.append("- Notes:")
        lines.extend(f"  - {note}" for note in item.notes)

    lines.append("")
    return lines


def _render_source(source: RulesMatrixSource) -> list[str]:
    lines = [
        f"### {source.title}",
        "",
        f"- Id: `{source.id}`",
        f"- Kind: `{source.kind}`",
        f"- Reviewed on: `{source.reviewed_on}`",
        f"- URL: [{source.url}]({source.url})",
    ]
    if source.evidence:
        lines.append("- Evidence:")
        lines.extend(f"  - {evidence}" for evidence in source.evidence)
    lines.append("")
    return lines


def _render_risk(risk_entry: RulesMatrixRiskEntry) -> list[str]:
    risk = risk_entry.risk
    lines = [
        f"### {risk.title}",
        "",
        f"- Id: `{risk.id}`",
        f"- Severity: `{risk.severity}`",
        f"- Description: {risk.description}",
        f"- Owner: {risk.owner}",
        f"- Clarification channel: {risk.clarification_channel}",
        f"- Mitigation: {risk.mitigation}",
    ]
    if risk_entry.related_items:
        lines.append("- Related items:")
        lines.extend(
            f"  - `{item.id}` {item.name} (`{item.decision}`)" for item in risk_entry.related_items
        )
    lines.append("")
    return lines


def _build_path_check(*, project_root: Path, configured_path: str) -> RulesMatrixPathCheck:
    resolved_path = resolve_project_path(str(project_root), configured_path)
    if resolved_path.is_dir():
        path_type = "dir"
    elif resolved_path.is_file():
        path_type = "file"
    else:
        path_type = "missing"
    return RulesMatrixPathCheck(
        configured_path=configured_path,
        resolved_path=str(resolved_path),
        exists=resolved_path.exists(),
        path_type=path_type,
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def _require_literal[T: str](
    data: dict[str, object],
    key: str,
    allowed_values: tuple[T, ...],
) -> T:
    value = _require_str(data, key)
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"Rules matrix field '{key}' must be one of: {allowed}.")
    return cast(T, value)


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Rules matrix field '{key}' is missing or invalid.")
    return value


def _require_str_list(data: dict[str, object], key: str) -> list[str]:
    values = _optional_str_list(data, key)
    if not values:
        raise ValueError(f"Rules matrix field '{key}' must contain at least one string.")
    return values


def _optional_str_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Rules matrix field '{key}' must be a list of strings.")
    return cast(list[str], list(value))


def _require_table_list(data: dict[str, object], key: str) -> list[dict[str, object]]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Rules matrix plan must define at least one [[{key}]] entry.")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Rules matrix field '{key}' must contain TOML tables.")
    return cast(list[dict[str, object]], list(value))


def _ensure_unique_ids(kind: str, ids: list[str]) -> None:
    duplicates = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {kind} ids are not allowed: {', '.join(duplicates)}.")


def _format_counts(counts: dict[str, int]) -> str:
    populated = {name: count for name, count in counts.items() if count > 0}
    if not populated:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in populated.items())


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
