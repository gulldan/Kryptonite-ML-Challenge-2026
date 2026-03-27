"""Render and write release postmortem reports."""

from __future__ import annotations

import json
from pathlib import Path

from .release_postmortem_builder import sort_backlog_items
from .release_postmortem_models import (
    RELEASE_POSTMORTEM_JSON_NAME,
    RELEASE_POSTMORTEM_MARKDOWN_NAME,
    ReleaseBacklogItem,
    ReleasePostmortemFinding,
    ReleasePostmortemReport,
    WrittenReleasePostmortem,
)


def write_release_postmortem(report: ReleasePostmortemReport) -> WrittenReleasePostmortem:
    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / RELEASE_POSTMORTEM_JSON_NAME
    markdown_path = output_root / RELEASE_POSTMORTEM_MARKDOWN_NAME

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_release_postmortem_markdown(report) + "\n",
        encoding="utf-8",
    )

    return WrittenReleasePostmortem(
        output_root=str(output_root),
        report_json_path=str(json_path),
        report_markdown_path=str(markdown_path),
        summary=report.summary,
    )


def render_release_postmortem_markdown(report: ReleasePostmortemReport) -> str:
    lines = [f"# {report.title}", ""]
    if report.summary_text:
        lines.extend([report.summary_text, ""])
    lines.extend(
        [
            "## Summary",
            "",
            f"- Release id: `{report.release_id}`",
            f"- Release tag: `{report.release_tag or 'unversioned'}`",
            f"- Worked findings: `{report.summary.worked_count}`",
            f"- Missed findings: `{report.summary.missed_count}`",
            f"- Risk findings: `{report.summary.risk_count}`",
            f"- Next-iteration items: `{report.summary.next_iteration_count}`",
            f"- De-scoped items: `{report.summary.de_scoped_count}`",
            f"- Monitor items: `{report.summary.monitor_count}`",
        ]
    )
    if report.summary.highest_priority_next_items:
        lines.extend(
            [
                (
                    "- Highest-priority next items: "
                    + ", ".join(f"`{item}`" for item in report.summary.highest_priority_next_items)
                ),
                "",
            ]
        )
    else:
        lines.append("")

    lines.extend(
        [
            "## Evidence",
            "",
            _markdown_table(
                headers=["Label", "Kind", "Path", "Files", "SHA256"],
                rows=[
                    [
                        item.label,
                        item.kind,
                        item.path,
                        str(item.file_count),
                        item.sha256 or "-",
                    ]
                    for item in report.evidence
                ],
            ),
            "",
            "## What Worked",
            "",
        ]
    )
    lines.extend(_render_findings(_filter_findings(report.findings, outcome="worked")))
    lines.extend(["", "## What Did Not Ship", ""])
    lines.extend(_render_findings(_filter_findings(report.findings, outcome="missed")))
    lines.extend(["", "## Risks", ""])
    lines.extend(_render_findings(_filter_findings(report.findings, outcome="risk")))

    ordered_backlog = sort_backlog_items(report.backlog_items)
    lines.extend(
        [
            "",
            "## Backlog v2",
            "",
            _markdown_table(
                headers=["Priority", "Disposition", "Area", "Linear", "Item", "Dependencies"],
                rows=[
                    [
                        item.priority,
                        _humanize(item.disposition),
                        _humanize(item.area),
                        item.related_issue or "-",
                        item.title,
                        ", ".join(item.dependencies) if item.dependencies else "-",
                    ]
                    for item in ordered_backlog
                ],
            ),
            "",
        ]
    )
    for section_title, disposition in (
        ("Next Iteration", "next_iteration"),
        ("Monitor", "monitor"),
        ("De-Scoped", "de_scoped"),
    ):
        section_items = [item for item in ordered_backlog if item.disposition == disposition]
        lines.extend([f"### {section_title}", ""])
        if not section_items:
            lines.append("- None.")
        else:
            lines.extend(_render_backlog_items(section_items))
        lines.append("")

    if report.validation_commands:
        lines.extend(["## Validation", ""])
        for command in report.validation_commands:
            lines.extend(["```bash", command, "```", ""])

    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    if report.source_config_path is not None:
        lines.extend(["## Source Config", ""])
        lines.append(f"- Config: `{report.source_config_path}`")
        lines.append(f"- SHA256: `{report.source_config_sha256 or '-'}`")
        lines.append("")

    return "\n".join(lines).rstrip()


def _filter_findings(
    findings: tuple[ReleasePostmortemFinding, ...],
    *,
    outcome: str,
) -> tuple[ReleasePostmortemFinding, ...]:
    return tuple(item for item in findings if item.outcome == outcome)


def _render_findings(findings: tuple[ReleasePostmortemFinding, ...]) -> list[str]:
    if not findings:
        return ["- None."]
    lines: list[str] = []
    for item in findings:
        headline = f"- [{_humanize(item.area)}] {item.title}"
        if item.related_issues:
            headline += " (" + ", ".join(f"`{issue}`" for issue in item.related_issues) + ")"
        lines.append(headline)
        lines.append(f"  {item.detail}")
        if item.evidence_labels:
            lines.append("  Evidence: " + ", ".join(f"`{label}`" for label in item.evidence_labels))
    return lines


def _render_backlog_items(items: list[ReleaseBacklogItem]) -> list[str]:
    lines: list[str] = []
    for item in items:
        headline = (
            f"- `{item.priority}` [{_humanize(item.area)}] {item.title}"
            f" ({item.related_issue or 'no Linear issue'})"
        )
        lines.append(headline)
        lines.append(f"  {item.rationale}")
        if item.dependencies:
            dependency_text = ", ".join(f"`{value}`" for value in item.dependencies)
            lines.append(f"  Dependencies: {dependency_text}")
    return lines


def _markdown_table(*, headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _humanize(value: str) -> str:
    return value.replace("_", " ").title()


__all__ = [
    "render_release_postmortem_markdown",
    "write_release_postmortem",
]
