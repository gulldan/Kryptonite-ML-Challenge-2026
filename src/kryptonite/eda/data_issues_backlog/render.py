"""Markdown and file rendering for the cleanup backlog."""

from __future__ import annotations

import json
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .common import format_counts, markdown_table
from .models import DataIssuesBacklogReport, WrittenDataIssuesBacklogReport


def render_data_issues_backlog_markdown(report: DataIssuesBacklogReport) -> str:
    lines = [
        "# Data Issues Backlog",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Manifests root: `{report.manifests_root}`",
        "",
        "## Source Inputs",
        "",
        markdown_table(
            ["Source", "Script", "Artifact"],
            [[source.name, source.script_path, source.artifact_path] for source in report.sources],
        ),
        "",
    ]

    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")

    lines.extend(
        [
            "## Overview",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["Issues", str(report.issue_count)],
                    ["By severity", format_counts(report.issue_counts_by_severity)],
                    ["By action", format_counts(report.issue_counts_by_action)],
                    ["Stop rules", str(len(report.stop_rules))],
                    ["Profiled manifests", str(report.profile_manifest_count)],
                    ["Leakage findings", str(report.leakage_finding_count)],
                    ["Audio-quality patterns", str(report.audio_pattern_count)],
                    ["Quarantine manifests", str(report.quarantine_manifest_count)],
                    ["Quarantined rows", str(report.quarantine_row_count)],
                ],
            ),
            "",
            "## Decision Summary",
            "",
            markdown_table(
                ["Severity", "Action", "Category", "Code", "Summary"],
                [
                    [
                        issue.severity,
                        issue.action,
                        issue.category,
                        issue.code,
                        issue.summary,
                    ]
                    for issue in report.issues
                ]
                or [
                    [
                        "-",
                        "-",
                        "-",
                        "-",
                        "No active cleanup or documentation decisions were generated.",
                    ]
                ],
            ),
            "",
            "## Stop Rules",
            "",
        ]
    )

    if report.stop_rules:
        lines.extend(f"- {rule}" for rule in report.stop_rules)
    else:
        lines.append("_No stop-rules were triggered by the current manifests._")
    lines.append("")

    lines.extend(["## Issues", ""])
    if not report.issues:
        lines.append("_No active data issues were generated from the current manifests._")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    for issue in report.issues:
        lines.extend(
            [
                f"### {issue.title}",
                "",
                f"- Severity: `{issue.severity}`",
                f"- Action: `{issue.action}`",
                f"- Category: `{issue.category}`",
                f"- Code: `{issue.code}`",
                f"- Summary: {issue.summary}",
                f"- Rationale: {issue.rationale}",
            ]
        )
        if issue.stop_rule is not None:
            lines.append(f"- Stop rule: {issue.stop_rule}")
        if issue.evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - {evidence}" for evidence in issue.evidence)
        if issue.references:
            lines.append("- References:")
            lines.extend(f"  - `{reference}`" for reference in issue.references)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_data_issues_backlog_report(
    *,
    report: DataIssuesBacklogReport,
    output_root: Path | str,
) -> WrittenDataIssuesBacklogReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "data_issues_backlog.json"
    markdown_path = output_root_path / "data_issues_backlog.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_data_issues_backlog_markdown(report))
    return WrittenDataIssuesBacklogReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )
