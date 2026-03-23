"""Helpers for reading quarantine manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .common import relative_to_project, sorted_counts
from .models import QuarantineManifestSummary, QuarantineSummary


def collect_quarantine_summary(
    *,
    project_root: Path | str,
    manifests_root: Path | str,
) -> QuarantineSummary:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifests_root_path = resolve_project_path(str(project_root_path), str(manifests_root))
    manifests: list[QuarantineManifestSummary] = []
    issue_counts: Counter[str] = Counter()
    invalid_line_count = 0

    if manifests_root_path.exists():
        for manifest_path in sorted(manifests_root_path.rglob("*quarantine*.jsonl")):
            row_count = 0
            for raw_line in manifest_path.read_text().splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    invalid_line_count += 1
                    continue
                row_count += 1
                issue_code = payload.get("quality_issue_code") or payload.get("issue_code")
                if isinstance(issue_code, str) and issue_code:
                    issue_counts[issue_code] += 1
                else:
                    issue_counts["unspecified"] += 1
            manifests.append(
                QuarantineManifestSummary(
                    manifest_path=relative_to_project(manifest_path, project_root_path),
                    row_count=row_count,
                )
            )

    return QuarantineSummary(
        manifests=tuple(manifests),
        issue_counts=sorted_counts(issue_counts),
        invalid_line_count=invalid_line_count,
    )
