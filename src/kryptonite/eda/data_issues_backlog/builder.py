"""Build the cleanup backlog from the available EDA reports."""

from __future__ import annotations

from pathlib import Path

from kryptonite.deployment import resolve_project_path

from ..dataset_audio_quality import build_dataset_audio_quality_report
from ..dataset_leakage import build_dataset_leakage_report
from ..dataset_profile import build_dataset_profile_report
from .common import merge_warnings, utc_now
from .issues import (
    build_audio_quality_issues,
    build_leakage_issues,
    build_profile_issues,
    build_quarantine_issues,
    build_sources,
    build_stop_rules,
    deduplicate_issues,
    issue_sort_key,
)
from .models import DataIssuesBacklogReport
from .quarantine import collect_quarantine_summary


def build_data_issues_backlog_report(
    *,
    project_root: Path | str,
    manifests_root: Path | str,
) -> DataIssuesBacklogReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifests_root_path = resolve_project_path(str(project_root_path), str(manifests_root))

    profile = build_dataset_profile_report(
        project_root=project_root_path,
        manifests_root=manifests_root_path,
    )
    leakage = build_dataset_leakage_report(
        project_root=project_root_path,
        manifests_root=manifests_root_path,
    )
    audio_quality = build_dataset_audio_quality_report(
        project_root=project_root_path,
        manifests_root=manifests_root_path,
    )
    quarantine = collect_quarantine_summary(
        project_root=project_root_path,
        manifests_root=manifests_root_path,
    )

    issues = [
        *build_profile_issues(profile),
        *build_leakage_issues(leakage),
        *build_quarantine_issues(quarantine),
        *build_audio_quality_issues(audio_quality),
    ]
    issues = deduplicate_issues(issues)
    issues.sort(key=issue_sort_key)

    warnings = merge_warnings(
        profile.warnings,
        leakage.warnings,
        audio_quality.warnings,
        _quarantine_warnings(quarantine.invalid_line_count),
    )

    return DataIssuesBacklogReport(
        generated_at=utc_now(),
        project_root=str(project_root_path),
        manifests_root=str(manifests_root_path),
        profile_manifest_count=profile.manifest_count,
        leakage_finding_count=leakage.finding_count,
        audio_pattern_count=len(audio_quality.patterns),
        quarantine_manifest_count=quarantine.manifest_count,
        quarantine_row_count=quarantine.row_count,
        sources=build_sources(quarantine),
        issues=issues,
        stop_rules=build_stop_rules(issues),
        warnings=warnings,
    )


def _quarantine_warnings(invalid_line_count: int) -> list[str]:
    if invalid_line_count == 0:
        return []
    return [(f"{invalid_line_count} invalid JSONL lines were skipped in quarantine manifests.")]
