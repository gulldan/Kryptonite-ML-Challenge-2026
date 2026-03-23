"""Turn EDA outputs into an executable data-cleanup backlog."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .audio_quality.models import AudioQualityPattern, DatasetAudioQualityReport
from .dataset_audio_quality import build_dataset_audio_quality_report
from .dataset_leakage import AuditFinding, DatasetLeakageReport, build_dataset_leakage_report
from .dataset_profile import DatasetProfileReport, build_dataset_profile_report

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
        counts = Counter(issue.action for issue in self.issues)
        ordered = sorted(counts.items(), key=lambda item: ACTION_ORDER.get(item[0], 999))
        return {action: count for action, count in ordered}

    @property
    def issue_counts_by_severity(self) -> dict[str, int]:
        counts = Counter(issue.severity for issue in self.issues)
        ordered = sorted(counts.items(), key=lambda item: SEVERITY_ORDER.get(item[0], 999))
        return {severity: count for severity, count in ordered}

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
        *_build_profile_issues(profile),
        *_build_leakage_issues(leakage),
        *_build_quarantine_issues(quarantine),
        *_build_audio_quality_issues(audio_quality),
    ]
    issues = _deduplicate_issues(issues)
    issues.sort(key=_issue_sort_key)

    stop_rules = _build_stop_rules(issues)
    warnings = _merge_warnings(
        profile.warnings,
        leakage.warnings,
        audio_quality.warnings,
        [
            (
                f"{quarantine.invalid_line_count} invalid JSONL lines were skipped in "
                "quarantine manifests."
            )
        ]
        if quarantine.invalid_line_count
        else [],
    )
    sources = _build_sources(quarantine)

    return DataIssuesBacklogReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        manifests_root=str(manifests_root_path),
        profile_manifest_count=profile.manifest_count,
        leakage_finding_count=leakage.finding_count,
        audio_pattern_count=len(audio_quality.patterns),
        quarantine_manifest_count=quarantine.manifest_count,
        quarantine_row_count=quarantine.row_count,
        sources=sources,
        issues=issues,
        stop_rules=stop_rules,
        warnings=warnings,
    )


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
        _markdown_table(
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
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Issues", str(report.issue_count)],
                    ["By severity", _format_counts(report.issue_counts_by_severity)],
                    ["By action", _format_counts(report.issue_counts_by_action)],
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
            _markdown_table(
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
                    manifest_path=_relative_to_project(manifest_path, project_root_path),
                    row_count=row_count,
                )
            )

    return QuarantineSummary(
        manifests=tuple(manifests),
        issue_counts=_sorted_counts(issue_counts),
        invalid_line_count=invalid_line_count,
    )


def _build_profile_issues(report: DatasetProfileReport) -> list[DataIssue]:
    issues: list[DataIssue] = []
    references = (
        "scripts/dataset_profile_report.py",
        "artifacts/eda/dataset-profile/dataset_profile.md",
    )

    if report.invalid_line_count:
        issues.append(
            DataIssue(
                code="invalid_json_lines",
                severity="high",
                action="fix",
                category="integrity",
                title="Manifest JSONL must stay parse-clean",
                summary=(
                    f"{report.invalid_line_count} invalid JSONL lines were skipped by profiling."
                ),
                rationale=(
                    "Invalid manifest rows break reproducibility and can silently change corpus "
                    "shape between runs."
                ),
                stop_rule=(
                    "Regenerate manifests until the profile report shows zero invalid JSONL lines."
                ),
                evidence=(f"profiled_manifests={report.manifest_count}",),
                references=references,
            )
        )

    missing_audio_paths = report.total_summary.missing_audio_path_count
    if missing_audio_paths:
        issues.append(
            DataIssue(
                code="missing_audio_path",
                severity="high",
                action="fix",
                category="integrity",
                title="Rows without `audio_path` must be repaired before use",
                summary=f"{missing_audio_paths} rows are missing `audio_path` metadata.",
                rationale=(
                    "Rows without resolvable audio paths cannot be validated, resampled, or fed "
                    "into downstream preprocessing."
                ),
                stop_rule=(
                    "Rows missing `audio_path` cannot appear in active manifests for baseline "
                    "or training."
                ),
                evidence=(f"rows_with_audio_path={report.total_summary.entries_with_audio_path}",),
                references=references,
            )
        )

    return issues


def _build_leakage_issues(report: DatasetLeakageReport) -> list[DataIssue]:
    issues: list[DataIssue] = []
    references = (
        "scripts/dataset_leakage_report.py",
        "artifacts/eda/dataset-leakage/dataset_leakage.md",
    )

    for finding in report.findings:
        action = _leakage_action(finding.code)
        issues.append(
            DataIssue(
                code=finding.code,
                severity=finding.severity,
                action=action,
                category=finding.category,
                title=finding.title,
                summary=finding.summary,
                rationale=finding.impact,
                stop_rule=_leakage_stop_rule(finding.code),
                evidence=tuple(_finding_evidence(finding)),
                references=references,
            )
        )
    return issues


def _build_quarantine_issues(summary: QuarantineSummary) -> list[DataIssue]:
    if summary.row_count == 0:
        return []

    severity = "medium"
    if any(
        code in {"duplicate_audio_content", "cross_split_audio_overlap"}
        for code in summary.issue_counts
    ):
        severity = "high"

    evidence = [
        f"{manifest.manifest_path}: {manifest.row_count} rows" for manifest in summary.manifests[:3]
    ]
    if summary.issue_counts:
        evidence.append(f"issue_counts={_format_counts(summary.issue_counts)}")

    references = [
        "scripts/prepare_ffsvc2022_surrogate.py",
        *(manifest.manifest_path for manifest in summary.manifests[:2]),
    ]
    return [
        DataIssue(
            code="quarantined_rows",
            severity=severity,
            action="quarantine",
            category="cleanup",
            title="Confirmed bad rows stay in quarantine manifests",
            summary=(
                f"{summary.row_count} rows are already parked in {summary.manifest_count} "
                "quarantine manifest(s)."
            ),
            rationale=(
                "Confirmed integrity failures should remain auditable in quarantine rather than "
                "silently disappearing or leaking back into active train/dev coverage."
            ),
            stop_rule=(
                "Do not reintroduce quarantined rows into active manifests or trial bundles "
                "without a fresh audit and an explicit canonical replacement."
            ),
            evidence=tuple(evidence),
            references=tuple(references),
        )
    ]


def _build_audio_quality_issues(report: DatasetAudioQualityReport) -> list[DataIssue]:
    issues: list[DataIssue] = []
    references = (
        "scripts/dataset_audio_quality_report.py",
        "artifacts/eda/dataset-audio-quality/dataset_audio_quality.md",
    )

    for pattern in report.patterns:
        action = _audio_action(pattern.code)
        issues.append(
            DataIssue(
                code=pattern.code,
                severity=_audio_severity(pattern.code),
                action=action,
                category="audio-quality",
                title=_audio_title(pattern.code),
                summary=pattern.summary,
                rationale=pattern.implication,
                stop_rule=_audio_stop_rule(pattern.code),
                evidence=tuple(_audio_pattern_evidence(pattern, report)),
                references=references,
            )
        )
    return issues


def _build_sources(quarantine: QuarantineSummary) -> list[BacklogSource]:
    sources = [
        BacklogSource(
            name="dataset profile",
            script_path="scripts/dataset_profile_report.py",
            artifact_path="artifacts/eda/dataset-profile/dataset_profile.md",
        ),
        BacklogSource(
            name="dataset leakage audit",
            script_path="scripts/dataset_leakage_report.py",
            artifact_path="artifacts/eda/dataset-leakage/dataset_leakage.md",
        ),
        BacklogSource(
            name="dataset audio quality",
            script_path="scripts/dataset_audio_quality_report.py",
            artifact_path="artifacts/eda/dataset-audio-quality/dataset_audio_quality.md",
        ),
    ]
    if quarantine.manifest_count:
        sources.append(
            BacklogSource(
                name="quarantine manifests",
                script_path="scripts/prepare_ffsvc2022_surrogate.py",
                artifact_path=quarantine.manifests[0].manifest_path,
            )
        )
    return sources


def _deduplicate_issues(issues: list[DataIssue]) -> list[DataIssue]:
    deduplicated: dict[str, DataIssue] = {}
    for issue in issues:
        existing = deduplicated.get(issue.code)
        if existing is None or _issue_sort_key(issue) < _issue_sort_key(existing):
            deduplicated[issue.code] = issue
    return list(deduplicated.values())


def _build_stop_rules(issues: list[DataIssue]) -> list[str]:
    ordered: list[str] = []
    for issue in issues:
        if issue.stop_rule is None or issue.stop_rule in ordered:
            continue
        ordered.append(issue.stop_rule)
    return ordered


def _merge_warnings(*warning_sets: list[str]) -> list[str]:
    merged: list[str] = []
    for warnings in warning_sets:
        for warning in warnings:
            if warning not in merged:
                merged.append(warning)
    return merged


def _leakage_action(code: str) -> str:
    return {
        "no_data_manifests": "fix",
        "missing_required_split": "fix",
        "unknown_split_rows": "fix",
        "missing_speaker_id": "fix",
        "duplicate_row_within_manifest": "fix",
        "cross_split_audio_overlap": "fix",
        "unexpected_same_split_overlap": "fix",
        "duplicate_audio_content": "quarantine",
        "speaker_overlap": "fix",
        "session_overlap": "fix",
        "manifest_split_mismatch": "fix",
        "all_manifest_missing_rows": "fix",
        "all_manifest_extra_rows": "fix",
        "trial_reference_missing": "fix",
        "speaker_disjoint_trial_split_violation": "fix",
    }.get(code, "document")


def _leakage_stop_rule(code: str) -> str | None:
    return {
        "no_data_manifests": (
            "Baseline, threshold tuning, and evaluation stay blocked until active manifests "
            "exist under the configured manifests root."
        ),
        "missing_required_split": (
            "Do not start baseline, threshold tuning, or training until active train/dev "
            "manifests exist and the leakage audit reruns clean."
        ),
        "unknown_split_rows": "Rows without an auditable split cannot stay in active manifests.",
        "missing_speaker_id": (
            "Rows without `speaker_id` cannot be used for speaker-disjoint split generation "
            "or verification trials."
        ),
        "duplicate_row_within_manifest": (
            "Regenerate manifests until duplicate rows are removed from active train/dev coverage."
        ),
        "cross_split_audio_overlap": (
            "Any row appearing in more than one split blocks train/dev evaluation until split "
            "assignment is corrected."
        ),
        "unexpected_same_split_overlap": (
            "Unexpected same-split duplicates must be explained or removed before corpus-wide "
            "counts become a source of truth."
        ),
        "duplicate_audio_content": (
            "Confirmed duplicate-content rows must remain quarantined until a canonical row is "
            "selected and manifests are rebuilt."
        ),
        "speaker_overlap": (
            "Speaker-disjoint evaluation is invalid while the same speaker appears in more than "
            "one active split."
        ),
        "session_overlap": (
            "Channel/session overlap must be resolved or explicitly accepted before dev metrics "
            "are treated as held-out."
        ),
        "manifest_split_mismatch": (
            "Manifest filenames and row-level split labels must agree before downstream loaders "
            "can be trusted."
        ),
        "all_manifest_missing_rows": (
            "Aggregate `all_manifest` coverage must stay aligned with split-specific manifests "
            "before audits and loaders consume it as truth."
        ),
        "all_manifest_extra_rows": (
            "Aggregate-only rows must be reconciled before `all_manifest` is used as the "
            "canonical corpus index."
        ),
        "trial_reference_missing": (
            "Verification trials cannot be treated as executable until every referenced audio "
            "identifier resolves to a manifest row."
        ),
        "speaker_disjoint_trial_split_violation": (
            "Speaker-disjoint trial bundles must reference dev-only audio before threshold tuning "
            "or verification benchmarking."
        ),
    }.get(code)


def _finding_evidence(finding: AuditFinding) -> list[str]:
    evidence: list[str] = []
    for example in finding.examples[:2]:
        details = ", ".join(
            f"{key}={_format_value(value)}" for key, value in sorted(example.details.items())
        )
        evidence.append(f"{example.label}: {details}")
    return evidence


def _audio_action(code: str) -> str:
    return {
        "mixed_sample_rates": "fix",
        "non_mono_audio": "fix",
        "silence_heavy_tail": "keep",
        "low_level_recordings": "keep",
        "clipping_present": "document",
        "duration_long_tail": "document",
        "inspection_gaps": "quarantine",
    }.get(code, "document")


def _audio_severity(code: str) -> str:
    return {
        "mixed_sample_rates": "high",
        "non_mono_audio": "medium",
        "silence_heavy_tail": "high",
        "low_level_recordings": "high",
        "clipping_present": "medium",
        "duration_long_tail": "medium",
        "inspection_gaps": "high",
    }.get(code, "medium")


def _audio_title(code: str) -> str:
    return {
        "mixed_sample_rates": "Resample active audio to the training target rate",
        "non_mono_audio": "Fold non-mono audio deterministically",
        "silence_heavy_tail": "Keep silence-heavy rows as part of the robust setting",
        "low_level_recordings": "Keep quiet recordings and normalize them explicitly",
        "clipping_present": "Track clipping as a documented robustness slice",
        "duration_long_tail": "Document chunking policy for long-tail durations",
        "inspection_gaps": "Rows with missing or unreadable audio must leave the active set",
    }.get(code, code.replace("_", " "))


def _audio_stop_rule(code: str) -> str | None:
    return {
        "mixed_sample_rates": (
            "Feature extraction and augmentation cannot rely on implicit backend resampling; "
            "the preprocessing contract must resample explicitly to 16 kHz."
        ),
        "non_mono_audio": (
            "Multi-channel rows must be folded down deterministically before scoring, export, "
            "or demo evaluation."
        ),
        "silence_heavy_tail": (
            "Silence-heavy rows stay in the corpus; compare `no VAD`, `light trimming`, and "
            "`aggressive trimming` instead of deleting them."
        ),
        "low_level_recordings": (
            "Quiet rows stay in the corpus; any loudness normalization must be bounded and "
            "logged as an explicit preprocessing choice."
        ),
        "clipping_present": (
            "Clipping remains a tracked slice in reports and benchmarks; do not silently drop "
            "those rows unless a later audit proves they are corrupt."
        ),
        "duration_long_tail": (
            "Chunking or truncation policy must be explicit before large-batch training or "
            "latency benchmarking."
        ),
        "inspection_gaps": (
            "Rows missing audio files or failing waveform inspection cannot appear in active "
            "preprocessing, training, or evaluation manifests."
        ),
    }.get(code)


def _audio_pattern_evidence(
    pattern: AudioQualityPattern,
    report: DatasetAudioQualityReport,
) -> list[str]:
    summary = report.total_summary
    evidence: list[str] = []
    if pattern.code == "mixed_sample_rates":
        evidence.append(f"sample_rate_counts={_format_counts(summary.sample_rate_counts)}")
    elif pattern.code == "non_mono_audio":
        evidence.append(f"channel_counts={_format_counts(summary.channel_counts)}")
    elif pattern.code == "silence_heavy_tail":
        evidence.append(
            "flag_counts="
            + _format_counts(
                _select_counts(
                    summary.flag_counts,
                    ("high_silence_ratio", "moderate_silence_ratio"),
                )
            )
        )
        evidence.append(f"silence_p95={_format_value(summary.silence_summary.p95)}")
    elif pattern.code == "low_level_recordings":
        evidence.append(
            "flag_counts="
            + _format_counts(
                _select_counts(summary.flag_counts, ("very_low_loudness", "low_loudness"))
            )
        )
        evidence.append(f"mean_loudness={_format_value(summary.loudness_summary.mean)}")
    elif pattern.code == "clipping_present":
        evidence.append(f"clipping_risk={summary.flag_counts.get('clipping_risk', 0)}")
        evidence.append(f"peak_max={_format_value(summary.peak_summary.maximum)}")
    elif pattern.code == "duration_long_tail":
        evidence.append(f"duration_p95={_format_value(summary.duration_summary.p95)}")
        evidence.append(f"duration_max={_format_value(summary.duration_summary.maximum)}")
    elif pattern.code == "inspection_gaps":
        evidence.append(f"missing_audio_file_count={summary.missing_audio_file_count}")
        evidence.append(f"audio_inspection_error_count={summary.audio_inspection_error_count}")
    return evidence


def _select_counts(counts: dict[str, int], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: counts[key] for key in keys if counts.get(key)}


def _issue_sort_key(issue: DataIssue) -> tuple[int, int, str]:
    return (
        SEVERITY_ORDER.get(issue.severity, 999),
        ACTION_ORDER.get(issue.action, 999),
        issue.code,
    )


def _sorted_counts(counts: Counter[str] | dict[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _format_value(value: object) -> str:
    if isinstance(value, dict):
        integer_counts = {str(key): int(val) for key, val in value.items() if isinstance(val, int)}
        if integer_counts:
            return _format_counts(integer_counts)
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value[:5]) + "]"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
