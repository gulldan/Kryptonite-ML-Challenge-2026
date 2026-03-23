"""Issue classification policy for the cleanup backlog."""

from __future__ import annotations

from ..audio_quality.models import AudioQualityPattern, DatasetAudioQualityReport
from ..dataset_leakage import AuditFinding, DatasetLeakageReport
from ..dataset_profile import DatasetProfileReport
from .common import format_counts, format_value
from .models import (
    ACTION_ORDER,
    SEVERITY_ORDER,
    BacklogSource,
    DataIssue,
    QuarantineSummary,
)


def build_profile_issues(report: DatasetProfileReport) -> list[DataIssue]:
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


def build_leakage_issues(report: DatasetLeakageReport) -> list[DataIssue]:
    references = (
        "scripts/dataset_leakage_report.py",
        "artifacts/eda/dataset-leakage/dataset_leakage.md",
    )
    return [
        DataIssue(
            code=finding.code,
            severity=finding.severity,
            action=_leakage_action(finding.code),
            category=finding.category,
            title=finding.title,
            summary=finding.summary,
            rationale=finding.impact,
            stop_rule=_leakage_stop_rule(finding.code),
            evidence=tuple(_finding_evidence(finding)),
            references=references,
        )
        for finding in report.findings
    ]


def build_quarantine_issues(summary: QuarantineSummary) -> list[DataIssue]:
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
        evidence.append(f"issue_counts={format_counts(summary.issue_counts)}")

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


def build_audio_quality_issues(report: DatasetAudioQualityReport) -> list[DataIssue]:
    references = (
        "scripts/dataset_audio_quality_report.py",
        "artifacts/eda/dataset-audio-quality/dataset_audio_quality.md",
    )
    return [
        DataIssue(
            code=pattern.code,
            severity=_audio_severity(pattern.code),
            action=_audio_action(pattern.code),
            category="audio-quality",
            title=_audio_title(pattern.code),
            summary=pattern.summary,
            rationale=pattern.implication,
            stop_rule=_audio_stop_rule(pattern.code),
            evidence=tuple(_audio_pattern_evidence(pattern, report)),
            references=references,
        )
        for pattern in report.patterns
    ]


def build_sources(quarantine: QuarantineSummary) -> list[BacklogSource]:
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


def deduplicate_issues(issues: list[DataIssue]) -> list[DataIssue]:
    deduplicated: dict[str, DataIssue] = {}
    for issue in issues:
        existing = deduplicated.get(issue.code)
        if existing is None or issue_sort_key(issue) < issue_sort_key(existing):
            deduplicated[issue.code] = issue
    return list(deduplicated.values())


def build_stop_rules(issues: list[DataIssue]) -> list[str]:
    ordered: list[str] = []
    for issue in issues:
        if issue.stop_rule is None or issue.stop_rule in ordered:
            continue
        ordered.append(issue.stop_rule)
    return ordered


def issue_sort_key(issue: DataIssue) -> tuple[int, int, str]:
    return (
        SEVERITY_ORDER.get(issue.severity, 999),
        ACTION_ORDER.get(issue.action, 999),
        issue.code,
    )


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
            "Speaker-disjoint trial bundles must reference dev-only audio before threshold "
            "tuning or verification benchmarking."
        ),
    }.get(code)


def _finding_evidence(finding: AuditFinding) -> list[str]:
    evidence: list[str] = []
    for example in finding.examples[:2]:
        details = ", ".join(
            f"{key}={format_value(value)}" for key, value in sorted(example.details.items())
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
        evidence.append(f"sample_rate_counts={format_counts(summary.sample_rate_counts)}")
    elif pattern.code == "non_mono_audio":
        evidence.append(f"channel_counts={format_counts(summary.channel_counts)}")
    elif pattern.code == "silence_heavy_tail":
        evidence.append(
            "flag_counts="
            + format_counts(
                _select_counts(
                    summary.flag_counts,
                    ("high_silence_ratio", "moderate_silence_ratio"),
                )
            )
        )
        evidence.append(f"silence_p95={format_value(summary.silence_summary.p95)}")
    elif pattern.code == "low_level_recordings":
        evidence.append(
            "flag_counts="
            + format_counts(
                _select_counts(
                    summary.flag_counts,
                    ("very_low_loudness", "low_loudness"),
                )
            )
        )
        evidence.append(f"mean_loudness={format_value(summary.loudness_summary.mean)}")
    elif pattern.code == "clipping_present":
        evidence.append(f"clipping_risk={summary.flag_counts.get('clipping_risk', 0)}")
        evidence.append(f"peak_max={format_value(summary.peak_summary.maximum)}")
    elif pattern.code == "duration_long_tail":
        evidence.append(f"duration_p95={format_value(summary.duration_summary.p95)}")
        evidence.append(f"duration_max={format_value(summary.duration_summary.maximum)}")
    elif pattern.code == "inspection_gaps":
        evidence.append(f"missing_audio_file_count={summary.missing_audio_file_count}")
        evidence.append(f"audio_inspection_error_count={summary.audio_inspection_error_count}")
    return evidence


def _select_counts(counts: dict[str, int], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: counts[key] for key in keys if counts.get(key)}
