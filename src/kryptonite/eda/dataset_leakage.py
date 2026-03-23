"""Reproducible duplicate, leakage, and split-integrity audit for manifests-backed corpora."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kryptonite.data.schema import normalize_manifest_entry
from kryptonite.deployment import resolve_project_path

KNOWN_DATA_SPLITS: tuple[str, ...] = ("train", "dev", "demo")
REQUIRED_SPLITS: tuple[str, ...] = ("train", "dev")
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


@dataclass(frozen=True, slots=True)
class AuditRecord:
    manifest_path: str
    line_number: int
    dataset_name: str
    split_name: str
    speaker_id: str | None
    session_key: str | None
    utterance_id: str | None
    audio_path: str | None
    audio_basename: str | None
    identity_key: str
    audio_exists: bool
    file_size_bytes: int | None
    duration_seconds: float | None

    @property
    def location(self) -> str:
        return f"{self.manifest_path}:{self.line_number}"


@dataclass(frozen=True, slots=True)
class TrialRecord:
    manifest_path: str
    line_number: int
    label: int | None
    left_audio: str | None
    right_audio: str | None

    @property
    def location(self) -> str:
        return f"{self.manifest_path}:{self.line_number}"


@dataclass(frozen=True, slots=True)
class AuditExample:
    label: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "details": dict(self.details)}


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    severity: str
    category: str
    title: str
    summary: str
    impact: str
    examples: list[AuditExample] = field(default_factory=list)

    @property
    def example_count(self) -> int:
        return len(self.examples)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "summary": self.summary,
            "impact": self.impact,
            "example_count": self.example_count,
            "examples": [example.to_dict() for example in self.examples],
        }


@dataclass(slots=True)
class DatasetLeakageReport:
    generated_at: str
    project_root: str
    manifests_root: str
    raw_record_count: int
    deduplicated_record_count: int
    data_manifest_count: int
    trial_manifest_count: int
    trial_count: int
    invalid_line_count: int
    hashed_audio_file_count: int
    split_counts: dict[str, int]
    findings: list[AuditFinding]
    warnings: list[str]

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def finding_counts_by_severity(self) -> dict[str, int]:
        counts: Counter[str] = Counter(finding.severity for finding in self.findings)
        ordered = sorted(counts.items(), key=lambda item: SEVERITY_ORDER.get(item[0], 999))
        return {severity: count for severity, count in ordered}

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "manifests_root": self.manifests_root,
            "raw_record_count": self.raw_record_count,
            "deduplicated_record_count": self.deduplicated_record_count,
            "data_manifest_count": self.data_manifest_count,
            "trial_manifest_count": self.trial_manifest_count,
            "trial_count": self.trial_count,
            "invalid_line_count": self.invalid_line_count,
            "hashed_audio_file_count": self.hashed_audio_file_count,
            "split_counts": dict(self.split_counts),
            "finding_count": self.finding_count,
            "finding_counts_by_severity": self.finding_counts_by_severity,
            "warnings": list(self.warnings),
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class WrittenDatasetLeakageReport:
    output_root: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def build_dataset_leakage_report(
    *,
    project_root: Path | str,
    manifests_root: Path | str,
) -> DatasetLeakageReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifests_root_path = resolve_project_path(str(project_root_path), str(manifests_root))

    raw_records: list[AuditRecord] = []
    trial_records: list[TrialRecord] = []
    data_manifest_count = 0
    trial_manifest_count = 0
    invalid_line_count = 0
    warnings: list[str] = []

    if manifests_root_path.exists():
        for manifest_path in sorted(manifests_root_path.rglob("*.jsonl")):
            objects, invalid_lines = _load_jsonl_objects(manifest_path)
            invalid_line_count += invalid_lines
            manifest_name = manifest_path.name.lower()

            if "trial" in manifest_name:
                trial_manifest_count += 1
                trial_records.extend(
                    _build_trial_record(
                        manifest_path=manifest_path,
                        entry=entry,
                        line_number=line_number,
                        project_root=project_root_path,
                    )
                    for line_number, entry in enumerate(objects, start=1)
                )
                continue
            if "quarantine" in manifest_name:
                continue

            if "manifest" not in manifest_name and not any("audio_path" in row for row in objects):
                continue

            data_manifest_count += 1
            raw_records.extend(
                _build_audit_record(
                    manifest_path=manifest_path,
                    entry=entry,
                    line_number=line_number,
                    project_root=project_root_path,
                    manifests_root=manifests_root_path,
                )
                for line_number, entry in enumerate(objects, start=1)
            )
    else:
        warnings.append("Configured manifests root does not exist.")

    coverage_records = _deduplicate_records(raw_records)
    audio_hashes, hashed_audio_count = _build_audio_hashes(coverage_records, project_root_path)
    findings: list[AuditFinding] = []

    if not raw_records:
        findings.append(
            AuditFinding(
                code="no_data_manifests",
                severity="critical",
                category="coverage",
                title="No data manifests found",
                summary=(
                    "No data-manifest JSONL files were discovered under the configured "
                    "manifests root."
                ),
                impact=(
                    "Split leakage, duplicate detection, and baseline-evaluation integrity "
                    "cannot be audited until train/dev manifests are materialized."
                ),
            )
        )

    findings.extend(_build_coverage_findings(coverage_records))
    findings.extend(
        _build_duplicate_findings(
            coverage_records,
            raw_records,
            audio_hashes=audio_hashes,
            hashed_audio_count=hashed_audio_count,
        )
    )
    findings.extend(_build_overlap_findings(coverage_records))
    findings.extend(_build_manifest_integrity_findings(coverage_records, raw_records))
    findings.extend(_build_trial_findings(coverage_records, trial_records))

    if invalid_line_count:
        warnings.append(f"{invalid_line_count} invalid JSONL lines were skipped during the audit.")
    if not trial_records:
        warnings.append("No trial JSONL files were discovered under the manifests root.")

    findings.sort(key=_finding_sort_key)
    return DatasetLeakageReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        manifests_root=str(manifests_root_path),
        raw_record_count=len(raw_records),
        deduplicated_record_count=len(coverage_records),
        data_manifest_count=data_manifest_count,
        trial_manifest_count=trial_manifest_count,
        trial_count=len(trial_records),
        invalid_line_count=invalid_line_count,
        hashed_audio_file_count=hashed_audio_count,
        split_counts=_sorted_counts(Counter(record.split_name for record in coverage_records)),
        findings=findings,
        warnings=warnings,
    )


def render_dataset_leakage_markdown(report: DatasetLeakageReport) -> str:
    lines = [
        "# Dataset Leakage Audit",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Manifests root: `{report.manifests_root}`",
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
                    ["Data manifests", str(report.data_manifest_count)],
                    ["Trial manifests", str(report.trial_manifest_count)],
                    ["Raw rows", str(report.raw_record_count)],
                    ["Deduplicated rows", str(report.deduplicated_record_count)],
                    ["Trials", str(report.trial_count)],
                    ["Invalid JSON lines", str(report.invalid_line_count)],
                    ["Split coverage", _format_counts(report.split_counts)],
                    ["Findings", str(report.finding_count)],
                    ["By severity", _format_counts(report.finding_counts_by_severity)],
                    ["Hashed audio files", str(report.hashed_audio_file_count)],
                ],
            ),
            "",
            "## Findings",
            "",
            _markdown_table(
                ["Severity", "Category", "Code", "Summary"],
                [
                    [
                        finding.severity,
                        finding.category,
                        finding.code,
                        finding.summary,
                    ]
                    for finding in report.findings
                ]
                or [
                    [
                        "-",
                        "-",
                        "-",
                        "No duplicate, leakage, or split-integrity issues were detected.",
                    ]
                ],
            ),
            "",
        ]
    )

    for finding in report.findings:
        lines.extend(
            [
                f"## {finding.title}",
                "",
                f"- Severity: `{finding.severity}`",
                f"- Category: `{finding.category}`",
                f"- Code: `{finding.code}`",
                f"- Summary: {finding.summary}",
                f"- Impact: {finding.impact}",
                "",
            ]
        )
        if finding.examples:
            lines.append(_markdown_table(["Example", "Details"], _finding_example_rows(finding)))
            lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            (
                "- Split coverage is deduplicated by `(identity, split)` so that "
                "`all/train/dev` manifest overlap does not inflate the audit summary."
            ),
            (
                "- Explicit duplicate checks use canonical row identity; implicit "
                "duplicate checks hash audio contents for candidate files with matching "
                "sizes."
            ),
            (
                "- Trial manifests are audited separately from data manifests because "
                "they describe evaluation pairs rather than corpus rows."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_dataset_leakage_report(
    *,
    report: DatasetLeakageReport,
    output_root: Path | str,
) -> WrittenDatasetLeakageReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "dataset_leakage.json"
    markdown_path = output_root_path / "dataset_leakage.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_dataset_leakage_markdown(report))
    return WrittenDatasetLeakageReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def _build_audit_record(
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    line_number: int,
    project_root: Path,
    manifests_root: Path,
) -> AuditRecord:
    normalized_entry = normalize_manifest_entry(entry)
    audio_path = _coerce_str(normalized_entry.get("audio_path"))
    resolved_audio_path = (
        resolve_project_path(str(project_root), audio_path) if audio_path is not None else None
    )

    file_size_bytes: int | None = None
    audio_exists = False
    if resolved_audio_path is not None and resolved_audio_path.exists():
        audio_exists = True
        try:
            file_size_bytes = resolved_audio_path.stat().st_size
        except OSError:
            file_size_bytes = None

    speaker_id = _coerce_str(normalized_entry.get("speaker_id"))
    utterance_id = (
        _coerce_str(normalized_entry.get("utterance_id"))
        or _coerce_str(normalized_entry.get("original_name"))
        or _coerce_str(normalized_entry.get("recording_id"))
    )
    manifest_relative_path = _relative_to_project(manifest_path, project_root)
    return AuditRecord(
        manifest_path=manifest_relative_path,
        line_number=line_number,
        dataset_name=_infer_dataset_name(
            entry=normalized_entry,
            audio_path=audio_path,
            manifest_path=manifest_path,
            manifests_root=manifests_root,
        ),
        split_name=_infer_split_name(entry=normalized_entry, manifest_path=manifest_path),
        speaker_id=speaker_id,
        session_key=_infer_session_key(entry=normalized_entry, speaker_id=speaker_id),
        utterance_id=utterance_id,
        audio_path=audio_path,
        audio_basename=PurePosixPath(audio_path).name if audio_path is not None else None,
        identity_key=_build_identity_key(
            audio_path=audio_path,
            entry=normalized_entry,
            manifest_path=manifest_path,
            line_number=line_number,
        ),
        audio_exists=audio_exists,
        file_size_bytes=file_size_bytes,
        duration_seconds=_coerce_float(normalized_entry.get("duration_seconds")),
    )


def _build_trial_record(
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    line_number: int,
    project_root: Path,
) -> TrialRecord:
    return TrialRecord(
        manifest_path=_relative_to_project(manifest_path, project_root),
        line_number=line_number,
        label=_coerce_int(entry.get("label")),
        left_audio=_coerce_str(entry.get("left_audio")),
        right_audio=_coerce_str(entry.get("right_audio")),
    )


def _build_coverage_findings(records: list[AuditRecord]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    observed_splits = {record.split_name for record in records}
    missing_splits = [
        split_name for split_name in REQUIRED_SPLITS if split_name not in observed_splits
    ]
    if missing_splits:
        findings.append(
            AuditFinding(
                code="missing_required_split",
                severity="critical",
                category="coverage",
                title="Required leakage splits are missing",
                summary=(
                    "The manifest set is missing the required train/dev split coverage: "
                    + ", ".join(missing_splits)
                    + "."
                ),
                impact=(
                    "Speaker leakage and split-integrity checks cannot prove baseline/eval "
                    "correctness until train and dev manifests exist in the checkout."
                ),
                examples=[
                    AuditExample(
                        label="observed splits",
                        details={
                            "split_counts": _sorted_counts(
                                Counter(record.split_name for record in records)
                            )
                        },
                    )
                ],
            )
        )

    unknown_split_rows = [record for record in records if record.split_name == "unknown"]
    if unknown_split_rows:
        findings.append(
            AuditFinding(
                code="unknown_split_rows",
                severity="high",
                category="coverage",
                title="Rows without an auditable split",
                summary=f"{len(unknown_split_rows)} rows could not be assigned to train/dev/demo.",
                impact=(
                    "Unknown split assignments make speaker/session leakage checks incomplete "
                    "and can hide train/dev contamination."
                ),
                examples=_record_examples(unknown_split_rows),
            )
        )

    missing_speaker_rows = [record for record in records if record.speaker_id is None]
    if missing_speaker_rows:
        findings.append(
            AuditFinding(
                code="missing_speaker_id",
                severity="high",
                category="coverage",
                title="Rows missing speaker identifiers",
                summary=f"{len(missing_speaker_rows)} rows do not define `speaker_id`.",
                impact=(
                    "Speaker-disjoint split validation is impossible for rows that do not carry "
                    "speaker identity."
                ),
                examples=_record_examples(missing_speaker_rows),
            )
        )
    return findings


def _build_duplicate_findings(
    records: list[AuditRecord],
    raw_records: list[AuditRecord],
    *,
    audio_hashes: dict[str, str],
    hashed_audio_count: int,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    duplicate_rows_within_manifest: list[tuple[tuple[str, str], list[AuditRecord]]] = []
    grouped_raw: dict[tuple[str, str], list[AuditRecord]] = defaultdict(list)
    for record in raw_records:
        grouped_raw[(record.manifest_path, record.identity_key)].append(record)
    for key, cluster in grouped_raw.items():
        if len(cluster) > 1:
            duplicate_rows_within_manifest.append((key, cluster))

    if duplicate_rows_within_manifest:
        findings.append(
            AuditFinding(
                code="duplicate_row_within_manifest",
                severity="high",
                category="duplicates",
                title="Duplicate rows found inside a single manifest",
                summary=(
                    f"{len(duplicate_rows_within_manifest)} manifest-level duplicate "
                    "groups were found."
                ),
                impact=(
                    "Intra-manifest duplication inflates split sizes and can distort sampling, "
                    "class balance, and evaluation metrics."
                ),
                examples=[
                    AuditExample(
                        label=f"{manifest_path} x{len(cluster)}",
                        details={
                            "manifest_path": manifest_path,
                            "identity_key": identity_key,
                            "locations": [record.location for record in cluster[:5]],
                        },
                    )
                    for (manifest_path, identity_key), cluster in duplicate_rows_within_manifest[
                        :10
                    ]
                ],
            )
        )

    records_by_identity: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        records_by_identity[record.identity_key].append(record)

    cross_split_clusters = [
        cluster
        for cluster in records_by_identity.values()
        if len({record.split_name for record in cluster}) > 1
    ]
    if cross_split_clusters:
        findings.append(
            AuditFinding(
                code="cross_split_audio_overlap",
                severity="critical",
                category="duplicates",
                title="The same row appears in multiple splits",
                summary=(
                    f"{len(cross_split_clusters)} canonical rows were found in more than one split."
                ),
                impact=(
                    "The same audio or row identity leaking across train/dev splits invalidates "
                    "offline evaluation and threshold tuning."
                ),
                examples=[
                    AuditExample(
                        label=cluster[0].audio_path or cluster[0].identity_key,
                        details={
                            "identity_key": cluster[0].identity_key,
                            "splits": sorted({record.split_name for record in cluster}),
                            "locations": [record.location for record in cluster[:5]],
                            "speaker_ids": sorted(
                                {record.speaker_id for record in cluster if record.speaker_id}
                            ),
                        },
                    )
                    for cluster in cross_split_clusters[:10]
                ],
            )
        )

    unexpected_same_split_overlap = [
        cluster
        for cluster in records_by_identity.values()
        if len(cluster) > 1
        and len({record.split_name for record in cluster}) == 1
        and not _is_expected_aggregate_overlap(cluster)
    ]
    if unexpected_same_split_overlap:
        findings.append(
            AuditFinding(
                code="unexpected_same_split_overlap",
                severity="medium",
                category="duplicates",
                title="Unexpected duplicate coverage across manifests",
                summary=(
                    f"{len(unexpected_same_split_overlap)} duplicate groups were found "
                    "outside expected "
                    "`all_manifest` overlap."
                ),
                impact=(
                    "Unexpected same-split duplication suggests redundant manifests or duplicate "
                    "rows that can silently bias data loading."
                ),
                examples=[
                    AuditExample(
                        label=cluster[0].audio_path or cluster[0].identity_key,
                        details={
                            "split": cluster[0].split_name,
                            "manifests": sorted({record.manifest_path for record in cluster}),
                            "locations": [record.location for record in cluster[:5]],
                        },
                    )
                    for cluster in unexpected_same_split_overlap[:10]
                ],
            )
        )

    duplicate_hash_groups: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        if record.audio_path is None:
            continue
        audio_hash = audio_hashes.get(record.audio_path)
        if audio_hash is None:
            continue
        duplicate_hash_groups[audio_hash].append(record)

    implicit_duplicate_clusters = [
        cluster
        for cluster in duplicate_hash_groups.values()
        if len({record.audio_path for record in cluster if record.audio_path is not None}) > 1
    ]
    if implicit_duplicate_clusters:
        severity = "high"
        if any(
            len({record.split_name for record in cluster}) > 1
            for cluster in implicit_duplicate_clusters
        ):
            severity = "critical"
        findings.append(
            AuditFinding(
                code="duplicate_audio_content",
                severity=severity,
                category="duplicates",
                title="Different paths resolve to identical audio content",
                summary=(
                    f"{len(implicit_duplicate_clusters)} audio-content duplicate groups were found "
                    f"after hashing {hashed_audio_count} candidate files."
                ),
                impact=(
                    "Implicit content duplicates can bypass simple path-based deduplication and "
                    "still leak evaluation audio or duplicate speaker evidence."
                ),
                examples=[
                    AuditExample(
                        label=audio_hashes[cluster[0].audio_path or ""][:12],
                        details={
                            "hash": audio_hashes[cluster[0].audio_path or ""],
                            "audio_paths": sorted(
                                {
                                    record.audio_path
                                    for record in cluster
                                    if record.audio_path is not None
                                }
                            )[:5],
                            "splits": sorted({record.split_name for record in cluster}),
                            "speaker_ids": sorted(
                                {
                                    record.speaker_id
                                    for record in cluster
                                    if record.speaker_id is not None
                                }
                            ),
                        },
                    )
                    for cluster in implicit_duplicate_clusters[:10]
                ],
            )
        )

    return findings


def _build_overlap_findings(records: list[AuditRecord]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    speaker_clusters = _clusters_with_multi_split(
        records,
        key=lambda record: record.speaker_id,
    )
    if speaker_clusters:
        findings.append(
            AuditFinding(
                code="speaker_overlap",
                severity="critical",
                category="leakage",
                title="Speakers overlap across splits",
                summary=f"{len(speaker_clusters)} speakers were observed in more than one split.",
                impact=(
                    "Speaker overlap between train and dev breaks speaker-disjoint evaluation and "
                    "makes reported verification quality unreliable."
                ),
                examples=[
                    AuditExample(
                        label=speaker_id,
                        details={
                            "splits": sorted({record.split_name for record in cluster}),
                            "locations": [record.location for record in cluster[:5]],
                        },
                    )
                    for speaker_id, cluster in speaker_clusters[:10]
                ],
            )
        )

    session_clusters = _clusters_with_multi_split(
        records,
        key=lambda record: record.session_key,
    )
    if session_clusters:
        findings.append(
            AuditFinding(
                code="session_overlap",
                severity="high",
                category="leakage",
                title="Sessions overlap across splits",
                summary=f"{len(session_clusters)} sessions were observed in more than one split.",
                impact=(
                    "Session overlap can leak channel, room, or device conditions across "
                    "splits even "
                    "when speaker ids look disjoint."
                ),
                examples=[
                    AuditExample(
                        label=session_key,
                        details={
                            "splits": sorted({record.split_name for record in cluster}),
                            "locations": [record.location for record in cluster[:5]],
                        },
                    )
                    for session_key, cluster in session_clusters[:10]
                ],
            )
        )
    return findings


def _build_manifest_integrity_findings(
    records: list[AuditRecord],
    raw_records: list[AuditRecord],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    split_mismatches = [
        record
        for record in raw_records
        if (expected_split := _expected_split_from_manifest(record.manifest_path)) is not None
        and expected_split != record.split_name
    ]
    if split_mismatches:
        findings.append(
            AuditFinding(
                code="manifest_split_mismatch",
                severity="high",
                category="integrity",
                title="Manifest filename and row split disagree",
                summary=(
                    f"{len(split_mismatches)} rows disagree with the split implied by "
                    "their manifest filename."
                ),
                impact=(
                    "Split-label mismatches cause silent train/dev contamination and make "
                    "downstream "
                    "sampling logic unreliable."
                ),
                examples=_record_examples(split_mismatches),
            )
        )

    grouped_by_parent: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        grouped_by_parent[str(PurePosixPath(record.manifest_path).parent)].append(record)

    missing_from_all_examples: list[AuditExample] = []
    only_in_all_examples: list[AuditExample] = []
    for parent, cluster in grouped_by_parent.items():
        all_rows = {
            record.identity_key
            for record in cluster
            if _manifest_role(record.manifest_path) == "all"
        }
        split_rows = {
            record.identity_key
            for record in cluster
            if _manifest_role(record.manifest_path) in KNOWN_DATA_SPLITS
        }
        if all_rows and split_rows:
            missing_from_all = sorted(split_rows - all_rows)
            only_in_all = sorted(all_rows - split_rows)
            if missing_from_all:
                missing_from_all_examples.append(
                    AuditExample(
                        label=parent,
                        details={
                            "missing_identity_keys": missing_from_all[:5],
                            "count": len(missing_from_all),
                        },
                    )
                )
            if only_in_all:
                only_in_all_examples.append(
                    AuditExample(
                        label=parent,
                        details={
                            "only_in_all_identity_keys": only_in_all[:5],
                            "count": len(only_in_all),
                        },
                    )
                )

    if missing_from_all_examples:
        findings.append(
            AuditFinding(
                code="all_manifest_missing_rows",
                severity="medium",
                category="integrity",
                title="Split manifests contain rows missing from all-manifest coverage",
                summary=(
                    f"{len(missing_from_all_examples)} manifest groups have split rows "
                    "that are absent "
                    "from `all_manifest`."
                ),
                impact=(
                    "Inconsistent aggregate manifests make corpus-wide audits and "
                    "downstream loaders "
                    "report different dataset shapes."
                ),
                examples=missing_from_all_examples[:10],
            )
        )

    if only_in_all_examples:
        findings.append(
            AuditFinding(
                code="all_manifest_extra_rows",
                severity="medium",
                category="integrity",
                title="All-manifests contain rows outside split coverage",
                summary=(
                    f"{len(only_in_all_examples)} manifest groups have `all_manifest` "
                    "rows that do not "
                    "appear in split-specific manifests."
                ),
                impact=(
                    "Aggregate-only rows suggest split generation drift or incomplete "
                    "split materialization."
                ),
                examples=only_in_all_examples[:10],
            )
        )

    return findings


def _build_trial_findings(
    records: list[AuditRecord],
    trial_records: list[TrialRecord],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if not trial_records:
        return findings

    lookup: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        if record.audio_path is not None:
            lookup[record.audio_path].append(record)
        if record.audio_basename is not None:
            lookup[record.audio_basename].append(record)

    missing_reference_examples: list[AuditExample] = []
    split_violation_examples: list[AuditExample] = []
    for trial in trial_records:
        left_matches = _match_trial_reference(trial.left_audio, lookup)
        right_matches = _match_trial_reference(trial.right_audio, lookup)
        if not left_matches or not right_matches:
            missing_reference_examples.append(
                AuditExample(
                    label=trial.location,
                    details={
                        "left_audio": trial.left_audio,
                        "left_match_count": len(left_matches),
                        "right_audio": trial.right_audio,
                        "right_match_count": len(right_matches),
                    },
                )
            )
            continue

        if _trial_requires_dev_only(trial.manifest_path):
            trial_splits = {record.split_name for record in [*left_matches, *right_matches]}
            if trial_splits != {"dev"}:
                split_violation_examples.append(
                    AuditExample(
                        label=trial.location,
                        details={
                            "left_audio": trial.left_audio,
                            "right_audio": trial.right_audio,
                            "splits": sorted(trial_splits),
                        },
                    )
                )

    if missing_reference_examples:
        findings.append(
            AuditFinding(
                code="trial_reference_missing",
                severity="high",
                category="integrity",
                title="Trials reference audio that is absent from the manifests",
                summary=(
                    f"{len(missing_reference_examples)} trial rows reference audio "
                    "identifiers that do not "
                    "resolve to corpus rows."
                ),
                impact=(
                    "Verification trials that cannot be resolved against the manifests "
                    "make dev/eval "
                    "scoring incomplete or silently inconsistent."
                ),
                examples=missing_reference_examples[:10],
            )
        )

    if split_violation_examples:
        findings.append(
            AuditFinding(
                code="speaker_disjoint_trial_split_violation",
                severity="high",
                category="integrity",
                title="Speaker-disjoint trials reference non-dev audio",
                summary=(
                    f"{len(split_violation_examples)} speaker-disjoint trial rows are not "
                    "restricted "
                    "to dev split audio."
                ),
                impact=(
                    "Dev-only trial bundles must not pull train/demo rows or the "
                    "verification gate stops "
                    "measuring held-out performance."
                ),
                examples=split_violation_examples[:10],
            )
        )
    return findings


def _deduplicate_records(records: list[AuditRecord]) -> list[AuditRecord]:
    deduplicated: dict[tuple[str, str], AuditRecord] = {}
    for record in records:
        key = (record.identity_key, record.split_name)
        existing = deduplicated.get(key)
        if existing is None or _record_score(record) > _record_score(existing):
            deduplicated[key] = record
    return list(deduplicated.values())


def _record_score(record: AuditRecord) -> int:
    return sum(
        (
            8 if record.audio_exists else 0,
            4 if record.speaker_id is not None else 0,
            2 if record.session_key is not None else 0,
            1 if record.utterance_id is not None else 0,
        )
    )


def _build_audio_hashes(
    records: list[AuditRecord],
    project_root: Path,
) -> tuple[dict[str, str], int]:
    records_by_size: dict[int, list[AuditRecord]] = defaultdict(list)
    for record in records:
        if record.audio_path is None or not record.audio_exists or record.file_size_bytes is None:
            continue
        records_by_size[record.file_size_bytes].append(record)

    hash_cache: dict[Path, str] = {}
    audio_hashes: dict[str, str] = {}
    hashed_audio_count = 0
    for cluster in records_by_size.values():
        unique_paths = {record.audio_path for record in cluster if record.audio_path is not None}
        if len(unique_paths) < 2:
            continue
        for audio_path in sorted(unique_paths):
            resolved_path = resolve_project_path(str(project_root), audio_path)
            if resolved_path in hash_cache:
                audio_hashes[audio_path] = hash_cache[resolved_path]
                continue
            digest = hashlib.sha256()
            with resolved_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hash_cache[resolved_path] = digest.hexdigest()
            audio_hashes[audio_path] = hash_cache[resolved_path]
            hashed_audio_count += 1
    return audio_hashes, hashed_audio_count


def _clusters_with_multi_split(
    records: list[AuditRecord],
    *,
    key: Any,
) -> list[tuple[str, list[AuditRecord]]]:
    clusters: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        group_key = key(record)
        if group_key is None:
            continue
        clusters[group_key].append(record)

    result = [
        (group_key, cluster)
        for group_key, cluster in clusters.items()
        if len({record.split_name for record in cluster}) > 1
    ]
    result.sort(key=lambda item: (-len(item[1]), item[0]))
    return result


def _is_expected_aggregate_overlap(cluster: list[AuditRecord]) -> bool:
    split_names = {record.split_name for record in cluster}
    if len(split_names) != 1:
        return False
    split_name = next(iter(split_names))
    if split_name == "unknown":
        return False
    manifest_roles = {_manifest_role(record.manifest_path) for record in cluster}
    return manifest_roles.issubset({"all", split_name})


def _manifest_role(manifest_path: str) -> str:
    stem = PurePosixPath(manifest_path).stem.lower()
    if "trial" in stem:
        return "trial"
    if "all" in stem:
        return "all"
    for split_name in KNOWN_DATA_SPLITS:
        if split_name in stem:
            return split_name
    return "other"


def _expected_split_from_manifest(manifest_path: str) -> str | None:
    role = _manifest_role(manifest_path)
    if role in KNOWN_DATA_SPLITS:
        return role
    return None


def _trial_requires_dev_only(manifest_path: str) -> bool:
    stem = PurePosixPath(manifest_path).stem.lower()
    return "speaker_disjoint" in stem or "split_trial" in stem


def _match_trial_reference(
    reference: str | None,
    lookup: dict[str, list[AuditRecord]],
) -> list[AuditRecord]:
    if reference is None:
        return []
    matches = lookup.get(reference, [])
    if matches:
        return matches
    return lookup.get(PurePosixPath(reference).name, [])


def _record_examples(records: list[AuditRecord]) -> list[AuditExample]:
    return [
        AuditExample(
            label=record.location,
            details={
                "audio_path": record.audio_path,
                "split": record.split_name,
                "speaker_id": record.speaker_id,
                "session_key": record.session_key,
            },
        )
        for record in records[:10]
    ]


def _finding_example_rows(finding: AuditFinding) -> list[list[str]]:
    rows: list[list[str]] = []
    for example in finding.examples:
        detail_chunks = [
            f"{key}={_stringify_detail(value)}" for key, value in sorted(example.details.items())
        ]
        rows.append([example.label, ", ".join(detail_chunks)])
    return rows


def _load_jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], int]:
    objects: list[dict[str, Any]] = []
    invalid_line_count = 0
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            invalid_line_count += 1
            continue
        if isinstance(payload, dict):
            objects.append(payload)
        else:
            invalid_line_count += 1
    return objects, invalid_line_count


def _infer_dataset_name(
    *,
    entry: dict[str, Any],
    audio_path: str | None,
    manifest_path: Path,
    manifests_root: Path,
) -> str:
    for field_name in ("dataset", "source_dataset"):
        value = _coerce_str(entry.get(field_name))
        if value is not None:
            return value

    if audio_path is not None:
        parts = PurePosixPath(audio_path).parts
        if len(parts) >= 2 and parts[0] == "datasets":
            return parts[1]

    if manifest_path.parent != manifests_root:
        return manifest_path.parent.name

    stem = manifest_path.stem
    for suffix in ("_manifest", "-manifest"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem


def _infer_split_name(*, entry: dict[str, Any], manifest_path: Path) -> str:
    split_name = _coerce_str(entry.get("split"))
    if split_name is not None:
        return split_name
    if _coerce_str(entry.get("role")) is not None:
        return "demo"

    stem = manifest_path.stem.lower()
    for candidate in KNOWN_DATA_SPLITS:
        if candidate in stem:
            return candidate
    return "unknown"


def _infer_session_key(*, entry: dict[str, Any], speaker_id: str | None) -> str | None:
    session_id = _coerce_str(entry.get("session_id"))
    if session_id is not None:
        if speaker_id is not None and ":" not in session_id:
            return f"{speaker_id}:{session_id}"
        return session_id

    session_index = _coerce_str(entry.get("session_index"))
    if session_index is None:
        return None
    if speaker_id is not None:
        return f"{speaker_id}:{session_index}"
    return session_index


def _build_identity_key(
    *,
    audio_path: str | None,
    entry: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> str:
    if audio_path is not None:
        return f"audio:{audio_path}"
    for field_name in ("demo_subset_path", "utterance_id", "recording_id", "id"):
        value = _coerce_str(entry.get(field_name))
        if value is not None:
            return f"{field_name}:{value}"
    return f"manifest:{manifest_path.as_posix()}:{line_number}"


def _relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter, key=lambda item: (-counter[item], item))}


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _finding_sort_key(finding: AuditFinding) -> tuple[int, str, str]:
    return (
        SEVERITY_ORDER.get(finding.severity, 999),
        finding.category,
        finding.code,
    )


def _stringify_detail(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_stringify_detail(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key}:{_stringify_detail(item)}" for key, item in value.items())
        return "{" + inner + "}"
    return str(value)


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
