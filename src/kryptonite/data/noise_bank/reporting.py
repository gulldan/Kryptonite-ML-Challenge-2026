"""Build and render reproducible noise-bank artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from kryptonite.data.normalization import AudioNormalizationPolicy
from kryptonite.data.normalization.engine import ManifestAudioNormalizer
from kryptonite.data.normalization.models import QuarantineDecision
from kryptonite.deployment import resolve_project_path

from .models import (
    ALLOWED_NOISE_CATEGORIES,
    ALLOWED_NOISE_SEVERITIES,
    MANIFEST_JSONL_NAME,
    MIX_MODE_BY_CATEGORY,
    QUARANTINE_JSONL_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    SUPPORTED_AUDIO_SUFFIXES,
    NoiseBankEntry,
    NoiseBankPlan,
    NoiseBankQuarantineRecord,
    NoiseBankReport,
    NoiseBankSummary,
    NoiseSourceStatus,
    WrittenNoiseBankArtifacts,
)


def build_noise_bank(
    *,
    project_root: Path | str,
    dataset_root: Path | str,
    output_root: Path | str,
    plan: NoiseBankPlan,
    policy: AudioNormalizationPolicy,
    plan_path: Path | str | None = None,
) -> NoiseBankReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    dataset_root_path = resolve_project_path(str(project_root_path), str(dataset_root))
    output_root_path = resolve_project_path(str(project_root_path), str(output_root))
    normalizer = ManifestAudioNormalizer(
        project_root=project_root_path,
        dataset_root=dataset_root_path,
        audio_output_root=output_root_path / "audio",
        policy=policy,
    )

    source_statuses: list[NoiseSourceStatus] = []
    entries: list[NoiseBankEntry] = []
    quarantined: list[NoiseBankQuarantineRecord] = []

    for source in plan.sources:
        resolved_root = _resolve_source_root(project_root_path, source.root_candidates)
        if resolved_root is None:
            source_statuses.append(
                NoiseSourceStatus(
                    source_id=source.id,
                    name=source.name,
                    inventory_source_id=source.inventory_source_id,
                    configured_roots=source.root_candidates,
                    resolved_root=None,
                    status="missing",
                    discovered_audio_count=0,
                )
            )
            continue

        audio_files = _discover_audio_files(resolved_root)
        source_statuses.append(
            NoiseSourceStatus(
                source_id=source.id,
                name=source.name,
                inventory_source_id=source.inventory_source_id,
                configured_roots=source.root_candidates,
                resolved_root=str(resolved_root),
                status="present",
                discovered_audio_count=len(audio_files),
            )
        )
        for audio_path in audio_files:
            relative_path = audio_path.relative_to(resolved_root).as_posix()
            classification = source.classify(relative_path)
            normalized = normalizer.normalize_row(
                {"audio_path": _relative_to_project(audio_path, project_root_path)}
            )
            if isinstance(normalized, QuarantineDecision):
                quarantined.append(
                    NoiseBankQuarantineRecord(
                        source_id=source.id,
                        source_name=source.name,
                        inventory_source_id=source.inventory_source_id,
                        source_audio_path=_relative_to_project(audio_path, project_root_path),
                        category=classification.category,
                        severity=classification.severity,
                        issue_code=normalized.issue_code,
                        reason=normalized.reason,
                    )
                )
                continue

            severity_profile = plan.severity_profiles[classification.severity]
            entries.append(
                NoiseBankEntry(
                    noise_id=_noise_id(source.id, audio_path),
                    source_id=source.id,
                    source_name=source.name,
                    inventory_source_id=source.inventory_source_id,
                    source_audio_path=normalized.source_audio_path,
                    normalized_audio_path=normalized.normalized_audio_path,
                    relative_path=relative_path,
                    category=classification.category,
                    severity=classification.severity,
                    mix_mode=MIX_MODE_BY_CATEGORY[classification.category],
                    tags=classification.tags,
                    sampling_weight=round(
                        source.base_weight * severity_profile.weight_multiplier,
                        6,
                    ),
                    recommended_snr_db_min=severity_profile.snr_db_min,
                    recommended_snr_db_max=severity_profile.snr_db_max,
                    source_sample_rate_hz=normalized.source_sample_rate_hz,
                    source_num_channels=normalized.source_num_channels,
                    source_duration_seconds=normalized.source_duration_seconds,
                    normalized_duration_seconds=normalized.normalized_duration_seconds,
                    normalization_profile=policy.normalization_profile,
                    normalization_resampled=normalized.resampled,
                    normalization_downmixed=normalized.downmixed,
                    normalization_peak_scaled=normalized.peak_scaled,
                    normalization_loudness_applied=normalized.loudness_applied,
                )
            )

    summary = NoiseBankSummary(
        source_count=len(source_statuses),
        present_source_count=sum(status.status == "present" for status in source_statuses),
        missing_source_count=sum(status.status == "missing" for status in source_statuses),
        entry_count=len(entries),
        quarantine_count=len(quarantined),
        total_duration_seconds=round(
            sum(entry.normalized_duration_seconds for entry in entries),
            6,
        ),
        category_counts=_ordered_counts(
            (entry.category for entry in entries),
            ALLOWED_NOISE_CATEGORIES,
        ),
        severity_counts=_ordered_counts(
            (entry.severity for entry in entries),
            ALLOWED_NOISE_SEVERITIES,
        ),
        source_entry_counts=dict(
            sorted(Counter(entry.source_id for entry in entries).items(), key=lambda item: item[0])
        ),
    )
    resolved_plan_path = (
        str(resolve_project_path(str(project_root_path), str(plan_path)))
        if plan_path is not None
        else None
    )
    return NoiseBankReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        dataset_root=str(dataset_root_path),
        output_root=str(output_root_path),
        plan_path=resolved_plan_path,
        policy=policy,
        notes=plan.notes,
        severity_profiles=plan.severity_profiles,
        sources=tuple(source_statuses),
        entries=tuple(entries),
        quarantined=tuple(quarantined),
        summary=summary,
    )


def render_noise_bank_markdown(report: NoiseBankReport) -> str:
    lines = [
        "# Noise Bank Report",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Dataset root: `{report.dataset_root}`",
        f"- Output root: `{report.output_root}`",
        f"- Plan path: `{report.plan_path or '-'}`",
        "",
    ]
    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    lines.extend(
        [
            "## Overview",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Sources", str(report.summary.source_count)],
                    ["Present sources", str(report.summary.present_source_count)],
                    ["Missing sources", str(report.summary.missing_source_count)],
                    ["Noise entries", str(report.summary.entry_count)],
                    ["Quarantined rows", str(report.summary.quarantine_count)],
                    [
                        "Total normalized duration (s)",
                        f"{report.summary.total_duration_seconds:.3f}",
                    ],
                    ["Categories", _format_counts(report.summary.category_counts)],
                    ["Severities", _format_counts(report.summary.severity_counts)],
                ],
            ),
            "",
            "## Source Status",
            "",
            _markdown_table(
                ["Source", "Inventory id", "Status", "Resolved root", "Audio files"],
                [
                    [
                        source.name,
                        source.inventory_source_id,
                        source.status,
                        source.resolved_root or "-",
                        str(source.discovered_audio_count),
                    ]
                    for source in report.sources
                ],
            ),
            "",
        ]
    )

    if report.entries:
        lines.extend(
            [
                "## Bank Coverage",
                "",
                _markdown_table(
                    ["Noise id", "Source", "Category", "Severity", "Mix mode", "Relative path"],
                    [
                        [
                            entry.noise_id,
                            entry.source_id,
                            entry.category,
                            entry.severity,
                            entry.mix_mode,
                            entry.relative_path,
                        ]
                        for entry in report.entries[:25]
                    ],
                ),
                "",
            ]
        )

    if report.quarantined:
        lines.extend(["## Quarantine", ""])
        lines.extend(
            f"- `{record.source_audio_path}`: `{record.issue_code}` ({record.reason})"
            for record in report.quarantined
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_noise_bank_report(
    *,
    report: NoiseBankReport,
    output_root: Path | str | None = None,
) -> WrittenNoiseBankArtifacts:
    output_root_path = (
        Path(report.output_root)
        if output_root is None
        else resolve_project_path(report.project_root, str(output_root))
    )
    manifests_root = output_root_path / "manifests"
    reports_root = output_root_path / "reports"
    manifests_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_root / MANIFEST_JSONL_NAME
    quarantine_path = manifests_root / QUARANTINE_JSONL_NAME
    json_path = reports_root / REPORT_JSON_NAME
    markdown_path = reports_root / REPORT_MARKDOWN_NAME

    manifest_path.write_text(
        "".join(json.dumps(entry.to_dict(), sort_keys=True) + "\n" for entry in report.entries)
    )
    quarantine_path.write_text(
        "".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in report.quarantined
        )
    )
    json_path.write_text(
        json.dumps(report.to_dict(include_quarantine=True), indent=2, sort_keys=True) + "\n"
    )
    markdown_path.write_text(render_noise_bank_markdown(report))
    return WrittenNoiseBankArtifacts(
        output_root=str(output_root_path),
        manifest_path=str(manifest_path),
        quarantine_path=str(quarantine_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def _discover_audio_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )


def _resolve_source_root(project_root: Path, root_candidates: tuple[str, ...]) -> Path | None:
    for candidate in root_candidates:
        resolved = resolve_project_path(str(project_root), candidate)
        if resolved.is_dir():
            return resolved
    return None


def _ordered_counts(items: Iterable[str], order: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(items)
    return {name: counts.get(name, 0) for name in order if counts.get(name, 0)}


def _relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _noise_id(source_id: str, audio_path: Path) -> str:
    digest = hashlib.blake2b(
        f"{source_id}\0{audio_path.as_posix()}".encode(),
        digest_size=8,
    ).hexdigest()
    return f"{source_id}-{digest}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = ["| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


__all__ = [
    "build_noise_bank",
    "render_noise_bank_markdown",
    "write_noise_bank_report",
]
