"""Build and render reproducible RIR-bank artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from kryptonite.data.normalization import AudioNormalizationPolicy
from kryptonite.data.normalization.engine import ManifestAudioNormalizer
from kryptonite.data.normalization.models import QuarantineDecision
from kryptonite.deployment import resolve_project_path

from .analysis import analyze_rir_file
from .models import (
    ALLOWED_RIR_DIRECT_CONDITIONS,
    ALLOWED_RIR_FAMILIES,
    ALLOWED_RIR_FIELDS,
    ALLOWED_RIR_ROOM_SIZES,
    ALLOWED_RIR_RT60_BUCKETS,
    MANIFEST_JSONL_NAME,
    QUARANTINE_JSONL_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    ROOM_CONFIG_JSONL_NAME,
    SUPPORTED_AUDIO_SUFFIXES,
    RIRBankEntry,
    RIRBankPlan,
    RIRBankQuarantineRecord,
    RIRBankReport,
    RIRBankSummary,
    RIRClassification,
    RIRClassificationOverride,
    RIRRoomSize,
    RIRSourceStatus,
    RoomSimulationConfig,
    WrittenRIRBankArtifacts,
)


def build_rir_bank(
    *,
    project_root: Path | str,
    dataset_root: Path | str,
    output_root: Path | str,
    plan: RIRBankPlan,
    policy: AudioNormalizationPolicy,
    plan_path: Path | str | None = None,
) -> RIRBankReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    dataset_root_path = resolve_project_path(str(project_root_path), str(dataset_root))
    output_root_path = resolve_project_path(str(project_root_path), str(output_root))
    normalizer = ManifestAudioNormalizer(
        project_root=project_root_path,
        dataset_root=dataset_root_path,
        audio_output_root=output_root_path / "audio",
        policy=policy,
    )

    source_statuses: list[RIRSourceStatus] = []
    entries: list[RIRBankEntry] = []
    quarantined: list[RIRBankQuarantineRecord] = []

    for source in plan.sources:
        resolved_root = _resolve_source_root(project_root_path, source.root_candidates)
        if resolved_root is None:
            source_statuses.append(
                RIRSourceStatus(
                    source_id=source.id,
                    name=source.name,
                    inventory_source_id=source.inventory_source_id,
                    room_family=source.room_family,
                    configured_roots=source.root_candidates,
                    resolved_root=None,
                    status="missing",
                    discovered_audio_count=0,
                )
            )
            continue

        audio_files = _discover_audio_files(resolved_root)
        source_statuses.append(
            RIRSourceStatus(
                source_id=source.id,
                name=source.name,
                inventory_source_id=source.inventory_source_id,
                room_family=source.room_family,
                configured_roots=source.root_candidates,
                resolved_root=str(resolved_root),
                status="present",
                discovered_audio_count=len(audio_files),
            )
        )
        for audio_path in audio_files:
            relative_path = audio_path.relative_to(resolved_root).as_posix()
            override = source.classify(relative_path)
            normalized = normalizer.normalize_row(
                {"audio_path": _relative_to_project(audio_path, project_root_path)}
            )
            if isinstance(normalized, QuarantineDecision):
                quarantined.append(
                    RIRBankQuarantineRecord(
                        source_id=source.id,
                        source_name=source.name,
                        inventory_source_id=source.inventory_source_id,
                        room_family=source.room_family,
                        source_audio_path=_relative_to_project(audio_path, project_root_path),
                        relative_path=relative_path,
                        room_size=override.room_size or source.default_room_size,
                        issue_code=normalized.issue_code,
                        reason=normalized.reason,
                    )
                )
                continue

            normalized_path = resolve_project_path(
                str(project_root_path),
                normalized.normalized_audio_path,
            )
            try:
                metrics = analyze_rir_file(normalized_path, plan.analysis)
                classification = _resolve_classification(
                    source_default_room_size=source.default_room_size,
                    override=override,
                    estimated_rt60_seconds=metrics.estimated_rt60_seconds,
                    estimated_drr_db=metrics.estimated_drr_db,
                    plan=plan,
                )
            except ValueError as exc:
                normalized_path.unlink(missing_ok=True)
                quarantined.append(
                    RIRBankQuarantineRecord(
                        source_id=source.id,
                        source_name=source.name,
                        inventory_source_id=source.inventory_source_id,
                        room_family=source.room_family,
                        source_audio_path=normalized.source_audio_path,
                        relative_path=relative_path,
                        room_size=override.room_size or source.default_room_size,
                        issue_code="rir_analysis_error",
                        reason=str(exc),
                    )
                )
                continue

            entries.append(
                RIRBankEntry(
                    rir_id=_rir_id(source.id, audio_path),
                    source_id=source.id,
                    source_name=source.name,
                    inventory_source_id=source.inventory_source_id,
                    room_family=source.room_family,
                    source_audio_path=normalized.source_audio_path,
                    normalized_audio_path=normalized.normalized_audio_path,
                    relative_path=relative_path,
                    room_size=classification.room_size,
                    field=classification.field,
                    rt60_bucket=classification.rt60_bucket,
                    direct_condition=classification.direct_condition,
                    tags=classification.tags,
                    sampling_weight=round(source.base_weight, 6),
                    peak_time_ms=metrics.peak_time_ms,
                    tail_duration_ms=metrics.tail_duration_ms,
                    energy_centroid_ms=metrics.energy_centroid_ms,
                    estimated_rt60_seconds=metrics.estimated_rt60_seconds,
                    estimated_drr_db=metrics.estimated_drr_db,
                    envelope_preview=metrics.envelope_preview,
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

    room_configs = _build_room_configs(entries)
    summary = RIRBankSummary(
        source_count=len(source_statuses),
        present_source_count=sum(status.status == "present" for status in source_statuses),
        missing_source_count=sum(status.status == "missing" for status in source_statuses),
        entry_count=len(entries),
        config_count=len(room_configs),
        quarantine_count=len(quarantined),
        total_duration_seconds=round(
            sum(entry.normalized_duration_seconds for entry in entries),
            6,
        ),
        room_size_counts=_ordered_counts(
            (entry.room_size for entry in entries),
            ALLOWED_RIR_ROOM_SIZES,
        ),
        field_counts=_ordered_counts((entry.field for entry in entries), ALLOWED_RIR_FIELDS),
        rt60_counts=_ordered_counts(
            (entry.rt60_bucket for entry in entries),
            ALLOWED_RIR_RT60_BUCKETS,
        ),
        direct_condition_counts=_ordered_counts(
            (entry.direct_condition for entry in entries),
            ALLOWED_RIR_DIRECT_CONDITIONS,
        ),
        room_family_counts=_ordered_counts(
            (entry.room_family for entry in entries),
            ALLOWED_RIR_FAMILIES,
        ),
    )
    resolved_plan_path = (
        str(resolve_project_path(str(project_root_path), str(plan_path)))
        if plan_path is not None
        else None
    )
    return RIRBankReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        dataset_root=str(dataset_root_path),
        output_root=str(output_root_path),
        plan_path=resolved_plan_path,
        policy=policy,
        analysis=plan.analysis,
        notes=plan.notes,
        sources=tuple(source_statuses),
        entries=tuple(entries),
        room_configs=tuple(room_configs),
        quarantined=tuple(quarantined),
        summary=summary,
    )


def render_rir_bank_markdown(report: RIRBankReport) -> str:
    lines = [
        "# RIR Bank Report",
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

    missing_fields = report.summary.missing_field_coverage
    coverage_text = "complete" if not missing_fields else ", ".join(missing_fields)
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
                    ["RIR entries", str(report.summary.entry_count)],
                    ["Room configs", str(report.summary.config_count)],
                    ["Quarantined rows", str(report.summary.quarantine_count)],
                    [
                        "Total normalized duration (s)",
                        f"{report.summary.total_duration_seconds:.3f}",
                    ],
                    ["Room sizes", _format_counts(report.summary.room_size_counts)],
                    ["Field coverage", coverage_text],
                    ["RT60 buckets", _format_counts(report.summary.rt60_counts)],
                    [
                        "Direct conditions",
                        _format_counts(report.summary.direct_condition_counts),
                    ],
                    ["Room families", _format_counts(report.summary.room_family_counts)],
                ],
            ),
            "",
            "## Source Status",
            "",
            _markdown_table(
                ["Source", "Family", "Inventory id", "Status", "Resolved root", "Audio files"],
                [
                    [
                        source.name,
                        source.room_family,
                        source.inventory_source_id,
                        source.status,
                        source.resolved_root or "-",
                        str(source.discovered_audio_count),
                    ]
                    for source in report.sources
                ],
            ),
            "",
            "## Coverage Matrix",
            "",
            "### Room Size x Field",
            "",
            _coverage_matrix(
                entries=report.entries,
                row_order=ALLOWED_RIR_ROOM_SIZES,
                col_order=ALLOWED_RIR_FIELDS,
                row_getter=lambda entry: entry.room_size,
                col_getter=lambda entry: entry.field,
                row_header="Room size",
            ),
            "",
            "### RT60 x Direct Condition",
            "",
            _coverage_matrix(
                entries=report.entries,
                row_order=ALLOWED_RIR_RT60_BUCKETS,
                col_order=ALLOWED_RIR_DIRECT_CONDITIONS,
                row_getter=lambda entry: entry.rt60_bucket,
                col_getter=lambda entry: entry.direct_condition,
                row_header="RT60",
            ),
            "",
        ]
    )

    if report.room_configs:
        lines.extend(
            [
                "## Room Configs",
                "",
                _markdown_table(
                    [
                        "Config",
                        "Room size",
                        "Field",
                        "RT60",
                        "Direct",
                        "RIRs",
                        "RT60 range (s)",
                        "DRR range (dB)",
                    ],
                    [
                        [
                            config.config_id,
                            config.room_size,
                            config.field,
                            config.rt60_bucket,
                            config.direct_condition,
                            str(config.rir_count),
                            f"{config.min_rt60_seconds:.3f}-{config.max_rt60_seconds:.3f}",
                            f"{config.min_drr_db:.3f}-{config.max_drr_db:.3f}",
                        ]
                        for config in report.room_configs[:20]
                    ],
                ),
                "",
            ]
        )

    if report.entries:
        lines.extend(
            [
                "## Visual Sanity Checks",
                "",
                _markdown_table(
                    [
                        "Field",
                        "RIR id",
                        "Room size",
                        "RT60 (s)",
                        "DRR (dB)",
                        "Preview",
                    ],
                    [
                        [
                            entry.field,
                            entry.rir_id,
                            entry.room_size,
                            f"{entry.estimated_rt60_seconds:.3f}",
                            f"{entry.estimated_drr_db:.3f}",
                            entry.envelope_preview,
                        ]
                        for entry in _visual_examples(report.entries)
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


def write_rir_bank_report(
    *,
    report: RIRBankReport,
    output_root: Path | str | None = None,
) -> WrittenRIRBankArtifacts:
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
    room_config_path = manifests_root / ROOM_CONFIG_JSONL_NAME
    quarantine_path = manifests_root / QUARANTINE_JSONL_NAME
    json_path = reports_root / REPORT_JSON_NAME
    markdown_path = reports_root / REPORT_MARKDOWN_NAME

    manifest_path.write_text(
        "".join(json.dumps(entry.to_dict(), sort_keys=True) + "\n" for entry in report.entries)
    )
    room_config_path.write_text(
        "".join(
            json.dumps(config.to_dict(), sort_keys=True) + "\n" for config in report.room_configs
        )
    )
    quarantine_path.write_text(
        "".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in report.quarantined
        )
    )
    json_path.write_text(
        json.dumps(
            report.to_dict(include_configs=True, include_quarantine=True),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_path.write_text(render_rir_bank_markdown(report))
    return WrittenRIRBankArtifacts(
        output_root=str(output_root_path),
        manifest_path=str(manifest_path),
        room_config_path=str(room_config_path),
        quarantine_path=str(quarantine_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def _resolve_classification(
    *,
    source_default_room_size: RIRRoomSize,
    override: RIRClassificationOverride,
    estimated_rt60_seconds: float,
    estimated_drr_db: float,
    plan: RIRBankPlan,
) -> RIRClassification:
    room_size = override.room_size or source_default_room_size
    field = override.field or plan.analysis.bucket_field(estimated_drr_db)
    rt60_bucket = override.rt60_bucket or plan.analysis.bucket_rt60(estimated_rt60_seconds)
    direct_condition = override.direct_condition or plan.analysis.bucket_direct_condition(
        estimated_drr_db
    )
    return RIRClassification(
        room_size=room_size,
        field=field,
        rt60_bucket=rt60_bucket,
        direct_condition=direct_condition,
        tags=override.tags,
    )


def _build_room_configs(entries: list[RIRBankEntry]) -> list[RoomSimulationConfig]:
    grouped: dict[tuple[str, str, str, str], list[RIRBankEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.room_size, entry.field, entry.rt60_bucket, entry.direct_condition)].append(
            entry
        )

    configs: list[RoomSimulationConfig] = []
    for room_size in ALLOWED_RIR_ROOM_SIZES:
        for field in ALLOWED_RIR_FIELDS:
            for rt60_bucket in ALLOWED_RIR_RT60_BUCKETS:
                for direct_condition in ALLOWED_RIR_DIRECT_CONDITIONS:
                    group = grouped.get((room_size, field, rt60_bucket, direct_condition))
                    if not group:
                        continue
                    configs.append(
                        RoomSimulationConfig(
                            config_id=(
                                f"{room_size}-{field}-{rt60_bucket}-{direct_condition}".replace(
                                    "_", "-"
                                )
                            ),
                            room_size=room_size,
                            field=field,
                            rt60_bucket=rt60_bucket,
                            direct_condition=direct_condition,
                            rir_count=len(group),
                            sample_rir_ids=tuple(entry.rir_id for entry in group[:8]),
                            min_rt60_seconds=round(
                                min(entry.estimated_rt60_seconds for entry in group),
                                6,
                            ),
                            max_rt60_seconds=round(
                                max(entry.estimated_rt60_seconds for entry in group),
                                6,
                            ),
                            min_drr_db=round(min(entry.estimated_drr_db for entry in group), 6),
                            max_drr_db=round(max(entry.estimated_drr_db for entry in group), 6),
                            room_families=tuple(sorted(set(entry.room_family for entry in group))),
                            source_ids=tuple(sorted(set(entry.source_id for entry in group))),
                        )
                    )
    return configs


def _visual_examples(entries: tuple[RIRBankEntry, ...]) -> list[RIRBankEntry]:
    examples: list[RIRBankEntry] = []
    for field in ALLOWED_RIR_FIELDS:
        subset = [entry for entry in entries if entry.field == field]
        if not subset:
            continue
        if field == "near":
            chosen = max(subset, key=lambda entry: entry.estimated_drr_db)
        elif field == "far":
            chosen = min(subset, key=lambda entry: entry.estimated_drr_db)
        else:
            chosen = min(subset, key=lambda entry: abs(entry.estimated_drr_db))
        examples.append(chosen)
    if examples:
        return examples
    return list(entries[:3])


def _coverage_matrix(
    *,
    entries: tuple[RIRBankEntry, ...],
    row_order: tuple[str, ...],
    col_order: tuple[str, ...],
    row_getter: Callable[[RIRBankEntry], str],
    col_getter: Callable[[RIRBankEntry], str],
    row_header: str,
) -> str:
    counts = Counter((row_getter(entry), col_getter(entry)) for entry in entries)
    rows: list[list[str]] = []
    for row_name in row_order:
        row_values = [row_name]
        total = 0
        for col_name in col_order:
            count = counts.get((row_name, col_name), 0)
            total += count
            row_values.append(str(count))
        row_values.append(str(total))
        rows.append(row_values)
    return _markdown_table(
        [row_header, *col_order, "total"],
        rows,
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


def _rir_id(source_id: str, audio_path: Path) -> str:
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
    "build_rir_bank",
    "render_rir_bank_markdown",
    "write_rir_bank_report",
]
