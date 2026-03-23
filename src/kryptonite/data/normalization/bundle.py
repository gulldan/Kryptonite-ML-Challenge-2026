"""Bundle-level orchestration for manifest-driven audio normalization."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from ..manifest_artifacts import (
    FileArtifact,
    TabularArtifact,
    build_file_artifact,
    write_manifest_inventory,
    write_tabular_artifact,
)
from ..schema import ManifestRow, normalize_manifest_entry
from .audio_io import validate_audio_normalization_policy
from .common import (
    coerce_float,
    coerce_str,
    deduplicate_rows,
    detect_dataset_name,
    read_jsonl_rows,
    relative_to_project,
    row_identity_key,
    source_manifest_sort_key,
)
from .constants import (
    QUARANTINE_MANIFEST_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    TRIAL_FIELD_ORDER,
)
from .engine import ManifestAudioNormalizer
from .models import (
    AudioNormalizationPolicy,
    AudioNormalizationSummary,
    AuxiliaryTable,
    NormalizedAudioRecord,
    QuarantineDecision,
    SourceManifestTable,
)
from .reporting import render_summary_markdown


def normalize_audio_manifest_bundle(
    *,
    project_root: str,
    dataset_root: str,
    source_manifests_root: str,
    output_root: str,
    policy: AudioNormalizationPolicy,
) -> AudioNormalizationSummary:
    validate_audio_normalization_policy(policy)

    project_root_path = resolve_project_path(project_root, ".")
    dataset_root_path = resolve_project_path(project_root, dataset_root)
    source_root_path = resolve_project_path(project_root, source_manifests_root)
    output_root_path = resolve_project_path(project_root, output_root)
    manifests_output_root = output_root_path / "manifests"
    audio_output_root = output_root_path / "audio"
    reports_output_root = output_root_path / "reports"

    source_tables, carried_quarantine_rows, auxiliary_tables, metadata_files = _load_source_bundle(
        source_root_path
    )
    if not source_tables:
        raise FileNotFoundError(
            f"No active data manifests found under source manifests root {source_root_path}"
        )

    dataset_name = detect_dataset_name(source_tables)
    normalizer = ManifestAudioNormalizer(
        project_root=project_root_path,
        dataset_root=dataset_root_path,
        audio_output_root=audio_output_root,
        policy=policy,
    )

    manifest_artifacts = []
    unique_source_row_keys: set[str] = set()
    unique_normalized_row_keys: set[str] = set()
    generated_quarantine_rows_by_key: dict[str, dict[str, object]] = {}

    for table in source_tables:
        normalized_rows = []
        for row in table.rows:
            row_key = row_identity_key(row)
            unique_source_row_keys.add(row_key)
            decision = normalizer.normalize_row(row)
            if isinstance(decision, QuarantineDecision):
                generated_quarantine_rows_by_key.setdefault(
                    row_key,
                    _build_generated_quarantine_row(
                        row,
                        decision=decision,
                    ),
                )
                continue

            unique_normalized_row_keys.add(row_key)
            normalized_rows.append(
                _build_normalized_manifest_row(
                    row,
                    record=decision,
                    policy=policy,
                )
            )

        manifest_artifacts.append(
            write_tabular_artifact(
                name=table.name.removesuffix(".jsonl"),
                kind="data_manifest",
                rows=normalized_rows,
                jsonl_path=manifests_output_root / table.name,
                project_root=str(project_root_path),
            )
        )

    prepared_carried_quarantine_rows = deduplicate_rows(
        _prepare_carried_quarantine_row(row) for row in carried_quarantine_rows
    )
    generated_quarantine_rows = list(generated_quarantine_rows_by_key.values())
    all_quarantine_rows = [*prepared_carried_quarantine_rows, *generated_quarantine_rows]
    manifest_artifacts.append(
        write_tabular_artifact(
            name=QUARANTINE_MANIFEST_NAME.removesuffix(".jsonl"),
            kind="data_manifest",
            rows=all_quarantine_rows,
            jsonl_path=manifests_output_root / QUARANTINE_MANIFEST_NAME,
            project_root=str(project_root_path),
        )
    )

    auxiliary_artifacts = _write_auxiliary_tables(
        auxiliary_tables=auxiliary_tables,
        manifests_output_root=manifests_output_root,
        project_root_path=project_root_path,
        basename_mapping=normalizer.output_basename_by_source_basename,
    )
    metadata_artifacts = _copy_metadata_files(
        metadata_files=metadata_files,
        manifests_output_root=manifests_output_root,
        project_root_path=project_root_path,
    )

    reports_output_root.mkdir(parents=True, exist_ok=True)
    report_json_path = reports_output_root / REPORT_JSON_NAME
    report_markdown_path = reports_output_root / REPORT_MARKDOWN_NAME
    quarantine_issue_counts = Counter(
        str(row.get("quality_issue_code", "unknown")) for row in generated_quarantine_rows
    )

    report_summary = AudioNormalizationSummary(
        dataset=dataset_name,
        source_manifests_root=relative_to_project(source_root_path, project_root_path),
        output_root=relative_to_project(output_root_path, project_root_path),
        output_manifests_root=relative_to_project(manifests_output_root, project_root_path),
        output_audio_root=relative_to_project(audio_output_root, project_root_path),
        manifest_inventory_file="",
        report_json_file=relative_to_project(report_json_path, project_root_path),
        report_markdown_file=relative_to_project(report_markdown_path, project_root_path),
        source_manifest_count=len(source_tables),
        source_row_count=len(unique_source_row_keys),
        normalized_row_count=len(unique_normalized_row_keys),
        normalized_audio_count=len(normalizer.success_by_source_audio_path),
        generated_quarantine_row_count=len(generated_quarantine_rows),
        carried_quarantine_row_count=len(prepared_carried_quarantine_rows),
        auxiliary_table_count=len(auxiliary_tables),
        copied_metadata_file_count=len(metadata_files),
        resampled_row_count=sum(
            1 for record in normalizer.success_by_source_audio_path.values() if record.resampled
        ),
        downmixed_row_count=sum(
            1 for record in normalizer.success_by_source_audio_path.values() if record.downmixed
        ),
        dc_offset_fixed_row_count=sum(
            1
            for record in normalizer.success_by_source_audio_path.values()
            if record.dc_offset_removed
        ),
        peak_scaled_row_count=sum(
            1 for record in normalizer.success_by_source_audio_path.values() if record.peak_scaled
        ),
        source_clipping_row_count=sum(
            1
            for record in normalizer.success_by_source_audio_path.values()
            if record.source_clipped_sample_ratio > 0.0
        ),
        quarantine_issue_counts=dict(sorted(quarantine_issue_counts.items())),
        policy=policy,
    )
    _write_report_files(
        report_summary=report_summary,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
    )

    report_artifacts = [
        build_file_artifact(
            name="audio_normalization_report_json",
            kind="report",
            path=report_json_path,
            project_root=str(project_root_path),
        ),
        build_file_artifact(
            name="audio_normalization_report_markdown",
            kind="report",
            path=report_markdown_path,
            project_root=str(project_root_path),
        ),
    ]

    inventory_path = manifests_output_root / "manifest_inventory.json"
    report_summary.manifest_inventory_file = write_manifest_inventory(
        dataset=dataset_name,
        inventory_path=inventory_path,
        project_root=str(project_root_path),
        manifest_tables=manifest_artifacts,
        auxiliary_tables=auxiliary_artifacts,
        auxiliary_files=(*metadata_artifacts, *report_artifacts),
    )
    _write_report_files(
        report_summary=report_summary,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
    )
    return report_summary


def _load_source_bundle(
    source_root: Path,
) -> tuple[list[SourceManifestTable], list[dict[str, object]], list[AuxiliaryTable], list[Path]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source manifests root does not exist: {source_root}")

    source_tables: list[SourceManifestTable] = []
    carried_quarantine_rows: list[dict[str, object]] = []
    auxiliary_tables: list[AuxiliaryTable] = []
    metadata_files: list[Path] = []

    for path in sorted(source_root.iterdir()):
        if path.suffix == ".jsonl":
            rows = read_jsonl_rows(path)
            lower_name = path.name.lower()
            if "trial" in lower_name:
                auxiliary_tables.append(AuxiliaryTable(name=path.name, path=path, rows=rows))
            elif "quarantine" in lower_name:
                carried_quarantine_rows.extend(rows)
            else:
                source_tables.append(SourceManifestTable(name=path.name, path=path, rows=rows))
        elif path.is_file() and path.name != "manifest_inventory.json":
            metadata_files.append(path)

    source_tables.sort(key=source_manifest_sort_key)
    auxiliary_tables.sort(key=lambda table: table.name)
    metadata_files.sort(key=lambda path: path.name)
    return source_tables, carried_quarantine_rows, auxiliary_tables, metadata_files


def _write_auxiliary_tables(
    *,
    auxiliary_tables: Sequence[AuxiliaryTable],
    manifests_output_root: Path,
    project_root_path: Path,
    basename_mapping: Mapping[str, str],
) -> list[TabularArtifact]:
    artifacts: list[TabularArtifact] = []
    for table in auxiliary_tables:
        artifacts.append(
            write_tabular_artifact(
                name=table.name.removesuffix(".jsonl"),
                kind="trial_list",
                rows=_rewrite_trial_rows(
                    rows=table.rows,
                    basename_mapping=basename_mapping,
                ),
                jsonl_path=manifests_output_root / table.name,
                project_root=str(project_root_path),
                field_order=TRIAL_FIELD_ORDER,
            )
        )
    return artifacts


def _copy_metadata_files(
    *,
    metadata_files: Sequence[Path],
    manifests_output_root: Path,
    project_root_path: Path,
) -> list[FileArtifact]:
    artifacts: list[FileArtifact] = []
    for path in metadata_files:
        copied = manifests_output_root / path.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
        artifacts.append(
            build_file_artifact(
                name=path.name,
                kind="metadata",
                path=copied,
                project_root=str(project_root_path),
            )
        )
    return artifacts


def _write_report_files(
    *,
    report_summary: AudioNormalizationSummary,
    report_json_path: Path,
    report_markdown_path: Path,
) -> None:
    report_json_path.write_text(
        json.dumps(report_summary.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    report_markdown_path.write_text(render_summary_markdown(report_summary))


def _build_normalized_manifest_row(
    row: Mapping[str, object],
    *,
    record: NormalizedAudioRecord,
    policy: AudioNormalizationPolicy,
) -> dict[str, object]:
    normalized_entry = normalize_manifest_entry(row)
    base_row = ManifestRow(
        dataset=str(normalized_entry["dataset"]),
        source_dataset=str(normalized_entry.get("source_dataset") or normalized_entry["dataset"]),
        speaker_id=str(normalized_entry["speaker_id"]),
        audio_path=record.normalized_audio_path,
        utterance_id=coerce_str(normalized_entry.get("utterance_id")),
        session_id=coerce_str(normalized_entry.get("session_id")),
        split=coerce_str(normalized_entry.get("split")),
        role=coerce_str(normalized_entry.get("role")),
        language=coerce_str(normalized_entry.get("language")),
        device=coerce_str(normalized_entry.get("device")),
        channel="mono"
        if policy.target_channels == 1
        else coerce_str(normalized_entry.get("channel")),
        snr_db=coerce_float(normalized_entry.get("snr_db")),
        rir_id=coerce_str(normalized_entry.get("rir_id")),
        duration_seconds=record.normalized_duration_seconds,
        sample_rate_hz=policy.target_sample_rate_hz,
        num_channels=policy.target_channels,
    )
    canonical_fields = set(base_row.to_dict())
    extra_fields = {key: value for key, value in row.items() if key not in canonical_fields}
    extra_fields.update(
        {
            "source_audio_path": record.source_audio_path,
            "source_sample_rate_hz": record.source_sample_rate_hz,
            "source_num_channels": record.source_num_channels,
            "source_peak_amplitude": round(record.source_peak_amplitude, 6),
            "source_dc_offset_ratio": round(record.source_dc_offset_ratio, 6),
            "source_clipped_sample_ratio": round(record.source_clipped_sample_ratio, 6),
            "normalization_profile": policy.normalization_profile,
            "normalization_resampled": record.resampled,
            "normalization_downmixed": record.downmixed,
            "normalization_dc_offset_removed": record.dc_offset_removed,
            "normalization_peak_scaled": record.peak_scaled,
        }
    )
    return base_row.to_dict(extra_fields=extra_fields)


def _build_generated_quarantine_row(
    row: Mapping[str, object],
    *,
    decision: QuarantineDecision,
) -> dict[str, object]:
    payload = dict(row)
    payload["quality_issue_code"] = decision.issue_code
    payload["quarantine_policy"] = "quarantine"
    payload["quarantine_stage"] = "audio_normalization"
    payload["quarantine_reason"] = decision.reason
    if decision.source_audio_path is not None:
        payload.setdefault("source_audio_path", decision.source_audio_path)
    return payload


def _prepare_carried_quarantine_row(row: Mapping[str, object]) -> dict[str, object]:
    payload = dict(row)
    payload.setdefault("quarantine_stage", "source_manifest")
    payload.setdefault("quarantine_policy", "quarantine")
    return payload


def _rewrite_trial_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    basename_mapping: Mapping[str, str],
) -> list[dict[str, object]]:
    rewritten_rows: list[dict[str, object]] = []
    for row in rows:
        payload = dict(row)
        for field_name in ("left_audio", "right_audio"):
            current = coerce_str(payload.get(field_name))
            if current is None:
                continue
            payload[field_name] = basename_mapping.get(current, current)
        rewritten_rows.append(payload)
    return rewritten_rows
