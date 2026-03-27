"""Builder for the internal verification-protocol snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kryptonite.deployment import resolve_project_path
from kryptonite.repro import fingerprint_path

from .verification_data import (
    build_trial_item_index,
    load_verification_metadata_rows,
    load_verification_trial_rows,
    resolve_trial_side_identifier,
)
from .verification_protocol import (
    VerificationProtocolBundle,
    VerificationProtocolReport,
    VerificationProtocolSliceDefinition,
    VerificationProtocolSliceObservation,
    VerificationProtocolSummary,
    VerificationProtocolTrialSource,
)
from .verification_protocol_config import VerificationProtocolConfig
from .verification_slices import (
    VERIFICATION_SLICE_FIELD_DESCRIPTIONS,
    VERIFICATION_SLICE_FIELD_TITLES,
    derive_slice_value,
)

_SAMPLE_SLICE_VALUES = 5


def build_verification_protocol_report(
    config: VerificationProtocolConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> VerificationProtocolReport:
    resolved_project_root = _resolve_project_root(project_root)
    clean_bundles = tuple(
        _build_bundle(
            bundle_id=clean.bundle_id,
            stage=clean.stage,
            family="clean",
            description=clean.description,
            metadata_manifest_path=clean.metadata_manifest_path,
            trial_manifest_paths=(clean.trial_manifest_path,),
            notes=clean.notes,
            required_slice_fields=config.required_slice_fields,
            project_root=resolved_project_root,
        )
        for clean in config.clean_sets
    )
    production_bundles, production_warnings = _load_production_bundles(
        config=config,
        project_root=resolved_project_root,
    )

    all_bundles = (*clean_bundles, *production_bundles)
    covered_required_slice_fields = tuple(
        sorted(
            {
                field_name
                for bundle in all_bundles
                for field_name in bundle.available_slice_fields
                if field_name in config.required_slice_fields
            }
        )
    )
    missing_required_slice_fields = tuple(
        field_name
        for field_name in config.required_slice_fields
        if field_name not in covered_required_slice_fields
    )
    warning_count = len(production_warnings) + sum(len(bundle.warnings) for bundle in all_bundles)

    source_config_file = None if config_path is None else Path(config_path).resolve()
    source_config_sha256 = None
    if source_config_file is not None:
        source_fingerprint = fingerprint_path(source_config_file)
        source_config_sha256 = (
            None if not bool(source_fingerprint["exists"]) else str(source_fingerprint["sha256"])
        )

    slice_definitions = tuple(
        VerificationProtocolSliceDefinition(
            field_name=field_name,
            title=VERIFICATION_SLICE_FIELD_TITLES.get(field_name, field_name),
            description=VERIFICATION_SLICE_FIELD_DESCRIPTIONS.get(
                field_name,
                "Protocol-defined slice field.",
            ),
            required=field_name in config.required_slice_fields,
        )
        for field_name in config.required_slice_fields
    )
    notes = tuple((*config.notes, *production_warnings))
    return VerificationProtocolReport(
        title=config.title,
        ticket_id=config.ticket_id,
        protocol_id=config.protocol_id,
        summary_text=config.summary,
        output_root=config.output_root,
        source_config_path=None if source_config_file is None else str(source_config_file),
        source_config_sha256=source_config_sha256,
        required_slice_fields=config.required_slice_fields,
        slice_definitions=slice_definitions,
        clean_bundles=clean_bundles,
        production_bundles=production_bundles,
        validation_commands=config.validation_commands,
        notes=notes,
        summary=VerificationProtocolSummary(
            clean_bundle_count=len(clean_bundles),
            production_bundle_count=len(production_bundles),
            covered_required_slice_fields=covered_required_slice_fields,
            missing_required_slice_fields=missing_required_slice_fields,
            warning_count=warning_count,
            is_complete=not missing_required_slice_fields and warning_count == 0,
        ),
    )


def _build_bundle(
    *,
    bundle_id: str,
    stage: str,
    family: str,
    description: str,
    metadata_manifest_path: str | None,
    trial_manifest_paths: tuple[str, ...],
    notes: tuple[str, ...],
    required_slice_fields: tuple[str, ...],
    project_root: Path,
) -> VerificationProtocolBundle:
    warnings: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    metadata_exists = False
    metadata_sha256 = None
    metadata_row_count = 0
    speaker_count = 0
    audio_count = 0

    if metadata_manifest_path is not None:
        resolved_metadata_path = resolve_project_path(str(project_root), metadata_manifest_path)
        metadata_fingerprint = fingerprint_path(resolved_metadata_path)
        metadata_exists = bool(metadata_fingerprint["exists"])
        metadata_sha256 = None if not metadata_exists else str(metadata_fingerprint["sha256"])
        if metadata_exists:
            try:
                metadata_rows = load_verification_metadata_rows(resolved_metadata_path)
                metadata_row_count = len(metadata_rows)
                speaker_count = len(
                    {
                        str(row.get("speaker_id")).strip()
                        for row in metadata_rows
                        if str(row.get("speaker_id")).strip()
                    }
                )
                audio_count = len(
                    {
                        str(row.get("audio_path")).strip()
                        for row in metadata_rows
                        if str(row.get("audio_path")).strip()
                    }
                )
            except ValueError as exc:
                warnings.append(
                    f"Failed to read metadata manifest `{metadata_manifest_path}`: {exc}"
                )
        else:
            warnings.append(f"Missing metadata manifest `{metadata_manifest_path}`.")

    trial_sources: list[VerificationProtocolTrialSource] = []
    combined_trial_rows: list[dict[str, Any]] = []
    available_trial_fields: set[str] = set()
    for raw_path in trial_manifest_paths:
        resolved_trial_path = resolve_project_path(str(project_root), raw_path)
        fingerprint = fingerprint_path(resolved_trial_path)
        exists = bool(fingerprint["exists"])
        trial_count = 0
        positive_count = 0
        negative_count = 0
        if exists:
            try:
                rows = load_verification_trial_rows(resolved_trial_path)
                combined_trial_rows.extend(rows)
                trial_count = len(rows)
                positive_count = sum(1 for row in rows if int(row.get("label", -1)) == 1)
                negative_count = sum(1 for row in rows if int(row.get("label", -1)) == 0)
                for row in rows:
                    available_trial_fields.update(str(key) for key in row.keys())
            except ValueError as exc:
                warnings.append(f"Failed to read trial manifest `{raw_path}`: {exc}")
        else:
            warnings.append(f"Missing trial manifest `{raw_path}`.")

        trial_sources.append(
            VerificationProtocolTrialSource(
                path=raw_path,
                exists=exists,
                sha256=None if not exists else str(fingerprint["sha256"]),
                trial_count=trial_count,
                positive_count=positive_count,
                negative_count=negative_count,
            )
        )

    slice_observations = _collect_slice_observations(
        trial_rows=combined_trial_rows,
        metadata_rows=metadata_rows,
        required_slice_fields=required_slice_fields,
    )
    available_slice_fields = tuple(item.field_name for item in slice_observations)
    missing_required_slice_fields = tuple(
        field_name
        for field_name in required_slice_fields
        if field_name not in available_slice_fields
    )
    return VerificationProtocolBundle(
        bundle_id=bundle_id,
        stage=stage,
        family=family,
        description=description,
        metadata_path=metadata_manifest_path,
        metadata_exists=metadata_exists,
        metadata_sha256=metadata_sha256,
        metadata_row_count=metadata_row_count,
        speaker_count=speaker_count,
        audio_count=audio_count,
        trial_sources=tuple(trial_sources),
        available_trial_fields=tuple(sorted(available_trial_fields)),
        available_slice_fields=available_slice_fields,
        missing_required_slice_fields=missing_required_slice_fields,
        slice_observations=slice_observations,
        notes=notes,
        warnings=tuple(warnings),
    )


def _load_production_bundles(
    *,
    config: VerificationProtocolConfig,
    project_root: Path,
) -> tuple[tuple[VerificationProtocolBundle, ...], tuple[str, ...]]:
    if config.corrupted_suite_catalog_path is None:
        return (), ()

    catalog_path = resolve_project_path(str(project_root), config.corrupted_suite_catalog_path)
    if not catalog_path.exists():
        return (), (f"Missing corrupted-suite catalog `{config.corrupted_suite_catalog_path}`.",)

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (), (f"Failed to read corrupted-suite catalog `{catalog_path}`: {exc}",)
    if not isinstance(payload, dict):
        return (), (f"Corrupted-suite catalog `{catalog_path}` is not a JSON object.",)

    raw_suites = payload.get("suites", [])
    if not isinstance(raw_suites, list):
        return (), (f"Corrupted-suite catalog `{catalog_path}` has invalid `suites` data.",)

    bundles: list[VerificationProtocolBundle] = []
    warnings: list[str] = []
    for raw_suite in raw_suites:
        if not isinstance(raw_suite, dict):
            warnings.append("Ignored non-object corrupted-suite entry in catalog.")
            continue
        bundle_id = str(raw_suite.get("suite_id", "")).strip()
        family = str(raw_suite.get("family", "")).strip() or "prod_like"
        description = str(raw_suite.get("description", "")).strip() or bundle_id
        manifest_path = str(raw_suite.get("manifest_path", "")).strip()
        trial_manifest_paths = raw_suite.get("trial_manifest_paths", [])
        if not bundle_id or not manifest_path or not isinstance(trial_manifest_paths, list):
            warnings.append("Ignored incomplete corrupted-suite entry in catalog.")
            continue
        bundle = _build_bundle(
            bundle_id=bundle_id,
            stage="prod_like",
            family=family,
            description=description,
            metadata_manifest_path=manifest_path,
            trial_manifest_paths=tuple(
                str(item).strip() for item in trial_manifest_paths if str(item).strip()
            ),
            notes=(),
            required_slice_fields=config.required_slice_fields,
            project_root=project_root,
        )
        bundles.append(bundle)
    return tuple(bundles), tuple(warnings)


def _collect_slice_observations(
    *,
    trial_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    required_slice_fields: tuple[str, ...],
) -> tuple[VerificationProtocolSliceObservation, ...]:
    if not trial_rows or not metadata_rows:
        return ()

    metadata_index = build_trial_item_index(metadata_rows)
    values_by_field: dict[str, set[str]] = {
        field_name: set() for field_name in required_slice_fields
    }
    for row in trial_rows:
        left_identifier = resolve_trial_side_identifier(row, "left")
        right_identifier = resolve_trial_side_identifier(row, "right")
        left_metadata = None if left_identifier is None else metadata_index.get(left_identifier)
        right_metadata = None if right_identifier is None else metadata_index.get(right_identifier)
        for field_name in required_slice_fields:
            field_value = derive_slice_value(
                field_name,
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            )
            if field_value is not None:
                values_by_field[field_name].add(field_value)

    observations: list[VerificationProtocolSliceObservation] = []
    for field_name in required_slice_fields:
        values = tuple(sorted(values_by_field[field_name]))
        if not values:
            continue
        observations.append(
            VerificationProtocolSliceObservation(
                field_name=field_name,
                value_count=len(values),
                sample_values=values[:_SAMPLE_SLICE_VALUES],
            )
        )
    return tuple(observations)


def _resolve_project_root(project_root: Path | str | None) -> Path:
    if project_root is None:
        return Path.cwd().resolve()
    return Path(project_root).resolve()


__all__ = ["build_verification_protocol_report"]
