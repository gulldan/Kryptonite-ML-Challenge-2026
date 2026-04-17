"""Manifest validation reports for manifests-backed audio datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kryptonite.deployment import resolve_project_path

from .schema import MANIFEST_RECORD_TYPE, validate_manifest_entry


@dataclass(frozen=True, slots=True)
class ManifestRowIssue:
    manifest_path: str
    line_number: int
    field: str
    code: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "line_number": self.line_number,
            "field": self.field,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class SkippedManifest:
    manifest_path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_path": self.manifest_path,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ManifestValidationReport:
    generated_at: str
    project_root: str
    manifests_root: str
    validated_manifest_count: int
    valid_row_count: int
    invalid_row_count: int
    invalid_json_line_count: int
    skipped_manifests: list[SkippedManifest]
    issues: list[ManifestRowIssue]

    @property
    def skipped_manifest_count(self) -> int:
        return len(self.skipped_manifests)

    @property
    def passed(self) -> bool:
        return self.invalid_row_count == 0 and self.invalid_json_line_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "manifests_root": self.manifests_root,
            "validated_manifest_count": self.validated_manifest_count,
            "skipped_manifest_count": self.skipped_manifest_count,
            "valid_row_count": self.valid_row_count,
            "invalid_row_count": self.invalid_row_count,
            "invalid_json_line_count": self.invalid_json_line_count,
            "passed": self.passed,
            "skipped_manifests": [manifest.to_dict() for manifest in self.skipped_manifests],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def build_manifest_validation_report(
    *,
    project_root: Path | str,
    manifests_root: Path | str,
) -> ManifestValidationReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifests_root_path = resolve_project_path(str(project_root_path), str(manifests_root))

    validated_manifest_count = 0
    valid_row_count = 0
    invalid_row_count = 0
    invalid_json_line_count = 0
    skipped_manifests: list[SkippedManifest] = []
    issues: list[ManifestRowIssue] = []

    if manifests_root_path.exists():
        for manifest_path in sorted(manifests_root_path.rglob("*.jsonl")):
            relative_manifest_path = _relative_to_project(manifest_path, project_root_path)
            manifest_name = manifest_path.name.lower()
            if "trial" in manifest_name:
                skipped_manifests.append(
                    SkippedManifest(
                        manifest_path=relative_manifest_path,
                        reason="trial-only JSONL is validated by task-specific readers",
                    )
                )
                continue

            objects, file_invalid_json_line_count = _load_jsonl_objects(manifest_path)
            invalid_json_line_count += file_invalid_json_line_count
            if not _looks_like_data_manifest(manifest_name=manifest_name, rows=objects):
                skipped_manifests.append(
                    SkippedManifest(
                        manifest_path=relative_manifest_path,
                        reason="JSONL does not look like a data manifest",
                    )
                )
                continue

            validated_manifest_count += 1
            for line_number, row in enumerate(objects, start=1):
                row_issues = validate_manifest_entry(row, require_schema_version=True)
                if row_issues:
                    invalid_row_count += 1
                    issues.extend(
                        ManifestRowIssue(
                            manifest_path=relative_manifest_path,
                            line_number=line_number,
                            field=issue.field,
                            code=issue.code,
                            message=issue.message,
                        )
                        for issue in row_issues
                    )
                else:
                    valid_row_count += 1

    return ManifestValidationReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        manifests_root=str(manifests_root_path),
        validated_manifest_count=validated_manifest_count,
        valid_row_count=valid_row_count,
        invalid_row_count=invalid_row_count,
        invalid_json_line_count=invalid_json_line_count,
        skipped_manifests=skipped_manifests,
        issues=issues,
    )


def _looks_like_data_manifest(*, manifest_name: str, rows: list[dict[str, Any]]) -> bool:
    if "manifest" in manifest_name:
        return True
    return any(
        "audio_path" in row or row.get("record_type") == MANIFEST_RECORD_TYPE for row in rows
    )


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


def _relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
