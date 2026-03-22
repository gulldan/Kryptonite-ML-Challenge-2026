"""Dataset acquisition helpers for server-side surrogate data bundles."""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import tarfile
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from kryptonite.deployment import resolve_project_path

ChecksumAlgorithm = Literal["md5", "sha256"]


@dataclass(frozen=True, slots=True)
class DownloadArtifact:
    name: str
    url: str
    target_path: str
    checksum: str | None = None
    checksum_algorithm: ChecksumAlgorithm = "sha256"
    size_bytes: int | None = None
    extract_to: str | None = None
    required: bool = True
    description: str | None = None


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    name: str
    dataset_root: str
    notes: list[str] = field(default_factory=list)
    artifacts: list[DownloadArtifact] = field(default_factory=list)


@dataclass(slots=True)
class ArtifactAcquisitionResult:
    name: str
    url: str
    target_path: str
    required: bool
    executed: bool
    exists: bool
    size_bytes: int | None
    expected_size_bytes: int | None
    checksum_algorithm: ChecksumAlgorithm
    checksum: str | None
    expected_checksum: str | None
    extracted_to: str | None
    description: str | None = None
    download_command: str | None = None
    exit_code: int | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        if not self.exists:
            return False
        if self.expected_size_bytes is not None and self.size_bytes != self.expected_size_bytes:
            return False
        if self.expected_checksum is not None and self.checksum != self.expected_checksum:
            return False
        return self.error is None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "target_path": self.target_path,
            "required": self.required,
            "executed": self.executed,
            "exists": self.exists,
            "passed": self.passed,
            "size_bytes": self.size_bytes,
            "expected_size_bytes": self.expected_size_bytes,
            "checksum_algorithm": self.checksum_algorithm,
            "checksum": self.checksum,
            "expected_checksum": self.expected_checksum,
            "extracted_to": self.extracted_to,
            "description": self.description,
            "download_command": self.download_command,
            "exit_code": self.exit_code,
            "error": self.error,
        }


@dataclass(slots=True)
class AcquisitionReport:
    name: str
    dataset_root: str
    executed: bool
    notes: list[str]
    artifacts: list[ArtifactAcquisitionResult]
    started_at: str
    completed_at: str

    @property
    def missing_required(self) -> list[str]:
        return [
            artifact.name
            for artifact in self.artifacts
            if artifact.required and not artifact.passed
        ]

    @property
    def passed(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dataset_root": self.dataset_root,
            "executed": self.executed,
            "notes": list(self.notes),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "passed": self.passed,
            "missing_required": self.missing_required,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def load_acquisition_plan(path: Path | str) -> AcquisitionPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())
    artifacts_data = data.get("artifacts")
    if not isinstance(artifacts_data, list) or not artifacts_data:
        raise ValueError("Acquisition plan must define at least one [[artifacts]] entry.")

    artifacts: list[DownloadArtifact] = []
    for artifact_data in artifacts_data:
        if not isinstance(artifact_data, dict):
            raise ValueError("Each artifact entry must be a TOML table.")
        artifacts.append(
            DownloadArtifact(
                name=_require_str(artifact_data, "name"),
                url=_require_str(artifact_data, "url"),
                target_path=_require_str(artifact_data, "target_path"),
                checksum=_optional_str(artifact_data, "checksum"),
                checksum_algorithm=cast(
                    ChecksumAlgorithm,
                    _optional_literal(
                        artifact_data, "checksum_algorithm", ("md5", "sha256"), "sha256"
                    ),
                ),
                size_bytes=_optional_int(artifact_data, "size_bytes"),
                extract_to=_optional_str(artifact_data, "extract_to"),
                required=bool(artifact_data.get("required", True)),
                description=_optional_str(artifact_data, "description"),
            )
        )

    return AcquisitionPlan(
        name=_require_str(data, "name"),
        dataset_root=_require_str(data, "dataset_root"),
        notes=_optional_str_list(data, "notes"),
        artifacts=artifacts,
    )


def acquire_plan(
    *,
    project_root: Path | str,
    plan: AcquisitionPlan,
    execute: bool,
) -> AcquisitionReport:
    started_at = _utc_now()
    dataset_root = resolve_project_path(str(project_root), plan.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    artifact_results: list[ArtifactAcquisitionResult] = []
    for artifact in plan.artifacts:
        target_path = resolve_project_path(str(dataset_root), artifact.target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        download_command: str | None = None
        exit_code: int | None = None
        error: str | None = None

        if execute:
            download_command = shlex.join(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "5",
                    "-C",
                    "-",
                    "-o",
                    str(target_path),
                    artifact.url,
                ]
            )
            completed = subprocess.run(
                shlex.split(download_command),
                text=True,
                capture_output=True,
                check=False,
            )
            exit_code = completed.returncode
            error = _format_process_error(completed)

            if error is None and artifact.extract_to is not None:
                error = _extract_archive(
                    archive_path=target_path,
                    extract_root=resolve_project_path(str(dataset_root), artifact.extract_to),
                )

        exists = target_path.exists()
        size_bytes = target_path.stat().st_size if exists else None
        checksum = (
            _compute_checksum(target_path, artifact.checksum_algorithm)
            if exists and (artifact.checksum is not None or not execute)
            else None
        )
        if exists and checksum is None and artifact.checksum is not None:
            checksum = _compute_checksum(target_path, artifact.checksum_algorithm)

        artifact_results.append(
            ArtifactAcquisitionResult(
                name=artifact.name,
                url=artifact.url,
                target_path=str(target_path),
                required=artifact.required,
                executed=execute,
                exists=exists,
                size_bytes=size_bytes,
                expected_size_bytes=artifact.size_bytes,
                checksum_algorithm=artifact.checksum_algorithm,
                checksum=checksum,
                expected_checksum=artifact.checksum,
                extracted_to=(
                    str(resolve_project_path(str(dataset_root), artifact.extract_to))
                    if artifact.extract_to is not None
                    else None
                ),
                description=artifact.description,
                download_command=download_command,
                exit_code=exit_code,
                error=error,
            )
        )

    return AcquisitionReport(
        name=plan.name,
        dataset_root=str(dataset_root),
        executed=execute,
        notes=list(plan.notes),
        artifacts=artifact_results,
        started_at=started_at,
        completed_at=_utc_now(),
    )


def render_acquisition_report(report: AcquisitionReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    mode = "execute" if report.executed else "inspect-only"
    lines = [
        f"Dataset acquisition ({report.name}): {status}",
        f"Mode: {mode}",
        f"Dataset root: {report.dataset_root}",
        f"Started at: {report.started_at}",
        f"Completed at: {report.completed_at}",
    ]
    for artifact in report.artifacts:
        readiness = "ok" if artifact.passed else "missing-or-invalid"
        line = (
            f"- {artifact.name}: {readiness} "
            f"(path={artifact.target_path}, size={artifact.size_bytes or 0}, "
            f"checksum_algorithm={artifact.checksum_algorithm})"
        )
        if artifact.description:
            line = f"{line}; {artifact.description}"
        if artifact.extracted_to:
            line = f"{line}; extracted_to={artifact.extracted_to}"
        if artifact.error:
            line = f"{line}; error={artifact.error}"
        lines.append(line)

    if report.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in report.notes)

    if report.passed:
        lines.append("All required surrogate artifacts are present and verified.")
    else:
        lines.append(f"Missing or invalid required artifacts: {', '.join(report.missing_required)}")
    return "\n".join(lines)


def _extract_archive(*, archive_path: Path, extract_root: Path) -> str | None:
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(path=extract_root, filter="data")
    except (OSError, tarfile.TarError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _compute_checksum(path: Path, algorithm: ChecksumAlgorithm) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_process_error(completed: subprocess.CompletedProcess[str]) -> str | None:
    if completed.returncode == 0:
        return None
    output = completed.stderr.strip() or completed.stdout.strip() or "command failed"
    return f"exit_code={completed.returncode}: {output}"


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for '{key}'.")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected string for '{key}'.")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Expected non-negative integer for '{key}'.")
    return value


def _optional_literal(
    data: dict[str, Any],
    key: str,
    allowed: tuple[str, ...],
    default: str,
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Expected '{key}' to be one of {allowed}.")
    return value


def _optional_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected string list for '{key}'.")
    return [str(item) for item in value]
