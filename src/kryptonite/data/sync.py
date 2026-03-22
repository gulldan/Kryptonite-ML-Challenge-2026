"""Dataset synchronization and remote readiness reporting."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from textwrap import dedent
from typing import Any, Literal, cast

from kryptonite.deployment import resolve_project_path

ChecksumMode = Literal["catalog", "sha256"]
PathType = Literal["dir", "file"]

REMOTE_INVENTORY_SCRIPT = dedent(
    """
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--path-type", choices=("dir", "file"), required=True)
    parser.add_argument("--checksum-mode", choices=("catalog", "sha256"), required=True)
    parser.add_argument("--sample-limit", type=int, required=True)
    args = parser.parse_args()


    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def build_checksum(entries, checksum_mode: str) -> str:
        digest = hashlib.sha256()
        for relative_path, size_bytes, absolute_path in entries:
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\\t")
            if checksum_mode == "catalog":
                digest.update(str(size_bytes).encode("utf-8"))
            else:
                digest.update(hash_file(absolute_path).encode("utf-8"))
            digest.update(b"\\n")
        return digest.hexdigest()


    def collect_snapshot(
        path: Path,
        path_type: str,
        checksum_mode: str,
        sample_limit: int,
    ) -> dict[str, object]:
        exists = path.exists()
        error = None
        entries = []
        if exists:
            if path_type == "dir":
                if not path.is_dir():
                    error = "expected directory"
                else:
                    entries = sorted(
                        (
                            (
                                relative_path.as_posix(),
                                child.stat().st_size,
                                child,
                            )
                            for child in path.rglob("*")
                            if child.is_file()
                            for relative_path in (child.relative_to(path),)
                        ),
                        key=lambda item: item[0],
                    )
            elif path_type == "file":
                if not path.is_file():
                    error = "expected file"
                else:
                    entries = [(path.name, path.stat().st_size, path)]
            else:
                error = f"unsupported path_type={path_type}"

        checksum = None
        if exists and error is None:
            checksum = build_checksum(entries, checksum_mode)

        payload = {
            "path": str(path),
            "path_type": path_type,
            "exists": exists,
            "file_count": len(entries),
            "total_bytes": sum(size_bytes for _, size_bytes, _ in entries),
            "checksum_mode": checksum_mode,
            "checksum": checksum,
            "samples": [relative_path for relative_path, _, _ in entries[:sample_limit]],
            "error": error,
        }
        print(json.dumps(payload, sort_keys=True))


    collect_snapshot(
        path=Path(args.path),
        path_type=args.path_type,
        checksum_mode=args.checksum_mode,
        sample_limit=args.sample_limit,
    )
    """
).strip()


@dataclass(frozen=True, slots=True)
class RemoteSyncConfig:
    host: str
    project_root: str
    ssh_options: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReportConfig:
    local_path: str | None = None
    remote_path: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPayloadSpec:
    name: str
    source: str
    target: str
    path_type: PathType
    checksum_mode: ChecksumMode
    required: bool = True
    description: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPlan:
    remote: RemoteSyncConfig
    report: ReportConfig
    payloads: list[SyncPayloadSpec]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InventorySnapshot:
    path: str
    path_type: PathType
    exists: bool
    file_count: int
    total_bytes: int
    checksum_mode: ChecksumMode
    checksum: str | None
    samples: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.exists and self.error is None

    def matches(self, other: InventorySnapshot) -> bool:
        return (
            self.ready
            and other.ready
            and self.path_type == other.path_type
            and self.file_count == other.file_count
            and self.total_bytes == other.total_bytes
            and self.checksum_mode == other.checksum_mode
            and self.checksum == other.checksum
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "path_type": self.path_type,
            "exists": self.exists,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "checksum_mode": self.checksum_mode,
            "checksum": self.checksum,
            "samples": list(self.samples),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InventorySnapshot:
        return cls(
            path=str(payload["path"]),
            path_type=cast(PathType, payload["path_type"]),
            exists=bool(payload["exists"]),
            file_count=int(payload["file_count"]),
            total_bytes=int(payload["total_bytes"]),
            checksum_mode=cast(ChecksumMode, payload["checksum_mode"]),
            checksum=str(payload["checksum"]) if payload["checksum"] is not None else None,
            samples=[str(item) for item in payload.get("samples", [])],
            error=str(payload["error"]) if payload.get("error") else None,
        )


@dataclass(slots=True)
class PayloadSyncResult:
    name: str
    required: bool
    description: str | None
    local_path: str
    remote_path: str
    executed: bool
    local: InventorySnapshot
    remote_before: InventorySnapshot
    remote_after: InventorySnapshot
    sync_command: str | None = None
    sync_exit_code: int | None = None
    sync_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.remote_after.matches(self.local)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required": self.required,
            "description": self.description,
            "local_path": self.local_path,
            "remote_path": self.remote_path,
            "executed": self.executed,
            "ready": self.ready,
            "sync_command": self.sync_command,
            "sync_exit_code": self.sync_exit_code,
            "sync_error": self.sync_error,
            "local": self.local.to_dict(),
            "remote_before": self.remote_before.to_dict(),
            "remote_after": self.remote_after.to_dict(),
        }


@dataclass(slots=True)
class DataSyncReport:
    remote_host: str
    remote_project_root: str
    executed: bool
    payloads: list[PayloadSyncResult]
    notes: list[str]
    started_at: str
    completed_at: str
    local_report_path: str | None = None
    remote_report_path: str | None = None
    report_upload_error: str | None = None

    @property
    def missing_required(self) -> list[str]:
        return [payload.name for payload in self.payloads if payload.required and not payload.ready]

    @property
    def passed(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict[str, object]:
        return {
            "remote_host": self.remote_host,
            "remote_project_root": self.remote_project_root,
            "executed": self.executed,
            "passed": self.passed,
            "missing_required": self.missing_required,
            "notes": list(self.notes),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "local_report_path": self.local_report_path,
            "remote_report_path": self.remote_report_path,
            "report_upload_error": self.report_upload_error,
            "payloads": [payload.to_dict() for payload in self.payloads],
        }


def load_sync_plan(path: Path | str) -> SyncPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())
    remote_data = _require_dict(data, "remote")
    report_data = _optional_dict(data, "report")
    payloads_data = data.get("payloads")
    if not isinstance(payloads_data, list) or not payloads_data:
        raise ValueError("Sync plan must define at least one [[payloads]] entry.")

    payloads: list[SyncPayloadSpec] = []
    for payload_data in payloads_data:
        if not isinstance(payload_data, dict):
            raise ValueError("Each payload entry must be a TOML table.")
        payloads.append(
            SyncPayloadSpec(
                name=_require_str(payload_data, "name"),
                source=_require_str(payload_data, "source"),
                target=_require_str(payload_data, "target"),
                path_type=cast(
                    PathType,
                    _require_literal(payload_data, "path_type", ("dir", "file")),
                ),
                checksum_mode=cast(
                    ChecksumMode,
                    _require_literal(payload_data, "checksum_mode", ("catalog", "sha256")),
                ),
                required=bool(payload_data.get("required", True)),
                description=_optional_str(payload_data, "description"),
            )
        )

    return SyncPlan(
        remote=RemoteSyncConfig(
            host=_require_str(remote_data, "host"),
            project_root=_require_str(remote_data, "project_root"),
            ssh_options=_optional_str_list(remote_data, "ssh_options"),
        ),
        report=ReportConfig(
            local_path=_optional_str(report_data, "local_path"),
            remote_path=_optional_str(report_data, "remote_path"),
        ),
        payloads=payloads,
        notes=_optional_str_list(data, "notes"),
    )


def run_data_sync(
    *,
    project_root: Path | str,
    plan: SyncPlan,
    execute: bool,
    sample_limit: int = 10,
) -> DataSyncReport:
    started_at = _utc_now()
    resolved_project_root = resolve_project_path(str(project_root), ".")
    payload_results: list[PayloadSyncResult] = []

    for payload in plan.payloads:
        local_path = resolve_project_path(str(resolved_project_root), payload.source)
        remote_path = resolve_remote_path(plan.remote.project_root, payload.target)
        local_snapshot = collect_local_inventory(
            path=local_path,
            path_type=payload.path_type,
            checksum_mode=payload.checksum_mode,
            sample_limit=sample_limit,
        )
        remote_before = collect_remote_inventory(
            remote=plan.remote,
            path=remote_path,
            path_type=payload.path_type,
            checksum_mode=payload.checksum_mode,
            sample_limit=sample_limit,
        )

        sync_command: str | None = None
        sync_exit_code: int | None = None
        sync_error: str | None = None
        if execute and local_snapshot.ready:
            sync_command, sync_exit_code, sync_error = sync_payload_to_remote(
                remote=plan.remote,
                local_path=local_path,
                remote_path=remote_path,
                path_type=payload.path_type,
            )

        remote_after = collect_remote_inventory(
            remote=plan.remote,
            path=remote_path,
            path_type=payload.path_type,
            checksum_mode=payload.checksum_mode,
            sample_limit=sample_limit,
        )

        payload_results.append(
            PayloadSyncResult(
                name=payload.name,
                required=payload.required,
                description=payload.description,
                local_path=str(local_path),
                remote_path=remote_path,
                executed=execute,
                local=local_snapshot,
                remote_before=remote_before,
                remote_after=remote_after,
                sync_command=sync_command,
                sync_exit_code=sync_exit_code,
                sync_error=sync_error,
            )
        )

    return DataSyncReport(
        remote_host=plan.remote.host,
        remote_project_root=plan.remote.project_root,
        executed=execute,
        payloads=payload_results,
        notes=list(plan.notes),
        started_at=started_at,
        completed_at=_utc_now(),
    )


def write_sync_report(
    *,
    report: DataSyncReport,
    plan: SyncPlan,
    project_root: Path | str,
) -> DataSyncReport:
    resolved_project_root = resolve_project_path(str(project_root), ".")

    if plan.report.local_path:
        local_report_path = resolve_project_path(str(resolved_project_root), plan.report.local_path)
        report.local_report_path = str(local_report_path)

    if plan.report.remote_path:
        report.remote_report_path = resolve_remote_path(
            plan.remote.project_root, plan.report.remote_path
        )

    rendered_json = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"

    if report.remote_report_path:
        upload_error = upload_remote_report(
            remote=plan.remote,
            remote_path=report.remote_report_path,
            content=rendered_json,
        )
        report.report_upload_error = upload_error

    if report.local_report_path:
        local_report_path = Path(report.local_report_path)
        local_report_path.parent.mkdir(parents=True, exist_ok=True)
        local_report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")

    return report


def render_sync_report(report: DataSyncReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    mode = "execute" if report.executed else "inspect-only"
    lines = [
        f"Dataset sync ({report.remote_host}): {status}",
        f"Mode: {mode}",
        f"Remote root: {report.remote_project_root}",
        f"Started at: {report.started_at}",
        f"Completed at: {report.completed_at}",
    ]

    if report.local_report_path:
        lines.append(f"Local report: {report.local_report_path}")
    if report.remote_report_path:
        remote_line = f"Remote report: {report.remote_report_path}"
        if report.report_upload_error:
            remote_line = f"{remote_line} (upload error: {report.report_upload_error})"
        lines.append(remote_line)

    for payload in report.payloads:
        readiness = "ok" if payload.ready else "mismatch"
        line = (
            f"- {payload.name}: {readiness} "
            f"(local={payload.local.file_count} files/{payload.local.total_bytes} bytes, "
            f"remote={payload.remote_after.file_count} files/"
            f"{payload.remote_after.total_bytes} bytes, "
            f"checksum_mode={payload.local.checksum_mode})"
        )
        if payload.description:
            line = f"{line}; {payload.description}"
        if payload.sync_error:
            line = f"{line}; sync_error={payload.sync_error}"
        elif payload.sync_exit_code not in (None, 0):
            line = f"{line}; sync_exit_code={payload.sync_exit_code}"
        elif payload.executed and payload.sync_command:
            line = f"{line}; synced"
        if payload.remote_after.error:
            line = f"{line}; remote_error={payload.remote_after.error}"
        if payload.local.error:
            line = f"{line}; local_error={payload.local.error}"
        lines.append(line)

    if report.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in report.notes)

    if report.passed:
        lines.append("All required payloads match between local source paths and gpu-server.")
    else:
        lines.append(
            f"Missing or mismatched required payloads: {', '.join(report.missing_required)}"
        )
    return "\n".join(lines)


def collect_local_inventory(
    *,
    path: Path,
    path_type: PathType,
    checksum_mode: ChecksumMode,
    sample_limit: int,
) -> InventorySnapshot:
    exists = path.exists()
    error: str | None = None
    entries: list[tuple[str, int, Path]] = []

    if exists:
        if path_type == "dir":
            if not path.is_dir():
                error = "expected directory"
            else:
                entries = _walk_directory(path)
        elif path_type == "file":
            if not path.is_file():
                error = "expected file"
            else:
                entries = [(path.name, path.stat().st_size, path)]
        else:
            error = f"unsupported path_type={path_type}"

    checksum = None
    if exists and error is None:
        checksum = _build_checksum(entries=entries, checksum_mode=checksum_mode)

    return InventorySnapshot(
        path=str(path),
        path_type=path_type,
        exists=exists,
        file_count=len(entries),
        total_bytes=sum(size_bytes for _, size_bytes, _ in entries),
        checksum_mode=checksum_mode,
        checksum=checksum,
        samples=[relative_path for relative_path, _, _ in entries[:sample_limit]],
        error=error,
    )


def collect_remote_inventory(
    *,
    remote: RemoteSyncConfig,
    path: str,
    path_type: PathType,
    checksum_mode: ChecksumMode,
    sample_limit: int,
) -> InventorySnapshot:
    command = [
        "ssh",
        *remote.ssh_options,
        remote.host,
        "python3",
        "-",
        "--path",
        path,
        "--path-type",
        path_type,
        "--checksum-mode",
        checksum_mode,
        "--sample-limit",
        str(sample_limit),
    ]
    completed = subprocess.run(
        command,
        input=REMOTE_INVENTORY_SCRIPT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return InventorySnapshot(
            path=path,
            path_type=path_type,
            exists=False,
            file_count=0,
            total_bytes=0,
            checksum_mode=checksum_mode,
            checksum=None,
            error=_format_process_error(completed),
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return InventorySnapshot(
            path=path,
            path_type=path_type,
            exists=False,
            file_count=0,
            total_bytes=0,
            checksum_mode=checksum_mode,
            checksum=None,
            error=f"Invalid remote inventory payload: {exc}",
        )
    return InventorySnapshot.from_dict(payload)


def sync_payload_to_remote(
    *,
    remote: RemoteSyncConfig,
    local_path: Path,
    remote_path: str,
    path_type: PathType,
) -> tuple[str, int, str | None]:
    mkdir_error = _ensure_remote_directory(
        remote=remote,
        remote_directory=remote_path
        if path_type == "dir"
        else str(PurePosixPath(remote_path).parent),
    )
    if mkdir_error is not None:
        return "", 1, mkdir_error

    ssh_command = shlex.join(["ssh", *remote.ssh_options])
    source_argument = f"{local_path}/" if path_type == "dir" else str(local_path)
    target_argument = (
        f"{remote.host}:{_with_trailing_slash(remote_path)}"
        if path_type == "dir"
        else f"{remote.host}:{remote_path}"
    )
    command = ["rsync", "-a", "-e", ssh_command, source_argument, target_argument]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return shlex.join(command), completed.returncode, _format_process_error(completed)


def upload_remote_report(
    *,
    remote: RemoteSyncConfig,
    remote_path: str,
    content: str,
) -> str | None:
    parent = str(PurePosixPath(remote_path).parent)
    mkdir_error = _ensure_remote_directory(remote=remote, remote_directory=parent)
    if mkdir_error is not None:
        return mkdir_error

    remote_command = f"cat > {shlex.quote(remote_path)}"
    completed = subprocess.run(
        ["ssh", *remote.ssh_options, remote.host, remote_command],
        input=content,
        text=True,
        capture_output=True,
        check=False,
    )
    return _format_process_error(completed)


def resolve_remote_path(project_root: str, configured_path: str) -> str:
    candidate = PurePosixPath(configured_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(PurePosixPath(project_root) / candidate)


def _walk_directory(path: Path) -> list[tuple[str, int, Path]]:
    entries = [
        (child.relative_to(path).as_posix(), child.stat().st_size, child)
        for child in path.rglob("*")
        if child.is_file()
    ]
    return sorted(entries, key=lambda item: item[0])


def _build_checksum(
    *,
    entries: list[tuple[str, int, Path]],
    checksum_mode: ChecksumMode,
) -> str:
    digest = hashlib.sha256()
    for relative_path, size_bytes, absolute_path in entries:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\t")
        if checksum_mode == "catalog":
            digest.update(str(size_bytes).encode("utf-8"))
        else:
            digest.update(_hash_file(absolute_path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_remote_directory(*, remote: RemoteSyncConfig, remote_directory: str) -> str | None:
    remote_command = f"mkdir -p {shlex.quote(remote_directory)}"
    completed = subprocess.run(
        ["ssh", *remote.ssh_options, remote.host, remote_command],
        text=True,
        capture_output=True,
        check=False,
    )
    return _format_process_error(completed)


def _format_process_error(completed: subprocess.CompletedProcess[str]) -> str | None:
    if completed.returncode == 0:
        return None
    output = completed.stderr.strip() or completed.stdout.strip() or "command failed"
    return f"exit_code={completed.returncode}: {output}"


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _with_trailing_slash(path: str) -> str:
    return path if path.endswith("/") else f"{path}/"


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected [{key}] table in sync plan.")
    return value


def _optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected [{key}] table in sync plan.")
    return value


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


def _optional_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected string list for '{key}'.")
    return [str(item) for item in value]


def _require_literal(
    data: dict[str, Any],
    key: str,
    allowed: tuple[str, ...],
) -> str:
    value = _require_str(data, key)
    if value not in allowed:
        raise ValueError(f"Expected '{key}' to be one of {allowed}, got {value!r}.")
    return value
