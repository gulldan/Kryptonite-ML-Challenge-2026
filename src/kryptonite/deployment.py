"""Shared deployment artifact preflight helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    name: str
    configured_path: str
    path_type: str
    required_on_target: bool = True
    require_non_empty: bool = False
    description: str | None = None


@dataclass(slots=True)
class ArtifactCheck:
    name: str
    configured_path: str
    resolved_path: str
    path_type: str
    required_on_target: bool
    require_non_empty: bool
    exists: bool
    non_empty: bool
    description: str | None = None
    details: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ready(self) -> bool:
        if not self.exists:
            return False
        if self.require_non_empty and not self.non_empty:
            return False
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "configured_path": self.configured_path,
            "resolved_path": self.resolved_path,
            "path_type": self.path_type,
            "required_on_target": self.required_on_target,
            "require_non_empty": self.require_non_empty,
            "exists": self.exists,
            "non_empty": self.non_empty,
            "ready": self.ready,
            "description": self.description,
            "details": dict(self.details),
            "error": self.error,
        }


@dataclass(slots=True)
class ArtifactReport:
    scope: str
    strict: bool
    checks: list[ArtifactCheck]

    @property
    def missing_required(self) -> list[str]:
        return [check.name for check in self.checks if check.required_on_target and not check.ready]

    @property
    def passed(self) -> bool:
        return not self.strict or not self.missing_required

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "strict": self.strict,
            "passed": self.passed,
            "missing_required": self.missing_required,
            "checks": [check.to_dict() for check in self.checks],
        }


def build_artifact_report(
    *,
    scope: str,
    strict: bool,
    project_root: str,
    specs: list[ArtifactSpec],
) -> ArtifactReport:
    checks = [_build_artifact_check(project_root=project_root, spec=spec) for spec in specs]
    return ArtifactReport(scope=scope, strict=strict, checks=checks)


def render_artifact_report(report: ArtifactReport) -> str:
    if report.strict:
        status = "PASS" if report.passed else "FAIL"
    else:
        status = "ADVISORY"

    lines = [
        f"Artifact preflight ({report.scope}): {status}",
        f"Mode: {'strict' if report.strict else 'advisory'}",
    ]
    for check in report.checks:
        if check.ready:
            readiness = "ok"
        elif check.exists:
            readiness = "incomplete"
        else:
            readiness = "missing"
        line = (
            f"- {check.name}: {readiness} "
            f"(type={check.path_type}, path={check.resolved_path}, "
            f"required_on_target={check.required_on_target})"
        )
        if check.description:
            line = f"{line}; {check.description}"
        if check.details:
            detail_text = ", ".join(
                f"{key}={value}" for key, value in sorted(check.details.items())
            )
            line = f"{line}; {detail_text}"
        if check.error:
            line = f"{line}; error={check.error}"
        lines.append(line)

    if report.strict:
        if report.passed:
            lines.append("All target artifacts are present.")
        else:
            missing = ", ".join(report.missing_required)
            lines.append(f"Missing required target artifacts: {missing}")
    else:
        lines.append(
            "Artifacts were inspected in advisory mode only; missing paths do not fail the run."
        )
    return "\n".join(lines)


def resolve_project_path(project_root: str, configured_path: str) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate

    root = Path(project_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return (root / candidate).resolve()


def _build_artifact_check(*, project_root: str, spec: ArtifactSpec) -> ArtifactCheck:
    path = resolve_project_path(project_root, spec.configured_path)
    exists = path.exists()
    non_empty = False
    details: dict[str, str] = {}
    error: str | None = None
    try:
        if spec.path_type == "dir":
            if exists and path.is_dir():
                non_empty = any(path.iterdir()) if spec.require_non_empty else True
                details["entries"] = str(sum(1 for _ in path.iterdir()))
            elif exists:
                error = "expected directory"
        elif spec.path_type == "file":
            if exists and path.is_file():
                non_empty = path.stat().st_size > 0 if spec.require_non_empty else True
                details["size_bytes"] = str(path.stat().st_size)
            elif exists:
                error = "expected file"
        else:
            error = f"unsupported path_type={spec.path_type}"
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"

    if not exists:
        details["entries"] = "0" if spec.path_type == "dir" else "-"
    return ArtifactCheck(
        name=spec.name,
        configured_path=spec.configured_path,
        resolved_path=str(path),
        path_type=spec.path_type,
        required_on_target=spec.required_on_target,
        require_non_empty=spec.require_non_empty,
        exists=exists,
        non_empty=non_empty,
        description=spec.description,
        details=details,
        error=error,
    )
