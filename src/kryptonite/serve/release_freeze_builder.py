"""Helpers for building a release freeze snapshot from staged submission artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from kryptonite.deployment import resolve_project_path
from kryptonite.repro import fingerprint_path

from .release_freeze_models import ReleaseFreezeScope, SubmissionBundleReleaseFreeze
from .submission_bundle_config import (
    DEFAULT_SUBMISSION_BUNDLE_CODE_FINGERPRINT_PATHS,
    SubmissionBundleConfig,
)
from .submission_bundle_models import SubmissionBundleArtifactRef


def build_submission_bundle_release_freeze(
    *,
    config: SubmissionBundleConfig,
    project_root: Path,
    model_version: str,
    artifacts: tuple[SubmissionBundleArtifactRef, ...],
) -> SubmissionBundleReleaseFreeze:
    resolved_release_tag = _resolve_release_tag(
        configured_release_tag=config.release_tag,
        project_root=project_root,
    )
    scopes = [
        _build_code_scope(
            code_fingerprint_paths=config.code_fingerprint_paths,
            project_root=project_root,
            configured_release_tag=config.release_tag,
            resolved_release_tag=resolved_release_tag,
        )
    ]
    data_artifacts = tuple(artifact for artifact in artifacts if artifact.kind == "data_manifest")
    if data_artifacts:
        scopes.append(
            _build_artifact_scope(
                scope="data",
                version_tag=resolved_release_tag or "unversioned-data",
                artifacts=data_artifacts,
                metadata={"release_tag": resolved_release_tag},
            )
        )
    scopes.append(
        _build_artifact_scope(
            scope="model",
            version_tag=model_version
            if model_version != "unknown"
            else resolved_release_tag or "unknown-model",
            artifacts=tuple(
                artifact
                for artifact in artifacts
                if artifact.kind in {"model_bundle_metadata", "checkpoint"}
            ),
            metadata={"model_version": model_version},
        )
    )
    scopes.append(
        _build_artifact_scope(
            scope="engine",
            version_tag=resolved_release_tag or model_version or "unversioned-engine",
            artifacts=tuple(
                artifact
                for artifact in artifacts
                if artifact.kind in {"onnx_model", "tensorrt_plan", "triton_repository"}
            ),
            metadata={
                "onnx_included": any(artifact.kind == "onnx_model" for artifact in artifacts),
                "tensorrt_included": any(
                    artifact.kind == "tensorrt_plan" for artifact in artifacts
                ),
                "triton_repository_included": any(
                    artifact.kind == "triton_repository" for artifact in artifacts
                ),
            },
        )
    )
    return SubmissionBundleReleaseFreeze(
        release_tag=resolved_release_tag,
        scopes=tuple(scopes),
    )


def _build_code_scope(
    *,
    code_fingerprint_paths: tuple[str, ...],
    project_root: Path,
    configured_release_tag: str | None,
    resolved_release_tag: str | None,
) -> ReleaseFreezeScope:
    resolved_paths = tuple(
        resolve_project_path(str(project_root), configured_path)
        for configured_path in (
            code_fingerprint_paths or DEFAULT_SUBMISSION_BUNDLE_CODE_FINGERPRINT_PATHS
        )
    )
    fingerprints = [fingerprint_path(path) for path in resolved_paths]
    payload = json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(payload).hexdigest()
    version_tag = (
        resolved_release_tag
        or _get_exact_git_tag(project_root)
        or _build_git_fallback_tag(project_root)
    )
    return ReleaseFreezeScope(
        scope="code",
        version_tag=version_tag,
        checksum_algorithm="sha256-catalog",
        checksum=checksum,
        file_count=sum(int(fingerprint["file_count"]) for fingerprint in fingerprints),
        source_paths=tuple(str(path) for path in resolved_paths),
        staged_paths=(),
        metadata={
            "configured_release_tag": configured_release_tag,
            "git_commit": _run_git_command(project_root, "rev-parse", "HEAD"),
            "git_branch": _run_git_command(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_exact_tag": _get_exact_git_tag(project_root),
            "fingerprints": fingerprints,
        },
    )


def _build_artifact_scope(
    *,
    scope: str,
    version_tag: str,
    artifacts: tuple[SubmissionBundleArtifactRef, ...],
    metadata: dict[str, object],
) -> ReleaseFreezeScope:
    if not artifacts:
        raise ValueError(f"Release freeze scope {scope!r} must include at least one artifact.")
    digest = hashlib.sha256()
    for artifact in sorted(artifacts, key=lambda candidate: candidate.staged_path):
        digest.update(artifact.staged_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(artifact.sha256.encode("utf-8"))
        digest.update(b"\n")
    return ReleaseFreezeScope(
        scope=scope,
        version_tag=version_tag,
        checksum_algorithm="sha256-artifact-manifest",
        checksum=digest.hexdigest(),
        file_count=sum(artifact.file_count for artifact in artifacts),
        source_paths=tuple(artifact.original_path for artifact in artifacts),
        staged_paths=tuple(artifact.staged_path for artifact in artifacts),
        metadata={
            "artifact_count": len(artifacts),
            "artifact_kinds": [artifact.kind for artifact in artifacts],
            **metadata,
        },
    )


def _resolve_release_tag(
    *,
    configured_release_tag: str | None,
    project_root: Path,
) -> str | None:
    if configured_release_tag is not None and configured_release_tag.strip():
        return configured_release_tag.strip()
    return _get_exact_git_tag(project_root)


def _get_exact_git_tag(project_root: Path) -> str | None:
    return _run_git_command(project_root, "describe", "--tags", "--exact-match")


def _build_git_fallback_tag(project_root: Path) -> str:
    commit = _run_git_command(project_root, "rev-parse", "--short", "HEAD")
    if commit is not None:
        return f"git-{commit}"
    return "working-tree"


def _run_git_command(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


__all__ = ["build_submission_bundle_release_freeze"]
