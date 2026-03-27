"""Datamodels for the release freeze snapshot packaged with submission bundles."""

from __future__ import annotations

from dataclasses import dataclass

SUBMISSION_BUNDLE_RELEASE_FREEZE_JSON_NAME = "release_freeze.json"
SUBMISSION_BUNDLE_RELEASE_FREEZE_MARKDOWN_NAME = "release_freeze.md"


@dataclass(frozen=True, slots=True)
class ReleaseFreezeScope:
    scope: str
    version_tag: str
    checksum_algorithm: str
    checksum: str
    file_count: int
    source_paths: tuple[str, ...]
    staged_paths: tuple[str, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "version_tag": self.version_tag,
            "checksum_algorithm": self.checksum_algorithm,
            "checksum": self.checksum,
            "file_count": self.file_count,
            "source_paths": list(self.source_paths),
            "staged_paths": list(self.staged_paths),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SubmissionBundleReleaseFreeze:
    release_tag: str | None
    scopes: tuple[ReleaseFreezeScope, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "release_tag": self.release_tag,
            "scopes": [scope.to_dict() for scope in self.scopes],
        }


__all__ = [
    "ReleaseFreezeScope",
    "SUBMISSION_BUNDLE_RELEASE_FREEZE_JSON_NAME",
    "SUBMISSION_BUNDLE_RELEASE_FREEZE_MARKDOWN_NAME",
    "SubmissionBundleReleaseFreeze",
]
