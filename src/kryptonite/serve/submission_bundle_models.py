"""Datamodels for the self-contained submission/release bundle."""

from __future__ import annotations

from dataclasses import dataclass

from .release_freeze_models import (
    SUBMISSION_BUNDLE_RELEASE_FREEZE_JSON_NAME,
    SUBMISSION_BUNDLE_RELEASE_FREEZE_MARKDOWN_NAME,
    ReleaseFreezeScope,
    SubmissionBundleReleaseFreeze,
)

SUBMISSION_BUNDLE_JSON_NAME = "submission_bundle.json"
SUBMISSION_BUNDLE_MARKDOWN_NAME = "submission_bundle.md"
SUBMISSION_BUNDLE_README_NAME = "README.md"


@dataclass(frozen=True, slots=True)
class SubmissionBundleArtifactRef:
    kind: str
    original_path: str
    staged_path: str
    path_type: str
    sha256: str
    file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "original_path": self.original_path,
            "staged_path": self.staged_path,
            "path_type": self.path_type,
            "sha256": self.sha256,
            "file_count": self.file_count,
        }


@dataclass(frozen=True, slots=True)
class SubmissionBundleSummary:
    bundle_mode: str
    release_tag: str | None
    model_version: str
    structural_stub: bool
    config_count: int
    data_manifest_count: int
    benchmark_artifact_count: int
    checkpoint_count: int
    documentation_count: int
    supporting_artifact_count: int
    threshold_calibration_included: bool
    tensorrt_plan_included: bool
    triton_repository_included: bool
    demo_assets_included: bool
    source_artifact_count: int
    release_freeze_scope_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_mode": self.bundle_mode,
            "release_tag": self.release_tag,
            "model_version": self.model_version,
            "structural_stub": self.structural_stub,
            "config_count": self.config_count,
            "data_manifest_count": self.data_manifest_count,
            "benchmark_artifact_count": self.benchmark_artifact_count,
            "checkpoint_count": self.checkpoint_count,
            "documentation_count": self.documentation_count,
            "supporting_artifact_count": self.supporting_artifact_count,
            "threshold_calibration_included": self.threshold_calibration_included,
            "tensorrt_plan_included": self.tensorrt_plan_included,
            "triton_repository_included": self.triton_repository_included,
            "demo_assets_included": self.demo_assets_included,
            "source_artifact_count": self.source_artifact_count,
            "release_freeze_scope_count": self.release_freeze_scope_count,
        }


@dataclass(frozen=True, slots=True)
class SubmissionBundleReport:
    title: str
    bundle_id: str
    bundle_mode: str
    summary_text: str
    output_root: str
    source_config_artifact: SubmissionBundleArtifactRef | None
    notes: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: SubmissionBundleSummary
    artifacts: tuple[SubmissionBundleArtifactRef, ...]
    release_freeze: SubmissionBundleReleaseFreeze

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "bundle_id": self.bundle_id,
            "bundle_mode": self.bundle_mode,
            "summary_text": self.summary_text,
            "output_root": self.output_root,
            "source_config_artifact": (
                None
                if self.source_config_artifact is None
                else self.source_config_artifact.to_dict()
            ),
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "summary": self.summary.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "release_freeze": self.release_freeze.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenSubmissionBundle:
    output_root: str
    readme_path: str
    report_json_path: str
    report_markdown_path: str
    release_freeze_json_path: str
    release_freeze_markdown_path: str
    archive_path: str | None
    summary: SubmissionBundleSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "readme_path": self.readme_path,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "release_freeze_json_path": self.release_freeze_json_path,
            "release_freeze_markdown_path": self.release_freeze_markdown_path,
            "archive_path": self.archive_path,
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "ReleaseFreezeScope",
    "SUBMISSION_BUNDLE_JSON_NAME",
    "SUBMISSION_BUNDLE_MARKDOWN_NAME",
    "SUBMISSION_BUNDLE_README_NAME",
    "SUBMISSION_BUNDLE_RELEASE_FREEZE_JSON_NAME",
    "SUBMISSION_BUNDLE_RELEASE_FREEZE_MARKDOWN_NAME",
    "SubmissionBundleArtifactRef",
    "SubmissionBundleReport",
    "SubmissionBundleReleaseFreeze",
    "SubmissionBundleSummary",
    "WrittenSubmissionBundle",
]
