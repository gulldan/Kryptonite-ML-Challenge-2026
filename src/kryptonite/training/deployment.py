"""Training-container artifact checks."""

from __future__ import annotations

from kryptonite.config import ProjectConfig
from kryptonite.deployment import ArtifactReport, ArtifactSpec, build_artifact_report


def build_training_artifact_report(
    *,
    config: ProjectConfig,
    strict: bool,
) -> ArtifactReport:
    specs = [
        ArtifactSpec(
            name="dataset_root",
            configured_path=config.paths.dataset_root,
            path_type="dir",
            require_non_empty=True,
            description="Training datasets must be mounted or synced before GPU runs.",
        ),
        ArtifactSpec(
            name="manifests_root",
            configured_path=config.paths.manifests_root,
            path_type="dir",
            require_non_empty=True,
            description="Dataset manifests must exist next to the runtime checkout.",
        ),
        ArtifactSpec(
            name="demo_manifest_file",
            configured_path=f"{config.paths.manifests_root}/demo_manifest.jsonl",
            path_type="file",
            require_non_empty=True,
            description="Canonical manifest for the generated mini-demo dataset.",
        ),
    ]
    return build_artifact_report(
        scope="train",
        strict=strict,
        project_root=config.paths.project_root,
        specs=specs,
    )
