"""Inference/demo deployment artifact checks."""

from __future__ import annotations

from kryptonite.config import ProjectConfig
from kryptonite.deployment import ArtifactReport, ArtifactSpec, build_artifact_report

from .enrollment_cache import (
    ENROLLMENT_EMBEDDINGS_NPZ_NAME,
    ENROLLMENT_METADATA_PARQUET_NAME,
    ENROLLMENT_SUMMARY_JSON_NAME,
)


def build_infer_artifact_report(
    *,
    config: ProjectConfig,
    strict: bool,
) -> ArtifactReport:
    specs = [
        ArtifactSpec(
            name="manifests_root",
            configured_path=config.paths.manifests_root,
            path_type="dir",
            require_non_empty=True,
            description="Resolved manifests must be present for demo/runtime validation.",
        ),
        ArtifactSpec(
            name="demo_manifest_file",
            configured_path=f"{config.paths.manifests_root}/demo_manifest.jsonl",
            path_type="file",
            require_non_empty=True,
            description="Canonical manifest that binds dataset audio to the demo subset.",
        ),
        ArtifactSpec(
            name="model_bundle_root",
            configured_path=config.deployment.model_bundle_root,
            path_type="dir",
            require_non_empty=True,
            description="Exported model bundle for the selected inference backend.",
        ),
        ArtifactSpec(
            name="model_onnx_file",
            configured_path=f"{config.deployment.model_bundle_root}/model.onnx",
            path_type="file",
            require_non_empty=True,
            description="Deployable ONNX graph mounted into the demo container.",
        ),
        ArtifactSpec(
            name="model_metadata_file",
            configured_path=f"{config.deployment.model_bundle_root}/metadata.json",
            path_type="file",
            require_non_empty=True,
            description="Model bundle metadata that documents the mounted graph.",
        ),
        ArtifactSpec(
            name="demo_subset_root",
            configured_path=config.deployment.demo_subset_root,
            path_type="dir",
            require_non_empty=True,
            description="Enrollment/test demo subset prepared for the containerized flow.",
        ),
        ArtifactSpec(
            name="demo_enrollment_dir",
            configured_path=f"{config.deployment.demo_subset_root}/enrollment",
            path_type="dir",
            require_non_empty=True,
            description="Enrollment samples that the demo flow can mount explicitly.",
        ),
        ArtifactSpec(
            name="demo_test_dir",
            configured_path=f"{config.deployment.demo_subset_root}/test",
            path_type="dir",
            require_non_empty=True,
            description="Held-out test samples that mirror the enrollment subset.",
        ),
        ArtifactSpec(
            name="demo_subset_file",
            configured_path=f"{config.deployment.demo_subset_root}/demo_subset.json",
            path_type="file",
            require_non_empty=True,
            description="Subset metadata that maps enrollment/test samples for the demo run.",
        ),
        ArtifactSpec(
            name="enrollment_cache_root",
            configured_path=config.deployment.enrollment_cache_root,
            path_type="dir",
            require_non_empty=True,
            description="Offline enrollment cache consumed by the runtime verify flow.",
        ),
        ArtifactSpec(
            name="enrollment_embeddings_file",
            configured_path=(
                f"{config.deployment.enrollment_cache_root}/{ENROLLMENT_EMBEDDINGS_NPZ_NAME}"
            ),
            path_type="file",
            require_non_empty=True,
            description=(
                "Normalized enrollment embedding centroids prepared before runtime startup."
            ),
        ),
        ArtifactSpec(
            name="enrollment_metadata_file",
            configured_path=(
                f"{config.deployment.enrollment_cache_root}/{ENROLLMENT_METADATA_PARQUET_NAME}"
            ),
            path_type="file",
            require_non_empty=True,
            description="Enrollment cache metadata aligned with the stored centroid table.",
        ),
        ArtifactSpec(
            name="enrollment_summary_file",
            configured_path=(
                f"{config.deployment.enrollment_cache_root}/{ENROLLMENT_SUMMARY_JSON_NAME}"
            ),
            path_type="file",
            require_non_empty=True,
            description="Enrollment cache provenance and model-bundle compatibility summary.",
        ),
    ]
    return build_artifact_report(
        scope="infer",
        strict=strict,
        project_root=config.paths.project_root,
        specs=specs,
    )
