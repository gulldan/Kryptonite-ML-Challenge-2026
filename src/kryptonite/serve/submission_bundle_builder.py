"""Builder for the self-contained submission/release bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

from kryptonite.deployment import (
    ArtifactReport,
    ArtifactSpec,
    build_artifact_report,
    render_artifact_report,
    resolve_project_path,
)
from kryptonite.project import get_project_layout

from .release_freeze_builder import build_submission_bundle_release_freeze
from .submission_bundle_config import SubmissionBundleConfig
from .submission_bundle_models import (
    SubmissionBundleArtifactRef,
    SubmissionBundleReport,
    SubmissionBundleSummary,
)


def build_submission_bundle_source_report(
    config: SubmissionBundleConfig,
    *,
    project_root: Path | str | None = None,
) -> ArtifactReport:
    resolved_project_root = _resolve_project_root(project_root)
    specs: list[ArtifactSpec] = [
        ArtifactSpec(
            name="repository_readme",
            configured_path=config.repository_readme_path,
            path_type="file",
            require_non_empty=True,
            description="Repository-level README staged into the release bundle docs.",
        ),
        ArtifactSpec(
            name="model_card",
            configured_path=config.model_card_path,
            path_type="file",
            require_non_empty=True,
            description="Release model card copied into the bundle docs.",
        ),
        ArtifactSpec(
            name="release_runbook",
            configured_path=config.runbook_path,
            path_type="file",
            require_non_empty=True,
            description="Release runbook copied into the bundle docs.",
        ),
        ArtifactSpec(
            name="model_bundle_metadata",
            configured_path=config.model_bundle_metadata_path,
            path_type="file",
            require_non_empty=True,
            description="Model bundle metadata used to stamp the bundle manifest and README.",
        ),
        ArtifactSpec(
            name="onnx_model",
            configured_path=config.onnx_model_path,
            path_type="file",
            require_non_empty=True,
            description="ONNX export staged as the primary portable model artifact.",
        ),
        ArtifactSpec(
            name="demo_assets_root",
            configured_path=config.demo_assets_root,
            path_type="dir",
            require_non_empty=True,
            description="Demo subset shipped with the handoff bundle.",
        ),
    ]
    specs.extend(
        ArtifactSpec(
            name=f"config_{index:02d}",
            configured_path=path,
            path_type="file",
            require_non_empty=True,
            description="Deployment or runtime config copied into the bundle.",
        )
        for index, path in enumerate(config.config_paths, start=1)
    )
    specs.extend(
        ArtifactSpec(
            name=f"documentation_{index:02d}",
            configured_path=path,
            path_type="file",
            require_non_empty=True,
            description="Additional release documentation copied into the bundle.",
        )
        for index, path in enumerate(config.documentation_paths, start=1)
    )
    specs.extend(
        ArtifactSpec(
            name=f"data_manifest_{index:02d}",
            configured_path=path,
            path_type=_infer_artifact_path_type(resolved_project_root, path),
            require_non_empty=True,
            description=(
                "Frozen data manifest or manifest directory staged into the release bundle."
            ),
        )
        for index, path in enumerate(config.data_manifest_paths, start=1)
    )
    specs.extend(
        ArtifactSpec(
            name=f"benchmark_{index:02d}",
            configured_path=path,
            path_type="file",
            require_non_empty=True,
            description="Frozen benchmark summary or companion report copied into the bundle.",
        )
        for index, path in enumerate(config.benchmark_paths, start=1)
    )
    specs.extend(
        ArtifactSpec(
            name=f"checkpoint_{index:02d}",
            configured_path=path,
            path_type="file",
            require_non_empty=True,
            description="Frozen checkpoint shipped with the bundle.",
        )
        for index, path in enumerate(config.checkpoint_paths, start=1)
    )
    specs.extend(
        ArtifactSpec(
            name=f"supporting_{index:02d}",
            configured_path=path,
            path_type="file",
            require_non_empty=True,
            description="Extra release notes or supporting metadata copied into the bundle.",
        )
        for index, path in enumerate(config.supporting_paths, start=1)
    )
    if config.threshold_calibration_path is not None:
        specs.append(
            ArtifactSpec(
                name="threshold_calibration",
                configured_path=config.threshold_calibration_path,
                path_type="file",
                require_non_empty=True,
                description="Frozen threshold calibration bundle for the active candidate.",
            )
        )
    if config.tensorrt_plan_path is not None:
        specs.append(
            ArtifactSpec(
                name="tensorrt_plan",
                configured_path=config.tensorrt_plan_path,
                path_type="file",
                require_non_empty=True,
                description="Optional TensorRT handoff artifact.",
            )
        )
    if config.triton_repository_root is not None:
        specs.append(
            ArtifactSpec(
                name="triton_repository",
                configured_path=config.triton_repository_root,
                path_type="dir",
                require_non_empty=True,
                description="Optional Triton model repository staged for deployment handoff.",
            )
        )
    return build_artifact_report(
        scope="submission_bundle",
        strict=True,
        project_root=str(resolved_project_root),
        specs=specs,
    )


def build_submission_bundle(
    config: SubmissionBundleConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> SubmissionBundleReport:
    resolved_project_root = _resolve_project_root(project_root)
    source_report = build_submission_bundle_source_report(
        config, project_root=resolved_project_root
    )
    if not source_report.passed:
        raise RuntimeError(render_artifact_report(source_report))

    output_root = resolve_project_path(str(resolved_project_root), config.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_config_artifact = None
    if config_path is not None:
        source_config_artifact = _copy_file_artifact(
            source_path=Path(config_path).resolve(),
            destination_path=output_root / "sources" / "submission_bundle_config.toml",
            output_root=output_root,
            kind="bundle_config",
        )

    metadata_path = _resolve_file(resolved_project_root, config.model_bundle_metadata_path)
    metadata_payload = _load_json_object(metadata_path)
    model_version = str(metadata_payload.get("model_version", "unknown"))
    structural_stub = bool(metadata_payload.get("structural_stub", False))

    artifacts: list[SubmissionBundleArtifactRef] = [
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, config.repository_readme_path),
            destination_path=output_root / "docs" / "repository-readme.md",
            output_root=output_root,
            kind="repository_readme",
        ),
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, config.model_card_path),
            destination_path=output_root / "docs" / "model-card.md",
            output_root=output_root,
            kind="model_card",
        ),
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, config.runbook_path),
            destination_path=output_root / "docs" / "release-runbook.md",
            output_root=output_root,
            kind="release_runbook",
        ),
        _copy_file_artifact(
            source_path=metadata_path,
            destination_path=output_root / "model" / "metadata.json",
            output_root=output_root,
            kind="model_bundle_metadata",
        ),
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, config.onnx_model_path),
            destination_path=output_root / "model" / "model.onnx",
            output_root=output_root,
            kind="onnx_model",
        ),
        _copy_dir_artifact(
            source_path=_resolve_dir(resolved_project_root, config.demo_assets_root),
            destination_path=output_root / "demo",
            output_root=output_root,
            kind="demo_assets",
        ),
    ]
    artifacts.extend(
        _copy_path_artifact(
            source_path=resolve_project_path(str(resolved_project_root), path),
            destination_path=output_root
            / "data-manifests"
            / f"manifest_{index:02d}_{Path(path).name}",
            output_root=output_root,
            kind="data_manifest",
        )
        for index, path in enumerate(config.data_manifest_paths, start=1)
    )
    artifacts.extend(
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, path),
            destination_path=output_root / "docs" / f"supporting_{index:02d}_{Path(path).name}",
            output_root=output_root,
            kind="documentation",
        )
        for index, path in enumerate(config.documentation_paths, start=1)
    )
    artifacts.extend(
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, path),
            destination_path=output_root / "benchmark" / f"benchmark_{index:02d}_{Path(path).name}",
            output_root=output_root,
            kind="benchmark",
        )
        for index, path in enumerate(config.benchmark_paths, start=1)
    )
    artifacts.extend(
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, path),
            destination_path=output_root / "configs" / f"config_{index:02d}_{Path(path).name}",
            output_root=output_root,
            kind="config",
        )
        for index, path in enumerate(config.config_paths, start=1)
    )
    artifacts.extend(
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, path),
            destination_path=output_root
            / "checkpoints"
            / f"checkpoint_{index:02d}_{Path(path).name}",
            output_root=output_root,
            kind="checkpoint",
        )
        for index, path in enumerate(config.checkpoint_paths, start=1)
    )
    artifacts.extend(
        _copy_file_artifact(
            source_path=_resolve_file(resolved_project_root, path),
            destination_path=output_root
            / "supporting"
            / f"supporting_{index:02d}_{Path(path).name}",
            output_root=output_root,
            kind="supporting",
        )
        for index, path in enumerate(config.supporting_paths, start=1)
    )
    if config.threshold_calibration_path is not None:
        artifacts.append(
            _copy_file_artifact(
                source_path=_resolve_file(resolved_project_root, config.threshold_calibration_path),
                destination_path=output_root
                / "thresholds"
                / "verification_threshold_calibration.json",
                output_root=output_root,
                kind="threshold_calibration",
            )
        )
    if config.tensorrt_plan_path is not None:
        artifacts.append(
            _copy_file_artifact(
                source_path=_resolve_file(resolved_project_root, config.tensorrt_plan_path),
                destination_path=output_root / "model" / "model.plan",
                output_root=output_root,
                kind="tensorrt_plan",
            )
        )
    if config.triton_repository_root is not None:
        artifacts.append(
            _copy_dir_artifact(
                source_path=_resolve_dir(resolved_project_root, config.triton_repository_root),
                destination_path=output_root / "triton-model-repository",
                output_root=output_root,
                kind="triton_repository",
            )
        )

    warnings = _build_warnings(
        bundle_mode=config.bundle_mode,
        release_tag=config.release_tag,
        structural_stub=structural_stub,
        has_data_manifests=bool(config.data_manifest_paths),
        has_threshold=config.threshold_calibration_path is not None,
        has_benchmarks=bool(config.benchmark_paths),
        has_tensorrt=config.tensorrt_plan_path is not None,
    )
    release_freeze = build_submission_bundle_release_freeze(
        config=config,
        project_root=resolved_project_root,
        model_version=model_version,
        artifacts=tuple(artifacts),
    )
    summary = SubmissionBundleSummary(
        bundle_mode=config.bundle_mode,
        release_tag=release_freeze.release_tag,
        model_version=model_version,
        structural_stub=structural_stub,
        config_count=len(config.config_paths),
        data_manifest_count=len(config.data_manifest_paths),
        benchmark_artifact_count=len(config.benchmark_paths),
        checkpoint_count=len(config.checkpoint_paths),
        documentation_count=3 + len(config.documentation_paths),
        supporting_artifact_count=len(config.supporting_paths),
        threshold_calibration_included=config.threshold_calibration_path is not None,
        tensorrt_plan_included=config.tensorrt_plan_path is not None,
        triton_repository_included=config.triton_repository_root is not None,
        demo_assets_included=True,
        source_artifact_count=len(artifacts) + (1 if source_config_artifact is not None else 0),
        release_freeze_scope_count=len(release_freeze.scopes),
    )
    return SubmissionBundleReport(
        title=config.title,
        bundle_id=config.bundle_id,
        bundle_mode=config.bundle_mode,
        summary_text=config.summary,
        output_root=str(output_root),
        source_config_artifact=source_config_artifact,
        notes=config.notes,
        warnings=warnings,
        summary=summary,
        artifacts=tuple(artifacts),
        release_freeze=release_freeze,
    )


def _build_warnings(
    *,
    bundle_mode: str,
    release_tag: str | None,
    structural_stub: bool,
    has_data_manifests: bool,
    has_threshold: bool,
    has_benchmarks: bool,
    has_tensorrt: bool,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if bundle_mode == "smoke":
        warnings.append(
            "Smoke mode allows candidate-only artifacts to be omitted; "
            "do not treat this bundle as a final submission."
        )
    if release_tag is None:
        warnings.append(
            "No explicit release_tag was recorded; release freeze metadata falls back to git "
            "metadata where possible."
        )
    if structural_stub:
        warnings.append(
            "Model bundle metadata marks the staged encoder as a structural smoke "
            "stub, not a production-grade release artifact."
        )
    if not has_data_manifests:
        warnings.append("No frozen data manifests were staged.")
    if not has_threshold:
        warnings.append("No threshold calibration artifact was staged.")
    if not has_benchmarks:
        warnings.append("No frozen benchmark summary was staged.")
    if not has_tensorrt:
        warnings.append("No TensorRT plan was staged.")
    return tuple(warnings)


def _resolve_project_root(project_root: Path | str | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return get_project_layout().root.resolve()


def _resolve_file(project_root: Path, configured_path: str) -> Path:
    resolved = resolve_project_path(str(project_root), configured_path)
    if not resolved.is_file():
        raise ValueError(f"Expected file at {resolved}, but it does not exist.")
    return resolved


def _resolve_dir(project_root: Path, configured_path: str) -> Path:
    resolved = resolve_project_path(str(project_root), configured_path)
    if not resolved.is_dir():
        raise ValueError(f"Expected directory at {resolved}, but it does not exist.")
    return resolved


def _copy_file_artifact(
    *,
    source_path: Path,
    destination_path: Path,
    output_root: Path,
    kind: str,
) -> SubmissionBundleArtifactRef:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    return SubmissionBundleArtifactRef(
        kind=kind,
        original_path=str(source_path),
        staged_path=_relative_to(destination_path, output_root),
        path_type="file",
        sha256=_sha256_file(destination_path),
        file_count=1,
    )


def _copy_dir_artifact(
    *,
    source_path: Path,
    destination_path: Path,
    output_root: Path,
    kind: str,
) -> SubmissionBundleArtifactRef:
    if destination_path.exists():
        shutil.rmtree(destination_path)
    shutil.copytree(source_path, destination_path)
    directory_sha256, file_count = _sha256_dir(destination_path)
    return SubmissionBundleArtifactRef(
        kind=kind,
        original_path=str(source_path),
        staged_path=_relative_to(destination_path, output_root),
        path_type="dir",
        sha256=directory_sha256,
        file_count=file_count,
    )


def _copy_path_artifact(
    *,
    source_path: Path,
    destination_path: Path,
    output_root: Path,
    kind: str,
) -> SubmissionBundleArtifactRef:
    if source_path.is_dir():
        return _copy_dir_artifact(
            source_path=source_path,
            destination_path=destination_path,
            output_root=output_root,
            kind=kind,
        )
    if source_path.is_file():
        return _copy_file_artifact(
            source_path=source_path,
            destination_path=destination_path,
            output_root=output_root,
            kind=kind,
        )
    raise ValueError(f"Expected file or directory at {source_path}, but it does not exist.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_dir(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for file_path in files:
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(file_path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _relative_to(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _infer_artifact_path_type(project_root: Path, configured_path: str) -> str:
    resolved = resolve_project_path(str(project_root), configured_path)
    if resolved.is_dir():
        return "dir"
    return "file"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return cast(dict[str, Any], payload)


__all__ = [
    "build_submission_bundle",
    "build_submission_bundle_source_report",
]
