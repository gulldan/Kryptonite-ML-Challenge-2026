"""Typed config loader for the submission/release bundle workflow."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_SUBMISSION_BUNDLE_MODES = frozenset({"candidate", "smoke"})
DEFAULT_SUBMISSION_BUNDLE_CODE_FINGERPRINT_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src",
    "apps",
    "scripts",
    "configs",
)


def normalize_submission_bundle_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SUBMISSION_BUNDLE_MODES:
        raise ValueError(
            "bundle_mode must be one of "
            f"{sorted(SUPPORTED_SUBMISSION_BUNDLE_MODES)}, got {value!r}."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class SubmissionBundleConfig:
    title: str
    bundle_id: str
    bundle_mode: str
    summary: str
    output_root: str
    release_tag: str | None
    create_archive: bool
    require_tensorrt_plan: bool
    repository_readme_path: str
    model_card_path: str
    runbook_path: str
    documentation_paths: tuple[str, ...]
    data_manifest_paths: tuple[str, ...]
    code_fingerprint_paths: tuple[str, ...]
    benchmark_paths: tuple[str, ...]
    config_paths: tuple[str, ...]
    checkpoint_paths: tuple[str, ...]
    supporting_paths: tuple[str, ...]
    model_bundle_metadata_path: str
    onnx_model_path: str
    tensorrt_plan_path: str | None
    threshold_calibration_path: str | None
    demo_assets_root: str
    triton_repository_root: str | None
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_mode = normalize_submission_bundle_mode(self.bundle_mode)
        object.__setattr__(self, "bundle_mode", normalized_mode)

        if not self.title.strip():
            raise ValueError("title must not be empty.")
        if not self.bundle_id.strip():
            raise ValueError("bundle_id must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")
        if not self.repository_readme_path.strip():
            raise ValueError("repository_readme_path must not be empty.")
        if not self.model_card_path.strip():
            raise ValueError("model_card_path must not be empty.")
        if not self.runbook_path.strip():
            raise ValueError("runbook_path must not be empty.")
        if not self.model_bundle_metadata_path.strip():
            raise ValueError("model_bundle_metadata_path must not be empty.")
        if not self.onnx_model_path.strip():
            raise ValueError("onnx_model_path must not be empty.")
        if not self.demo_assets_root.strip():
            raise ValueError("demo_assets_root must not be empty.")
        if not self.config_paths:
            raise ValueError("config_paths must include at least one config file.")

        if self.bundle_mode == "candidate":
            if self.release_tag is None or not self.release_tag.strip():
                raise ValueError("candidate bundle_mode requires release_tag.")
            if not self.benchmark_paths:
                raise ValueError(
                    "candidate bundle_mode requires at least one benchmark_paths entry."
                )
            if not self.data_manifest_paths:
                raise ValueError(
                    "candidate bundle_mode requires at least one data_manifest_paths entry."
                )
            if self.threshold_calibration_path is None:
                raise ValueError("candidate bundle_mode requires threshold_calibration_path.")
            if not self.checkpoint_paths:
                raise ValueError(
                    "candidate bundle_mode requires at least one checkpoint_paths entry."
                )

        if self.require_tensorrt_plan and self.tensorrt_plan_path is None:
            raise ValueError(
                "require_tensorrt_plan=true requires tensorrt_plan_path to be provided."
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["documentation_paths"] = list(self.documentation_paths)
        payload["data_manifest_paths"] = list(self.data_manifest_paths)
        payload["code_fingerprint_paths"] = list(self.code_fingerprint_paths)
        payload["benchmark_paths"] = list(self.benchmark_paths)
        payload["config_paths"] = list(self.config_paths)
        payload["checkpoint_paths"] = list(self.checkpoint_paths)
        payload["supporting_paths"] = list(self.supporting_paths)
        payload["notes"] = list(self.notes)
        return payload


def load_submission_bundle_config(*, config_path: Path | str) -> SubmissionBundleConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    bundle_id = str(raw.get("bundle_id", "")).strip()
    output_root = str(raw.get("output_root", "")).strip() or (
        f"artifacts/release-bundles/{bundle_id}"
    )
    release_tag = str(raw.get("release_tag", "")).strip() or None
    tensorrt_plan_path = str(raw.get("tensorrt_plan_path", "")).strip() or None
    threshold_calibration_path = str(raw.get("threshold_calibration_path", "")).strip() or None
    triton_repository_root = str(raw.get("triton_repository_root", "")).strip() or None
    return SubmissionBundleConfig(
        title=str(raw.get("title", "")).strip(),
        bundle_id=bundle_id,
        bundle_mode=str(raw.get("bundle_mode", "candidate")).strip(),
        summary=str(raw.get("summary", "")).strip(),
        output_root=output_root,
        release_tag=release_tag,
        create_archive=bool(raw.get("create_archive", True)),
        require_tensorrt_plan=bool(raw.get("require_tensorrt_plan", False)),
        repository_readme_path=str(raw.get("repository_readme_path", "README.md")).strip(),
        model_card_path=str(raw.get("model_card_path", "docs/model-card.md")).strip(),
        runbook_path=str(raw.get("runbook_path", "docs/release-runbook.md")).strip(),
        documentation_paths=tuple(
            _coerce_string_list(raw.get("documentation_paths", []), "documentation_paths")
        ),
        data_manifest_paths=tuple(
            _coerce_string_list(raw.get("data_manifest_paths", []), "data_manifest_paths")
        ),
        code_fingerprint_paths=tuple(
            _coerce_string_list(
                raw.get(
                    "code_fingerprint_paths",
                    list(DEFAULT_SUBMISSION_BUNDLE_CODE_FINGERPRINT_PATHS),
                ),
                "code_fingerprint_paths",
            )
        ),
        benchmark_paths=tuple(
            _coerce_string_list(raw.get("benchmark_paths", []), "benchmark_paths")
        ),
        config_paths=tuple(_coerce_string_list(raw.get("config_paths", []), "config_paths")),
        checkpoint_paths=tuple(
            _coerce_string_list(raw.get("checkpoint_paths", []), "checkpoint_paths")
        ),
        supporting_paths=tuple(
            _coerce_string_list(raw.get("supporting_paths", []), "supporting_paths")
        ),
        model_bundle_metadata_path=str(
            raw.get("model_bundle_metadata_path", "artifacts/model-bundle/metadata.json")
        ).strip(),
        onnx_model_path=str(
            raw.get("onnx_model_path", "artifacts/model-bundle/model.onnx")
        ).strip(),
        tensorrt_plan_path=tensorrt_plan_path,
        threshold_calibration_path=threshold_calibration_path,
        demo_assets_root=str(raw.get("demo_assets_root", "artifacts/demo-subset")).strip(),
        triton_repository_root=triton_repository_root,
        notes=tuple(_coerce_string_list(raw.get("notes", []), "notes")),
    )


def _coerce_string_list(raw: object, field_name: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array of strings.")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string.")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{field_name}[{index}] must not be empty.")
        values.append(stripped)
    return values


__all__ = [
    "DEFAULT_SUBMISSION_BUNDLE_CODE_FINGERPRINT_PATHS",
    "SUPPORTED_SUBMISSION_BUNDLE_MODES",
    "SubmissionBundleConfig",
    "load_submission_bundle_config",
    "normalize_submission_bundle_mode",
]
