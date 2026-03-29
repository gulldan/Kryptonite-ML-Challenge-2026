"""Typed config loader for reproducible ONNX Runtime parity reports."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from kryptonite.common.parsing import (
    coerce_optional_float as _coerce_optional_float,
    coerce_required_float as _coerce_required_float,
    coerce_required_int as _coerce_required_int,
    coerce_string_list as _coerce_string_list,
    coerce_table as _coerce_table,
)

_SUPPORTED_PARITY_VARIANTS = frozenset({"identity", "gaussian_noise", "clip", "pause"})
_SUPPORTED_PARITY_ROLES = frozenset({"enrollment", "test"})


@dataclass(frozen=True, slots=True)
class ONNXParityArtifactsConfig:
    model_bundle_metadata_path: str
    trial_rows_path: str
    metadata_rows_path: str

    def __post_init__(self) -> None:
        for field_name in (
            "model_bundle_metadata_path",
            "trial_rows_path",
            "metadata_rows_path",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"artifacts.{field_name} must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ONNXParityEvaluationConfig:
    seed: int
    prefer_demo_subset: bool
    max_trial_count: int | None
    score_normalize: bool
    promote_validated_backend: bool

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("evaluation.seed must be non-negative.")
        if self.max_trial_count is not None and self.max_trial_count <= 0:
            raise ValueError("evaluation.max_trial_count must be positive when provided.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ONNXParityVariantConfig:
    variant_id: str
    kind: str
    description: str
    apply_to_roles: tuple[str, ...]
    snr_db: float | None = None
    pre_gain_db: float | None = None
    clip_amplitude: float = 0.8
    pause_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.variant_id.strip():
            raise ValueError("variants[].id must not be empty.")
        if self.kind not in _SUPPORTED_PARITY_VARIANTS:
            raise ValueError(
                "variants[].kind must be one of "
                f"{sorted(_SUPPORTED_PARITY_VARIANTS)}, got {self.kind!r}."
            )
        if self.kind == "identity":
            return
        if not self.apply_to_roles:
            raise ValueError("variants[].apply_to_roles must not be empty for corrupting variants.")
        unknown_roles = [
            role for role in self.apply_to_roles if role not in _SUPPORTED_PARITY_ROLES
        ]
        if unknown_roles:
            raise ValueError(
                "variants[].apply_to_roles contains unsupported roles: "
                + ", ".join(sorted(unknown_roles))
            )
        if self.kind == "gaussian_noise":
            if self.snr_db is None or self.snr_db <= 0.0:
                raise ValueError("gaussian_noise variants require a positive snr_db.")
        elif self.kind == "clip":
            if self.pre_gain_db is None or self.pre_gain_db <= 0.0:
                raise ValueError("clip variants require a positive pre_gain_db.")
            if not 0.0 < self.clip_amplitude <= 1.0:
                raise ValueError("clip variants require clip_amplitude within (0.0, 1.0].")
        elif self.kind == "pause":
            if self.pause_ratio is None or not 0.0 < self.pause_ratio < 1.0:
                raise ValueError("pause variants require pause_ratio within (0.0, 1.0).")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["apply_to_roles"] = list(self.apply_to_roles)
        return payload


@dataclass(frozen=True, slots=True)
class ONNXParityTolerancesConfig:
    max_chunk_max_abs_diff: float
    max_pooled_max_abs_diff: float
    max_pooled_cosine_distance: float
    max_score_abs_diff: float
    max_eer_delta: float
    max_min_dcf_delta: float

    def __post_init__(self) -> None:
        for field_name in (
            "max_chunk_max_abs_diff",
            "max_pooled_max_abs_diff",
            "max_pooled_cosine_distance",
            "max_score_abs_diff",
            "max_eer_delta",
            "max_min_dcf_delta",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"tolerances.{field_name} must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ONNXParityConfig:
    title: str
    report_id: str
    summary: str
    project_root: str
    output_root: str
    artifacts: ONNXParityArtifactsConfig
    evaluation: ONNXParityEvaluationConfig
    variants: tuple[ONNXParityVariantConfig, ...]
    tolerances: ONNXParityTolerancesConfig
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty.")
        if not self.report_id.strip():
            raise ValueError("report_id must not be empty.")
        if not self.project_root.strip():
            raise ValueError("project_root must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")
        if not self.variants:
            raise ValueError("At least one parity variant must be configured.")
        seen_ids: set[str] = set()
        for variant in self.variants:
            if variant.variant_id in seen_ids:
                raise ValueError(f"Duplicate parity variant id: {variant.variant_id!r}.")
            seen_ids.add(variant.variant_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "summary": self.summary,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "artifacts": self.artifacts.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "tolerances": self.tolerances.to_dict(),
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
        }


def load_onnx_parity_config(*, config_path: Path | str) -> ONNXParityConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    artifacts = _coerce_table(raw.get("artifacts"), "artifacts")
    evaluation = _coerce_table(raw.get("evaluation"), "evaluation")
    tolerances = _coerce_table(raw.get("tolerances"), "tolerances")
    raw_variants = raw.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("variants must be a non-empty array of tables.")
    variants: list[ONNXParityVariantConfig] = []
    for index, variant in enumerate(raw_variants):
        if not isinstance(variant, dict):
            raise ValueError(f"variants[{index}] must be a table.")
        variant_table = _coerce_table(variant, f"variants[{index}]")
        variants.append(
            ONNXParityVariantConfig(
                variant_id=str(variant_table.get("id", "")).strip(),
                kind=str(variant_table.get("kind", "")).strip(),
                description=str(variant_table.get("description", "")).strip(),
                apply_to_roles=tuple(
                    _coerce_string_list(
                        variant_table.get("apply_to_roles", []),
                        f"variants[{index}].apply_to_roles",
                    )
                ),
                snr_db=_coerce_optional_float(variant_table.get("snr_db")),
                pre_gain_db=_coerce_optional_float(variant_table.get("pre_gain_db")),
                clip_amplitude=_coerce_optional_float(variant_table.get("clip_amplitude")) or 0.8,
                pause_ratio=_coerce_optional_float(variant_table.get("pause_ratio")),
            )
        )
    return ONNXParityConfig(
        title=str(raw.get("title", "")).strip(),
        report_id=str(raw.get("report_id", "")).strip(),
        summary=str(raw.get("summary", "")).strip(),
        project_root=str(raw.get("project_root", ".")).strip() or ".",
        output_root=str(raw.get("output_root", "")).strip(),
        artifacts=ONNXParityArtifactsConfig(
            model_bundle_metadata_path=str(artifacts.get("model_bundle_metadata_path", "")).strip(),
            trial_rows_path=str(artifacts.get("trial_rows_path", "")).strip(),
            metadata_rows_path=str(artifacts.get("metadata_rows_path", "")).strip(),
        ),
        evaluation=ONNXParityEvaluationConfig(
            seed=_coerce_required_int(evaluation.get("seed", 0), "evaluation.seed"),
            prefer_demo_subset=bool(evaluation.get("prefer_demo_subset", True)),
            max_trial_count=_coerce_optional_int(evaluation.get("max_trial_count")),
            score_normalize=bool(evaluation.get("score_normalize", True)),
            promote_validated_backend=bool(evaluation.get("promote_validated_backend", False)),
        ),
        variants=tuple(variants),
        tolerances=ONNXParityTolerancesConfig(
            max_chunk_max_abs_diff=_coerce_required_float(
                tolerances.get("max_chunk_max_abs_diff"),
                "tolerances.max_chunk_max_abs_diff",
            ),
            max_pooled_max_abs_diff=_coerce_required_float(
                tolerances.get("max_pooled_max_abs_diff"),
                "tolerances.max_pooled_max_abs_diff",
            ),
            max_pooled_cosine_distance=_coerce_required_float(
                tolerances.get("max_pooled_cosine_distance"),
                "tolerances.max_pooled_cosine_distance",
            ),
            max_score_abs_diff=_coerce_required_float(
                tolerances.get("max_score_abs_diff"),
                "tolerances.max_score_abs_diff",
            ),
            max_eer_delta=_coerce_required_float(
                tolerances.get("max_eer_delta"),
                "tolerances.max_eer_delta",
            ),
            max_min_dcf_delta=_coerce_required_float(
                tolerances.get("max_min_dcf_delta"),
                "tolerances.max_min_dcf_delta",
            ),
        ),
        validation_commands=tuple(
            _coerce_string_list(raw.get("validation_commands", []), "validation_commands")
        ),
        notes=tuple(_coerce_string_list(raw.get("notes", []), "notes")),
    )
def _coerce_optional_int(raw: object) -> int | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("configured integer values must be integers.")
    return raw


__all__ = [
    "ONNXParityArtifactsConfig",
    "ONNXParityConfig",
    "ONNXParityEvaluationConfig",
    "ONNXParityTolerancesConfig",
    "ONNXParityVariantConfig",
    "load_onnx_parity_config",
]
