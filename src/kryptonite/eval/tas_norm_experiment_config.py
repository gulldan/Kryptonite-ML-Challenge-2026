"""Typed config loader for reproducible TAS-norm experiment reports."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from .cohort_bank import CohortEmbeddingBankSelection
from .tas_norm import TasNormTrainingConfig


@dataclass(frozen=True, slots=True)
class TasNormExperimentArtifactsConfig:
    scores_path: str
    trials_path: str
    metadata_path: str
    embeddings_path: str
    cohort_bank_output_root: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TasNormExperimentRuntimeConfig:
    top_k: int
    std_epsilon: float
    exclude_matching_speakers: bool
    eval_fraction: float
    split_seed: int

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("tas_norm.top_k must be positive.")
        if self.std_epsilon <= 0.0:
            raise ValueError("tas_norm.std_epsilon must be positive.")
        if not 0.0 < self.eval_fraction < 1.0:
            raise ValueError("tas_norm.eval_fraction must be within (0, 1).")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TasNormExperimentGatesConfig:
    min_train_trials: int
    min_eval_trials: int
    min_train_positives: int
    min_train_negatives: int
    min_eval_positives: int
    min_eval_negatives: int
    min_eer_gain_vs_raw: float
    min_min_dcf_gain_vs_raw: float
    min_eer_gain_vs_as_norm: float
    min_min_dcf_gain_vs_as_norm: float
    max_train_eval_eer_gap: float | None
    max_train_eval_min_dcf_gap: float | None

    def __post_init__(self) -> None:
        for field_name in (
            "min_train_trials",
            "min_eval_trials",
            "min_train_positives",
            "min_train_negatives",
            "min_eval_positives",
            "min_eval_negatives",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"gates.{field_name} must be positive.")
        for field_name in (
            "min_eer_gain_vs_raw",
            "min_min_dcf_gain_vs_raw",
            "min_eer_gain_vs_as_norm",
            "min_min_dcf_gain_vs_as_norm",
            "max_train_eval_eer_gap",
            "max_train_eval_min_dcf_gap",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0.0:
                raise ValueError(f"gates.{field_name} must be non-negative when provided.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TasNormExperimentConfig:
    title: str
    report_id: str
    candidate_label: str
    summary: str
    output_root: str
    artifacts: TasNormExperimentArtifactsConfig
    cohort_selection: CohortEmbeddingBankSelection
    runtime: TasNormExperimentRuntimeConfig
    training: TasNormTrainingConfig
    gates: TasNormExperimentGatesConfig
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("title", "report_id", "candidate_label", "output_root"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "candidate_label": self.candidate_label,
            "summary": self.summary,
            "output_root": self.output_root,
            "artifacts": self.artifacts.to_dict(),
            "cohort_selection": self.cohort_selection.to_dict(),
            "runtime": self.runtime.to_dict(),
            "training": self.training.to_dict(),
            "gates": self.gates.to_dict(),
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
        }


def load_tas_norm_experiment_config(*, config_path: Path | str) -> TasNormExperimentConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    artifacts = _coerce_table(raw.get("artifacts"), "artifacts")
    cohort_selection = _coerce_table(raw.get("cohort_selection", {}), "cohort_selection")
    runtime = _coerce_table(raw.get("tas_norm"), "tas_norm")
    training = _coerce_table(raw.get("training"), "training")
    gates = _coerce_table(raw.get("gates"), "gates")
    report_id = str(raw.get("report_id", "")).strip()
    output_root = str(raw.get("output_root", "")).strip() or (
        f"artifacts/release-decisions/{report_id}"
    )
    return TasNormExperimentConfig(
        title=str(raw.get("title", "")).strip(),
        report_id=report_id,
        candidate_label=str(raw.get("candidate_label", "")).strip(),
        summary=str(raw.get("summary", "")).strip(),
        output_root=output_root,
        artifacts=TasNormExperimentArtifactsConfig(
            scores_path=_require_string(artifacts, "scores_path"),
            trials_path=_require_string(artifacts, "trials_path"),
            metadata_path=_require_string(artifacts, "metadata_path"),
            embeddings_path=_require_string(artifacts, "embeddings_path"),
            cohort_bank_output_root=_require_string(artifacts, "cohort_bank_output_root"),
        ),
        cohort_selection=CohortEmbeddingBankSelection(
            include_roles=tuple(
                _coerce_string_list(
                    cohort_selection.get("include_roles", []),
                    "cohort_selection.include_roles",
                )
            ),
            include_splits=tuple(
                _coerce_string_list(
                    cohort_selection.get("include_splits", []),
                    "cohort_selection.include_splits",
                )
            ),
            include_datasets=tuple(
                _coerce_string_list(
                    cohort_selection.get("include_datasets", []),
                    "cohort_selection.include_datasets",
                )
            ),
            min_embeddings_per_speaker=_coerce_positive_int(
                cohort_selection.get("min_embeddings_per_speaker", 1),
                "cohort_selection.min_embeddings_per_speaker",
            ),
            max_embeddings_per_speaker=_coerce_optional_positive_int(
                cohort_selection.get("max_embeddings_per_speaker"),
                "cohort_selection.max_embeddings_per_speaker",
            ),
            max_embeddings=_coerce_optional_positive_int(
                cohort_selection.get("max_embeddings"),
                "cohort_selection.max_embeddings",
            ),
            trial_paths=tuple([_require_string(artifacts, "trials_path")]),
            validation_manifest_paths=tuple(
                _coerce_string_list(
                    cohort_selection.get("validation_manifest_paths", []),
                    "cohort_selection.validation_manifest_paths",
                )
            ),
            strict_speaker_disjointness=bool(
                cohort_selection.get("strict_speaker_disjointness", False)
            ),
            allow_trial_overlap_fallback=bool(
                cohort_selection.get("allow_trial_overlap_fallback", True)
            ),
            point_id_field=str(cohort_selection.get("point_id_field", "atlas_point_id")).strip(),
            embeddings_key=str(cohort_selection.get("embeddings_key", "embeddings")).strip(),
            ids_key=_coerce_optional_string(cohort_selection.get("ids_key", "point_ids")),
        ),
        runtime=TasNormExperimentRuntimeConfig(
            top_k=_coerce_positive_int(runtime.get("top_k", 100), "tas_norm.top_k"),
            std_epsilon=_coerce_positive_float(
                runtime.get("std_epsilon", 1e-6),
                "tas_norm.std_epsilon",
            ),
            exclude_matching_speakers=bool(runtime.get("exclude_matching_speakers", True)),
            eval_fraction=_coerce_fraction(
                runtime.get("eval_fraction", 0.5),
                "tas_norm.eval_fraction",
            ),
            split_seed=_coerce_int(runtime.get("split_seed", 528), "tas_norm.split_seed"),
        ),
        training=TasNormTrainingConfig(
            learning_rate=_coerce_positive_float(
                training.get("learning_rate", 0.1),
                "training.learning_rate",
            ),
            max_steps=_coerce_positive_int(training.get("max_steps", 300), "training.max_steps"),
            l2_regularization=_coerce_non_negative_float(
                training.get("l2_regularization", 1e-3),
                "training.l2_regularization",
            ),
            early_stopping_patience=_coerce_positive_int(
                training.get("early_stopping_patience", 25),
                "training.early_stopping_patience",
            ),
            min_relative_loss_improvement=_coerce_non_negative_float(
                training.get("min_relative_loss_improvement", 1e-4),
                "training.min_relative_loss_improvement",
            ),
            balance_classes=bool(training.get("balance_classes", True)),
        ),
        gates=TasNormExperimentGatesConfig(
            min_train_trials=_coerce_positive_int(
                gates.get("min_train_trials", 4),
                "gates.min_train_trials",
            ),
            min_eval_trials=_coerce_positive_int(
                gates.get("min_eval_trials", 4),
                "gates.min_eval_trials",
            ),
            min_train_positives=_coerce_positive_int(
                gates.get("min_train_positives", 1),
                "gates.min_train_positives",
            ),
            min_train_negatives=_coerce_positive_int(
                gates.get("min_train_negatives", 1),
                "gates.min_train_negatives",
            ),
            min_eval_positives=_coerce_positive_int(
                gates.get("min_eval_positives", 1),
                "gates.min_eval_positives",
            ),
            min_eval_negatives=_coerce_positive_int(
                gates.get("min_eval_negatives", 1),
                "gates.min_eval_negatives",
            ),
            min_eer_gain_vs_raw=_coerce_non_negative_float(
                gates.get("min_eer_gain_vs_raw", 0.0),
                "gates.min_eer_gain_vs_raw",
            ),
            min_min_dcf_gain_vs_raw=_coerce_non_negative_float(
                gates.get("min_min_dcf_gain_vs_raw", 0.0),
                "gates.min_min_dcf_gain_vs_raw",
            ),
            min_eer_gain_vs_as_norm=_coerce_non_negative_float(
                gates.get("min_eer_gain_vs_as_norm", 0.0),
                "gates.min_eer_gain_vs_as_norm",
            ),
            min_min_dcf_gain_vs_as_norm=_coerce_non_negative_float(
                gates.get("min_min_dcf_gain_vs_as_norm", 0.0),
                "gates.min_min_dcf_gain_vs_as_norm",
            ),
            max_train_eval_eer_gap=_coerce_optional_non_negative_float(
                gates.get("max_train_eval_eer_gap"),
                "gates.max_train_eval_eer_gap",
            ),
            max_train_eval_min_dcf_gap=_coerce_optional_non_negative_float(
                gates.get("max_train_eval_min_dcf_gap"),
                "gates.max_train_eval_min_dcf_gap",
            ),
        ),
        validation_commands=tuple(
            _coerce_string_list(raw.get("validation_commands", []), "validation_commands")
        ),
        notes=tuple(_coerce_string_list(raw.get("notes", []), "notes")),
    )


def _coerce_table(raw: object, field_name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a table.")
    return {str(key): value for key, value in raw.items()}


def _require_string(raw: dict[str, object], field_name: str) -> str:
    value = _coerce_optional_string(raw.get(field_name))
    if value is None:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _coerce_optional_string(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("expected a string.")
    stripped = raw.strip()
    return stripped or None


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


def _coerce_positive_int(raw: object, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field_name} must be a positive integer.")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _coerce_optional_positive_int(raw: object, field_name: str) -> int | None:
    if raw is None or raw == "":
        return None
    return _coerce_positive_int(raw, field_name)


def _coerce_int(raw: object, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field_name} must be an integer.")
    return int(raw)


def _coerce_positive_float(raw: object, field_name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field_name} must be a positive number.")
    value = float(raw)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be a positive number.")
    return value


def _coerce_non_negative_float(raw: object, field_name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field_name} must be a non-negative number.")
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{field_name} must be a non-negative number.")
    return value


def _coerce_optional_non_negative_float(raw: object, field_name: str) -> float | None:
    if raw is None or raw == "":
        return None
    return _coerce_non_negative_float(raw, field_name)


def _coerce_fraction(raw: object, field_name: str) -> float:
    value = _coerce_positive_float(raw, field_name)
    if value >= 1.0:
        raise ValueError(f"{field_name} must be within (0, 1).")
    return value


__all__ = [
    "TasNormExperimentArtifactsConfig",
    "TasNormExperimentConfig",
    "TasNormExperimentGatesConfig",
    "TasNormExperimentRuntimeConfig",
    "load_tas_norm_experiment_config",
]
