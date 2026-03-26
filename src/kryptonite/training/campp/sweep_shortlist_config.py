"""Typed config loader for the CAM++ stage-3 hyperparameter sweep shortlist."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from .config import _coerce_string_list


@dataclass(frozen=True, slots=True)
class SweepSelectionConfig:
    """Primary ranking weights for the shortlist winner."""

    clean_weight: float = 0.25
    corrupted_weight: float = 0.75
    eer_weight: float = 0.7
    min_dcf_weight: float = 0.3

    def __post_init__(self) -> None:
        for field_name in ("clean_weight", "corrupted_weight", "eer_weight", "min_dcf_weight"):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.clean_weight + self.corrupted_weight <= 0.0:
            raise ValueError("At least one of clean_weight/corrupted_weight must be positive.")
        if self.eer_weight + self.min_dcf_weight <= 0.0:
            raise ValueError("At least one of eer_weight/min_dcf_weight must be positive.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SweepBudgetConfig:
    """Explicit sweep-budget declaration checked into the repo."""

    max_candidates: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError("budget.max_candidates must be at least 1.")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_candidates": self.max_candidates,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class CorruptedSuitesConfig:
    """Frozen robust-dev suites used for shortlist ranking."""

    catalog_path: str
    suite_ids: tuple[str, ...] = ()
    run_clean_dev: bool = True

    def __post_init__(self) -> None:
        if not self.catalog_path.strip():
            raise ValueError("corrupted_suites.catalog_path must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "suite_ids": list(self.suite_ids),
            "run_clean_dev": self.run_clean_dev,
        }


@dataclass(frozen=True, slots=True)
class SweepMarginScheduleOverride:
    """Optional stage-3 margin-schedule overrides for one candidate."""

    enabled: bool | None = None
    start_margin: float | None = None
    end_margin: float | None = None
    ramp_epochs: int | None = None

    def __post_init__(self) -> None:
        if self.start_margin is not None and self.start_margin < 0.0:
            raise ValueError("margin_schedule.start_margin must be non-negative when provided.")
        if self.end_margin is not None and self.end_margin < 0.0:
            raise ValueError("margin_schedule.end_margin must be non-negative when provided.")
        if (
            self.start_margin is not None
            and self.end_margin is not None
            and self.start_margin > self.end_margin
        ):
            raise ValueError("margin_schedule.start_margin must not exceed end_margin.")
        if self.ramp_epochs is not None and self.ramp_epochs < 0:
            raise ValueError("margin_schedule.ramp_epochs must be non-negative when provided.")

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class SweepCropCurriculumOverride:
    """Optional stage-3 crop-curriculum overrides for one candidate."""

    enabled: bool | None = None
    start_crop_seconds: float | None = None
    end_crop_seconds: float | None = None
    curriculum_epochs: int | None = None

    def __post_init__(self) -> None:
        if self.start_crop_seconds is not None and self.start_crop_seconds <= 0.0:
            raise ValueError("crop_curriculum.start_crop_seconds must be positive when provided.")
        if self.end_crop_seconds is not None and self.end_crop_seconds <= 0.0:
            raise ValueError("crop_curriculum.end_crop_seconds must be positive when provided.")
        if (
            self.start_crop_seconds is not None
            and self.end_crop_seconds is not None
            and self.start_crop_seconds > self.end_crop_seconds
        ):
            raise ValueError("crop_curriculum.start_crop_seconds must not exceed end_crop_seconds.")
        if self.curriculum_epochs is not None and self.curriculum_epochs < 0:
            raise ValueError(
                "crop_curriculum.curriculum_epochs must be non-negative when provided."
            )

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class SweepCandidateConfig:
    """One shortlist candidate expressed as repo-native overrides."""

    candidate_id: str
    description: str
    project_overrides: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    margin_schedule: SweepMarginScheduleOverride = SweepMarginScheduleOverride()
    crop_curriculum: SweepCropCurriculumOverride = SweepCropCurriculumOverride()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty.")
        if not self.description.strip():
            raise ValueError("candidate description must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "description": self.description,
            "project_overrides": list(self.project_overrides),
            "notes": list(self.notes),
            "margin_schedule": self.margin_schedule.to_dict(),
            "crop_curriculum": self.crop_curriculum.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CAMPPlusSweepShortlistConfig:
    """Full config for the CAM++ stage-3 shortlist runner."""

    base_stage3_config_path: str
    output_root: str
    project_overrides: tuple[str, ...]
    selection: SweepSelectionConfig
    budget: SweepBudgetConfig
    corrupted_suites: CorruptedSuitesConfig
    candidates: tuple[SweepCandidateConfig, ...]

    def __post_init__(self) -> None:
        if not self.base_stage3_config_path.strip():
            raise ValueError("base_stage3_config_path must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")
        if not self.candidates:
            raise ValueError("At least one shortlist candidate must be defined.")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Shortlist candidate ids must be unique.")
        if len(self.candidates) > self.budget.max_candidates:
            raise ValueError(
                "Configured shortlist exceeds budget.max_candidates: "
                f"{len(self.candidates)} > {self.budget.max_candidates}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "base_stage3_config_path": self.base_stage3_config_path,
            "output_root": self.output_root,
            "project_overrides": list(self.project_overrides),
            "selection": self.selection.to_dict(),
            "budget": self.budget.to_dict(),
            "corrupted_suites": self.corrupted_suites.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def load_campp_sweep_shortlist_config(*, config_path: Path | str) -> CAMPPlusSweepShortlistConfig:
    """Parse the checked-in sweep shortlist TOML config."""

    raw = tomllib.loads(Path(config_path).read_text())
    selection = _load_selection_section(raw.get("selection"))
    budget = _load_budget_section(raw.get("budget"))
    corrupted_suites = _load_corrupted_suites_section(raw.get("corrupted_suites"))
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("At least one [[candidates]] table is required.")

    return CAMPPlusSweepShortlistConfig(
        base_stage3_config_path=str(
            raw.get("base_stage3_config_path", "configs/training/campp-stage3.toml")
        ),
        output_root=str(raw.get("output_root", "artifacts/sweeps/campp-stage3-shortlist")),
        project_overrides=tuple(_coerce_string_list(raw.get("project_overrides"))),
        selection=selection,
        budget=budget,
        corrupted_suites=corrupted_suites,
        candidates=tuple(_load_candidate_section(section) for section in candidates_raw),
    )


def _load_selection_section(section: object) -> SweepSelectionConfig:
    if section is None:
        return SweepSelectionConfig()
    section_dict = _require_table(section, name="selection")
    return SweepSelectionConfig(
        clean_weight=_coerce_float_value(
            section_dict.get("clean_weight", 0.25),
            field_name="selection.clean_weight",
        ),
        corrupted_weight=_coerce_float_value(
            section_dict.get("corrupted_weight", 0.75),
            field_name="selection.corrupted_weight",
        ),
        eer_weight=_coerce_float_value(
            section_dict.get("eer_weight", 0.7),
            field_name="selection.eer_weight",
        ),
        min_dcf_weight=_coerce_float_value(
            section_dict.get("min_dcf_weight", 0.3),
            field_name="selection.min_dcf_weight",
        ),
    )


def _load_budget_section(section: object) -> SweepBudgetConfig:
    section_dict = _require_table(section, name="budget")
    notes = tuple(_coerce_string_list(section_dict.get("notes")))
    max_candidates = section_dict.get("max_candidates")
    if not isinstance(max_candidates, int):
        raise ValueError("budget.max_candidates must be an integer.")
    return SweepBudgetConfig(max_candidates=max_candidates, notes=notes)


def _load_corrupted_suites_section(section: object) -> CorruptedSuitesConfig:
    section_dict = _require_table(section, name="corrupted_suites")
    catalog_path = section_dict.get("catalog_path")
    if not isinstance(catalog_path, str):
        raise ValueError("corrupted_suites.catalog_path must be a string.")
    run_clean_dev = section_dict.get("run_clean_dev", True)
    if not isinstance(run_clean_dev, bool):
        raise ValueError("corrupted_suites.run_clean_dev must be a boolean.")
    return CorruptedSuitesConfig(
        catalog_path=catalog_path,
        suite_ids=tuple(_coerce_string_list(section_dict.get("suite_ids"))),
        run_clean_dev=run_clean_dev,
    )


def _load_candidate_section(section: object) -> SweepCandidateConfig:
    section_dict = _require_table(section, name="candidates[]")
    candidate_id = section_dict.get("candidate_id")
    description = section_dict.get("description")
    if not isinstance(candidate_id, str):
        raise ValueError("candidates.candidate_id must be a string.")
    if not isinstance(description, str):
        raise ValueError("candidates.description must be a string.")
    return SweepCandidateConfig(
        candidate_id=candidate_id.strip(),
        description=description.strip(),
        project_overrides=tuple(_coerce_string_list(section_dict.get("project_overrides"))),
        notes=tuple(_coerce_string_list(section_dict.get("notes"))),
        margin_schedule=_load_margin_schedule_override(section_dict.get("margin_schedule")),
        crop_curriculum=_load_crop_curriculum_override(section_dict.get("crop_curriculum")),
    )


def _load_margin_schedule_override(section: object) -> SweepMarginScheduleOverride:
    if section is None:
        return SweepMarginScheduleOverride()
    section_dict = _require_table(section, name="candidates.margin_schedule")
    return SweepMarginScheduleOverride(
        enabled=_coerce_optional_bool(
            section_dict.get("enabled"),
            field_name="candidates.margin_schedule.enabled",
        ),
        start_margin=(
            None
            if section_dict.get("start_margin") is None
            else _coerce_float_value(
                section_dict.get("start_margin"),
                field_name="candidates.margin_schedule.start_margin",
            )
        ),
        end_margin=(
            None
            if section_dict.get("end_margin") is None
            else _coerce_float_value(
                section_dict.get("end_margin"),
                field_name="candidates.margin_schedule.end_margin",
            )
        ),
        ramp_epochs=(
            None
            if section_dict.get("ramp_epochs") is None
            else _coerce_int_value(
                section_dict.get("ramp_epochs"),
                field_name="candidates.margin_schedule.ramp_epochs",
            )
        ),
    )


def _load_crop_curriculum_override(section: object) -> SweepCropCurriculumOverride:
    if section is None:
        return SweepCropCurriculumOverride()
    section_dict = _require_table(section, name="candidates.crop_curriculum")
    return SweepCropCurriculumOverride(
        enabled=_coerce_optional_bool(
            section_dict.get("enabled"),
            field_name="candidates.crop_curriculum.enabled",
        ),
        start_crop_seconds=(
            None
            if section_dict.get("start_crop_seconds") is None
            else _coerce_float_value(
                section_dict.get("start_crop_seconds"),
                field_name="candidates.crop_curriculum.start_crop_seconds",
            )
        ),
        end_crop_seconds=(
            None
            if section_dict.get("end_crop_seconds") is None
            else _coerce_float_value(
                section_dict.get("end_crop_seconds"),
                field_name="candidates.crop_curriculum.end_crop_seconds",
            )
        ),
        curriculum_epochs=(
            None
            if section_dict.get("curriculum_epochs") is None
            else _coerce_int_value(
                section_dict.get("curriculum_epochs"),
                field_name="candidates.crop_curriculum.curriculum_epochs",
            )
        ),
    )


def _require_table(section: object, *, name: str) -> dict[str, object]:
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a table.")
    return cast("dict[str, object]", dict(section))


def _coerce_float_value(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _coerce_int_value(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _coerce_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean when provided.")
    return value


__all__ = [
    "CAMPPlusSweepShortlistConfig",
    "CorruptedSuitesConfig",
    "SweepBudgetConfig",
    "SweepCandidateConfig",
    "SweepCropCurriculumOverride",
    "SweepMarginScheduleOverride",
    "SweepSelectionConfig",
    "load_campp_sweep_shortlist_config",
]
