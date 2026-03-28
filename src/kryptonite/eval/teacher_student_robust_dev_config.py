"""Typed config loader for teacher-vs-student robust-dev comparisons."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

CandidateFamily = Literal["campp", "eres2netv2", "teacher_peft"]
CandidateRole = Literal["student", "teacher"]

ALLOWED_CANDIDATE_FAMILIES: tuple[CandidateFamily, ...] = (
    "campp",
    "eres2netv2",
    "teacher_peft",
)
ALLOWED_CANDIDATE_ROLES: tuple[CandidateRole, ...] = ("student", "teacher")


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevSelectionConfig:
    clean_weight: float = 0.25
    corrupted_weight: float = 0.75
    eer_weight: float = 0.7
    min_dcf_weight: float = 0.3

    def __post_init__(self) -> None:
        for name in ("clean_weight", "corrupted_weight", "eer_weight", "min_dcf_weight"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.clean_weight + self.corrupted_weight <= 0.0:
            raise ValueError("At least one of clean_weight/corrupted_weight must be positive.")
        if self.eer_weight + self.min_dcf_weight <= 0.0:
            raise ValueError("At least one of eer_weight/min_dcf_weight must be positive.")

    def to_dict(self) -> dict[str, float]:
        return {
            "clean_weight": self.clean_weight,
            "corrupted_weight": self.corrupted_weight,
            "eer_weight": self.eer_weight,
            "min_dcf_weight": self.min_dcf_weight,
        }


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevCorruptedSuitesConfig:
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
class TeacherStudentRobustDevCandidateConfig:
    candidate_id: str
    label: str
    role: CandidateRole
    family: CandidateFamily
    run_root: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty.")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        if self.role not in ALLOWED_CANDIDATE_ROLES:
            raise ValueError(f"role must be one of {sorted(ALLOWED_CANDIDATE_ROLES)}.")
        if self.family not in ALLOWED_CANDIDATE_FAMILIES:
            raise ValueError(f"family must be one of {sorted(ALLOWED_CANDIDATE_FAMILIES)}.")
        if not self.run_root.strip():
            raise ValueError("run_root must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "role": self.role,
            "family": self.family,
            "run_root": self.run_root,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class TeacherStudentRobustDevConfig:
    title: str
    ticket_id: str
    report_id: str
    output_root: str
    device: str
    selection: TeacherStudentRobustDevSelectionConfig
    corrupted_suites: TeacherStudentRobustDevCorruptedSuitesConfig
    candidates: tuple[TeacherStudentRobustDevCandidateConfig, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty.")
        if not self.ticket_id.strip():
            raise ValueError("ticket_id must not be empty.")
        if not self.report_id.strip():
            raise ValueError("report_id must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")
        if not self.device.strip():
            raise ValueError("device must not be empty.")
        if not self.candidates:
            raise ValueError("At least one candidate must be declared.")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_id values must be unique.")
        teacher_count = sum(1 for candidate in self.candidates if candidate.role == "teacher")
        student_count = sum(1 for candidate in self.candidates if candidate.role == "student")
        if teacher_count != 1:
            raise ValueError("Exactly one teacher candidate must be declared.")
        if student_count < 1:
            raise ValueError("At least one student candidate must be declared.")

    @property
    def teacher_candidate_id(self) -> str:
        for candidate in self.candidates:
            if candidate.role == "teacher":
                return candidate.candidate_id
        raise ValueError("Teacher candidate is missing.")

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "ticket_id": self.ticket_id,
            "report_id": self.report_id,
            "output_root": self.output_root,
            "device": self.device,
            "selection": self.selection.to_dict(),
            "corrupted_suites": self.corrupted_suites.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "notes": list(self.notes),
        }


def load_teacher_student_robust_dev_config(
    *,
    config_path: Path | str,
) -> TeacherStudentRobustDevConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    candidates = tuple(
        _load_candidate_config(candidate)
        for candidate in _require_list(raw.get("candidate"), "candidate")
    )
    config = TeacherStudentRobustDevConfig(
        title=_require_string(raw.get("title"), "title"),
        ticket_id=_require_string(raw.get("ticket_id"), "ticket_id"),
        report_id=_require_string(raw.get("report_id"), "report_id"),
        output_root=_require_string(raw.get("output_root"), "output_root"),
        device=_optional_string(raw.get("device")) or "auto",
        selection=_load_selection_config(raw.get("selection")),
        corrupted_suites=_load_corrupted_suites_config(raw.get("corrupted_suites")),
        candidates=candidates,
        notes=_require_string_tuple(raw.get("notes"), "notes"),
    )
    return config


def _load_selection_config(raw: object) -> TeacherStudentRobustDevSelectionConfig:
    section = _require_mapping(raw, "selection")
    return TeacherStudentRobustDevSelectionConfig(
        clean_weight=_coerce_float(section.get("clean_weight"), "selection.clean_weight", 0.25),
        corrupted_weight=_coerce_float(
            section.get("corrupted_weight"),
            "selection.corrupted_weight",
            0.75,
        ),
        eer_weight=_coerce_float(section.get("eer_weight"), "selection.eer_weight", 0.7),
        min_dcf_weight=_coerce_float(
            section.get("min_dcf_weight"),
            "selection.min_dcf_weight",
            0.3,
        ),
    )


def _load_corrupted_suites_config(
    raw: object,
) -> TeacherStudentRobustDevCorruptedSuitesConfig:
    section = _require_mapping(raw, "corrupted_suites")
    return TeacherStudentRobustDevCorruptedSuitesConfig(
        catalog_path=_require_string(section.get("catalog_path"), "corrupted_suites.catalog_path"),
        suite_ids=_require_string_tuple(section.get("suite_ids"), "corrupted_suites.suite_ids"),
        run_clean_dev=_coerce_bool(
            section.get("run_clean_dev"),
            "corrupted_suites.run_clean_dev",
            True,
        ),
    )


def _load_candidate_config(raw: object) -> TeacherStudentRobustDevCandidateConfig:
    section = _require_mapping(raw, "candidate")
    return TeacherStudentRobustDevCandidateConfig(
        candidate_id=_require_string(section.get("candidate_id"), "candidate.candidate_id"),
        label=_require_string(section.get("label"), "candidate.label"),
        role=cast(CandidateRole, _require_string(section.get("role"), "candidate.role")),
        family=cast(CandidateFamily, _require_string(section.get("family"), "candidate.family")),
        run_root=_require_string(section.get("run_root"), "candidate.run_root"),
        notes=_require_string_tuple(section.get("notes"), "candidate.notes"),
    )


def _require_mapping(raw: object, field_name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a TOML table.")
    return cast(dict[str, object], dict(raw))


def _require_list(raw: object, field_name: str) -> list[object]:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a TOML array of tables.")
    return list(raw)


def _require_string(raw: object, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return raw.strip()


def _optional_string(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("Expected a string value.")
    normalized = raw.strip()
    return normalized or None


def _require_string_tuple(raw: object, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string.")
        values.append(item.strip())
    return tuple(values)


def _coerce_float(raw: object, field_name: str, default: float) -> float:
    if raw is None:
        return default
    if not isinstance(raw, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(raw)


def _coerce_bool(raw: object, field_name: str, default: bool) -> bool:
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return raw


__all__ = [
    "ALLOWED_CANDIDATE_FAMILIES",
    "ALLOWED_CANDIDATE_ROLES",
    "CandidateFamily",
    "CandidateRole",
    "TeacherStudentRobustDevCandidateConfig",
    "TeacherStudentRobustDevConfig",
    "TeacherStudentRobustDevCorruptedSuitesConfig",
    "TeacherStudentRobustDevSelectionConfig",
    "load_teacher_student_robust_dev_config",
]
