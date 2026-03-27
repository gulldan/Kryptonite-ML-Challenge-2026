"""Typed config loader for the final release benchmark pack."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class FinalBenchmarkCandidateConfig:
    """One frozen candidate included in the final benchmark pack."""

    candidate_id: str
    label: str
    family: str
    verification_report_path: str
    threshold_calibration_path: str | None
    stress_report_path: str
    model_bundle_metadata_path: str
    export_boundary_path: str | None
    config_paths: tuple[str, ...]
    supporting_paths: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate.candidate_id must not be empty.")
        if not self.label.strip():
            raise ValueError(f"candidate[{self.candidate_id}].label must not be empty.")
        if not self.family.strip():
            raise ValueError(f"candidate[{self.candidate_id}].family must not be empty.")
        if not self.verification_report_path.strip():
            raise ValueError(
                f"candidate[{self.candidate_id}].verification_report_path must not be empty."
            )
        if not self.stress_report_path.strip():
            raise ValueError(
                f"candidate[{self.candidate_id}].stress_report_path must not be empty."
            )
        if not self.model_bundle_metadata_path.strip():
            raise ValueError(
                f"candidate[{self.candidate_id}].model_bundle_metadata_path must not be empty."
            )
        if not self.config_paths:
            raise ValueError(
                f"candidate[{self.candidate_id}] must include at least one config_path."
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["config_paths"] = list(self.config_paths)
        payload["supporting_paths"] = list(self.supporting_paths)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class FinalBenchmarkPackConfig:
    """Top-level config for the self-contained release benchmark pack."""

    title: str
    summary: str
    output_root: str
    candidates: tuple[FinalBenchmarkCandidateConfig, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")
        if len(self.candidates) < 2:
            raise ValueError("final benchmark pack requires at least two candidates.")
        seen: set[str] = set()
        for candidate in self.candidates:
            if candidate.candidate_id in seen:
                raise ValueError(
                    f"candidate ids must be unique; duplicate {candidate.candidate_id!r} found."
                )
            seen.add(candidate.candidate_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "summary": self.summary,
            "output_root": self.output_root,
            "notes": list(self.notes),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def load_final_benchmark_pack_config(
    *,
    config_path: Path | str,
) -> FinalBenchmarkPackConfig:
    """Parse the checked-in TOML config for the final benchmark pack."""

    raw = tomllib.loads(Path(config_path).read_text())
    candidate_sections = raw.get("candidate", [])
    if not isinstance(candidate_sections, list):
        raise ValueError("[[candidate]] entries must be provided as TOML array-of-tables.")
    if not candidate_sections:
        raise ValueError("At least one [[candidate]] entry is required.")

    return FinalBenchmarkPackConfig(
        title=str(raw.get("title", "")).strip(),
        summary=str(raw.get("summary", "")).strip(),
        output_root=str(raw.get("output_root", "artifacts/benchmark-pack/final")).strip(),
        candidates=tuple(
            _parse_candidate(index, section) for index, section in enumerate(candidate_sections)
        ),
        notes=tuple(_coerce_string_list(raw.get("notes", []), "notes")),
    )


def _parse_candidate(index: int, raw: object) -> FinalBenchmarkCandidateConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"candidate[{index}] must be a TOML table.")
    section = cast(dict[str, object], raw)
    candidate_id = str(section.get("candidate_id", "")).strip()
    threshold_path = str(section.get("threshold_calibration_path", "")).strip() or None
    export_boundary_path = str(section.get("export_boundary_path", "")).strip() or None
    return FinalBenchmarkCandidateConfig(
        candidate_id=candidate_id,
        label=str(section.get("label", "")).strip(),
        family=str(section.get("family", "")).strip(),
        verification_report_path=str(section.get("verification_report_path", "")).strip(),
        threshold_calibration_path=threshold_path,
        stress_report_path=str(section.get("stress_report_path", "")).strip(),
        model_bundle_metadata_path=str(section.get("model_bundle_metadata_path", "")).strip(),
        export_boundary_path=export_boundary_path,
        config_paths=tuple(
            _coerce_string_list(
                section.get("config_paths", []),
                f"candidate[{candidate_id}].config_paths",
            )
        ),
        supporting_paths=tuple(
            _coerce_string_list(
                section.get("supporting_paths", []),
                f"candidate[{candidate_id}].supporting_paths",
            )
        ),
        notes=tuple(
            _coerce_string_list(section.get("notes", []), f"candidate[{candidate_id}].notes")
        ),
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
    "FinalBenchmarkCandidateConfig",
    "FinalBenchmarkPackConfig",
    "load_final_benchmark_pack_config",
]
