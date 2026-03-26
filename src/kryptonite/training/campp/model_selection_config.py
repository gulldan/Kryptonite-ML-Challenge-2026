"""Typed config loader for CAM++ stage-3 model selection and checkpoint averaging."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import _coerce_int_list


@dataclass(frozen=True, slots=True)
class CheckpointAveragingConfig:
    """Optional post-shortlist checkpoint averaging variants."""

    enabled: bool = True
    candidate_counts: tuple[int, ...] = (2, 3)
    weights: str = "uniform"

    def __post_init__(self) -> None:
        if self.weights != "uniform":
            raise ValueError("averaging.weights must be 'uniform'.")
        if self.enabled and not self.candidate_counts:
            raise ValueError(
                "averaging.candidate_counts must contain at least one entry "
                "when averaging is enabled."
            )
        seen: set[int] = set()
        previous = 1
        for count in self.candidate_counts:
            if count < 2:
                raise ValueError("averaging.candidate_counts values must be at least 2.")
            if count in seen:
                raise ValueError("averaging.candidate_counts must be unique.")
            if count <= previous:
                raise ValueError("averaging.candidate_counts must be strictly increasing.")
            seen.add(count)
            previous = count

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_counts"] = list(self.candidate_counts)
        return payload


@dataclass(frozen=True, slots=True)
class CAMPPlusModelSelectionConfig:
    """Full config for the CAM++ stage-3 final-candidate selector."""

    shortlist_report_path: str
    output_root: str
    averaging: CheckpointAveragingConfig

    def __post_init__(self) -> None:
        if not self.shortlist_report_path.strip():
            raise ValueError("shortlist_report_path must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "shortlist_report_path": self.shortlist_report_path,
            "output_root": self.output_root,
            "averaging": self.averaging.to_dict(),
        }


def load_campp_model_selection_config(
    *,
    config_path: Path | str,
) -> CAMPPlusModelSelectionConfig:
    """Parse the checked-in CAM++ model-selection TOML config."""

    raw = tomllib.loads(Path(config_path).read_text())
    averaging_section = raw.get("averaging", {})
    if not isinstance(averaging_section, dict):
        raise ValueError("[averaging] must be a TOML table when provided.")

    enabled = averaging_section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("averaging.enabled must be a boolean.")

    return CAMPPlusModelSelectionConfig(
        shortlist_report_path=str(raw.get("shortlist_report_path", "")).strip(),
        output_root=str(raw.get("output_root", "artifacts/model-selection/campp-stage3")).strip(),
        averaging=CheckpointAveragingConfig(
            enabled=enabled,
            candidate_counts=tuple(
                _coerce_int_list(
                    averaging_section.get("candidate_counts", [2, 3]),
                    "averaging.candidate_counts",
                )
            ),
            weights=str(averaging_section.get("weights", "uniform")).strip(),
        ),
    )


__all__ = [
    "CAMPPlusModelSelectionConfig",
    "CheckpointAveragingConfig",
    "load_campp_model_selection_config",
]
