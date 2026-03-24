"""Datamodels for frozen corrupted dev-suite generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SuiteFamily = Literal["noise", "reverb", "codec", "distance", "silence"]
SuiteSeverity = Literal["light", "medium", "heavy"]

ALLOWED_SUITE_FAMILIES: tuple[SuiteFamily, ...] = (
    "noise",
    "reverb",
    "codec",
    "distance",
    "silence",
)
ALLOWED_SUITE_SEVERITIES: tuple[SuiteSeverity, ...] = ("light", "medium", "heavy")


@dataclass(frozen=True, slots=True)
class SeverityWeights:
    light: float = 1.0
    medium: float = 1.0
    heavy: float = 1.0

    def __post_init__(self) -> None:
        for name in ALLOWED_SUITE_SEVERITIES:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} severity weight must be non-negative.")
        if self.total <= 0.0:
            raise ValueError("At least one severity weight must be positive.")

    @property
    def total(self) -> float:
        return round(self.light + self.medium + self.heavy, 6)

    def to_dict(self) -> dict[str, float]:
        return {
            "light": self.light,
            "medium": self.medium,
            "heavy": self.heavy,
        }


@dataclass(frozen=True, slots=True)
class ReverbDirectWeights:
    high: float = 1.0
    medium: float = 1.0
    low: float = 1.0

    def __post_init__(self) -> None:
        for name in ("high", "medium", "low"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} direct-condition weight must be non-negative.")
        if self.total <= 0.0:
            raise ValueError("At least one direct-condition weight must be positive.")

    @property
    def total(self) -> float:
        return round(self.high + self.medium + self.low, 6)

    def to_dict(self) -> dict[str, float]:
        return {
            "high": self.high,
            "medium": self.medium,
            "low": self.low,
        }


@dataclass(frozen=True, slots=True)
class DistanceFieldWeights:
    near: float = 1.0
    mid: float = 1.0
    far: float = 1.0

    def __post_init__(self) -> None:
        for name in ("near", "mid", "far"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} distance-field weight must be non-negative.")
        if self.total <= 0.0:
            raise ValueError("At least one distance-field weight must be positive.")

    @property
    def total(self) -> float:
        return round(self.near + self.mid + self.far, 6)

    def to_dict(self) -> dict[str, float]:
        return {
            "near": self.near,
            "mid": self.mid,
            "far": self.far,
        }


@dataclass(frozen=True, slots=True)
class CorruptedDevSuiteSpec:
    suite_id: str
    family: SuiteFamily
    description: str
    severity_weights: SeverityWeights
    codec_families: tuple[str, ...] = ()
    reverb_direct_weights: ReverbDirectWeights | None = None
    distance_field_weights: DistanceFieldWeights | None = None

    def __post_init__(self) -> None:
        if not self.suite_id.strip():
            raise ValueError("suite_id must be non-empty.")
        if not self.description.strip():
            raise ValueError("description must be non-empty.")
        if self.family not in ALLOWED_SUITE_FAMILIES:
            allowed = ", ".join(ALLOWED_SUITE_FAMILIES)
            raise ValueError(f"family must be one of: {allowed}")
        if self.family == "codec":
            if not self.codec_families:
                raise ValueError("codec suites must define at least one codec family.")
        elif self.codec_families:
            raise ValueError("codec_families may only be set for codec suites.")
        if self.family == "reverb":
            if self.reverb_direct_weights is None:
                raise ValueError("reverb suites must define reverb_direct_weights.")
        elif self.reverb_direct_weights is not None:
            raise ValueError("reverb_direct_weights may only be set for reverb suites.")
        if self.family == "distance":
            if self.distance_field_weights is None:
                raise ValueError("distance suites must define distance_field_weights.")
        elif self.distance_field_weights is not None:
            raise ValueError("distance_field_weights may only be set for distance suites.")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "suite_id": self.suite_id,
            "family": self.family,
            "description": self.description,
            "severity_weights": self.severity_weights.to_dict(),
        }
        if self.codec_families:
            payload["codec_families"] = list(self.codec_families)
        if self.reverb_direct_weights is not None:
            payload["reverb_direct_weights"] = self.reverb_direct_weights.to_dict()
        if self.distance_field_weights is not None:
            payload["distance_field_weights"] = self.distance_field_weights.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class CorruptedDevSuitesPlan:
    output_root: str
    source_manifest_path: str
    trial_manifest_paths: tuple[str, ...]
    seed: int
    suites: tuple[CorruptedDevSuiteSpec, ...]

    def __post_init__(self) -> None:
        if not self.output_root.strip():
            raise ValueError("output_root must be non-empty.")
        if not self.source_manifest_path.strip():
            raise ValueError("source_manifest_path must be non-empty.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not self.suites:
            raise ValueError("At least one suite must be defined.")
        suite_ids = [suite.suite_id for suite in self.suites]
        if len(set(suite_ids)) != len(suite_ids):
            raise ValueError("Suite ids must be unique.")

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "source_manifest_path": self.source_manifest_path,
            "trial_manifest_paths": list(self.trial_manifest_paths),
            "seed": self.seed,
            "suites": [suite.to_dict() for suite in self.suites],
        }


@dataclass(frozen=True, slots=True)
class BuiltCorruptedSuite:
    suite_id: str
    family: SuiteFamily
    description: str
    seed: int
    utterance_count: int
    speaker_count: int
    total_duration_seconds: float
    severity_counts: dict[str, int]
    candidate_counts: dict[str, int]
    output_root: str
    audio_root: str
    manifest_path: str
    inventory_path: str
    suite_summary_json_path: str
    suite_summary_markdown_path: str
    trial_manifest_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "family": self.family,
            "description": self.description,
            "seed": self.seed,
            "utterance_count": self.utterance_count,
            "speaker_count": self.speaker_count,
            "total_duration_seconds": self.total_duration_seconds,
            "severity_counts": dict(self.severity_counts),
            "candidate_counts": dict(self.candidate_counts),
            "output_root": self.output_root,
            "audio_root": self.audio_root,
            "manifest_path": self.manifest_path,
            "inventory_path": self.inventory_path,
            "suite_summary_json_path": self.suite_summary_json_path,
            "suite_summary_markdown_path": self.suite_summary_markdown_path,
            "trial_manifest_paths": list(self.trial_manifest_paths),
        }


@dataclass(frozen=True, slots=True)
class CorruptedDevSuitesReport:
    generated_at: str
    project_root: str
    plan_path: str | None
    source_manifest_path: str
    source_trial_manifest_paths: tuple[str, ...]
    output_root: str
    seed: int
    suites: tuple[BuiltCorruptedSuite, ...]
    catalog_json_path: str
    catalog_markdown_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "plan_path": self.plan_path,
            "source_manifest_path": self.source_manifest_path,
            "source_trial_manifest_paths": list(self.source_trial_manifest_paths),
            "output_root": self.output_root,
            "seed": self.seed,
            "suites": [suite.to_dict() for suite in self.suites],
            "catalog_json_path": self.catalog_json_path,
            "catalog_markdown_path": self.catalog_markdown_path,
        }
