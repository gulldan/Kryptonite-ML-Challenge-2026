"""Datamodels for augmentation scheduling and coverage reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from kryptonite.config import AugmentationSchedulerConfig, SilenceAugmentationConfig

AugmentationFamily = Literal["noise", "reverb", "distance", "codec", "silence"]
AugmentationIntensity = Literal["clean", "light", "medium", "heavy"]
SchedulerStage = Literal["warmup", "ramp", "steady"]

AUGMENTATION_FAMILY_ORDER: tuple[AugmentationFamily, ...] = (
    "noise",
    "reverb",
    "distance",
    "codec",
    "silence",
)
AUGMENTATION_INTENSITY_ORDER: tuple[AugmentationIntensity, ...] = (
    "clean",
    "light",
    "medium",
    "heavy",
)
NON_CLEAN_INTENSITY_ORDER: tuple[AugmentationIntensity, ...] = ("light", "medium", "heavy")
SCHEDULER_STAGE_ORDER: tuple[SchedulerStage, ...] = ("warmup", "ramp", "steady")


@dataclass(frozen=True, slots=True)
class BankManifestPaths:
    noise_manifest_path: str | None
    room_config_manifest_path: str | None
    distance_manifest_path: str | None
    codec_manifest_path: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "noise_manifest_path": self.noise_manifest_path,
            "room_config_manifest_path": self.room_config_manifest_path,
            "distance_manifest_path": self.distance_manifest_path,
            "codec_manifest_path": self.codec_manifest_path,
        }


@dataclass(frozen=True, slots=True)
class AugmentationCandidate:
    family: AugmentationFamily
    candidate_id: str
    label: str
    severity: Literal["light", "medium", "heavy"]
    weight: float
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise ValueError("Augmentation candidate weight must be positive.")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "severity": self.severity,
            "weight": self.weight,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AugmentationCatalog:
    candidates_by_family: dict[AugmentationFamily, tuple[AugmentationCandidate, ...]]

    @property
    def available_families(self) -> tuple[AugmentationFamily, ...]:
        return tuple(
            family
            for family in AUGMENTATION_FAMILY_ORDER
            if len(self.candidates_by_family.get(family, ())) > 0
        )

    @property
    def missing_families(self) -> tuple[AugmentationFamily, ...]:
        return tuple(
            family
            for family in AUGMENTATION_FAMILY_ORDER
            if len(self.candidates_by_family.get(family, ())) == 0
        )

    @property
    def candidate_counts_by_family(self) -> dict[str, int]:
        return {
            family: len(self.candidates_by_family.get(family, ()))
            for family in AUGMENTATION_FAMILY_ORDER
        }

    def to_dict(self, *, include_candidates: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "available_families": list(self.available_families),
            "missing_families": list(self.missing_families),
            "candidate_counts_by_family": dict(self.candidate_counts_by_family),
        }
        if include_candidates:
            payload["candidates_by_family"] = {
                family: [
                    candidate.to_dict() for candidate in self.candidates_by_family.get(family, ())
                ]
                for family in AUGMENTATION_FAMILY_ORDER
            }
        return payload


@dataclass(frozen=True, slots=True)
class EpochAugmentationPlan:
    epoch_index: int
    stage: SchedulerStage
    max_augmentations_per_sample: int
    intensity_probabilities: dict[AugmentationIntensity, float]
    family_probabilities: dict[AugmentationFamily, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch_index": self.epoch_index,
            "stage": self.stage,
            "max_augmentations_per_sample": self.max_augmentations_per_sample,
            "intensity_probabilities": {
                intensity: self.intensity_probabilities[intensity]
                for intensity in AUGMENTATION_INTENSITY_ORDER
            },
            "family_probabilities": {
                family: self.family_probabilities.get(family, 0.0)
                for family in AUGMENTATION_FAMILY_ORDER
            },
        }


@dataclass(frozen=True, slots=True)
class ScheduledAugmentation:
    family: AugmentationFamily
    candidate_id: str
    label: str
    severity: Literal["light", "medium", "heavy"]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScheduledSampleRecipe:
    epoch_index: int
    stage: SchedulerStage
    intensity: AugmentationIntensity
    clean_sample: bool
    augmentations: tuple[ScheduledAugmentation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch_index": self.epoch_index,
            "stage": self.stage,
            "intensity": self.intensity,
            "clean_sample": self.clean_sample,
            "augmentations": [augmentation.to_dict() for augmentation in self.augmentations],
        }


@dataclass(frozen=True, slots=True)
class EpochCoverageSummary:
    epoch_index: int
    stage: SchedulerStage
    sample_count: int
    clean_sample_count: int
    max_augmentations_per_sample: int
    intensity_probabilities: dict[AugmentationIntensity, float]
    empirical_intensity_ratios: dict[AugmentationIntensity, float]
    family_counts: dict[str, int]
    severity_counts: dict[str, int]
    top_augmentations: tuple[str, ...]

    @property
    def family_coverage(self) -> tuple[str, ...]:
        return tuple(
            family for family in AUGMENTATION_FAMILY_ORDER if self.family_counts.get(family, 0) > 0
        )

    @property
    def severity_coverage(self) -> tuple[str, ...]:
        return tuple(
            severity
            for severity in NON_CLEAN_INTENSITY_ORDER
            if self.severity_counts.get(severity, 0) > 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch_index": self.epoch_index,
            "stage": self.stage,
            "sample_count": self.sample_count,
            "clean_sample_count": self.clean_sample_count,
            "max_augmentations_per_sample": self.max_augmentations_per_sample,
            "intensity_probabilities": {
                intensity: self.intensity_probabilities[intensity]
                for intensity in AUGMENTATION_INTENSITY_ORDER
            },
            "empirical_intensity_ratios": {
                intensity: self.empirical_intensity_ratios[intensity]
                for intensity in AUGMENTATION_INTENSITY_ORDER
            },
            "family_counts": {
                family: self.family_counts.get(family, 0) for family in AUGMENTATION_FAMILY_ORDER
            },
            "family_coverage": list(self.family_coverage),
            "severity_counts": {
                severity: self.severity_counts.get(severity, 0)
                for severity in NON_CLEAN_INTENSITY_ORDER
            },
            "severity_coverage": list(self.severity_coverage),
            "top_augmentations": list(self.top_augmentations),
        }


@dataclass(frozen=True, slots=True)
class AugmentationSchedulerSummary:
    total_epochs: int
    samples_per_epoch: int
    candidate_counts_by_family: dict[str, int]
    missing_families: tuple[AugmentationFamily, ...]
    stage_epoch_counts: dict[str, int]
    overall_intensity_counts: dict[str, int]
    overall_family_counts: dict[str, int]
    overall_severity_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_epochs": self.total_epochs,
            "samples_per_epoch": self.samples_per_epoch,
            "candidate_counts_by_family": {
                family: self.candidate_counts_by_family.get(family, 0)
                for family in AUGMENTATION_FAMILY_ORDER
            },
            "missing_families": list(self.missing_families),
            "stage_epoch_counts": {
                stage: self.stage_epoch_counts.get(stage, 0) for stage in SCHEDULER_STAGE_ORDER
            },
            "overall_intensity_counts": {
                intensity: self.overall_intensity_counts.get(intensity, 0)
                for intensity in AUGMENTATION_INTENSITY_ORDER
            },
            "overall_family_counts": {
                family: self.overall_family_counts.get(family, 0)
                for family in AUGMENTATION_FAMILY_ORDER
            },
            "overall_severity_counts": {
                severity: self.overall_severity_counts.get(severity, 0)
                for severity in NON_CLEAN_INTENSITY_ORDER
            },
        }


@dataclass(frozen=True, slots=True)
class AugmentationSchedulerReport:
    generated_at: str
    project_root: str
    seed: int
    manifest_paths: BankManifestPaths
    scheduler_config: AugmentationSchedulerConfig
    silence_config: SilenceAugmentationConfig
    catalog: AugmentationCatalog
    epochs: tuple[EpochCoverageSummary, ...]
    summary: AugmentationSchedulerSummary

    def to_dict(self, *, include_candidates: bool = False) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "seed": self.seed,
            "manifest_paths": self.manifest_paths.to_dict(),
            "scheduler_config": asdict(self.scheduler_config),
            "silence_config": asdict(self.silence_config),
            "catalog": self.catalog.to_dict(include_candidates=include_candidates),
            "epochs": [epoch.to_dict() for epoch in self.epochs],
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenAugmentationSchedulerArtifacts:
    output_root: str
    json_path: str
    markdown_path: str
    epochs_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
            "epochs_path": self.epochs_path,
        }
