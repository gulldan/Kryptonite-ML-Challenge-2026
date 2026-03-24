"""Curriculum scheduler for clean/light/medium/heavy augmentation policies."""

from __future__ import annotations

import random
from dataclasses import dataclass

from kryptonite.config import AugmentationSchedulerConfig

from .models import (
    AUGMENTATION_FAMILY_ORDER,
    AUGMENTATION_INTENSITY_ORDER,
    AugmentationCatalog,
    AugmentationFamily,
    AugmentationIntensity,
    EpochAugmentationPlan,
    ScheduledAugmentation,
    ScheduledSampleRecipe,
    SchedulerStage,
)

_STAGE_FAMILY_MULTIPLIERS: dict[SchedulerStage, dict[AugmentationFamily, float]] = {
    "warmup": {
        "noise": 1.0,
        "reverb": 0.45,
        "distance": 0.35,
        "codec": 0.25,
        "silence": 0.65,
    },
    "ramp": {
        "noise": 1.0,
        "reverb": 0.85,
        "distance": 0.75,
        "codec": 0.60,
        "silence": 0.80,
    },
    "steady": {
        "noise": 1.0,
        "reverb": 1.0,
        "distance": 1.0,
        "codec": 0.90,
        "silence": 0.90,
    },
}
_SEVERITY_PREFERENCE: dict[AugmentationIntensity, tuple[str, ...]] = {
    "light": ("light", "medium", "heavy"),
    "medium": ("medium", "light", "heavy"),
    "heavy": ("heavy", "medium", "light"),
    "clean": (),
}


@dataclass(slots=True)
class AugmentationScheduler:
    config: AugmentationSchedulerConfig
    catalog: AugmentationCatalog
    total_epochs: int

    def __post_init__(self) -> None:
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")

    def plan_for_epoch(self, epoch_index: int) -> EpochAugmentationPlan:
        self._validate_epoch_index(epoch_index)
        if not self.config.enabled:
            return EpochAugmentationPlan(
                epoch_index=epoch_index,
                stage="steady",
                max_augmentations_per_sample=0,
                intensity_probabilities={
                    "clean": 1.0,
                    "light": 0.0,
                    "medium": 0.0,
                    "heavy": 0.0,
                },
                family_probabilities={},
            )

        stage = self._stage_for_epoch(epoch_index)
        progress = 0.0 if self.total_epochs == 1 else (epoch_index - 1) / (self.total_epochs - 1)
        intensity_probabilities = {
            intensity: _interpolate(
                start=getattr(self.config, f"{intensity}_probability_start"),
                end=getattr(self.config, f"{intensity}_probability_end"),
                progress=progress,
            )
            for intensity in AUGMENTATION_INTENSITY_ORDER
        }
        intensity_probabilities = _normalize_probabilities(intensity_probabilities)

        family_weights = {
            family: getattr(self.config.family_weights, family)
            * _STAGE_FAMILY_MULTIPLIERS[stage][family]
            for family in AUGMENTATION_FAMILY_ORDER
            if self.catalog.candidate_counts_by_family.get(family, 0) > 0
        }
        family_probabilities = _normalize_probabilities(family_weights)

        if stage == "warmup":
            max_augmentations = min(1, self.config.max_augmentations_per_sample)
        elif stage == "ramp":
            max_augmentations = min(2, self.config.max_augmentations_per_sample)
        else:
            max_augmentations = self.config.max_augmentations_per_sample

        return EpochAugmentationPlan(
            epoch_index=epoch_index,
            stage=stage,
            max_augmentations_per_sample=max_augmentations,
            intensity_probabilities=intensity_probabilities,
            family_probabilities=family_probabilities,
        )

    def sample_recipe(
        self,
        *,
        epoch_index: int,
        rng: random.Random,
    ) -> ScheduledSampleRecipe:
        plan = self.plan_for_epoch(epoch_index)
        intensity = _weighted_choice(
            plan.intensity_probabilities,
            rng=rng,
            fallback="clean",
        )
        intensity = intensity or "clean"
        if intensity == "clean" or not plan.family_probabilities:
            return ScheduledSampleRecipe(
                epoch_index=epoch_index,
                stage=plan.stage,
                intensity="clean",
                clean_sample=True,
                augmentations=(),
            )

        target_augmentation_count = _augmentation_count_for_intensity(
            intensity=intensity,
            max_augmentations_per_sample=plan.max_augmentations_per_sample,
            rng=rng,
        )
        selected_families: list[AugmentationFamily] = []
        scheduled: list[ScheduledAugmentation] = []
        for _ in range(target_augmentation_count):
            available_families = {
                family: probability
                for family, probability in plan.family_probabilities.items()
                if family not in selected_families
                and self._candidate_for_intensity(family=family, intensity=intensity, rng=rng)
                is not None
            }
            family = _weighted_choice(available_families, rng=rng, fallback=None)
            if family is None:
                break
            candidate = self._candidate_for_intensity(family=family, intensity=intensity, rng=rng)
            if candidate is None:
                continue
            selected_families.append(family)
            scheduled.append(
                ScheduledAugmentation(
                    family=candidate.family,
                    candidate_id=candidate.candidate_id,
                    label=candidate.label,
                    severity=candidate.severity,
                    metadata=dict(candidate.metadata),
                )
            )

        if not scheduled:
            return ScheduledSampleRecipe(
                epoch_index=epoch_index,
                stage=plan.stage,
                intensity="clean",
                clean_sample=True,
                augmentations=(),
            )
        return ScheduledSampleRecipe(
            epoch_index=epoch_index,
            stage=plan.stage,
            intensity=intensity,
            clean_sample=False,
            augmentations=tuple(scheduled),
        )

    def _stage_for_epoch(self, epoch_index: int) -> SchedulerStage:
        if epoch_index <= self.config.warmup_epochs:
            return "warmup"
        if epoch_index <= self.config.warmup_epochs + self.config.ramp_epochs:
            return "ramp"
        return "steady"

    def _candidate_for_intensity(
        self,
        *,
        family: AugmentationFamily,
        intensity: AugmentationIntensity,
        rng: random.Random,
    ):
        candidates = self.catalog.candidates_by_family.get(family, ())
        if intensity == "clean" or not candidates:
            return None

        for severity in _SEVERITY_PREFERENCE[intensity]:
            filtered = tuple(
                candidate for candidate in candidates if candidate.severity == severity
            )
            if filtered:
                return _weighted_sequence_choice(filtered, rng=rng)
        return None

    def _validate_epoch_index(self, epoch_index: int) -> None:
        if not 1 <= epoch_index <= self.total_epochs:
            raise ValueError(
                f"epoch_index must be within [1, {self.total_epochs}], got {epoch_index}."
            )


def _augmentation_count_for_intensity(
    *,
    intensity: AugmentationIntensity,
    max_augmentations_per_sample: int,
    rng: random.Random,
) -> int:
    if max_augmentations_per_sample <= 1:
        return 1
    if intensity == "light":
        return 1
    if intensity == "medium":
        return 1 if rng.random() < 0.65 else min(2, max_augmentations_per_sample)
    return min(2, max_augmentations_per_sample)


def _interpolate(*, start: float, end: float, progress: float) -> float:
    return round(start + ((end - start) * progress), 6)


def _normalize_probabilities[T: str](values: dict[T, float]) -> dict[T, float]:
    total = sum(value for value in values.values() if value > 0.0)
    if total <= 0.0:
        return {key: 0.0 for key in values}
    return {key: round(max(value, 0.0) / total, 6) for key, value in values.items()}


def _weighted_choice[T: str](
    values: dict[T, float],
    *,
    rng: random.Random,
    fallback: T | None,
) -> T | None:
    population = [key for key, weight in values.items() if weight > 0.0]
    if not population:
        return fallback
    weights = [values[key] for key in population]
    return rng.choices(population, weights=weights, k=1)[0]


def _weighted_sequence_choice(sequence, *, rng: random.Random):
    weights = [candidate.weight for candidate in sequence]
    return rng.choices(list(sequence), weights=weights, k=1)[0]
